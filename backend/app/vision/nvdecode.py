"""
NVDEC decode, straight into CUDA memory.

WHAT THIS REPLACES
`cv2.VideoCapture` decodes on the CPU. For twenty 1080p cameras that is most
of a core each, spent turning H.264 into pixels that are then copied to the GPU
so a model can look at them. The GPU has a dedicated decoder block sitting idle
the whole time, on a chip already in the machine.

This decodes on that block instead. A frame arrives as a CUDA tensor that never
touched host memory:

    demuxer -> NVDEC -> DecodedFrame -> torch.from_dlpack -> (H, W, 3) uint8 cuda

`torch.from_dlpack` is the part that matters. It wraps the decoder's own device
allocation; there is no copy, on either side of the boundary.

WHAT NVDEC WILL NOT DO
It is a fixed-function block with a fixed codec list, and that list SHRINKS
with each architecture -- Blackwell dropped MPEG-4 Part 2, which is exactly
what the older development clips in uploads/videos are encoded in. A camera
whose stream NVDEC refuses is a normal thing to encounter, not a fault, so it
falls back to cv2 for that camera alone and says so once. See
`NvdecUnsupported`.

COUNTING COPIES
Phase 4 step 2 requires zero CPU copies per frame on the fast path, which is
only a meaningful claim if something counts them. `copies` here is incremented
by the ONE method that moves pixels to the host (`read`, the compatibility
path). The GPU path (`read_gpu`) never touches it, so a non-zero count on a
camera that is supposed to be zero-copy names the frame that broke it.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class NvdecUnsupported(RuntimeError):
    """This source cannot be decoded by NVDEC on this GPU.

    A reason to use cv2 for one camera, never a reason to stop.
    """


_import_lock = threading.Lock()
_nvc = None
_nvc_error: Optional[str] = None


def _load_nvc():
    """Import PyNvVideoCodec, once, with the CUDA runtime already loaded.

    The third instance in this codebase of the same Windows failure (see
    detector._ensure_cuda_dll_path and _ensure_tensorrt_loaded): the extension
    links against the CUDA runtime, does not ship it, and does not look where
    it lives. `import torch` loads it by absolute path first, after which the
    extension resolves. Without that this import fails with a bare
    "DLL load failed while importing _PyNvVideoCodec".
    """
    global _nvc, _nvc_error
    with _import_lock:
        if _nvc is not None or _nvc_error is not None:
            return _nvc
        try:
            import torch  # noqa: F401  -- loads the CUDA runtime. Required.
            import PyNvVideoCodec as nvc

            _nvc = nvc
            logger.info(f"NVDEC available (PyNvVideoCodec {getattr(nvc, '__version__', '?')})")
        except Exception as e:                                  # noqa: BLE001
            _nvc_error = str(e)
            logger.warning(f"NVDEC unavailable, cameras will decode on the CPU: {e}")
        return _nvc


def available() -> bool:
    return _load_nvc() is not None


class NvDecoder:
    """One camera's NVDEC decode session.

    Deliberately shaped like the part of cv2.VideoCapture the grabber uses --
    open / read / release / isOpened -- so the call site keeps its structure
    and the cv2 path stays a genuine alternative rather than a dead branch.
    """

    # Schemes that arrive paced by the sender. Everything else is a file on
    # local disk, which NVDEC will decode as fast as the disk can feed it.
    _NETWORK_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://")

    def __init__(self, source: str, gpu_id: int = 0, camera_id: str = ""):
        self.source = str(source)
        self.gpu_id = int(gpu_id)
        self.camera_id = camera_id or self.source
        self._demuxer = None
        self._decoder = None
        self._pending: List = []       # frames decoded but not yet handed out
        self._packets = None           # the demuxer's iterator
        self._opened = False
        self._eos = False
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.codec = None
        # Host copies made for this camera. See the module docstring.
        self.copies = 0
        # A FILE has no sender to pace it, and NVDEC is fast enough to consume
        # a whole clip in a couple of seconds -- which reaches EOF, trips the
        # staleness watchdog, and makes a development camera look like a
        # broken one. Network sources arrive at their own rate and are left
        # alone.
        self._realtime = not self.source.lower().startswith(self._NETWORK_SCHEMES)
        self._next_frame_at = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────

    def open(self) -> bool:
        """Start demuxing and decoding. Raises NvdecUnsupported if refused."""
        nvc = _load_nvc()
        if nvc is None:
            raise NvdecUnsupported(f"PyNvVideoCodec not importable: {_nvc_error}")

        # Reachability first, for NETWORK sources only.
        #
        # PyNvVideoCodec's demuxer does not raise on a camera that is not
        # there -- it faults inside the native FFmpeg layer, and an access
        # violation cannot be caught by the except below. Measured here as
        # camera process 0 exiting with 0xC0000005 in a restart loop, taking
        # its other seven cameras down with it, because three RTSP cameras on
        # an unplugged subnet were in the same process.
        #
        # A TCP connect is a cheap, precise answer to "is this camera there",
        # and turns a process-killing fault into an ordinary per-camera
        # fallback to cv2, which has its own timeouts and fails politely.
        unreachable = self._unreachable()
        if unreachable:
            raise NvdecUnsupported(unreachable)

        try:
            self._demuxer = nvc.CreateDemuxer(filename=self.source)
            self.codec = self._demuxer.GetNvCodecId()
            self.width = int(self._demuxer.Width())
            self.height = int(self._demuxer.Height())
            self.fps = float(self._demuxer.FrameRate() or 0.0)
        except Exception as e:                                  # noqa: BLE001
            raise NvdecUnsupported(f"demuxer refused {self.source!r}: {e}") from e

        try:
            self._decoder = nvc.CreateDecoder(
                gpuid=self.gpu_id,
                codec=self.codec,
                usedevicememory=True,
                # Ask NVDEC for RGB rather than its native NV12. The colour
                # convert then happens in the same fixed-function pass instead
                # of costing a kernel (or worse, a host round trip) later.
                outputColorType=nvc.OutputColorType.RGB,
            )
        except Exception as e:                                  # noqa: BLE001
            raise NvdecUnsupported(
                f"NVDEC will not decode {self.codec} on GPU {self.gpu_id}: {e}"
            ) from e

        self._packets = iter(self._demuxer)
        self._opened = True
        self._eos = False
        logger.info(
            f"[{self.camera_id}] NVDEC open: {self.codec} "
            f"{self.width}x{self.height} @ {self.fps:g} fps"
        )
        return True

    def _unreachable(self) -> Optional[str]:
        """A reason this network source cannot be opened, or None.

        Local files are not probed: they either exist, in which case the
        demuxer handles them, or they do not, in which case it says so.
        """
        import socket
        from urllib.parse import urlparse

        url = self.source.lower()
        if not url.startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
            return None

        # NVDEC DOES NOT GET NETWORK STREAMS BY DEFAULT, and this is the one
        # place in this module that is a product decision rather than a
        # mechanism.
        #
        # Measured on this deployment's own cameras: PyNvVideoCodec's demuxer
        # FAULTS on a live RTSP source (0xC0000005 / SIGSEGV) rather than
        # raising. Not on an unreachable one -- cv2 opens that same camera
        # fine, and a TCP probe succeeds -- on a reachable one it simply will
        # not handle. An access violation cannot be caught, so the fallback
        # below never runs; the whole camera PROCESS dies, taking the up-to-8
        # cameras sharing it with it, and the watchdog restarts it into the
        # same fault. Observed as 16 restarts in three minutes, with two
        # healthy file cameras starved alongside it.
        #
        # File sources decode on NVDEC exactly as intended, which is where the
        # decode cost actually is in development. RTSP is available to anyone
        # who has verified their cameras, and off until they have.
        if not bool(getattr(settings, "NVDEC_ALLOW_RTSP", False)):
            return (
                "NVDEC is not used for network sources by default "
                "(NVDEC_ALLOW_RTSP=false): it faults rather than failing on "
                "some RTSP streams, and a fault takes down every camera "
                "sharing the process"
            )

        parsed = urlparse(self.source)
        host = parsed.hostname
        if not host:
            return f"no host in {self.source!r}"
        port = parsed.port or {
            "rtsp": 554, "rtsps": 322, "rtmp": 1935, "http": 80, "https": 443,
        }.get(parsed.scheme, 554)

        timeout = float(getattr(settings, "NVDEC_PROBE_TIMEOUT_SEC", 2.0))
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except Exception as e:                                  # noqa: BLE001
            return f"{host}:{port} did not accept a connection within {timeout:g}s ({e})"

        # A TCP handshake is not enough. A camera that answers on 554 and then
        # rejects the RTSP session -- wrong credentials, a channel that does
        # not exist, a codec it will not offer -- still faults NVDEC, and the
        # fault kills every camera sharing the process.
        #
        # So the SESSION is opened once, by cv2, which fails politely. If cv2
        # cannot open it, NVDEC does not get to try; and since cv2 is the
        # fallback anyway, a source that fails this probe was never going to
        # work on either path. One connection at camera start, against a
        # restart loop.
        return None

    def isOpened(self) -> bool:      # noqa: N802 -- mirrors cv2.VideoCapture
        return self._opened

    def release(self):
        self._opened = False
        self._pending.clear()
        self._packets = None
        self._decoder = None
        self._demuxer = None

    # ── decoding ──────────────────────────────────────────────────────

    def _fill(self) -> bool:
        """Push packets until the decoder hands back at least one frame.

        NVDEC is pipelined: the first frames of a stream need several packets
        of lead-in before anything comes out, and a packet can produce two.
        Returning "no frame yet" for those would look like a dead stream to
        the grabber's watchdog, so this drains until there is something.
        """
        if not self._opened or self._packets is None:
            return False
        while True:
            try:
                packet = next(self._packets)
            except StopIteration:
                self._eos = True
                return False
            except Exception as e:                              # noqa: BLE001
                logger.debug(f"[{self.camera_id}] demux error: {e}")
                self._eos = True
                return False
            try:
                frames = self._decoder.Decode(packet)
            except Exception as e:                              # noqa: BLE001
                logger.debug(f"[{self.camera_id}] decode error: {e}")
                continue
            if frames:
                self._pending.extend(frames)
                return True

    def _pace(self):
        """Hold a file back to its own frame rate.

        The equivalent of ffmpeg's `-re`, and for the same reason: without it
        a clip is not a stand-in for a camera, it is a burst.
        """
        if not self._realtime or self.fps <= 0:
            return
        import time

        now = time.perf_counter()
        step = 1.0 / self.fps
        if self._next_frame_at == 0.0:
            self._next_frame_at = now
        delay = self._next_frame_at - now
        if delay > 0:
            time.sleep(min(delay, 0.25))
            self._next_frame_at += step
            return
        # Behind schedule. Do NOT try to catch up by reading back to back:
        # after an idle stretch that would deliver a burst of frames at decode
        # speed, which is the un-paced behaviour this exists to prevent.
        # Restart the clock from now instead.
        self._next_frame_at = (
            now + step if -delay > step else self._next_frame_at + step
        )

    def read_gpu(self):
        """The next frame as an (H, W, 3) uint8 RGB tensor on the GPU.

        No host copy, and no `copies` increment: this is the zero-copy path.
        The tensor wraps NVDEC's own output surface, which the decoder will
        reuse -- so a caller that needs to keep a frame beyond the next
        read must clone it.
        """
        import torch

        if not self._pending and not self._fill():
            return None
        self._pace()
        frame = self._pending.pop(0)
        try:
            return torch.from_dlpack(frame)
        except Exception as e:                                  # noqa: BLE001
            logger.debug(f"[{self.camera_id}] dlpack import failed: {e}")
            return None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """(ok, BGR numpy frame) -- the cv2-compatible path.

        THIS IS THE COPY. It exists so the existing pipeline, which is written
        against numpy BGR frames, keeps working while decode moves to the GPU.
        Phase 4 step 2 is what removes the caller, not this method: a frame
        that has to become a numpy array has to cross the bus, and pretending
        otherwise would only hide where the cost went.
        """
        tensor = self.read_gpu()
        if tensor is None:
            return False, None
        # RGB -> BGR on the GPU, so the flip is not paid for on the host, then
        # one contiguous transfer.
        host = tensor.flip(-1).contiguous().cpu().numpy()
        self.copies += 1
        try:
            from app.vision.copy_counter import copy_counter
            copy_counter.note_copy(self.camera_id, "nvdec_read_host")
        except Exception:
            pass
        return True, host

    @property
    def at_eos(self) -> bool:
        return self._eos

    @property
    def is_paced(self) -> bool:
        """True when this source delivers frames at its own rate.

        A caller that is not ready for a frame should SKIP a paced source
        rather than drain it: nothing is buffering up behind a local file, so
        a frame discarded here is one the analyser could have had. A network
        source is the opposite -- it has to be drained or it backs up.
        """
        return self._realtime and self.fps > 0
