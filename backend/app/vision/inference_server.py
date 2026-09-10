"""
One inference loop per GPU.

Until now every camera thread called the detector itself: twenty threads, twenty
TensorRT execution contexts, twenty batch-1 calls, all serialised by the GIL and
by the GPU taking turns between contexts. Measured on this card, that is the
wrong shape twice over:

  * batch 1 costs 6.26 ms per frame against 1.96 ms at batch 8 -- roughly 3x
    thrown away by never batching;
  * an execution context sized for batch 32 costs 1,131 MiB, so twenty of them
    want 22.6 GB on a 12 GB card and simply do not load.

Both problems have the same answer: ONE consumer. This server owns the engines
and the ONNX Runtime sessions, collects whatever every camera has pending each
tick, runs one call per model over the whole batch, and routes the results back
by camera id. Camera workers keep decode, tracking and rules; they stop owning
models entirely.

TRANSPORT
Frames move through shared memory, never pickled: a 720p BGR frame is 2.7 MB and
pickling one per camera per tick would cost more than the inference. Each camera
owns a small ring of slots in one shared block; it writes a frame, puts a small
descriptor on the request queue, and waits for a result carrying the same frame
id. A slot is only reused once its result has come back, so the server can never
read a buffer that is being overwritten.

WHAT IS AND IS NOT BATCHED
YOLO is genuinely batched -- one call for every camera's pending frame. SCRFD and
ArcFace are not: insightface's Python pre/post runs per image, so they are looped
inside the tick. They are still centralised here, which is what gets them off the
camera threads and onto one CUDA context, but the batching win for faces needs
the GPU anchor decode and NMS of item 5. Said plainly so nobody reads "one
inference loop" as "faces are batched".
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Frame kinds a camera can ask for.
KIND_PERSONS = "persons"
KIND_FACES = "faces"


# ── Shared-memory frame transport ─────────────────────────────────────────

class FrameSlots:
    """A fixed pool of shared frame buffers.

    One block, divided into equal slots. The parent creates it, camera
    processes attach by name. Slots are handed out per camera so two cameras
    can never target the same buffer, and a camera only rewrites a slot after
    the result for its previous frame has returned.
    """

    def __init__(self, name: str, n_slots: int, slot_bytes: int, create: bool):
        from multiprocessing import shared_memory

        self.n_slots = n_slots
        self.slot_bytes = slot_bytes
        size = n_slots * slot_bytes
        if create:
            try:                       # a stale block from a crashed run
                old = shared_memory.SharedMemory(name=name)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        else:
            self.shm = shared_memory.SharedMemory(name=name)
        self.name = self.shm.name
        self._created = create

    def write(self, slot: int, frame: np.ndarray) -> Tuple[int, int, int]:
        """Copy a frame into a slot. Returns its shape for the descriptor."""
        h, w, c = frame.shape
        nbytes = frame.nbytes
        if nbytes > self.slot_bytes:
            raise ValueError(
                f"frame {h}x{w}x{c} ({nbytes} bytes) exceeds slot size "
                f"{self.slot_bytes}; raise INFERENCE_MAX_FRAME_PIXELS"
            )
        off = slot * self.slot_bytes
        buf = np.ndarray((nbytes,), dtype=np.uint8, buffer=self.shm.buf, offset=off)
        buf[:] = frame.reshape(-1).view(np.uint8)
        return h, w, c

    def read(self, slot: int, h: int, w: int, c: int) -> np.ndarray:
        """A view of a slot as a frame. Copied by the caller if it must outlive
        the tick -- the slot is reusable as soon as the result is sent."""
        off = slot * self.slot_bytes
        return np.ndarray((h, w, c), dtype=np.uint8, buffer=self.shm.buf, offset=off)

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass
        if self._created:
            try:
                self.shm.unlink()
            except Exception:
                pass


class CudaFrameSlots:
    """The same pool, in device memory instead of shared host memory.

    WHAT THIS REPLACES
    FrameSlots above moves pixels: the camera process copies a numpy frame
    into shared memory (host to host), and the server reads it back and hands
    it to a model, which copies it to the GPU. On the NVDEC path the frame was
    ALREADY on the GPU, so that is a device-to-host copy followed by a
    host-to-device one, per frame per camera, purely to cross a process
    boundary.

    Here the parent allocates persistent CUDA tensors and shares them with the
    camera processes by CUDA IPC handle -- torch does the handle passing when
    a tensor is sent through a torch.multiprocessing queue. The camera process
    copies its decoded frame into its slot (device to device, which never
    touches the CPU) and sends the slot INDEX. Nothing but the index crosses.

    WHY A COPY AT ALL, RATHER THAN SHARING THE DECODER'S SURFACE
    NVDEC hands back a view of a surface it owns and will reuse within a few
    frames. Sending that surface's handle would be a race the server loses
    silently -- it would map memory the decoder has since overwritten. Copying
    into a slot we own costs a device-to-device blit and makes the lifetime
    ours.
    """

    def __init__(self, tensors: List["torch.Tensor"], slots_per_camera: int):
        self.tensors = tensors
        self.slots_per_camera = slots_per_camera

    @classmethod
    def create(cls, n_slots: int, height: int, width: int, slots_per_camera: int):
        import torch

        tensors = [
            torch.empty((height, width, 3), dtype=torch.uint8, device="cuda")
            for _ in range(n_slots)
        ]
        for t in tensors:
            t.share_memory_()
        return cls(tensors, slots_per_camera)

    def write(self, slot: int, frame) -> Tuple[int, int, int]:
        """Device-to-device copy into a slot. Returns the shape descriptor."""
        h, w, c = int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])
        target = self.tensors[slot]
        if h > target.shape[0] or w > target.shape[1]:
            raise ValueError(
                f"frame {h}x{w} exceeds the GPU slot {tuple(target.shape)}; "
                f"raise INFERENCE_MAX_FRAME_PIXELS"
            )
        target[:h, :w, :c].copy_(frame, non_blocking=True)
        return h, w, c

    def read(self, slot: int, h: int, w: int, c: int):
        """A device view of a slot. Valid until the camera reuses the slot."""
        return self.tensors[slot][:h, :w, :c]


@dataclass
class Request:
    camera_id: str
    frame_id: int
    kind: str
    slot: int
    h: int
    w: int
    c: int
    worker: int          # which process to send the result back to
    gpu: bool = False    # slot indexes CudaFrameSlots, not FrameSlots
    meta: dict = field(default_factory=dict)


# ── The server ────────────────────────────────────────────────────────────

class InferenceServer:
    """Owns the models. Runs one tick loop. Never owned by a camera."""

    def __init__(
        self,
        request_q,
        result_queues: Dict[int, Any],
        slots: Optional[FrameSlots] = None,
        tick_ms: float = 20.0,
        max_batch: int = 32,
    ):
        self.request_q = request_q
        self.result_queues = result_queues
        self.slots = slots
        self.tick = max(0.010, min(0.033, tick_ms / 1000.0))
        self.max_batch = max_batch

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._person_model = None
        self._gpu_slots = None          # attached when ZERO_COPY_INFERENCE is on
        self._gpu_scrfd_obj = None
        self._weapon_model = None
        self._face_app = None
        self._models_lock = threading.Lock()

        # Batch statistics. Item 3 asks for tick fill and batch size to be
        # reported; they are also the only way to tell a server that is
        # batching from one that is doing twenty batch-1 calls in a trench
        # coat.
        self.ticks = 0
        self.frames = 0
        self.batch_hist: Dict[int, int] = {}
        self.max_batch_seen = 0

    # ── models ────────────────────────────────────────────────────────

    def _person(self):
        if self._person_model is not None:
            return self._person_model
        with self._models_lock:
            if self._person_model is None:
                from app.config import settings
                from app.vision.engines import get_engine
                from ultralytics import YOLO

                weights = getattr(settings, "HAZARD_MODEL_PATH", "yolov8n.pt")
                path = get_engine(
                    weights,
                    batch=int(getattr(settings, "TRT_MAX_BATCH", 32)),
                    imgsz=int(getattr(settings, "PERSON_DET_IMGSZ", 640)),
                    half=bool(getattr(settings, "PERSON_DET_HALF", True)),
                    build=bool(getattr(settings, "TRT_BUILD_ON_START", True)),
                ) or weights
                logger.info(f"InferenceServer: person model {path}")
                self._person_model = YOLO(path)
                # A .engine has its precision BAKED IN at build time, so
                # passing half= to predict() cannot change anything -- it only
                # earns ultralytics' "'half' is deprecated" warning, once per
                # call. At 25 fps per camera that is the loudest thing in the
                # log and it says nothing.
                self._person_is_engine = str(path).endswith(".engine")
        return self._person_model

    def _weapons(self):
        """yolo11m, loaded only if a camera actually asks for weapons."""
        if self._weapon_model is not None:
            return self._weapon_model
        with self._models_lock:
            if self._weapon_model is None:
                from app.config import settings
                from app.vision.engines import get_engine
                from ultralytics import YOLO

                weights = getattr(settings, "WEAPON_MODEL_PATH", "yolo11m.pt")
                path = get_engine(
                    weights,
                    batch=int(getattr(settings, "TRT_MAX_BATCH", 32)),
                    imgsz=int(getattr(settings, "PERSON_DET_IMGSZ", 640)),
                    half=True,
                    build=bool(getattr(settings, "TRT_BUILD_ON_START", True)),
                ) or weights
                logger.info(f"InferenceServer: weapons model {path}")
                self._weapon_model = YOLO(path)
        return self._weapon_model

    def _faces(self):
        if self._face_app is None:
            from app.vision.detector import _get_insightface_app
            self._face_app = _get_insightface_app()
        return self._face_app

    # NOTE: there is deliberately no plate or vehicle model here. Vehicle
    # monitoring is off by default (see settings.ENABLE_TRAFFIC_MONITOR) and
    # the server must not be the thing that quietly loads a model the product
    # has switched off.

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="inference-server",
                                        daemon=True)
        self._thread.start()
        logger.info(
            f"InferenceServer started (tick={self.tick * 1000:.0f}ms, "
            f"max_batch={self.max_batch})"
        )

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # ── the tick ──────────────────────────────────────────────────────

    def _collect(self) -> List[Request]:
        """Everything pending, up to the batch cap, without waiting past the tick."""
        batch: List[Request] = []
        deadline = time.perf_counter() + self.tick
        while len(batch) < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                req = self.request_q.get(timeout=remaining)
            except (queue.Empty, OSError):
                break
            if req is None:              # shutdown sentinel
                self._stop.set()
                break
            batch.append(req)
        return batch

    def _loop(self):
        while not self._stop.is_set():
            try:
                batch = self._collect()
                if not batch:
                    continue
                self.ticks += 1
                self.frames += len(batch)
                self.batch_hist[len(batch)] = self.batch_hist.get(len(batch), 0) + 1
                self.max_batch_seen = max(self.max_batch_seen, len(batch))
                self._run(batch)
            except Exception as e:
                logger.error(f"InferenceServer tick failed: {e}", exc_info=True)
                time.sleep(0.01)

    def _run(self, batch: List[Request]):
        by_kind: Dict[str, List[Request]] = {}
        for req in batch:
            by_kind.setdefault(req.kind, []).append(req)

        for kind, reqs in by_kind.items():
            try:
                if kind == KIND_PERSONS:
                    results = (
                        self._run_persons_gpu(reqs) if reqs[0].gpu
                        else self._run_persons(reqs)
                    )
                elif kind == KIND_FACES:
                    results = (
                        self._run_faces_gpu(reqs) if reqs[0].gpu
                        else self._run_faces(reqs)
                    )
                else:
                    results = [None] * len(reqs)
            except Exception as e:
                logger.error(f"{kind} inference failed: {e}", exc_info=True)
                results = [None] * len(reqs)

            for req, res in zip(reqs, results):
                self._reply(req, res)

    def attach_gpu_slots(self, slots):
        """The device-memory pool camera processes write frames into."""
        self._gpu_slots = slots

    def _gpu_frames_of(self, reqs: List[Request]):
        """Device views for GPU requests. No copy, no host memory."""
        return [self._gpu_slots.read(r.slot, r.h, r.w, r.c) for r in reqs]

    def _run_persons_gpu(self, reqs: List[Request]) -> List[list]:
        """YOLO over frames that are already on the GPU.

        Ultralytics letterboxes numpy input itself and then maps the boxes
        back. Handed a tensor it does neither -- it assumes the caller has
        preprocessed, and returns boxes in the letterboxed square. So the
        letterbox and its inverse are both done here, on the device, and the
        only thing that comes back to the host is the box list.
        """
        import torch
        from app.config import settings
        from app.vision.gpu_preproc import letterbox_gpu, unletterbox

        model = self._person()
        imgsz = int(getattr(settings, "PERSON_DET_IMGSZ", 640))
        half = bool(getattr(settings, "PERSON_DET_HALF", True))

        batch, geometry = [], []
        for frame in self._gpu_frames_of(reqs):
            chw, scale, px, py = letterbox_gpu(frame, imgsz, half=half)
            batch.append(chw)
            geometry.append((scale, px, py))
        if not batch:
            return []

        out = model.predict(
            torch.cat(batch, dim=0),
            conf=0.25, classes=[0], verbose=False, device=0, imgsz=imgsz,
        )

        results = []
        for res, (scale, px, py) in zip(out, geometry):
            persons = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes):
                xyxy = unletterbox(boxes.xyxy.float(), scale, px, py)
                # THE ONLY HOST TRANSFER on this path: N boxes and N scores.
                xyxy = xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clses = boxes.cls.cpu().numpy()
                for k in range(len(xyxy)):
                    if int(clses[k]) != 0:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in xyxy[k])
                    persons.append({
                        "confidence": float(confs[k]),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })
            results.append(persons)
        return results

    def _run_faces_gpu(self, reqs: List[Request]) -> List[tuple]:
        """SCRFD then ArcFace, both against device memory.

        Requires the GPU decode of item 5 (FACE_GPU_POSTPROCESS): without it
        SCRFD's postprocess is numpy and the frame would have to come back to
        the host, which is the copy this whole path exists to remove. That is
        checked once at startup, not per frame.
        """
        import torch
        from app.vision.gpu_preproc import align_faces_gpu

        decoder = self._gpu_scrfd()
        rec = self._arcface()
        results = []
        for frame in self._gpu_frames_of(reqs):
            if decoder is None:
                results.append(([], [], []))
                continue
            det, kpss = decoder.detect_gpu(frame)
            if det is None or len(det) == 0:
                results.append(([], [], []))
                continue

            locations, landmarks = [], []
            for k in range(len(det)):
                x1, y1, x2, y2 = (int(v) for v in det[k][:4])
                locations.append((y1, x2, y2, x1))
                landmarks.append(kpss[k] if kpss is not None else None)

            embeddings = [None] * len(locations)
            usable = [i for i, kp in enumerate(landmarks) if kp is not None]
            if rec is not None and usable:
                crops = align_faces_gpu(frame, [landmarks[i] for i in usable])
                if crops is not None:
                    embs = self._arcface_embed(crops)
                    for slot, i in enumerate(usable):
                        embeddings[i] = embs[slot]
            results.append((locations, embeddings, landmarks))
        return results

    def _gpu_scrfd(self):
        """The GPU SCRFD decoder, built once from the loaded insightface app."""
        if getattr(self, "_gpu_scrfd_obj", None) is not None:
            return self._gpu_scrfd_obj
        with self._models_lock:
            if getattr(self, "_gpu_scrfd_obj", None) is None:
                app = self._faces()
                if app is None:
                    return None
                from app.vision.scrfd_gpu import GpuScrfd
                self._gpu_scrfd_obj = GpuScrfd(app.models["detection"])
        return self._gpu_scrfd_obj

    def _arcface(self):
        app = self._faces()
        return None if app is None else app.models.get("recognition")

    def _arcface_embed(self, crops):
        """Embeddings for (N,3,112,112) crops that are already on the device.

        Bound through IOBinding from the tensor's own pointer, so the aligned
        crops are not copied to the host either -- only the 512-float
        embeddings come back.
        """
        import numpy as _np
        import torch

        rec = self._arcface()
        if rec is None:
            return [None] * int(crops.shape[0])

        x = crops.mul(255.0).sub_(rec.input_mean).div_(rec.input_std).contiguous()
        n = int(x.shape[0])
        name = rec.session.get_inputs()[0].name
        out_name = rec.session.get_outputs()[0].name
        out = torch.empty((n, 512), dtype=torch.float32, device=x.device)

        io_b = rec.session.io_binding()
        io_b.bind_input(name, "cuda", 0, _np.float32, tuple(x.shape), x.data_ptr())
        io_b.bind_output(out_name, "cuda", 0, _np.float32, tuple(out.shape), out.data_ptr())
        rec.session.run_with_iobinding(io_b)
        return list(out.cpu().numpy())

    def _frames_of(self, reqs: List[Request]) -> List[np.ndarray]:
        return [self.slots.read(r.slot, r.h, r.w, r.c) for r in reqs]

    def _run_persons(self, reqs: List[Request]) -> List[list]:
        """ONE call for every camera's pending frame. This is the batching."""
        from app.config import settings

        model = self._person()
        frames = self._frames_of(reqs)
        kwargs = dict(
            conf=0.25, classes=[0], verbose=False, device=0,
            imgsz=int(getattr(settings, "PERSON_DET_IMGSZ", 640)),
        )
        if not getattr(self, "_person_is_engine", False):
            kwargs["half"] = bool(getattr(settings, "PERSON_DET_HALF", True))
        out = model.predict(frames, **kwargs)
        results = []
        for res in out:
            persons = []
            boxes = getattr(res, "boxes", None)
            if boxes is not None:
                for b in boxes:
                    if int(b.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    persons.append({
                        "confidence": float(b.conf[0]),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })
            results.append(persons)
        return results

    def _run_faces(self, reqs: List[Request]) -> List[tuple]:
        """SCRFD + ArcFace, per image.

        Looped, not batched: insightface's pre/post is numpy and per-image, so
        a batched call would need the GPU anchor decode of item 5. Centralising
        it here still takes it off the camera threads and onto one context.
        """
        app = self._faces()
        results = []
        for req, frame in zip(reqs, self._frames_of(reqs)):
            if app is None:
                results.append(([], [], []))
                continue
            faces = app.get(frame)
            locations, embeddings, landmarks = [], [], []
            for f in faces:
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                locations.append((y1, x2, y2, x1))     # top, right, bottom, left
                emb = getattr(f, "embedding", None)
                embeddings.append(None if emb is None else np.asarray(emb))
                landmarks.append(getattr(f, "kps", None))
            results.append((locations, embeddings, landmarks))
        return results

    def _reply(self, req: Request, payload):
        q = self.result_queues.get(req.worker)
        if q is None:
            return
        try:
            q.put_nowait((req.camera_id, req.frame_id, req.kind, payload))
        except Exception:
            # A worker that has died, or whose queue is full, must not stall
            # the server or any other camera. Dropping is correct: the client
            # times out and treats it as an empty detection, exactly as it
            # would a frame the detector found nothing in.
            pass

    # ── reporting ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        fill = (self.frames / self.ticks) if self.ticks else 0.0
        return {
            "ticks": self.ticks,
            "frames": self.frames,
            "mean_batch": round(fill, 2),
            "max_batch_seen": self.max_batch_seen,
            "tick_ms": round(self.tick * 1000, 1),
            "batch_cap": self.max_batch,
            "batch_histogram": dict(sorted(self.batch_hist.items())),
        }


# ── Client, used inside camera processes ──────────────────────────────────

class InferenceClient:
    """The camera side of the server. One per camera.

    Blocks until its result comes back, which is what makes the camera loop
    read like it still owns a detector. Eight cameras share a process, so
    while one waits the other seven run.
    """

    def __init__(self, camera_id: str, worker: int, slots: FrameSlots,
                 request_q, inbox: "queue.Queue", first_slot: int,
                 n_slots: int, timeout: float = 2.0):
        self.camera_id = camera_id
        self.worker = worker
        self.slots = slots
        self.request_q = request_q
        self.inbox = inbox
        self.first_slot = first_slot
        self.n_slots = n_slots
        self.timeout = timeout
        self._frame_id = 0
        self._next_slot = 0
        self.gpu_slots = None      # set by the camera process when zero-copy is on
        # frame_id -> the factor its frame was shrunk by to fit a slot.
        self._scales = {}
        self._resize_logged = False

    def _fit_to_slot(self, frame: np.ndarray):
        """Shrink a frame that will not fit a slot. Returns (frame, scale).

        A 4 MP camera -- 2688x1520, which this deployment has several of --
        is 12.2 MB a frame against a 1080p slot's 6.2 MB, so every submit
        raised and that camera detected NOBODY, for as long as it ran. The
        error said "raise INFERENCE_MAX_FRAME_PIXELS", which would work and
        would also cost 45 slots x 12.2 MB of shared memory to carry pixels
        the detector is about to throw away.

        It throws them away because YOLO letterboxes to imgsz (640) whatever
        it is given. Downscaling to fit the slot changes the detector's input
        by a resample it was going to do anyway; the boxes come back in the
        shrunken frame's coordinates and are scaled up here, so nothing
        downstream sees a difference.
        """
        # getattr, not attribute access: `slots` is whatever the camera
        # process was handed, and a transport that does not describe a slot
        # size has no limit to respect. Requiring the attribute turned an
        # unknown transport into a crash on the submit path.
        limit = getattr(self.slots, "slot_bytes", 0)
        if not limit or frame.nbytes <= limit:
            return frame, 1.0

        import cv2

        scale = (limit / float(frame.nbytes)) ** 0.5
        # A hair under, so rounding cannot land back over the limit.
        scale *= 0.999
        h, w = frame.shape[:2]
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        if not self._resize_logged:
            self._resize_logged = True
            logger.info(
                f"[{self.camera_id}] frames are {w}x{h}, larger than an "
                f"inference slot; submitting at {new_w}x{new_h} and scaling "
                f"boxes back. Raise INFERENCE_MAX_FRAME_PIXELS to send them "
                f"whole."
            )
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA), (
            float(new_w) / w
        )

    def _submit(self, frame: np.ndarray, kind: str):
        frame, scale = self._fit_to_slot(frame)
        self._frame_id += 1
        slot = self.first_slot + (self._next_slot % self.n_slots)
        self._next_slot += 1
        h, w, c = self.slots.write(slot, np.ascontiguousarray(frame))
        self.request_q.put(Request(
            camera_id=self.camera_id, frame_id=self._frame_id, kind=kind,
            slot=slot, h=h, w=w, c=c, worker=self.worker,
        ))
        self._scales[self._frame_id] = scale
        # The map is only read by the next _await, so anything older is dead.
        if len(self._scales) > 4 * self.n_slots:
            for fid in sorted(self._scales)[:-self.n_slots]:
                self._scales.pop(fid, None)
        return self._frame_id

    def _submit_gpu(self, frame, kind: str):
        """Hand over a DEVICE frame: a slot index crosses, not pixels."""
        self._frame_id += 1
        slot = self.first_slot + (self._next_slot % self.n_slots)
        self._next_slot += 1
        h, w, c = self.gpu_slots.write(slot, frame)
        self.request_q.put(Request(
            camera_id=self.camera_id, frame_id=self._frame_id, kind=kind,
            slot=slot, h=h, w=w, c=c, worker=self.worker, gpu=True,
        ))
        return self._frame_id

    @property
    def zero_copy(self) -> bool:
        return self.gpu_slots is not None

    def _await(self, frame_id: int, kind: str, empty):
        """Wait for OUR result. Anything for another frame is stale and dropped."""
        deadline = time.perf_counter() + self.timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                logger.debug(f"[{self.camera_id}] inference timeout ({kind})")
                return empty
            try:
                cam, fid, k, payload = self.inbox.get(timeout=remaining)
            except queue.Empty:
                return empty
            if fid == frame_id and k == kind:
                return empty if payload is None else payload
            # Late result for an earlier frame: discard. Frames are analysed in
            # order per camera, so an older id can never become useful again.

    def persons(self, frame) -> list:
        try:
            fid = (
                self._submit_gpu(frame, KIND_PERSONS)
                if self.zero_copy and not isinstance(frame, np.ndarray)
                else self._submit(frame, KIND_PERSONS)
            )
        except Exception as e:
            logger.error(f"[{self.camera_id}] person submit failed: {e}")
            return []
        persons = self._await(fid, KIND_PERSONS, [])
        scale = self._scales.pop(fid, 1.0)
        if scale != 1.0 and persons:
            inv = 1.0 / scale
            for det in persons:
                b = det["bbox"]
                for k in ("x1", "y1", "x2", "y2"):
                    b[k] = int(b[k] * inv)
        return persons

    def faces(self, frame):
        try:
            fid = (
                self._submit_gpu(frame, KIND_FACES)
                if self.zero_copy and not isinstance(frame, np.ndarray)
                else self._submit(frame, KIND_FACES)
            )
        except Exception as e:
            logger.error(f"[{self.camera_id}] face submit failed: {e}")
            return [], [], []
        return self._await(fid, KIND_FACES, ([], [], []))

    @property
    def max_in_flight(self) -> int:
        """How many requests this camera may have outstanding at once.

        One per reserved slot. Submitting more would wrap `_next_slot` back
        onto a slot the server has not read yet, and the second write would
        replace a frame already queued -- silently, as a wrong answer rather
        than an error.
        """
        return self.n_slots

    def faces_many(self, crops: List[np.ndarray]):
        """SCRFD + ArcFace on several crops, submitted together.

        Item 6 sends person crops, not frames, and a camera has a handful of
        them per face cycle. Submitting them one at a time and blocking on
        each would cost one server tick EACH -- eight crops, eight ticks, up
        to a quarter of a second for work the server could have done in one
        pass. Submitting them all first means they arrive in the same tick and
        are collected into one batch, which is the whole point of having a
        server.

        Results come back in submission order. A crop whose result is lost to
        the timeout yields an empty triple rather than shifting its
        neighbours' answers onto the wrong track.
        """
        if not crops:
            return []
        crops = crops[: self.max_in_flight]

        ids = []
        for crop in crops:
            try:
                ids.append(self._submit(crop, KIND_FACES))
            except Exception as e:
                logger.error(f"[{self.camera_id}] face crop submit failed: {e}")
                ids.append(None)

        empty = ([], [], [])
        wanted = {fid for fid in ids if fid is not None}
        got = {}
        deadline = time.perf_counter() + self.timeout
        while wanted:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                logger.debug(
                    f"[{self.camera_id}] face batch timeout, "
                    f"{len(wanted)}/{len(ids)} crops unanswered"
                )
                break
            try:
                cam, fid, kind, payload = self.inbox.get(timeout=remaining)
            except queue.Empty:
                break
            if kind != KIND_FACES or fid not in wanted:
                continue  # stale, or a persons result for this camera
            wanted.discard(fid)
            got[fid] = empty if payload is None else payload

        return [got.get(fid, empty) for fid in ids]
