"""
Runs the server-mode deployment: one InferenceServer, N camera processes.

Lives beside CameraManager rather than inside it. CameraManager owns the thread
path and is reachable unchanged via INFERENCE_MODE=thread; this owns the process
path. Keeping them apart means the old shape stays exactly as it was rather than
becoming a branch inside every method of the new one.

The parent process holds the GPU. Camera processes hold no models at all, so a
camera process dying costs its eight cameras and nothing else -- the server and
every other process keep running, and the dead process is restarted.
"""

import logging
# torch.multiprocessing, not multiprocessing: it is the same API plus the
# reduction hooks that let a CUDA tensor cross a process boundary as an IPC
# handle instead of as pixels. That is what ZERO_COPY_INFERENCE needs, and it
# is a no-op for every other queue here.
import torch.multiprocessing as mp
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProcessCameraManager:
    """Camera processes plus the one inference server they share."""

    def __init__(self):
        self._specs: Dict[str, dict] = {}          # camera_id -> spec
        self._procs: Dict[int, mp.Process] = {}
        self._proc_specs: Dict[int, List[dict]] = {}
        self._result_queues: Dict[int, "mp.Queue"] = {}
        self._request_q: Optional["mp.Queue"] = None
        self._event_q: Optional["mp.Queue"] = None
        self._stop_evt = None
        self._slots = None
        self._server = None
        self._pump: Optional[threading.Thread] = None
        self._watchdog: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._cached_encodings: Optional[list] = None
        self._pending: List[str] = []
        self._debounce: Optional[threading.Timer] = None
        self._camera_slot_index: Dict[str, int] = {}
        self._slot_capacity = 0
        self._total_slots = 0
        self._slot_bytes = 0
        self._slot_layout: Dict[int, tuple] = {}

        self._on_event: Optional[Callable] = None
        self._on_unknown: Optional[Callable] = None
        # camera_id -> (monotonic time, frame count) from the last heartbeat.
        # The watchdog reads these; the pump writes them.
        self._beats: Dict[str, tuple] = {}
        # camera_id -> (monotonic time the count last CHANGED, that count).
        # Separate from _beats so a heartbeat arriving with an unchanged count
        # does not look like progress.
        self._stall_state: Dict[str, tuple] = {}
        self._gpu_slots = None
        # When each process was (re)spawned, for the warm-up grace above.
        self._proc_started_at: Dict[int, float] = {}

    # ── setup ─────────────────────────────────────────────────────────

    def set_callbacks(self, on_event=None, on_unknown=None, on_frame=None):
        # on_frame is not forwarded: a display frame would have to cross the
        # process boundary every frame, and Phase 1 item 1 established nobody
        # is watching most of the time. Live view is Phase 3's problem.
        self._on_event = on_event
        self._on_unknown = on_unknown

    def load_encodings_all(self, encodings: list):
        """Cached for processes started later. Running processes keep the
        gallery they were handed; a reload means a restart."""
        self._cached_encodings = encodings

    def add_camera(self, camera_id: str, stream_source: str):
        """Register a camera. Starts it too, if we are already running.

        Cameras arrive one at a time from the dashboard, but a process holds
        eight of them and cannot be extended once spawned. So a camera added
        while running goes on a pending list and a short debounce spawns one
        process for everything that arrived together -- which is what starting
        several cameras from the UI actually looks like. One camera on its own
        still gets its own process, and that is correct: it is the only way to
        run it without restarting its neighbours.
        """
        with self._lock:
            self._specs[camera_id] = {
                "camera_id": camera_id,
                "stream_source": stream_source,
            }
            if not self._running.is_set():
                # Not up yet. Debounce the FIRST start too, so several cameras
                # enabled together at boot land in one process instead of one
                # process each.
                self._schedule(self.start)
                return
            self._pending.append(camera_id)
            self._schedule(self._flush_pending)

    def _schedule(self, fn):
        if self._debounce is not None and self._debounce.is_alive():
            return
        self._debounce = threading.Timer(1.5, fn)
        self._debounce.daemon = True
        self._debounce.start()

    def _flush_pending(self):
        """Spawn one process for every camera registered since the last flush."""
        from app.config import settings

        with self._lock:
            ids, self._pending = self._pending, []
            if not ids or not self._running.is_set():
                return
            group = []
            for cid in ids:
                spec = dict(self._specs[cid])
                spec["encodings"] = self._cached_encodings
                idx = self._camera_slot_index.get(cid)
                if idx is None:
                    idx = self._claim_slot_index(cid)
                if idx is None:
                    logger.error(
                        f"no shared frame slots left for {cid}; raise MAX_CAMERAS"
                    )
                    continue
                spec["_slot_index"] = idx
                group.append(spec)
            if not group:
                return
            index = max(self._procs.keys(), default=-1) + 1
            self._proc_specs[index] = group
            ctx = mp.get_context("spawn")
            self._result_queues[index] = ctx.Queue(maxsize=256)
            self._server.result_queues = self._result_queues
            slots_per_cam = int(getattr(settings, "INFERENCE_SLOTS_PER_CAMERA", 3))
            self._spawn(index, group[0]["_slot_index"] * slots_per_cam,
                        slots_per_cam, self._total_slots, self._slot_bytes, ctx)

    def _claim_slot_index(self, camera_id: str):
        """Give a camera its reserved band of shared-memory slots.

        Bands are handed out by index and never reused while a camera is
        registered, so two cameras can never write into the same buffer.
        """
        used = set(self._camera_slot_index.values())
        for i in range(self._slot_capacity):
            if i not in used:
                self._camera_slot_index[camera_id] = i
                return i
        return None

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self):
        from app.config import settings
        from app.vision.inference_server import FrameSlots, InferenceServer

        if self._running.is_set():
            return
        if not self._specs:
            logger.info("ProcessCameraManager: no cameras to start")
            return

        per_proc = max(1, int(getattr(settings, "CAMERAS_PER_PROCESS", 8)))
        slots_per_cam = max(2, int(getattr(settings, "INFERENCE_SLOTS_PER_CAMERA", 3)))
        max_pixels = int(getattr(settings, "INFERENCE_MAX_FRAME_PIXELS", 1920 * 1080))
        slot_bytes = max_pixels * 3

        # Size the shared block for CAPACITY, not for the cameras registered
        # right now. Shared memory cannot grow, and cameras arrive from the
        # dashboard after start; a block sized to the current two would leave
        # the third with nowhere to write.
        self._slot_capacity = max(
            int(getattr(settings, "MAX_CAMERAS", 10)), len(self._specs)
        )
        total_slots = self._slot_capacity * slots_per_cam
        self._total_slots = total_slots
        self._slot_bytes = slot_bytes

        cameras = []
        for spec in self._specs.values():
            spec = dict(spec)
            spec["encodings"] = self._cached_encodings
            idx = self._claim_slot_index(spec["camera_id"])
            spec["_slot_index"] = idx
            cameras.append(spec)
        groups = [cameras[i:i + per_proc] for i in range(0, len(cameras), per_proc)]

        ctx = mp.get_context("spawn")
        self._request_q = ctx.Queue(maxsize=1024)
        self._event_q = ctx.Queue(maxsize=4096)
        self._stop_evt = ctx.Event()
        self._slots = FrameSlots(
            name=f"sentinel_frames_{int(time.time())}",
            n_slots=total_slots, slot_bytes=slot_bytes, create=True,
        )

        self._result_queues = {i: ctx.Queue(maxsize=256) for i in range(len(groups))}

        # The device-memory frame pool (Phase 4 step 2). Allocated HERE, in
        # the parent that owns the GPU, and handed to the camera processes as
        # CUDA IPC handles; they write decoded frames into it and send slot
        # indices. Never fatal: without it the shared-memory pixel path stands.
        if bool(getattr(settings, "ZERO_COPY_INFERENCE", False)):
            try:
                from app.vision.inference_server import CudaFrameSlots
                max_px = int(getattr(settings, "INFERENCE_MAX_FRAME_PIXELS", 1920 * 1080))
                height = int(getattr(settings, "ZERO_COPY_SLOT_HEIGHT", 1080))
                width = max(1, max_px // max(1, height))
                self._gpu_slots = CudaFrameSlots.create(
                    total_slots, height, width, slots_per_cam
                )
                logger.info(
                    f"zero-copy inference: {total_slots} CUDA frame slots of "
                    f"{height}x{width}x3 ({total_slots * height * width * 3 / 2**20:.0f} MiB)"
                )
            except Exception as e:                              # noqa: BLE001
                self._gpu_slots = None
                logger.warning(
                    f"zero-copy inference unavailable ({e}); camera processes "
                    f"will send pixels through shared memory."
                )

        self._server = InferenceServer(
            request_q=self._request_q,
            result_queues=self._result_queues,
            slots=self._slots,
            tick_ms=float(getattr(settings, "INFERENCE_TICK_MS", 20.0)),
            max_batch=int(getattr(settings, "INFERENCE_BATCH_CAP", 32)),
        )
        if self._gpu_slots is not None:
            self._server.attach_gpu_slots(self._gpu_slots)
        self._server.start()

        for i, group in enumerate(groups):
            self._proc_specs[i] = group
            # A group's slot base is its FIRST camera's reserved band, so the
            # bands stay owned by cameras rather than by whichever process
            # happens to hold them after a restart.
            self._spawn(i, group[0]["_slot_index"] * slots_per_cam,
                        slots_per_cam, total_slots, slot_bytes, ctx)

        self._running.set()
        self._pump = threading.Thread(target=self._pump_events, name="event-pump",
                                      daemon=True)
        self._pump.start()
        self._watchdog = threading.Thread(target=self._watch, name="camera-watchdog",
                                          daemon=True)
        self._watchdog.start()
        logger.info(
            f"ProcessCameraManager started: {len(cameras)} camera(s) in "
            f"{len(groups)} process(es), {total_slots} shared frame slots"
        )

    def _spawn(self, index, slot_base, slots_per_cam, total_slots, slot_bytes, ctx=None):
        from app.vision.camera_process import run_camera_process

        ctx = ctx or mp.get_context("spawn")
        group = self._proc_specs[index]
        p = ctx.Process(
            target=run_camera_process,
            args=(index, group, self._slots.name, total_slots, slot_bytes,
                  slots_per_cam, slot_base, self._request_q,
                  self._result_queues[index], self._event_q, self._stop_evt,
                  self._gpu_slots),
            name=f"cameras-{index}",
            daemon=True,
        )
        p.start()
        self._procs[index] = p
        self._proc_started_at[index] = time.monotonic()
        self._slot_layout = getattr(self, "_slot_layout", {})
        self._slot_layout[index] = (slot_base, slots_per_cam, total_slots, slot_bytes)
        logger.info(
            f"camera process {index} spawned (pid={p.pid}) with "
            f"{len(group)} camera(s)"
        )

    def stop(self):
        if not self._running.is_set():
            return
        self._running.clear()
        if self._stop_evt is not None:
            self._stop_evt.set()
        for i, p in list(self._procs.items()):
            try:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        self._procs.clear()
        if self._server:
            self._server.stop()
        if self._slots:
            self._slots.close()
        logger.info("ProcessCameraManager stopped")

    # ── the parent's two threads ──────────────────────────────────────

    def _pump_events(self):
        """Camera-process events into the EXISTING event pipeline.

        The callbacks here are the same ones the thread path uses, so nothing
        downstream can tell which path an event came from.
        """
        while self._running.is_set():
            try:
                event = self._event_q.get(timeout=0.2)
            except Exception:
                continue
            if event is None:
                break
            try:
                if event.get("event_type") == "_worker_died":
                    logger.error(
                        f"camera process {event.get('worker')} reported a fatal "
                        f"error: {event.get('error')}"
                    )
                    continue
                if event.get("event_type") == "_heartbeat":
                    cid = event.get("camera_id", "")
                    if cid:
                        self._beats[cid] = (time.monotonic(), int(event.get("frames", 0)))
                    continue
                if event.get("event_type") == "unknown_face" and self._on_unknown:
                    self._on_unknown(event)
                elif self._on_event:
                    self._on_event(event)
            except Exception as e:
                logger.error(f"event pump failed: {e}")

    def _watch(self):
        """Restart a dead camera process.

        A camera process holds no models and no GPU state, so losing one costs
        its cameras and nothing else. The server and the other processes are
        unaffected, which is the property that makes this shape safe to run.
        """
        while self._running.is_set():
            time.sleep(1.0)
            for index, p in list(self._procs.items()):
                if not self._running.is_set():
                    return

                if not p.is_alive():
                    reason = f"exited (code={p.exitcode})"
                elif self._stalled_cameras(index):
                    # A process that is alive and producing nothing is the
                    # worse failure: it holds its cameras, reports healthy to
                    # anything that only checks liveness, and shows a frozen
                    # picture. Dying is at least honest.
                    stalled = self._stalled_cameras(index)
                    reason = (
                        f"produced no frames for "
                        f"{self._stall_seconds():g}s on {', '.join(stalled)}"
                    )
                else:
                    continue

                logger.error(f"camera process {index} {reason}; restarting")
                try:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                    for spec in self._proc_specs.get(index, []):
                        self._beats.pop(spec["camera_id"], None)
                    base, per_cam, total, slot_bytes = self._slot_layout[index]
                    self._spawn(index, base, per_cam, total, slot_bytes)
                except Exception as e:
                    logger.error(f"could not restart camera process {index}: {e}")

    @staticmethod
    def _stall_seconds() -> float:
        from app.config import settings
        return float(getattr(settings, "CAMERA_STALL_RESTART_SEC", 5.0))

    def _stalled_cameras(self, index: int) -> List[str]:
        """Cameras in this process whose frame count has stopped moving.

        THE CLOCK STARTS AT THE FIRST FRAME, NOT THE FIRST HEARTBEAT.
        A camera process spends its first seconds loading TensorRT engines and
        the insightface models -- on this machine, comfortably longer than the
        five second stall limit -- and it heartbeats throughout with a frame
        count of zero. Treating that as a stall restarts the process before it
        can finish loading, which makes it load again, forever: measured here
        as a camera restarting every 5 seconds and never producing a frame.

        So a camera that has not yet produced its FIRST frame is warming up,
        not stalled, until CAMERA_START_GRACE_SEC has passed -- after which a
        process that has still produced nothing really is stuck.
        """
        limit = self._stall_seconds()
        if limit <= 0:
            return []
        from app.config import settings
        grace = float(getattr(settings, "CAMERA_START_GRACE_SEC", 120.0))

        now = time.monotonic()
        started = self._proc_started_at.get(index, now)
        stalled = []
        for spec in self._proc_specs.get(index, []):
            cid = spec["camera_id"]
            beat = self._beats.get(cid)
            if beat is None:
                continue
            _seen_at, frames = beat

            if frames == 0:
                if now - started >= grace:
                    stalled.append(cid)
                continue

            last = self._stall_state.get(cid)
            if last is None or last[1] != frames:
                self._stall_state[cid] = (now, frames)
                continue
            if now - last[0] >= limit:
                stalled.append(cid)
        return stalled

    # ── status ────────────────────────────────────────────────────────

    @property
    def server_stats(self) -> dict:
        return self._server.stats() if self._server else {}

    def get_active_camera_ids(self) -> list:
        return list(self._specs.keys())

    def get_all_status(self) -> dict:
        return {
            "mode": "server",
            "cameras": {
                cid: {
                    "camera_id": cid,
                    "running": True,
                    # The frame count its process last reported, so a caller
                    # can tell a camera that is running from one that is
                    # merely registered.
                    "frames": (self._beats.get(cid) or (0, 0))[1],
                }
                for cid in self._specs
            },
            "total_active": len(self._specs),
            "processes": {i: {"pid": p.pid, "alive": p.is_alive()}
                          for i, p in self._procs.items()},
            "inference": self.server_stats,
        }


process_camera_manager = ProcessCameraManager()
