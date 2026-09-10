"""
Camera workers as processes.

Twenty camera threads in one interpreter spend most of their time waiting for
the GIL, which is why the Phase 1 acceptance run left the GPU 81 percent idle
while missing its target. Decode, letterbox, ByteTrack, the identity ledger and
the drawing path are all Python, and Python only runs one of them at a time.

So cameras move into processes -- eight per process by default, each with its
own interpreter and its own GIL -- and inference moves out of them entirely,
into the one server that owns the GPU. A camera process does decode, tracking
and rules. It never loads a model.

WHAT CROSSES THE BOUNDARY
  frames   -> shared memory, never pickled (2.7 MB each at 720p)
  results  <- one queue per process, keyed by camera id and frame id
  events   <- one queue to the parent, which hands them to the SAME callbacks
              the thread path uses, so the event pipeline is untouched

WHY THE PIPELINE IS REUSED RATHER THAN REWRITTEN
VisionPipeline already holds the identity ledger, the unknown registry, the
crowd and loitering rules and the event shapes the rest of the system expects.
Reimplementing that for the process path would fork the product's behaviour in
two. Instead the pipeline keeps all of it and has only its two detector calls
redirected at the inference client, which is the single thing this change is
actually about.
"""

import logging
import os
import queue
import threading
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


def _drain_results(inbox_q, per_camera: dict, stop: threading.Event):
    """Fan the process's single result queue out to per-camera inboxes.

    One queue crosses the process boundary; each camera then waits on its own
    in-process queue. Without this every camera would have to inspect and
    re-queue results belonging to its seven neighbours.
    """
    while not stop.is_set():
        try:
            item = inbox_q.get(timeout=0.2)
        except (queue.Empty, OSError):
            continue
        if item is None:
            break
        camera_id = item[0]
        q = per_camera.get(camera_id)
        if q is not None:
            try:
                q.put_nowait(item)
            except queue.Full:
                pass


def run_camera_process(
    worker_index: int,
    camera_specs: List[dict],
    shm_name: str,
    n_slots: int,
    slot_bytes: int,
    slots_per_camera: int,
    slot_base: int,
    request_q,
    result_q,
    event_q,
    stop_evt,
):
    """Entry point for one camera process. Runs `camera_specs` to completion.

    Everything is wrapped: a process that dies takes its cameras with it, and
    the parent restarts it, but it must never take the server or the other
    processes down with it.
    """
    try:
        _run(worker_index, camera_specs, shm_name, n_slots, slot_bytes,
             slots_per_camera, slot_base, request_q, result_q, event_q, stop_evt)
    except Exception as e:                       # noqa: BLE001
        logger.error(f"camera process {worker_index} died: {e}", exc_info=True)
        try:
            event_q.put_nowait({
                "event_type": "_worker_died",
                "worker": worker_index,
                "error": str(e)[:400],
                "cameras": [s["camera_id"] for s in camera_specs],
            })
        except Exception:
            pass
        raise


def _run(worker_index, camera_specs, shm_name, n_slots, slot_bytes,
         slots_per_camera, slot_base, request_q, result_q, event_q, stop_evt):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from app.config import settings
    from app.vision.inference_server import FrameSlots, InferenceClient
    from app.vision.pipeline import VisionPipeline

    logging.basicConfig(level=logging.INFO)
    logger.info(
        f"camera process {worker_index} starting with "
        f"{len(camera_specs)} camera(s), pid={os.getpid()}"
    )

    slots = FrameSlots(shm_name, n_slots, slot_bytes, create=False)
    per_camera_inbox = {}
    stop = threading.Event()

    router = threading.Thread(
        target=_drain_results, args=(result_q, per_camera_inbox, stop),
        name=f"results-{worker_index}", daemon=True,
    )
    router.start()

    def emit(event):
        """Events go to the parent, which feeds the existing pipeline."""
        try:
            event_q.put_nowait(event)
        except Exception:
            pass

    pipelines = []
    threads = []
    for i, spec in enumerate(camera_specs):
        camera_id = spec["camera_id"]
        inbox = queue.Queue(maxsize=64)
        per_camera_inbox[camera_id] = inbox

        client = InferenceClient(
            camera_id=camera_id,
            worker=worker_index,
            slots=slots,
            request_q=request_q,
            inbox=inbox,
            # The camera's OWN reserved band, assigned by the parent. Not an
            # offset from this process's base: a camera keeps its band across
            # a process restart, so a slow frame written by the dying process
            # can never land in a slot the replacement has handed to someone
            # else.
            first_slot=spec.get("_slot_index", slot_base // max(1, slots_per_camera) + i)
                       * slots_per_camera,
            n_slots=slots_per_camera,
            timeout=float(getattr(settings, "INFERENCE_TIMEOUT_SEC", 2.0)),
        )

        try:
            pipe = VisionPipeline(
                camera_id=camera_id,
                stream_source=spec["stream_source"],
                on_event=emit,
                on_unknown=emit,
                on_frame=None,
            )
            pipe.attach_inference_client(client)
            if spec.get("encodings"):
                pipe.recognizer.load_encodings(spec["encodings"])
            pipe.start()
        except Exception as e:
            logger.error(f"[{camera_id}] failed to start: {e}", exc_info=True)
            continue

        pipelines.append(pipe)
        t = threading.Thread(target=_camera_loop, args=(pipe, stop, emit),
                             name=f"camera-{camera_id}", daemon=True)
        t.start()
        threads.append(t)

    logger.info(f"camera process {worker_index}: {len(pipelines)} camera(s) running")

    try:
        while not stop_evt.is_set():
            time.sleep(0.2)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop.set()
        for pipe in pipelines:
            try:
                pipe.stop()
            except Exception:
                pass
        slots.close()
        logger.info(f"camera process {worker_index} stopped")


def _camera_loop(pipe, stop: threading.Event, emit=None):
    """Decode, track, apply rules. No model is loaded on this thread.

    Also reports how many frames it has produced, about once a second. The
    parent restarts a process whose cameras stop counting -- see
    ProcessCameraManager._watch. A COUNT rather than a ping, because a loop
    that is alive and returning None on every frame is exactly the failure
    this is meant to catch, and a ping cannot tell the two apart.
    """
    target = float(getattr(pipe, "_analysis_fps", 0) or 0)
    # The pipeline gates its own analysis rate by wall clock; this loop only
    # needs to offer frames often enough not to be the limit.
    interval = 1.0 / max(target * 2.0, 30.0)
    frames = 0
    last_beat = 0.0
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            if pipe.process_frame() is not None:
                frames += 1
        except Exception as e:
            logger.error(f"[{pipe.camera_id}] frame failed: {e}")
            time.sleep(0.5)
            continue

        if emit is not None and t0 - last_beat >= 1.0:
            last_beat = t0
            emit({
                "event_type": "_heartbeat",
                "camera_id": pipe.camera_id,
                "frames": frames,
            })

        rest = interval - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)
