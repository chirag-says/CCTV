"""
Body appearance for cross-camera handoff.

WHY THIS EXISTS
---------------
Handoff can follow a person by where they are (see handoff.py and topology.py)
but geometry runs out exactly where it is needed most: three friends leave one
camera together and arrive on the next together. They took the same route in
the same second, so timing cannot say which is which, and the numbers get dealt
out in arrival order. What still tells them apart is what they are wearing.

This module turns a track's body crop into a vector with a network trained for
person re-identification (OSNet x1.0, trained on MSMT17), and hands the vector
to the handoff engine.

WHAT IT IS GOOD FOR, MEASURED
-----------------------------
On eight minutes of real COE footage, comparing whole tracks:

    same person, same side seen by both cameras     0.85 - 0.93
    same person, front on one camera, back on other  0.52 - 0.85
    two different people                             0.33 - 0.65

So it separates people cleanly when two cameras see them from the same side,
and only weakly when the cameras face each other. The engine uses it
accordingly: to choose among people who already fit by route or position, to
undo a stitch that joined two different people, and to carry a number on its
own only when the match is unmistakable.

It does NOT replace reid.py's ReIDTracker, and deliberately shares nothing
with it: that one follows settings.REID_MODEL, which defaults to an ImageNet
classifier that cannot tell two people apart (see ground_plane.py). This one
always loads re-identification weights or does not run at all.

WHERE IT RUNS
-------------
Camera processes never load a model (see camera_process.py). They cut the
crop, shrink it to the network's input size and send it as a JPEG on the event
queue they already have -- about 8 kB, a few per second per camera. The model
lives here in the main process, next to the engine that consumes its output,
and embeds whatever has arrived as one batch.

Crops are never written to disk and never leave the machine. Only the vector
reaches the engine, and it is forgotten with the track.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# The network's input, width x height. Crops travel at this size.
CROP_W, CROP_H = 128, 256

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Camera side: which crops are worth sending ───────────────────────────────

def usable_body_box(
    box: Sequence[float],
    frame_w: int,
    frame_h: int,
    others: Sequence[Sequence[float]] = (),
    min_height_px: int = 96,
) -> bool:
    """Is this (x1, y1, x2, y2) box a whole, unobstructed, readable person?

    A crop that fails here is worse than no crop: it produces a confident
    vector describing something other than the person.

      * too small    -- below about 96 px every small blurry figure looks like
                        every other. A through-glass figure 60 px tall scored
                        0.95 against a different man at a bench.
      * cut off      -- a box touching the frame edge is half a person.
      * not upright  -- someone crouching, or a box that has swallowed a table.
      * overlapped   -- two people in one crop is a vector for neither.
    """
    x1, y1, x2, y2 = (float(c) for c in box)
    w, h = x2 - x1, y2 - y1
    if h < min_height_px or w < 16 or h / w < 1.2:
        return False
    if x1 <= 2 or y1 <= 2 or x2 >= frame_w - 2 or y2 >= frame_h - 2:
        return False
    area = w * h
    for other in others:
        ox1, oy1, ox2, oy2 = (float(c) for c in other)
        iw = min(x2, ox2) - max(x1, ox1)
        ih = min(y2, oy2) - max(y1, oy1)
        if iw > 0 and ih > 0 and (iw * ih) / area > 0.35:
            return False
    return True


def encode_body_crop(frame: np.ndarray, box: Sequence[float], quality: int = 90) -> Optional[bytes]:
    """Cut a body box out of a BGR frame and pack it for the event queue."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(c))) for c in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    shrinking = crop.shape[0] > CROP_H
    crop = cv2.resize(crop, (CROP_W, CROP_H),
                      interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buf.tobytes() if ok else None


class CropSchedule:
    """Which tracks are due a crop. One per pipeline; not thread-safe.

    A track is looked at quickly when it is new, because that is when the
    handoff engine is deciding who it is, and rarely after that.
    """

    def __init__(self, early_interval: float = 0.8, early_count: int = 5,
                 late_interval: float = 4.0, per_frame: int = 3):
        self.early_interval = float(early_interval)
        self.early_count = int(early_count)
        self.late_interval = float(late_interval)
        self.per_frame = int(per_frame)
        self._sent: dict = {}          # track id -> (count, last time)

    def due(self, track_id: int, now: float) -> bool:
        count, last = self._sent.get(track_id, (0, float("-inf")))
        return now - last >= (self.early_interval if count < self.early_count else self.late_interval)

    def sent(self, track_id: int, now: float) -> None:
        count, _last = self._sent.get(track_id, (0, 0.0))
        self._sent[track_id] = (count + 1, now)

    def forget_except(self, live_track_ids) -> None:
        if len(self._sent) > 4 * max(8, len(live_track_ids)):
            keep = set(live_track_ids)
            self._sent = {k: v for k, v in self._sent.items() if k in keep}


# ── Main-process side: crops to vectors ──────────────────────────────────────

def default_weights_path() -> str:
    backend = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(backend, "models", "osnet_x1_0_msmt17.pth")


class AppearanceService:
    """Embeds queued body crops in batches and reports each vector."""

    def __init__(
        self,
        on_embedding: Callable[[str, int, np.ndarray, float], None],
        weights_path: Optional[str] = None,
        batch_size: int = 32,
        max_pending: int = 256,
        embed: Optional[Callable[[List[np.ndarray]], np.ndarray]] = None,
    ):
        """
        Args:
            on_embedding: called with (camera_id, track_id, unit vector, timestamp).
            embed: replaces the network (tests). Takes BGR crops, returns rows.
        """
        self._on_embedding = on_embedding
        self._weights_path = weights_path or default_weights_path()
        self._batch_size = int(batch_size)
        self._pending: "queue.Queue[tuple]" = queue.Queue(maxsize=int(max_pending))
        self._embed = embed
        self._model = None
        self._device = None
        self._failed = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.stats = {"submitted": 0, "dropped": 0, "embedded": 0, "batches": 0, "last_batch_ms": 0.0}

    @property
    def available(self) -> bool:
        return not self._failed

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="appearance", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def submit(self, camera_id: str, track_id: int, timestamp: float, jpeg: bytes) -> bool:
        """Queue one crop. Never blocks: this is called on the event pump, and
        a late look is worth less than a late box. Returns False if dropped."""
        if self._failed or not jpeg:
            return False
        try:
            self._pending.put_nowait((str(camera_id), int(track_id), float(timestamp), jpeg))
        except queue.Full:
            self.stats["dropped"] += 1
            return False
        self.stats["submitted"] += 1
        return True

    # -- worker --

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._pending.get_nowait())
                except queue.Empty:
                    break
            try:
                self._process(batch)
            except Exception as e:                      # noqa: BLE001
                # Handoff works without looks; a broken model must cost the
                # looks, not the event pump or the engine.
                self._failed = True
                logger.error(
                    f"[appearance] disabled after a failure ({e}). Cross-camera handoff "
                    f"continues on position and timing alone.", exc_info=True,
                )
                return

    def _process(self, batch: List[tuple]) -> None:
        import cv2

        started = time.perf_counter()
        crops, kept = [], []
        for item in batch:
            image = cv2.imdecode(np.frombuffer(item[3], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            if image.shape[0] != CROP_H or image.shape[1] != CROP_W:
                image = cv2.resize(image, (CROP_W, CROP_H))
            crops.append(image)
            kept.append(item)
        if not crops:
            return
        vectors = self._embed(crops) if self._embed is not None else self._embed_with_model(crops)
        for (camera_id, track_id, timestamp, _jpeg), vector in zip(kept, vectors):
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9 and np.isfinite(norm):
                self._on_embedding(camera_id, track_id, np.asarray(vector, dtype=np.float32) / norm, timestamp)
        self.stats["embedded"] += len(crops)
        self.stats["batches"] += 1
        self.stats["last_batch_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

    def _load(self) -> None:
        import torch
        import torchreid

        if not os.path.exists(self._weights_path):
            # Random weights would produce vectors that look fine and mean
            # nothing. Refusing is the only honest behaviour.
            raise FileNotFoundError(
                f"re-identification weights not found at {self._weights_path}"
            )
        model = torchreid.models.build_model(name="osnet_x1_0", num_classes=1, pretrained=False)
        state = torch.load(self._weights_path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        own = model.state_dict()
        matched = {
            k.removeprefix("module."): v for k, v in state.items()
            if k.removeprefix("module.") in own and v.shape == own[k.removeprefix("module.")].shape
        }
        # The classifier head is expected to be missing; the backbone is not.
        if len(matched) < 0.9 * len(own):
            raise RuntimeError(
                f"only {len(matched)} of {len(own)} weights matched {self._weights_path}"
            )
        own.update(matched)
        model.load_state_dict(own)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device).eval()
        logger.info(
            f"[appearance] OSNet x1.0 (MSMT17) ready on {self._device}: "
            f"{len(matched)}/{len(own)} weights"
        )

    def _embed_with_model(self, crops: List[np.ndarray]) -> np.ndarray:
        import torch

        if self._model is None:
            self._load()
        # BGR uint8 -> RGB float, normalised, NCHW.
        batch = np.stack(crops).astype(np.float32)[:, :, :, ::-1] / 255.0
        batch = (batch - _MEAN) / _STD
        tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(self._device)
        with torch.no_grad():
            out = self._model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
        return out.float().cpu().numpy()


# ── Singleton ────────────────────────────────────────────────────────────────

_service: Optional[AppearanceService] = None
_service_lock = threading.Lock()
_service_checked = False


def get_appearance() -> Optional[AppearanceService]:
    """Process-wide service, or None when handoff or appearance is switched off."""
    global _service, _service_checked
    if _service_checked:
        return _service
    with _service_lock:
        if _service_checked:
            return _service
        from app.config import settings
        from app.vision.handoff import get_handoff

        engine = get_handoff()
        if engine is not None and bool(getattr(settings, "HANDOFF_APPEARANCE_ENABLED", True)):
            def deliver(camera_id: str, track_id: int, vector: np.ndarray, timestamp: float) -> None:
                engine.add_appearance(camera_id, track_id, vector, timestamp)

            _service = AppearanceService(
                on_embedding=deliver,
                weights_path=getattr(settings, "HANDOFF_APPEARANCE_WEIGHTS", "") or None,
            )
            _service.start()
        _service_checked = True
    return _service
