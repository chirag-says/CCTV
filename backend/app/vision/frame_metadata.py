"""
Detections as metadata, not as pixels.

WHY THIS EXISTS
Until Phase 3 the only way to see what a camera saw was an annotated JPEG:
the pipeline drew boxes into the frame, an encoder thread turned that into
JPEG, and the browser displayed a picture of a decision. That made the
overlay unselectable, unstyleable and un-queryable, and it cost 8.9 ms of CPU
per frame per viewer to say something that fits in 200 bytes.

Video now comes from go2rtc over WebRTC, untouched by us. This module builds
the other half: the boxes, as data, in coordinates that do not depend on
whatever resolution the relay negotiated.

TWO RULES, BOTH ENFORCED HERE RATHER THAN AT THE CALL SITES
  1. Person and face only. The pipeline also tracks vehicles and reads number
     plates; neither belongs on a channel the live-view page subscribes to.
     A vehicle box is a movement record and a plate is personal data, and
     both would leak to every browser holding the monitor page open.
  2. Coordinates are normalised to 0..1 against the frame they were measured
     in. The analysis frame, the display frame and whatever WebRTC negotiates
     are three different sizes; a pixel box would be wrong in two of them.

Pure functions, no pipeline imports: this is the thing the test pins.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional

# The only kinds that may appear on the live metadata channel. Anything else
# -- "vehicle", "plate", "hazard" -- is dropped, silently and by design.
ALLOWED_KINDS = ("person", "face")

MESSAGE_TYPE = "detections"


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def normalise_box(
    location: tuple,
    frame_width: int,
    frame_height: int,
) -> Optional[List[float]]:
    """(top, right, bottom, left) pixels -> [x1, y1, x2, y2] in 0..1.

    Returns None for a box that survives clamping as zero-area. A box that is
    entirely off-frame, or a frame with a nonsense size, produces nothing
    rather than a degenerate rectangle the canvas would draw as a line.
    """
    if not frame_width or not frame_height or frame_width < 0 or frame_height < 0:
        return None
    try:
        top, right, bottom, left = (float(v) for v in location)
    except (TypeError, ValueError):
        return None

    x1 = _clamp01(min(left, right) / frame_width)
    x2 = _clamp01(max(left, right) / frame_width)
    y1 = _clamp01(min(top, bottom) / frame_height)
    y2 = _clamp01(max(top, bottom) / frame_height)

    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5)]


def build_frame_metadata(
    camera_id: str,
    frame_width: int,
    frame_height: int,
    entries: Iterable[dict],
    timestamp: Optional[float] = None,
    frame_id: Optional[int] = None,
) -> dict:
    """One message describing one analysed frame.

    `entries` are raw pipeline observations:
        {"kind": "person"|"face", "box": (top, right, bottom, left),
         "track_id": int|None, "label": str|None, "identity": str|None}

    Everything that is not a person or a face is dropped, every box is
    normalised, and nothing else from the entry is carried through -- the
    caller cannot widen this message by adding keys upstream.
    """
    objects = []
    for entry in entries or ():
        kind = entry.get("kind")
        if kind not in ALLOWED_KINDS:
            continue
        box = normalise_box(entry.get("box"), frame_width, frame_height)
        if box is None:
            continue

        track_id = entry.get("track_id")
        identity = entry.get("identity") or None
        label = entry.get("label") or identity or kind

        objects.append({
            "kind": kind,
            # Explicitly camera-local. Kept as `track_id` for compatibility.
            "track_id": int(track_id) if track_id is not None else None,
            "local_track_id": int(track_id) if track_id is not None else None,
            "global_id": str(entry["global_id"]) if entry.get("global_id") else None,
            "known_person_id": (
                str(entry["known_person_id"]) if entry.get("known_person_id") else None
            ),
            "identity_state": (
                str(entry["identity_state"]) if entry.get("identity_state") else None
            ),
            "identity_source": (
                str(entry["identity_source"]) if entry.get("identity_source") else None
            ),
            "last_camera": str(entry["last_camera"]) if entry.get("last_camera") else None,
            "last_seen": float(entry["last_seen"]) if entry.get("last_seen") is not None else None,
            "box": box,
            "label": str(label),
            "identity": str(identity) if identity else None,
            "coasting": bool(entry.get("coasting", False)),
        })

    return {
        "type": MESSAGE_TYPE,
        "camera_id": camera_id,
        # Wall clock, seconds. The browser matches this against the frame it
        # is displaying, so it has to be the same clock the video timestamps
        # come from -- not a monotonic counter.
        "timestamp": time.time() if timestamp is None else float(timestamp),
        "frame_id": frame_id,
        "objects": objects,
    }
