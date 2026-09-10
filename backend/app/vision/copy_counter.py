"""
Count host copies of a frame, per camera.

WHY THIS EXISTS AS A COUNTER AND NOT AS AN ASSERTION
"Zero copy" is a claim, and claims about performance rot silently. A frame
that starts on the GPU and reaches a model without touching host memory looks
identical, from the outside, to one that made two round trips -- same boxes,
same events, just slower and with a core busy. The only difference visible
from inside the process is whether anyone copied it, so that is what is
counted.

WHAT COUNTS AS A COPY
One increment per DEVICE-TO-HOST transfer of a whole frame. Not the boxes,
not the embeddings, not a 112x112 face crop -- those are the things that are
SUPPOSED to cross. Device-to-device blits do not count either: they never
touch the CPU, which is the property being measured.

Every increment names its reason, so a non-zero total says WHERE the copy
happened rather than merely that one did. Today there is exactly one such
reason on the NVDEC path, and it is honest about being unfinished:
`nvdec_read_host` in app/vision/nvdecode.py, which materialises the numpy
frame the tracking, drawing and rules code is written against.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from typing import Dict


class FrameCopyCounter:
    """Per-camera tallies of host copies, and why each happened."""

    def __init__(self):
        self._lock = threading.Lock()
        self._by_camera: Dict[str, Counter] = defaultdict(Counter)
        self._frames: Counter = Counter()

    def note_copy(self, camera_id: str, reason: str, n: int = 1):
        with self._lock:
            self._by_camera[camera_id][reason] += n

    def note_frame(self, camera_id: str, n: int = 1):
        """One analysed frame, the denominator for copies-per-frame."""
        with self._lock:
            self._frames[camera_id] += n

    def snapshot(self, camera_id: str) -> dict:
        with self._lock:
            reasons = dict(self._by_camera.get(camera_id, {}))
            frames = self._frames.get(camera_id, 0)
        total = sum(reasons.values())
        return {
            "frames": frames,
            "copies": total,
            "copies_per_frame": round(total / frames, 3) if frames else 0.0,
            "reasons": reasons,
        }

    def describe(self, camera_id: str) -> str:
        """One line for the log. Says zero plainly when it is zero."""
        s = self.snapshot(camera_id)
        if not s["frames"]:
            return f"[{camera_id}] no analysed frames yet"
        if s["copies"] == 0:
            return (
                f"[{camera_id}] host copies per frame: 0 "
                f"({s['frames']} frames, frame never left the GPU)"
            )
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(s["reasons"].items()))
        return (
            f"[{camera_id}] host copies per frame: {s['copies_per_frame']} "
            f"({s['copies']} over {s['frames']} frames; {reasons})"
        )

    def reset(self, camera_id: str = None):
        with self._lock:
            if camera_id is None:
                self._by_camera.clear()
                self._frames.clear()
            else:
                self._by_camera.pop(camera_id, None)
                self._frames.pop(camera_id, None)


# One per process. Camera processes each keep their own, which is correct:
# the copy being counted happens in the process that decodes.
copy_counter = FrameCopyCounter()
