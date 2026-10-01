"""
Ground-plane geometry — maps image pixels onto a zone's shared floor.

WHY THIS EXISTS
---------------
Two cameras facing each other down the COE corridor cannot associate a person
by appearance:

  * FACE is visible to at most ONE of them at a time. The camera behind you
    sees the back of your head, so there is no embedding to compare. ArcFace
    and the UnknownRegistry are structurally blind here, not mistuned.
  * BODY appearance is unusable at the far camera, where a person is a few
    dozen pixels tall. (reid.py would not save this either: it runs an
    ImageNet-pretrained ResNet50 with no re-identification training, so its
    features answer "is this a person", not "which person".)

What both cameras DO share is the floor. A person stands on exactly one point
of it. Give each camera a mapping from its image to that floor and both
compute the same (x, y) — so association becomes "same place, same person",
needing no face and no readable body. It survives turning away, being small in
frame, and having your face covered.

The position this produces is not a means to the association; it IS the
positioning the feature exists to provide. One mechanism, both goals.

WHAT THE MAPPING IS
-------------------
A homography: the 3x3 projective transform between two planes. It is exact for
points ON the floor and meaningless for anything off it, which is why the input
is a person's FOOT point (bottom-centre of the body box) rather than their
centroid. Step 1's switch to body-box tracking is what makes that point
available at all; face boxes never touch the ground.

ACCURACY, HONESTLY
------------------
Ground-plane error grows without bound as a pixel approaches the horizon: near
the vanishing line one pixel of image spans an unbounded stretch of floor. So a
distant person's position is genuinely poor, and any consumer that treats all
estimates as equal will be dominated by its worst one.

metres_per_pixel() measures that locally, by differencing the actual transform
rather than assuming anything about the camera, and position_sigma() turns it
into a usable standard deviation. Fusing inverse-variance then does the right
thing on its own: the near camera dominates the position while the far camera
still contributes enough to ASSOCIATE — a far lower bar, since it only has to
place you nearer the correct candidate than an incorrect one.

Points at or above the horizon return None. They are not floor points, and a
transform will happily return a confident, enormous, wrong number for them.

THREADING
---------
A GroundPlane is immutable once built, so it can be shared across every camera
thread without a lock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Minimum correspondences for a homography. Four is exact; more is
# least-squares and lets a careful operator improve the fit.
MIN_POINTS = 4

# Below this, the projective denominator means the pixel is on or beyond the
# horizon and does not correspond to any floor point.
_W_EPS = 1e-6


class CalibrationError(ValueError):
    """The supplied correspondences do not define a usable ground plane."""


def _as_points(points: Sequence[Sequence[float]], what: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise CalibrationError(f"{what} must be a list of [x, y] pairs, got shape {arr.shape}")
    if len(arr) < MIN_POINTS:
        raise CalibrationError(f"{what} needs at least {MIN_POINTS} points, got {len(arr)}")
    if not np.isfinite(arr).all():
        raise CalibrationError(f"{what} contains non-finite values")
    return arr


def _any_three_collinear(pts: np.ndarray) -> bool:
    """
    A homography needs the four points in 'general position'. Three collinear
    points make the system rank-deficient, and OpenCV will still return a
    matrix — one that produces confident nonsense. Cheaper to reject here than
    to debug a camera that silently reports wrong positions.

    Scale-aware: the cross product is compared against the triangle's own edge
    lengths, so this means "these are nearly a straight line" rather than
    "these are close together in pixels".
    """
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = pts[i], pts[j], pts[k]
                # 2-D cross product written out: np.cross on 2-vectors is
                # deprecated in NumPy 2.
                area2 = abs(
                    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                )
                scale = max(
                    float(np.linalg.norm(b - a)),
                    float(np.linalg.norm(c - a)),
                    float(np.linalg.norm(c - b)),
                )
                if scale <= 0:
                    return True  # duplicate points
                if area2 / (scale * scale) < 0.02:
                    return True
    return False


def fit_homography(
    image_points: Sequence[Sequence[float]],
    world_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """
    Solve for the image -> world (floor) transform.

    Raises CalibrationError rather than returning a degenerate matrix: a bad
    homography does not fail loudly at use time, it just reports wrong
    positions forever.
    """
    img = _as_points(image_points, "image_points")
    wld = _as_points(world_points, "world_points")
    if len(img) != len(wld):
        raise CalibrationError(
            f"point counts differ: {len(img)} image vs {len(wld)} world"
        )
    if _any_three_collinear(img):
        raise CalibrationError(
            "image points are nearly collinear — pick four floor points that "
            "form a wide quadrilateral, not a line"
        )
    if _any_three_collinear(wld):
        raise CalibrationError("world points are nearly collinear")

    import cv2

    H, _ = cv2.findHomography(img, wld, method=0)
    if H is None or not np.isfinite(H).all():
        raise CalibrationError("could not solve for a homography from these points")
    if abs(float(np.linalg.det(H))) < 1e-12:
        raise CalibrationError("homography is singular — check the point order matches in both views")
    H = np.asarray(H, dtype=np.float64)

    # A homography is defined only up to scale -- INCLUDING sign, which the
    # solver picks arbitrarily. The horizon test in project() reads the sign of
    # the projective denominator, so a negated H would make it reject every
    # real floor point. Calibration points are floor points by definition, so
    # normalise the sign to make their denominator positive.
    ws = (H @ np.hstack([img, np.ones((len(img), 1))]).T)[2]
    if np.median(ws) < 0:
        H = -H
    return H


@dataclass(frozen=True)
class GroundPlane:
    """
    One camera's mapping onto its zone's floor coordinate system.

    `world_points` are shared by every camera in the zone — the same physical
    floor points, entered in the same order — which is precisely what puts two
    cameras in a common frame. `image_points` are where that camera sees them.
    """

    camera_id: str
    zone: str
    image_points: np.ndarray
    world_points: np.ndarray
    H: np.ndarray
    residual: float          # worst calibration point reprojection error, metres
    updated_at: Optional[str] = None

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        camera_id: str,
        zone: str,
        image_points: Sequence[Sequence[float]],
        world_points: Sequence[Sequence[float]],
        updated_at: Optional[str] = None,
    ) -> "GroundPlane":
        img = _as_points(image_points, "image_points")
        wld = _as_points(world_points, "world_points")
        H = fit_homography(img, wld)
        plane = cls(
            camera_id=camera_id,
            zone=zone,
            image_points=img,
            world_points=wld,
            H=H,
            residual=0.0,
            updated_at=updated_at,
        )
        # Reprojection residual is the only honest quality signal available at
        # calibration time, and it is what tells an operator they misclicked.
        # Reported, not enforced: a slightly sloppy calibration is still far
        # better than none, and a hard threshold here would reject usable work.
        worst = 0.0
        for (u, v), target in zip(img, wld):
            got = plane.project(float(u), float(v))
            if got is None:
                raise CalibrationError(
                    "a calibration point maps to the horizon — the point order "
                    "probably differs between the two point sets"
                )
            worst = max(worst, float(np.hypot(got[0] - target[0], got[1] - target[1])))
        return cls(
            camera_id=camera_id,
            zone=zone,
            image_points=img,
            world_points=wld,
            H=H,
            residual=worst,
            updated_at=updated_at,
        )

    # ── Projection ───────────────────────────────────────────────────────

    def project(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        Map an image pixel to floor coordinates.

        Returns None for pixels on or beyond the horizon, which correspond to
        no floor point at all. Callers MUST handle None; treating it as an
        origin or a zero would place a distant person at the centre of the map.
        """
        p = self.H @ np.array([u, v, 1.0], dtype=np.float64)
        w = float(p[2])
        if w <= _W_EPS or not np.isfinite(w):
            return None
        x, y = float(p[0]) / w, float(p[1]) / w
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return x, y

    def metres_per_pixel(self, u: float, v: float) -> float:
        """
        Local scale: how much floor one pixel covers here.

        Measured by differencing the transform itself rather than modelling the
        camera, so it needs no intrinsics and stays correct for whatever lens
        and mounting this camera actually has. Returns inf where the estimate
        is meaningless (at the horizon, or where a neighbouring pixel crosses
        it) — which flows through position_sigma into a weight of zero, so a
        useless observation is ignored rather than trusted.
        """
        here = self.project(u, v)
        if here is None:
            return float("inf")
        right = self.project(u + 1.0, v)
        down = self.project(u, v + 1.0)
        if right is None or down is None:
            return float("inf")
        du = np.hypot(right[0] - here[0], right[1] - here[1])
        dv = np.hypot(down[0] - here[0], down[1] - here[1])
        # Max, not mean: near the horizon the vertical direction blows up while
        # the horizontal stays tame, and averaging would hide exactly the error
        # that matters.
        scale = float(max(du, dv))
        return scale if np.isfinite(scale) else float("inf")

    def position_sigma(self, u: float, v: float, noise_px: float) -> float:
        """
        Standard deviation of the floor position implied by `noise_px` pixels
        of uncertainty in the foot point. Feeds inverse-variance fusion.
        """
        return self.metres_per_pixel(u, v) * max(noise_px, 1e-3)

    def observe(
        self, u: float, v: float, noise_px: float
    ) -> Optional[Tuple[float, float, float]]:
        """Convenience: (x, y, sigma), or None if this pixel is not on the floor."""
        pos = self.project(u, v)
        if pos is None:
            return None
        sigma = self.position_sigma(u, v, noise_px)
        if not np.isfinite(sigma):
            return None
        return pos[0], pos[1], sigma

    # ── Persistence ──────────────────────────────────────────────────────

    def to_config(self) -> dict:
        """
        Serialised form for `cameras.config`, which is already a JSON column —
        so this needs no migration and no new table.

        The POINTS are the record, not the matrix: they are what an operator
        entered, they can be re-checked against a snapshot, and H is re-derived
        on load. Storing only the matrix would make a mis-click permanently
        opaque.
        """
        return {
            "zone": self.zone,
            "image_points": [[float(u), float(v)] for u, v in self.image_points],
            "world_points": [[float(x), float(y)] for x, y in self.world_points],
            "residual": float(self.residual),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_config(cls, camera_id: str, config: Optional[dict]) -> Optional["GroundPlane"]:
        """
        Rebuild from `cameras.config['ground_plane']`.

        Returns None when a camera is simply uncalibrated, which is the normal
        state for most of the fleet and must not be an error. A calibration
        that exists but no longer solves IS logged: silently degrading to
        "uncalibrated" would hide a real regression.
        """
        if not config:
            return None
        # Accept either the whole camera config or just the calibration
        # sub-dict. Anything else is an UNCALIBRATED camera, not a broken one:
        # cameras carry unrelated settings in `config`, and treating those as a
        # failed calibration logged a warning per camera per refresh forever.
        raw = config.get("ground_plane")
        if raw is None and "image_points" in config:
            raw = config
        if not raw:
            return None
        try:
            return cls.build(
                camera_id=camera_id,
                zone=raw.get("zone", ""),
                image_points=raw["image_points"],
                world_points=raw["world_points"],
                updated_at=raw.get("updated_at"),
            )
        except (CalibrationError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Camera {camera_id} has an unusable ground-plane calibration: {e}")
            return None


def foot_point(location: Sequence[float]) -> Tuple[float, float]:
    """
    Where a body box meets the floor: bottom-centre.

    `location` is the codebase's (top, right, bottom, left) box convention.
    The centroid is deliberately NOT used — it floats at chest height, off the
    plane the homography is defined on, and would place people metres away.
    """
    top, right, bottom, left = (float(x) for x in location[:4])
    return (left + right) / 2.0, bottom
