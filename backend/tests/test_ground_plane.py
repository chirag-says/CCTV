"""
Ground-plane geometry: the image -> floor mapping that lets two cameras facing
each other agree on where a person is standing.

The scenario throughout is the real one: a corridor section 4 m wide and 10 m
long, viewed obliquely, so the near edge is wide at the bottom of the image and
the far edge is narrow near the top. Every assertion below is a property the
cross-camera association depends on, not a restatement of the implementation.

Pure geometry — no database, no camera, no model.
"""

import math

import numpy as np
import pytest

from app.vision.ground_plane import (
    CalibrationError,
    GroundPlane,
    calibration_problems,
    fit_homography,
    foot_point,
)


# A 4 m x 10 m floor rectangle, corners ordered near-left, near-right,
# far-right, far-left.
WORLD = [[0.0, 0.0], [4.0, 0.0], [4.0, 10.0], [0.0, 10.0]]

# The same four corners as one camera sees them: a trapezoid, near edge wide
# and low, far edge narrow and high.
IMAGE = [[100.0, 400.0], [540.0, 400.0], [400.0, 150.0], [240.0, 150.0]]


@pytest.fixture
def plane() -> GroundPlane:
    return GroundPlane.build("cam-a", "COE", IMAGE, WORLD)


# ── The fit itself ───────────────────────────────────────────────────────────

def test_calibration_points_map_to_their_world_points(plane):
    for (u, v), (wx, wy) in zip(IMAGE, WORLD):
        got = plane.project(u, v)
        assert got is not None
        assert got[0] == pytest.approx(wx, abs=1e-6)
        assert got[1] == pytest.approx(wy, abs=1e-6)


def test_exact_four_point_fit_has_no_residual(plane):
    assert plane.residual < 1e-6


def test_midpoints_land_where_geometry_says_they_should(plane):
    """
    The midpoint of the near edge in the image must be the midpoint of the near
    edge on the floor. This is the property that makes two cameras agree: it
    holds for any point on the plane, not just the ones that were clicked.
    """
    near_mid = plane.project((100.0 + 540.0) / 2, 400.0)
    assert near_mid[0] == pytest.approx(2.0, abs=1e-6)
    assert near_mid[1] == pytest.approx(0.0, abs=1e-6)

    far_mid = plane.project((400.0 + 240.0) / 2, 150.0)
    assert far_mid[0] == pytest.approx(2.0, abs=1e-6)
    assert far_mid[1] == pytest.approx(10.0, abs=1e-6)


def test_two_cameras_agree_on_the_same_person():
    """
    The whole point. A second camera at the far end sees the SAME floor
    rectangle from the opposite side, so its near edge is the other one's far
    edge and its image points are ordered accordingly. A person standing at a
    given floor position must come out at the same world coordinate in both,
    even though nothing about the two images resembles the other.
    """
    cam_a = GroundPlane.build("cam-a", "COE", IMAGE, WORLD)
    # Facing the other way: this camera's near edge is the far edge (y=10),
    # and left/right swap because it is looking from the opposite direction.
    image_b = [[120.0, 420.0], [560.0, 420.0], [420.0, 170.0], [260.0, 170.0]]
    world_b = [[4.0, 10.0], [0.0, 10.0], [0.0, 0.0], [4.0, 0.0]]
    cam_b = GroundPlane.build("cam-b", "COE", image_b, world_b)

    # Pick a floor point and find where each camera would see it, by inverting
    # each mapping, then confirm both project back to the same world position.
    target = np.array([2.5, 6.0, 1.0])
    for cam in (cam_a, cam_b):
        inv = np.linalg.inv(cam.H)
        p = inv @ target
        u, v = p[0] / p[2], p[1] / p[2]
        got = cam.project(u, v)
        assert got is not None
        assert got[0] == pytest.approx(2.5, abs=1e-6)
        assert got[1] == pytest.approx(6.0, abs=1e-6)


# ── The horizon ──────────────────────────────────────────────────────────────

def test_pixels_beyond_the_horizon_are_rejected(plane):
    """
    Above the vanishing line a pixel corresponds to no floor point. The
    transform will still return a number, and it will be confident and wrong —
    so project() must refuse instead. Scan upward and require that it does.
    """
    rejected = [v for v in range(150, -400, -5) if plane.project(320.0, float(v)) is None]
    assert rejected, "no pixel above the far edge was ever rejected as off-plane"


def test_distance_grows_monotonically_up_the_image(plane):
    """Walking away from the camera must not make the reported distance jump around."""
    ys = []
    for v in range(400, 155, -10):
        pos = plane.project(320.0, float(v))
        assert pos is not None
        ys.append(pos[1])
    assert all(b > a for a, b in zip(ys, ys[1:])), "distance is not monotonic up the image"


# ── Uncertainty ──────────────────────────────────────────────────────────────

def test_a_pixel_covers_more_floor_further_away(plane):
    """
    This is what makes the far camera's estimate weak, and it is why fusion
    must weight by variance rather than averaging. If this ordering ever
    inverts, the weighting is backwards and the worse camera wins.
    """
    near = plane.metres_per_pixel(320.0, 395.0)
    mid = plane.metres_per_pixel(320.0, 275.0)
    far = plane.metres_per_pixel(320.0, 155.0)
    assert near < mid < far
    assert far > 3 * near, "far/near scale ratio is implausibly flat for a perspective view"


def test_sigma_scales_with_assumed_pixel_noise(plane):
    a = plane.position_sigma(320.0, 300.0, noise_px=1.0)
    b = plane.position_sigma(320.0, 300.0, noise_px=6.0)
    assert b == pytest.approx(6.0 * a, rel=1e-9)


def test_off_plane_pixels_have_no_usable_observation(plane):
    """A rejected pixel must yield no observation at all, not a zero or an origin."""
    v = next(v for v in range(150, -400, -5) if plane.project(320.0, float(v)) is None)
    assert plane.observe(320.0, float(v), noise_px=6.0) is None
    assert math.isinf(plane.metres_per_pixel(320.0, float(v)))


def test_observe_returns_position_and_sigma(plane):
    obs = plane.observe(320.0, 380.0, noise_px=6.0)
    assert obs is not None
    x, y, sigma = obs
    assert 0.0 < x < 4.0
    assert 0.0 <= y < 3.0
    assert sigma > 0.0


# ── Rejecting bad calibrations ───────────────────────────────────────────────

def test_collinear_image_points_are_rejected():
    with pytest.raises(CalibrationError, match="collinear"):
        fit_homography(
            [[0.0, 0.0], [100.0, 100.0], [200.0, 200.0], [300.0, 300.5]],
            WORLD,
        )


def test_duplicate_points_are_rejected():
    with pytest.raises(CalibrationError):
        fit_homography([[10.0, 10.0]] * 4, WORLD)


def test_too_few_points_are_rejected():
    with pytest.raises(CalibrationError, match="at least 4"):
        fit_homography(IMAGE[:3], WORLD[:3])


def test_mismatched_point_counts_are_rejected():
    with pytest.raises(CalibrationError, match="differ"):
        fit_homography(IMAGE + [[300.0, 300.0]], WORLD)


def test_non_finite_points_are_rejected():
    bad = [[float("nan"), 0.0], [540.0, 400.0], [400.0, 150.0], [240.0, 150.0]]
    with pytest.raises(CalibrationError, match="non-finite"):
        fit_homography(bad, WORLD)


def test_sign_convention_is_normalised():
    """
    The solver picks the homography's overall sign arbitrarily. If it lands
    negative and is not normalised, the horizon test rejects every real floor
    point and the camera silently reports nothing at all.
    """
    for image_points in (IMAGE, [list(reversed(p)) for p in IMAGE]):
        try:
            H = fit_homography(image_points, WORLD)
        except CalibrationError:
            continue
        ws = (H @ np.hstack([np.asarray(image_points), np.ones((4, 1))]).T)[2]
        assert np.median(ws) > 0


# ── Persistence ──────────────────────────────────────────────────────────────

def test_config_round_trip_preserves_projection(plane):
    restored = GroundPlane.from_config("cam-a", {"ground_plane": plane.to_config()})
    assert restored is not None
    assert restored.zone == "COE"
    for u, v in [(320.0, 380.0), (200.0, 300.0), (450.0, 200.0)]:
        assert restored.project(u, v) == pytest.approx(plane.project(u, v), abs=1e-9)


def test_uncalibrated_camera_is_not_an_error():
    assert GroundPlane.from_config("cam-x", None) is None
    assert GroundPlane.from_config("cam-x", {}) is None


def test_unrelated_camera_settings_are_not_a_broken_calibration():
    """
    Cameras keep other settings in `config`. Reading those as a failed
    calibration logged a warning per camera on every refresh, forever — noise
    that would bury a real calibration failure.
    """
    assert GroundPlane.from_config("cam-x", {"other_setting": 1}) is None
    assert GroundPlane.from_config("cam-x", {"rtsp_transport": "tcp", "fps": 12}) is None


def test_broken_calibration_degrades_to_none_not_an_exception():
    broken = {"ground_plane": {"zone": "COE", "image_points": [[0, 0]], "world_points": [[0, 0]]}}
    assert GroundPlane.from_config("cam-x", broken) is None


# ── Foot point ───────────────────────────────────────────────────────────────

def test_foot_point_is_bottom_centre_of_the_body_box():
    # (top, right, bottom, left)
    assert foot_point((100.0, 260.0, 400.0, 160.0)) == (210.0, 400.0)


def test_foot_point_differs_from_centroid():
    """
    Using a centroid would sample chest height — off the plane the homography
    is defined on — and place people metres from where they stand.
    """
    box = (100.0, 260.0, 400.0, 160.0)
    u, v = foot_point(box)
    centroid_v = (box[0] + box[2]) / 2
    assert v != centroid_v
    assert v > centroid_v


# ── A calibration that solves is not a calibration that works ─────────────────
#
# Four points always fit themselves, so the residual above is zero for ANY four
# clicks. Both COE corridor cameras were saved with four points up a door post:
# "fit error 0 cm", and every person projected forty metres outside the
# building or beyond the horizon. These tests pin the check that asks the real
# question -- can this mapping locate a person? -- in normalised coordinates,
# which is how calibrations are stored.

FLOOR_NORMALISED = [[0.16, 0.83], [0.84, 0.83], [0.63, 0.31], [0.38, 0.31]]
COE_LEFT_AS_SAVED = [[0.5847, 0.1642], [0.5767, 0.1500], [0.5738, 0.3430], [0.5796, 0.3648]]
COE_RIGHT_AS_SAVED = [[0.4711, 0.1156], [0.4660, 0.1271], [0.4675, 0.2776], [0.4718, 0.2583]]


def test_a_floor_rectangle_has_no_problems():
    assert calibration_problems(FLOOR_NORMALISED, WORLD) == []
    assert GroundPlane.build("cam-a", "COE", FLOOR_NORMALISED, WORLD).problems() == []


def test_a_small_but_real_floor_patch_far_down_a_corridor_is_accepted():
    """The check must not punish a careful operator with a distant rectangle."""
    far_patch = [[0.42, 0.45], [0.58, 0.45], [0.54, 0.28], [0.46, 0.28]]
    assert calibration_problems(far_patch, WORLD) == []


@pytest.mark.parametrize("saved", [COE_LEFT_AS_SAVED, COE_RIGHT_AS_SAVED])
def test_points_up_a_door_post_solve_perfectly_and_are_refused(saved):
    plane = GroundPlane.build("coe", "COE", saved, WORLD)
    assert plane.residual < 1e-3                 # the number the UI showed as "0 cm"
    problems = plane.problems()
    assert problems and "door post" in problems[0]
    # ... and this is what "0 cm" was hiding: most of where people stand is
    # not on this calibration's floor at all.
    lost = sum(plane.project(u / 10, 0.9) is None for u in range(1, 10))
    assert lost >= 3


def test_points_that_put_the_horizon_through_the_floor_are_refused():
    """Right four spots, wrong order: the fit is exact and the floor is inside out."""
    crossed = [FLOOR_NORMALISED[0], FLOOR_NORMALISED[2], FLOOR_NORMALISED[1], FLOOR_NORMALISED[3]]
    assert any("horizon" in p for p in calibration_problems(crossed, WORLD))


def test_unsolvable_points_are_reported_not_raised():
    problems = calibration_problems([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]], WORLD)
    assert len(problems) == 1 and "collinear" in problems[0]
