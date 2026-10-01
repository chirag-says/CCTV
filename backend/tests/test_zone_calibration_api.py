"""
Calibration payload validation.

The dangerous input here is not malformed JSON — it is a payload that LOOKS
fine and produces a confident, wrong homography. Pixels sent where normalised
coordinates are expected is exactly that: it solves, it saves, and every
position the camera reports afterwards is wrong by the frame size. So the
schema rejects it rather than trusting the caller.

Schema-level only; no database and no HTTP client required.
"""

import pytest
from pydantic import ValidationError

from app.routes.zones import CalibrationPayload


GOOD_IMAGE = [[0.16, 0.83], [0.84, 0.83], [0.63, 0.31], [0.38, 0.31]]
GOOD_WORLD = [[0.0, 0.0], [4.0, 0.0], [4.0, 10.0], [0.0, 10.0]]


def test_accepts_a_normalised_payload():
    payload = CalibrationPayload(zone="COE", image_points=GOOD_IMAGE, world_points=GOOD_WORLD)
    assert payload.zone == "COE"
    assert len(payload.image_points) == 4


def test_rejects_pixel_coordinates():
    """
    The failure this catches: a caller sends [320, 400] instead of [0.5, 0.83].
    It would solve happily and be silently wrong forever.
    """
    pixels = [[100.0, 400.0], [540.0, 400.0], [400.0, 150.0], [240.0, 150.0]]
    with pytest.raises(ValidationError, match="normalised"):
        CalibrationPayload(zone="COE", image_points=pixels, world_points=GOOD_WORLD)


def test_allows_a_small_overshoot_at_the_edges():
    """A click landing a pixel outside the frame is an operator, not an error."""
    edge = [[-0.01, 1.01], [1.01, 1.01], [0.63, 0.31], [0.38, 0.31]]
    assert CalibrationPayload(zone="COE", image_points=edge, world_points=GOOD_WORLD)


def test_requires_four_points():
    with pytest.raises(ValidationError):
        CalibrationPayload(zone="COE", image_points=GOOD_IMAGE[:3], world_points=GOOD_WORLD[:3])


def test_rejects_malformed_points():
    with pytest.raises(ValidationError):
        CalibrationPayload(
            zone="COE",
            image_points=[[0.1, 0.2, 0.3]] + GOOD_IMAGE[1:],
            world_points=GOOD_WORLD,
        )


def test_requires_a_zone():
    with pytest.raises(ValidationError):
        CalibrationPayload(zone="", image_points=GOOD_IMAGE, world_points=GOOD_WORLD)


def test_world_points_are_metres_not_normalised():
    """World points are real distances, so they are NOT range-checked to 0..1."""
    big = [[0.0, 0.0], [12.0, 0.0], [12.0, 40.0], [0.0, 40.0]]
    assert CalibrationPayload(zone="COE", image_points=GOOD_IMAGE, world_points=big)
