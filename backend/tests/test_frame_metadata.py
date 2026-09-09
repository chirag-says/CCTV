"""
The metadata message for one frame: person and face only, normalised boxes.

WHY THESE TWO PROPERTIES AND NOT OTHERS
This channel is a broadcast to every browser holding the monitor page open,
and the pipeline that feeds it also tracks vehicles and reads number plates.
A plate on this socket is personal data leaving the system to anyone with a
dashboard login and a devtools tab; a vehicle box is a movement record. The
filter is the security boundary, so it is pinned here rather than trusted to
the call sites.

The coordinates matter for a duller reason. Three different frame sizes exist
between the detector and the canvas -- the analysis frame, the display frame,
and whatever resolution WebRTC negotiates with go2rtc. A pixel box is correct
in exactly one of them. Normalised, it is correct in all three.

No pipeline, no models, no camera.
"""

import pytest

from app.vision.frame_metadata import (
    ALLOWED_KINDS,
    build_frame_metadata,
    normalise_box,
)

W, H = 1920, 1080


def entry(kind, box=(100, 300, 400, 100), **kw):
    base = {"kind": kind, "box": box, "track_id": 7,
            "label": "Abhitha", "identity": "Abhitha"}
    base.update(kw)
    return base


def msg(entries, width=W, height=H):
    return build_frame_metadata("cam-1", width, height, entries)


# ── the filter ────────────────────────────────────────────────────────────

def test_only_person_and_face_survive():
    """The one test the brief asks for: what the pipeline offers vs what the
    browser is allowed to receive."""
    out = msg([
        entry("person"),
        entry("face", box=(120, 260, 200, 180)),
        entry("vehicle", box=(500, 900, 800, 400)),
        entry("plate", box=(700, 880, 740, 600), label="KA01AB1234"),
        entry("hazard", box=(10, 50, 60, 20)),
    ])

    assert [o["kind"] for o in out["objects"]] == ["person", "face"]
    assert set(ALLOWED_KINDS) == {"person", "face"}

    # And nothing from a dropped entry survives anywhere in the message --
    # a plate must not leak through a label, either.
    assert "KA01AB1234" not in repr(out)


def test_message_carries_no_image_data():
    """Whatever else changes, no JPEG goes down this channel."""
    out = msg([entry("person")])
    blob = repr(out)
    assert "jpeg" not in blob.lower()
    assert "image" not in blob.lower()
    for obj in out["objects"]:
        assert set(obj) == {"kind", "track_id", "box", "label", "identity", "coasting"}


# ── the coordinates ───────────────────────────────────────────────────────

def test_boxes_are_normalised_to_the_unit_square():
    out = msg([entry("person", box=(108, 960, 540, 480))])  # t, r, b, l
    (obj,) = out["objects"]
    # x1, y1, x2, y2 = 480/1920, 108/1080, 960/1920, 540/1080
    assert obj["box"] == pytest.approx([0.25, 0.1, 0.5, 0.5])


def test_every_coordinate_stays_inside_zero_to_one():
    """A detector box can overhang the frame edge; the canvas must not be
    asked to draw outside the video."""
    out = msg([entry("person", box=(-40, W + 300, H + 90, -25))])
    (obj,) = out["objects"]
    assert obj["box"] == [0.0, 0.0, 1.0, 1.0]
    assert all(0.0 <= v <= 1.0 for v in obj["box"])


def test_resolution_change_does_not_move_the_box():
    """The same rectangle, measured on a 1080p frame and on a 4K one, has to
    land in the same place on the canvas -- that is the whole point."""
    small = msg([entry("person", box=(108, 960, 540, 480))], 1920, 1080)
    large = msg([entry("person", box=(216, 1920, 1080, 960))], 3840, 2160)
    assert small["objects"][0]["box"] == large["objects"][0]["box"]


def test_zero_area_and_unusable_boxes_are_dropped():
    assert msg([entry("person", box=(100, 200, 100, 200))])["objects"] == []
    assert msg([entry("person", box=(10, 20, 30, 5))], 0, 0)["objects"] == []
    assert msg([entry("person", box=None)])["objects"] == []
    # Entirely off-frame to the left: clamps to zero width, so it is nothing.
    assert msg([entry("person", box=(100, -10, 400, -300))])["objects"] == []


def test_normalise_box_orders_its_corners():
    """(top, right, bottom, left) with the pairs inverted still describes a
    rectangle, and must come back as x1<x2, y1<y2."""
    assert normalise_box((400, 100, 100, 300), W, H) == normalise_box(
        (100, 300, 400, 100), W, H
    )


# ── the rest of the message ───────────────────────────────────────────────

def test_identity_and_track_id_are_carried():
    out = msg([entry("person", track_id=42, label="Abhitha", identity="Abhitha")])
    (obj,) = out["objects"]
    assert obj["track_id"] == 42
    assert obj["identity"] == "Abhitha"
    assert obj["label"] == "Abhitha"


def test_an_unidentified_track_has_a_label_but_no_identity():
    out = msg([entry("person", track_id=3, label="Track #3", identity=None)])
    (obj,) = out["objects"]
    assert obj["identity"] is None
    assert obj["label"] == "Track #3"


def test_an_analysed_frame_with_nothing_in_it_still_produces_a_message():
    """An empty objects list is how the overlay learns the boxes are gone
    rather than merely late."""
    out = msg([])
    assert out["objects"] == []
    assert out["camera_id"] == "cam-1"
    assert out["type"] == "detections"
    assert out["timestamp"] > 0
