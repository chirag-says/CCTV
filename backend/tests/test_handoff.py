"""
Cross-camera handoff: one person keeps one number from camera to camera.

The scenario is COE. `room` (the lab camera) and `corridor` do not overlap;
people leave the room camera's view at its left edge and appear in the corridor
camera's view at its right edge a few seconds later. `left` and `right` face
each other down the corridor and see the same floor at the same moment.

The safety rules are pinned as hard as the happy path: the dangerous failure is
not a missed handoff (a fresh number), it is one number on two people.

No database, no cameras, no models.
"""

import numpy as np
import pytest

from app.vision.ground_plane import GroundPlane
from app.vision.handoff import HandoffConfig, HandoffEngine
from app.vision.zone_tracker import ZoneTracker

ROOM, CORRIDOR = "room", "corridor"

# Room camera: people exit at the left edge. Corridor camera: they enter at the right.
ROOM_EXIT = {"id": "exit", "name": "to corridor", "rect": [0.0, 0.0, 0.2, 1.0]}
CORRIDOR_ENTRY = {"id": "entry", "name": "from room", "rect": [0.8, 0.0, 1.0, 1.0]}

IN_ROOM_EXIT = (0.05, 0.3, 0.15, 0.9)       # foot at u=0.10 -> inside ROOM_EXIT
IN_CORRIDOR_ENTRY = (0.85, 0.3, 0.95, 0.9)  # foot at u=0.90 -> inside CORRIDOR_ENTRY
MIDDLE = (0.45, 0.3, 0.55, 0.9)             # foot in no area


def door_config(max_gap=10.0, extra_rows=()):
    rows = [
        (ROOM, "coe", {"handoff": {
            "areas": [ROOM_EXIT],
            "links": [{"area_id": "exit", "to_camera_id": CORRIDOR,
                       "to_area_id": "entry", "max_gap_sec": max_gap}],
        }}),
        (CORRIDOR, "coe left corridor", {"handoff": {"areas": [CORRIDOR_ENTRY]}}),
        *extra_rows,
    ]
    return HandoffConfig.from_rows(rows)


def engine(config=None, **kw):
    e = HandoffEngine(claim_window_sec=kw.pop("claim", 5.0), live_grace_sec=1.0,
                      memory_sec=60.0, **kw)
    e.set_config(config or door_config())
    return e


def person(track_id, box=MIDDLE, identity=None, coasting=False, label=None):
    return {"kind": "person", "track_id": track_id, "box": list(box),
            "identity": identity, "coasting": coasting,
            "label": label or identity or f"Track #{track_id}"}


def send(e, camera, t, *objects):
    message = {"camera_id": camera, "timestamp": t, "objects": list(objects)}
    e.process(message)
    return {o["track_id"]: o for o in message["objects"]}


def walk_out_of_room(e, track_id=7, start=100.0, identity=None):
    """A track seen in the middle of the room, then at the exit, then gone."""
    send(e, ROOM, start, person(track_id, MIDDLE, identity))
    out = send(e, ROOM, start + 1.0, person(track_id, IN_ROOM_EXIT, identity))
    return out[track_id]["global_track_id"]


# ── Basics ────────────────────────────────────────────────────────────────────

def test_a_track_keeps_its_number_and_is_labelled_with_it():
    e = engine()
    first = send(e, ROOM, 100.0, person(3))[3]
    later = send(e, ROOM, 101.0, person(3))[3]
    assert first["global_track_id"] == later["global_track_id"] == 1
    assert later["label"] == "#1"


def test_named_and_stranger_labels_keep_their_text():
    e = engine()
    out = send(e, ROOM, 100.0, person(1, identity="Chirag"),
               person(2, label="Unknown-3F2A"), person(3))
    assert out[1]["label"] == "Chirag #1"
    assert out[2]["label"] == "Unknown-3F2A #2"
    assert out[3]["label"] == "#3"


def test_two_tracks_on_one_camera_get_two_numbers():
    e = engine()
    out = send(e, ROOM, 100.0, person(1), person(2))
    assert out[1]["global_track_id"] != out[2]["global_track_id"]


def test_malformed_messages_are_ignored():
    e = engine()
    e.process({})
    e.process({"camera_id": ROOM, "objects": "nope"})
    no_box = {"kind": "person", "track_id": 1, "box": None}
    face = {"kind": "face", "track_id": None, "box": [0, 0, 1, 1]}
    e.process({"camera_id": ROOM, "timestamp": 100.0, "objects": [no_box, face, "junk"]})
    assert "global_track_id" not in no_box
    assert "global_track_id" not in face


# ── Handoff areas (non-overlapping cameras) ───────────────────────────────────

def test_room_to_corridor_carries_the_number():
    e = engine()
    gid = walk_out_of_room(e)
    out = send(e, CORRIDOR, 104.0, person(55, IN_CORRIDOR_ENTRY))
    assert out[55]["global_track_id"] == gid
    assert out[55]["label"] == f"#{gid}"
    assert e.stats["handoff_area"] == 1


def test_links_work_in_both_directions():
    e = engine()
    send(e, CORRIDOR, 100.0, person(55, MIDDLE))
    gid = send(e, CORRIDOR, 101.0, person(55, IN_CORRIDOR_ENTRY))[55]["global_track_id"]
    out = send(e, ROOM, 104.0, person(8, IN_ROOM_EXIT))
    assert out[8]["global_track_id"] == gid


def test_a_gap_longer_than_the_link_is_a_new_person():
    e = engine(door_config(max_gap=5.0))
    gid = walk_out_of_room(e)
    out = send(e, CORRIDOR, 101.0 + 8.0, person(55, IN_CORRIDOR_ENTRY))
    assert out[55]["global_track_id"] != gid


def test_appearing_outside_the_linked_area_is_a_new_person():
    e = engine()
    gid = walk_out_of_room(e)
    out = send(e, CORRIDOR, 104.0, person(55, MIDDLE))
    assert out[55]["global_track_id"] != gid


def test_leaving_from_the_middle_of_the_room_is_not_a_departure():
    """Last seen outside the exit area: they did not go through the door."""
    e = engine()
    send(e, ROOM, 100.0, person(7, MIDDLE))
    gid = send(e, ROOM, 101.0, person(7, MIDDLE))[7]["global_track_id"]
    out = send(e, CORRIDOR, 104.0, person(55, IN_CORRIDOR_ENTRY))
    assert out[55]["global_track_id"] != gid


def test_a_person_still_visible_cannot_be_handed_off():
    """Standing in the room's exit area is not having left the room."""
    e = engine()
    send(e, ROOM, 100.0, person(7, IN_ROOM_EXIT))
    out = send(e, CORRIDOR, 100.5, person(55, IN_CORRIDOR_ENTRY))
    send(e, ROOM, 100.6, person(7, IN_ROOM_EXIT))
    assert out[55]["global_track_id"] != 1


def test_first_out_first_in():
    e = engine()
    send(e, ROOM, 100.0, person(1, IN_ROOM_EXIT))
    send(e, ROOM, 101.0, person(2, IN_ROOM_EXIT))
    a = send(e, CORRIDOR, 104.0, person(51, IN_CORRIDOR_ENTRY))[51]["global_track_id"]
    b = send(e, CORRIDOR, 105.0, person(52, IN_CORRIDOR_ENTRY))[52]["global_track_id"]
    assert (a, b) == (1, 2)


def test_each_departure_is_claimed_once():
    e = engine()
    gid = walk_out_of_room(e)
    first = send(e, CORRIDOR, 104.0, person(51, IN_CORRIDOR_ENTRY))[51]["global_track_id"]
    # 51 goes out of view, then someone else appears in the same area.
    second = send(e, CORRIDOR, 107.0, person(52, IN_CORRIDOR_ENTRY))[52]["global_track_id"]
    assert first == gid
    assert second != gid


def test_recognition_outranks_timing():
    """A departure that was Ravi by face is not handed to someone recognised as Chirag."""
    e = engine()
    gid = walk_out_of_room(e, identity="Ravi")
    out = send(e, CORRIDOR, 104.0, person(55, IN_CORRIDOR_ENTRY, identity="Chirag"))
    assert out[55]["global_track_id"] != gid


def test_claim_window_closes():
    """A long-lived corridor track that wanders into the entry area keeps its number."""
    e = engine(claim=3.0)
    send(e, CORRIDOR, 90.0, person(55, MIDDLE))
    own = send(e, CORRIDOR, 99.0, person(55, MIDDLE))[55]["global_track_id"]
    walk_out_of_room(e, start=100.0)
    out = send(e, CORRIDOR, 104.0, person(55, IN_CORRIDOR_ENTRY))
    assert out[55]["global_track_id"] == own


def test_number_is_stable_after_a_handoff():
    e = engine()
    gid = walk_out_of_room(e)
    send(e, CORRIDOR, 104.0, person(55, IN_CORRIDOR_ENTRY))
    later = send(e, CORRIDOR, 120.0, person(55, MIDDLE))
    assert later[55]["global_track_id"] == gid


# ── Name ──────────────────────────────────────────────────────────────────────

def test_recognised_person_takes_the_number_they_hold_elsewhere():
    e = engine()
    gid = send(e, ROOM, 100.0, person(7, identity="Chirag"))[7]["global_track_id"]
    send(e, CORRIDOR, 110.0, person(55))
    out = send(e, CORRIDOR, 111.0, person(55, identity="Chirag"))
    assert out[55]["global_track_id"] == gid
    assert out[55]["label"] == f"Chirag #{gid}"


def test_name_never_merges_two_live_tracks_on_one_camera():
    e = engine()
    out = send(e, ROOM, 100.0, person(1, identity="Chirag"), person(2))
    out = send(e, ROOM, 100.5, person(1, identity="Chirag"), person(2, identity="Chirag"))
    assert out[1]["global_track_id"] != out[2]["global_track_id"]


def test_a_named_number_is_not_renamed():
    e = engine()
    send(e, ROOM, 100.0, person(7, identity="Chirag"))
    own = send(e, CORRIDOR, 110.0, person(55, identity="Ravi"))[55]["global_track_id"]
    out = send(e, CORRIDOR, 111.0, person(55, identity="Chirag"))
    assert out[55]["global_track_id"] == own


# ── Coasting ──────────────────────────────────────────────────────────────────

def test_coasting_box_shows_the_same_number_and_does_not_keep_the_person_visible():
    e = engine()
    gid = send(e, ROOM, 100.0, person(7, IN_ROOM_EXIT))[7]["global_track_id"]
    coast = send(e, ROOM, 102.0, person(7, IN_ROOM_EXIT, coasting=True))
    assert coast[7]["global_track_id"] == gid
    # Coasting is not a sighting, so the departure is still claimable.
    out = send(e, CORRIDOR, 103.0, person(55, IN_CORRIDOR_ENTRY))
    assert out[55]["global_track_id"] == gid


# ── Floor (overlapping cameras) ───────────────────────────────────────────────

IMG = [[0.2, 0.9], [0.8, 0.9], [0.6, 0.4], [0.4, 0.4]]
WORLD = [[0.0, 0.0], [4.0, 0.0], [4.0, 10.0], [0.0, 10.0]]
# Facing the other way: the same image corners see the opposite floor corners.
WORLD_OPPOSED = [[4.0, 10.0], [0.0, 10.0], [0.0, 0.0], [4.0, 0.0]]


def box_at(plane, x, y, height=0.2):
    """A body box whose foot projects to floor (x, y) in this camera."""
    p = np.linalg.inv(plane.H) @ np.array([x, y, 1.0])
    u, v = p[0] / p[2], p[1] / p[2]
    return (u - 0.03, v - height, u + 0.03, v)


@pytest.fixture
def floor():
    left = GroundPlane.build("left", "COE", IMG, WORLD)
    right = GroundPlane.build("right", "COE", IMG, WORLD_OPPOSED)
    config = HandoffConfig.from_rows([
        ("left", "coe left", {"ground_plane": left.to_config()}),
        ("right", "coe right", {"ground_plane": right.to_config()}),
    ])
    tracker = ZoneTracker("COE")
    e = HandoffEngine(claim_window_sec=5.0, live_grace_sec=1.0, memory_sec=60.0,
                      zone_lookup=lambda _zone: tracker)
    e.set_config(config)
    return e, left, right


def test_same_floor_spot_on_two_cameras_is_one_number(floor):
    e, left, right = floor
    gid = send(e, "left", 100.0, person(1, box_at(left, 2.0, 3.0)))[1]["global_track_id"]
    out = send(e, "right", 100.1, person(9, box_at(right, 2.0, 3.0)))
    assert out[9]["global_track_id"] == gid
    assert e.stats["handoff_floor"] == 1


def test_different_floor_spots_stay_two_numbers(floor):
    e, left, right = floor
    gid = send(e, "left", 100.0, person(1, box_at(left, 1.0, 2.0)))[1]["global_track_id"]
    out = send(e, "right", 100.1, person(9, box_at(right, 3.5, 8.0)))
    assert out[9]["global_track_id"] != gid


# ── Config parsing ────────────────────────────────────────────────────────────

def test_config_skips_bad_areas_and_links():
    config = HandoffConfig.from_rows([
        ("a", "A", {"handoff": {
            "areas": [
                {"id": "ok", "rect": [0, 0, 0.5, 0.5]},
                {"id": "pixels", "rect": [0, 0, 640, 480]},
                {"id": "", "rect": [0, 0, 0.5, 0.5]},
                {"rect": [0, 0, 0.5, 0.5]},
            ],
            "links": [
                {"area_id": "ok", "to_camera_id": "b", "to_area_id": "ok", "max_gap_sec": 10},
                {"area_id": "ok", "to_camera_id": "a", "to_area_id": "ok", "max_gap_sec": 10},
                {"area_id": "ok", "to_camera_id": "b", "to_area_id": "gone", "max_gap_sec": 10},
                {"area_id": "ok", "to_camera_id": "b", "to_area_id": "ok", "max_gap_sec": 9999},
            ],
        }}),
        ("b", "B", {"handoff": {"areas": [{"id": "ok", "rect": [0.5, 0.5, 1, 1]}]}}),
        ("c", "C", {"handoff": "garbage"}),
        ("d", "D", None),
    ])
    assert [a.area_id for a in config.areas["a"]] == ["ok"]
    assert config.links[("a", "ok")] == ((("b", "ok"), 10.0),)
    assert config.links[("b", "ok")] == ((("a", "ok"), 10.0),)
    assert config.max_gap == 10.0
