"""
Handoff evidence that needs no operator: stitching one camera's dropped
tracks, shared views learned from walkers, routes learned from departures and
arrivals, and body appearance.

test_handoff.py pins the operator-configured evidence and the safety rules.
This file pins the rest, in the same spirit: the dangerous failure is never a
missed handoff (a fresh number), it is one number on two people. So for every
kind of evidence there is a test that it joins what it should and a test that
it leaves alone what it should.

Geometry is the COE corridor as measured: `left` and `right` face each other,
so a walker climbs one picture while descending the other.

No database, no cameras, no models.
"""

import numpy as np
import pytest

from app.vision.handoff import HandoffConfig, HandoffEngine

LEFT, RIGHT, ROOM, CORRIDOR = "left", "right", "room", "corridor"
ROWS = [(c, c, {}, "COE") for c in (LEFT, RIGHT, ROOM, CORRIDOR)]


def engine(rows=ROWS, **kw):
    e = HandoffEngine(**kw)
    e.set_config(HandoffConfig.from_rows(rows))
    return e


def box(u, v, height=0.3):
    """A body box whose feet are at (u, v)."""
    return [u - height / 6, v - height, u + height / 6, v]


def person(track_id, u, v, height=0.3, identity=None):
    return {"kind": "person", "track_id": track_id, "box": box(u, v, height),
            "identity": identity, "label": identity or f"Track #{track_id}"}


def send(e, camera, t, *objects):
    message = {"camera_id": camera, "timestamp": t, "objects": list(objects)}
    e.process(message)
    return {o["track_id"]: o["global_track_id"] for o in message["objects"]}


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


RED, BLUE, GREEN = unit(1, 0, 0, 0), unit(0, 1, 0, 0), unit(0, 0, 1, 0)
REDDISH = unit(1, 0.2, 0, 0.1)          # the same person on another camera


def look(e, camera, track_id, vector, times=2, t=0.0):
    for _ in range(times):
        assert e.add_appearance(camera, track_id, vector, t)


# ── Stitch: one camera, a dropped track ──────────────────────────────────────

def stand(e, camera, track_id, t0, t1, u=0.5, v=0.6, step=0.2):
    out, t = {}, t0
    while t <= t1 + 1e-9:
        out = send(e, camera, t, person(track_id, u, v))
        t += step
    return out[track_id]


def test_a_track_that_starts_where_another_was_just_lost_keeps_the_number():
    e = engine()
    gid = stand(e, ROOM, 1, 100.0, 101.0)
    assert send(e, ROOM, 101.6, person(2, 0.5, 0.6))[2] == gid
    assert e.stats["handoff_stitch"] == 1


def test_a_track_that_starts_somewhere_else_is_someone_else():
    e = engine()
    gid = stand(e, ROOM, 1, 100.0, 101.0)
    assert send(e, ROOM, 101.6, person(2, 0.9, 0.6))[2] != gid


def test_a_track_lost_long_ago_is_not_carried_on():
    e = engine()
    gid = stand(e, ROOM, 1, 100.0, 101.0)
    assert send(e, ROOM, 110.0, person(2, 0.5, 0.6))[2] != gid


def test_a_walker_is_stitched_where_they_would_have_walked_to():
    e = engine()
    t, u = 100.0, 0.30
    while t <= 101.0 + 1e-9:                 # walking right at 0.1 per second
        gid = send(e, ROOM, t, person(1, u, 0.6))[1]
        t, u = t + 0.1, u + 0.01
    assert send(e, ROOM, 102.0, person(2, 0.50, 0.6))[2] == gid


def test_when_the_lost_track_comes_back_the_stitch_is_undone():
    """Both visible at once on one camera: they were two people all along."""
    e = engine()
    gid = stand(e, ROOM, 1, 100.0, 101.0)
    assert send(e, ROOM, 101.6, person(2, 0.5, 0.6))[2] == gid
    out = send(e, ROOM, 102.0, person(1, 0.45, 0.6), person(2, 0.55, 0.6))
    assert out[1] == gid and out[2] != gid
    assert e.stats["split"] == 1
    # ... and it stays undone.
    assert send(e, ROOM, 102.2, person(1, 0.45, 0.6), person(2, 0.55, 0.6))[2] == out[2]


def test_a_stitch_is_undone_when_the_two_halves_look_different():
    e = engine()
    gid = stand(e, ROOM, 1, 100.0, 101.0)
    look(e, ROOM, 1, RED)
    assert send(e, ROOM, 101.6, person(2, 0.5, 0.6))[2] == gid      # a blink: believed on position
    look(e, ROOM, 2, BLUE, times=1)
    assert send(e, ROOM, 101.8, person(2, 0.5, 0.6))[2] != gid
    assert e.stats["unstitch"] == 1


def test_a_stitch_stands_when_the_two_halves_look_alike():
    e = engine()
    gid = stand(e, ROOM, 1, 100.0, 101.0)
    look(e, ROOM, 1, RED)
    send(e, ROOM, 101.6, person(2, 0.5, 0.6))
    look(e, ROOM, 2, REDDISH, times=1)
    assert send(e, ROOM, 101.8, person(2, 0.5, 0.6))[2] == gid
    assert "unstitch" not in e.stats


def test_in_a_crowd_position_alone_does_not_stitch():
    """Someone was lost in a knot of people; which of them re-appeared is a
    guess unless there is a look to check it against."""
    e = engine()
    t = 100.0
    while t <= 101.0 + 1e-9:
        out = send(e, ROOM, t, person(1, 0.50, 0.6), person(9, 0.56, 0.6))
        t += 0.2
    look(e, ROOM, 9, GREEN)                         # looks exist, but not for track 1
    out2 = send(e, ROOM, 101.6, person(2, 0.50, 0.6), person(9, 0.56, 0.6))
    assert out2[2] != out[1]


# ── View: two cameras that see the same floor ────────────────────────────────

def facing(s):
    """Feet and height at fraction `s` along the corridor, on each camera."""
    return (0.5, 0.25 + 0.6 * s, 0.10 + 0.30 * s), (0.5, 0.85 - 0.6 * s, 0.40 - 0.30 * s)


def walk(e, t0, left_id, right_id, seconds=8.0, fps=10, start=0.0, end=1.0, extra_right=()):
    """One person walks the corridor, seen by both cameras. Returns the last
    numbers each camera showed."""
    out_left = out_right = {}
    steps = int(seconds * fps)
    for i in range(steps + 1):
        t = t0 + i / fps
        s = start + (end - start) * i / steps
        (lu, lv, lh), (ru, rv, rh) = facing(s)
        if left_id is not None:
            out_left = send(e, LEFT, t, person(left_id, lu, lv, lh))
        if right_id is not None:
            out_right = send(e, RIGHT, t + 0.02, person(right_id, ru, rv, rh), *extra_right)
    return out_left, out_right


def taught(walks=6):
    e = engine()
    for n in range(walks):
        walk(e, 100.0 + 20.0 * n, left_id=10 + n, right_id=50 + n)
    return e


def test_walkers_teach_the_cameras_that_they_overlap():
    e = taught()
    learned = e.snapshot()["learned"]["overlaps"]
    assert len(learned) == 1 and learned[0]["ready"]
    assert {learned[0]["a_name"], learned[0]["b_name"]} == {LEFT, RIGHT}


def test_once_learned_a_person_seen_by_both_cameras_has_one_number():
    e = taught()
    out_left, out_right = walk(e, 400.0, left_id=100, right_id=200)
    assert out_left[100] == out_right[200]
    assert e.stats["handoff_view"] >= 1


def test_a_shared_view_joins_people_however_long_they_have_been_tracked():
    """Unlike a handoff, this is not limited to a track's first seconds: the
    person walks into the shared stretch whenever they get there."""
    e = taught()
    # Thirty seconds alone on the left camera first, well past any claim window.
    for i in range(300):
        own = send(e, LEFT, 400.0 + i / 10, person(100, 0.2, 0.9))[100]
    out_left, out_right = walk(e, 430.0, left_id=100, right_id=200)
    assert out_left[100] == out_right[200] == own      # the older number survives


def test_too_few_walks_join_nobody():
    e = taught(walks=2)
    out_left, out_right = walk(e, 400.0, left_id=100, right_id=200)
    assert out_left[100] != out_right[200]
    assert "handoff_view" not in e.stats


def test_two_people_in_different_places_stay_two_people():
    e = taught()
    out_left, _ = walk(e, 400.0, left_id=100, right_id=None, start=0.0, end=0.4)
    # Seen only by the right camera, at the other end, at the same time.
    e2 = taught()
    for i in range(81):
        t = 400.0 + i / 10
        (lu, lv, lh), _ = facing(0.4 * i / 80)
        _, (ru, rv, rh) = facing(1.0 - 0.3 * i / 80)
        a = send(e2, LEFT, t, person(100, lu, lv, lh))
        b = send(e2, RIGHT, t + 0.02, person(200, ru, rv, rh))
    assert a[100] != b[200]


def test_a_walker_is_not_joined_to_someone_standing_where_they_pass():
    """Measured failure: seen through the lab's glass wall, hallway walkers
    cross right over the feet of a person standing at a bench."""
    e = taught()
    _, (su, sv, sh) = facing(0.5)
    bystander = person(300, su, sv, sh)
    for i in range(30):                              # standing there a while first
        send(e, RIGHT, 397.0 + i / 10, bystander)
    out_left, out_right = walk(e, 400.0, left_id=100, right_id=None, extra_right=())
    last = {}
    for i in range(81):
        t = 410.0 + i / 10
        (lu, lv, lh), _ = facing(i / 80)
        a = send(e, LEFT, t, person(101, lu, lv, lh))
        last = send(e, RIGHT, t + 0.02, bystander)
    assert a[101] != last[300]


def test_people_the_cameras_recognised_as_different_are_never_joined():
    e = taught()
    out_left = out_right = {}
    for i in range(81):
        t = 400.0 + i / 10
        (lu, lv, lh), (ru, rv, rh) = facing(i / 80)
        out_left = send(e, LEFT, t, person(100, lu, lv, lh, identity="Chirag"))
        out_right = send(e, RIGHT, t + 0.02, person(200, ru, rv, rh, identity="Ravi"))
    assert out_left[100] != out_right[200]


def test_learning_can_be_frozen():
    e = engine(learn=False)
    for n in range(6):
        walk(e, 100.0 + 20.0 * n, left_id=10 + n, right_id=50 + n)
    assert e.snapshot()["learned"] == {"overlaps": [], "routes": []}


# ── Route: people who leave one camera and arrive on another ─────────────────

def leave_room(e, track_id, t0):
    """Walk to the room's left edge and out. Returns (number, time last seen)."""
    gid, last = None, t0
    for i in range(31):                                  # three seconds, 0.7 -> 0.1
        last = t0 + i / 10
        gid = send(e, ROOM, last, person(track_id, 0.7 - 0.02 * i, 0.9))[track_id]
    return gid, last


def enter_corridor(e, track_id, t0, seconds=4.0):
    """Appear at the corridor's right edge and walk in. Returns numbers over time."""
    numbers, t, u = [], t0, 0.9
    while t <= t0 + seconds + 1e-9:
        numbers.append(send(e, CORRIDOR, t, person(track_id, u, 0.9))[track_id])
        t, u = t + 0.1, u - 0.015
    return numbers


def taught_route(episodes=6):
    e = engine()
    for n in range(episodes):
        _, gone = leave_room(e, 10 + n, 100.0 + 40.0 * n)
        enter_corridor(e, 50 + n, gone + 3.0)
    return e


def test_people_teach_the_route_between_two_cameras():
    e = taught_route()
    routes = e.snapshot()["learned"]["routes"]
    assert len(routes) == 1
    assert (routes[0]["from_name"], routes[0]["to_name"]) == (ROOM, CORRIDOR)
    assert routes[0]["min_sec"] <= 3.0 <= routes[0]["max_sec"]


def test_once_learned_the_number_follows_the_person_along_the_route():
    e = taught_route()
    before = e.stats["handoff_route"]
    gid, gone = leave_room(e, 100, 500.0)
    assert enter_corridor(e, 200, gone + 3.0)[-1] == gid
    assert e.stats["handoff_route"] == before + 1


def test_arriving_far_too_late_for_the_route_is_someone_else():
    e = taught_route()
    gid, gone = leave_room(e, 100, 500.0)
    assert enter_corridor(e, 200, gone + 20.0)[-1] != gid


def test_arriving_somewhere_the_route_does_not_lead_is_someone_else():
    e = taught_route()
    gid, gone = leave_room(e, 100, 500.0)
    out = {}
    for i in range(40):                                # appears mid-frame instead
        out = send(e, CORRIDOR, gone + 3.0 + i / 10, person(200, 0.3 + 0.01 * i, 0.4))
    assert out[200] != gid


def test_a_route_is_not_learned_from_a_tracker_that_stutters():
    """Tracks that blink in and out on the spot are neither arrivals nor
    departures, however many of them there are."""
    e = engine()
    for n in range(40):
        t0 = 100.0 + 6.0 * n
        stand(e, ROOM, 10 + n, t0, t0 + 2.5, u=0.1, v=0.9)
        stand(e, CORRIDOR, 500 + n, t0 + 5.5, t0 + 9.0, u=0.9, v=0.9)
    assert e.snapshot()["learned"]["routes"] == []


# ── Look: body appearance ────────────────────────────────────────────────────

def test_an_unmistakable_look_carries_the_number_with_no_route_known():
    e = engine()
    gid, gone = leave_room(e, 7, 100.0)
    look(e, ROOM, 7, RED)
    send(e, CORRIDOR, gone + 20.0, person(55, 0.5, 0.5))
    look(e, CORRIDOR, 55, REDDISH)
    assert send(e, CORRIDOR, gone + 20.5, person(55, 0.5, 0.5))[55] == gid
    assert e.stats["handoff_look"] == 1


def test_one_glance_is_not_enough_to_carry_a_number():
    e = engine()
    gid, gone = leave_room(e, 7, 100.0)
    look(e, ROOM, 7, RED)
    send(e, CORRIDOR, gone + 20.0, person(55, 0.5, 0.5))
    look(e, CORRIDOR, 55, REDDISH, times=1)
    assert send(e, CORRIDOR, gone + 20.5, person(55, 0.5, 0.5))[55] != gid


def test_a_look_that_fits_two_people_carries_neither():
    e = engine()
    first, _ = leave_room(e, 7, 100.0)
    second, gone = leave_room(e, 8, 110.0)
    look(e, ROOM, 7, RED)
    look(e, ROOM, 8, unit(1, 0.1, 0, 0))               # dressed alike
    send(e, CORRIDOR, gone + 20.0, person(55, 0.5, 0.5))
    look(e, CORRIDOR, 55, REDDISH)
    assert send(e, CORRIDOR, gone + 20.5, person(55, 0.5, 0.5))[55] not in (first, second)


def test_a_different_look_carries_nothing():
    e = engine()
    gid, gone = leave_room(e, 7, 100.0)
    look(e, ROOM, 7, RED)
    send(e, CORRIDOR, gone + 20.0, person(55, 0.5, 0.5))
    look(e, CORRIDOR, 55, BLUE)
    assert send(e, CORRIDOR, gone + 20.5, person(55, 0.5, 0.5))[55] != gid


def test_a_look_never_reaches_into_another_camera_group():
    rows = [(ROOM, ROOM, {}, "COE"), (CORRIDOR, CORRIDOR, {}, "Front Gate")]
    e = engine(rows)
    gid, gone = leave_room(e, 7, 100.0)
    look(e, ROOM, 7, RED)
    send(e, CORRIDOR, gone + 20.0, person(55, 0.5, 0.5))
    look(e, CORRIDOR, 55, RED)
    assert send(e, CORRIDOR, gone + 20.5, person(55, 0.5, 0.5))[55] != gid


def test_among_people_who_took_the_route_together_the_look_picks_who_is_who():
    """Three friends leave together and arrive together. Arrival order would
    deal the numbers out wrongly; what they are wearing does not."""
    e = taught_route()
    red, _ = leave_room(e, 101, 500.0)
    blue, gone = leave_room(e, 102, 500.4)
    look(e, ROOM, 101, RED)
    look(e, ROOM, 102, BLUE)
    t = gone + 3.0
    send(e, CORRIDOR, t, person(201, 0.9, 0.9))        # first to arrive ...
    look(e, CORRIDOR, 201, BLUE)                       # ... is the one in blue
    assert enter_corridor(e, 201, t + 0.1)[-1] == blue
    send(e, CORRIDOR, t + 0.5, person(202, 0.9, 0.9), person(201, 0.8, 0.9))
    look(e, CORRIDOR, 202, REDDISH)
    assert send(e, CORRIDOR, t + 0.7, person(202, 0.9, 0.9), person(201, 0.8, 0.9))[202] == red


def test_a_candidate_who_fits_the_route_but_visibly_is_not_them_is_refused():
    e = taught_route()
    gid, gone = leave_room(e, 100, 500.0)
    look(e, ROOM, 100, RED)
    # Alone on the route, so the number is taken at once, on timing ...
    assert send(e, CORRIDOR, gone + 3.0, person(200, 0.9, 0.9))[200] == gid
    # ... and given back when the first crop shows someone else.
    look(e, CORRIDOR, 200, GREEN, times=1)
    assert enter_corridor(e, 200, gone + 3.1)[-1] != gid
    assert e.stats["unclaim"] == 1


def test_a_look_for_a_track_that_already_ended_is_ignored():
    e = engine()
    assert e.add_appearance(ROOM, 999, RED) is False
    assert e.add_appearance(ROOM, 999, [0, 0, 0, 0]) is False
    assert e.add_appearance(ROOM, 999, "nonsense") is False


# ── What an operator is told ─────────────────────────────────────────────────

DOOR_POST = [[0.5847, 0.1642], [0.5767, 0.1500], [0.5738, 0.3430], [0.5796, 0.3648]]
FLOOR = [[0.2, 0.9], [0.8, 0.9], [0.6, 0.4], [0.4, 0.4]]
WORLD = [[0.0, 0.0], [4.0, 0.0], [4.0, 10.0], [0.0, 10.0]]


def test_a_calibration_that_is_not_on_the_floor_is_ignored_and_reported():
    """The stored COE calibrations: four points up a door post, 'fit error 0 cm'."""
    config = HandoffConfig.from_rows([
        (LEFT, "coe left", {"ground_plane": {"zone": "COE", "image_points": DOOR_POST, "world_points": WORLD}}, "COE"),
        (RIGHT, "coe right", {"ground_plane": {"zone": "COE", "image_points": FLOOR, "world_points": WORLD}}, "COE"),
    ])
    assert list(config.planes) == [RIGHT]
    assert "door post" in config.unusable_planes[LEFT][0]

    e = HandoffEngine()
    e.set_config(config)
    evidence = e.snapshot()["evidence"]
    assert evidence["floor_cameras"] == ["coe right"]
    assert "coe left" in evidence["floor_unusable"]
    assert evidence["appearance_active"] is False


def test_a_path_lists_arrivals_not_every_camera_that_reported_last():
    """Standing where two cameras overlap must not read left, right, left, right."""
    e = taught()
    for i in range(30):                                   # in the room first
        send(e, ROOM, 395.0 + i / 10, person(100, 0.5, 0.6))
    look(e, ROOM, 100, RED)
    walk_out = walk(e, 400.0, left_id=101, right_id=201)
    assert walk_out[0][101] == walk_out[1][201]
    path = next(p["path"] for p in e.snapshot()["people"] if p["global_track_id"] == walk_out[0][101])
    assert sorted(path) == [LEFT, RIGHT]


# ── What the live walk on 2026-11-20 exposed ─────────────────────────────────

def test_an_undone_stitch_is_not_made_again_on_the_next_frame():
    """Seen live: stitch, 'does not look like', stitch, 'does not look like' ...
    a dozen times in a second, each time under a new number."""
    e = engine()
    gid = stand(e, ROOM, 1, 100.0, 101.0)
    look(e, ROOM, 1, RED)
    send(e, ROOM, 101.6, person(2, 0.5, 0.6))
    look(e, ROOM, 2, BLUE, times=1)
    settled = send(e, ROOM, 101.8, person(2, 0.5, 0.6))[2]
    assert settled != gid
    for i in range(20):
        assert send(e, ROOM, 101.9 + i / 10, person(2, 0.5, 0.6))[2] == settled
    assert e.stats["unstitch"] == 1 and e.stats["handoff_stitch"] == 1


def test_a_stitch_is_checked_against_everything_known_about_the_person():
    """A poor crop of the track just before must not outvote a good one of the
    same person from another camera."""
    e = engine()
    send(e, CORRIDOR, 99.0, person(50, 0.5, 0.6, identity="Chirag"))
    look(e, CORRIDOR, 50, RED)                       # seen well here
    t = 100.0
    while t <= 101.0 + 1e-9:                         # the same person, named, in the room
        gid = send(e, ROOM, t, person(1, 0.5, 0.6, identity="Chirag"))[1]
        t += 0.2
    look(e, ROOM, 1, unit(1, 1.6, 0, 0))             # ... but only a poor, far-off crop of them
    assert send(e, ROOM, 101.6, person(2, 0.5, 0.6))[2] == gid
    look(e, ROOM, 2, RED, times=1)                   # 0.53 against the poor crop, 1.0 against the good one
    assert send(e, ROOM, 101.8, person(2, 0.5, 0.6))[2] == gid
    assert "unstitch" not in e.stats


def test_a_name_known_on_one_camera_is_shown_on_the_others():
    """The camera behind someone never sees their face; it still says who it is."""
    e = taught()
    last = {}
    for i in range(81):
        t = 400.0 + i / 10
        (lu, lv, lh), (ru, rv, rh) = facing(i / 80)
        e.process({"camera_id": LEFT, "timestamp": t,
                   "objects": [person(100, lu, lv, lh, identity="Chirag")]})
        last = {"camera_id": RIGHT, "timestamp": t + 0.02, "objects": [person(200, ru, rv, rh)]}
        e.process(last)
    seen = last["objects"][0]
    assert seen["label"].startswith("Chirag #")
    assert seen["global_name"] == "Chirag"
    assert seen["identity"] is None        # still not this camera's own recognition


def test_a_new_tracks_throwaway_number_can_be_held_back():
    e = engine(label_settle_sec=1.5)
    message = {"camera_id": ROOM, "timestamp": 100.0, "objects": [person(1, 0.5, 0.6)]}
    e.process(message)
    assert message["objects"][0]["label"] == ""
    assert "global_track_id" in message["objects"][0]
    message = {"camera_id": ROOM, "timestamp": 101.6, "objects": [person(1, 0.5, 0.6)]}
    e.process(message)
    assert message["objects"][0]["label"].startswith("#")
    # Someone this camera recognises is never held back.
    message = {"camera_id": ROOM, "timestamp": 101.6, "objects": [person(2, 0.2, 0.6, identity="Chirag")]}
    e.process(message)
    assert message["objects"][0]["label"].startswith("Chirag #")


# ── A recognised person's number lasts the day ───────────────────────────────

HOUR = 3600.0
NOON = 12 * HOUR + 86400.0 * 400          # midday on some day, local-time safe


def seen(e, camera, t, track_id, identity=None, u=0.5):
    return send(e, camera, t, person(track_id, u, 0.6, identity=identity))[track_id]


def test_a_recognised_person_keeps_their_number_all_day():
    e = engine()
    morning = seen(e, ROOM, NOON - 3 * HOUR, 1, "Chirag")
    for i in range(40):                                   # plenty of strangers in between
        seen(e, CORRIDOR, NOON - 2 * HOUR + 10 * i, 100 + i)
    # Three hours later, on another camera, under a new track.
    assert seen(e, CORRIDOR, NOON, 7, "Chirag") == morning
    assert e.stats["kept_for_day"] == 1


def test_the_number_comes_back_when_they_are_recognised_not_before():
    """Until the face is seen again this is just somebody; the reservation is
    not evidence that a track is that person."""
    e = engine()
    morning = seen(e, ROOM, NOON - HOUR, 1, "Chirag")
    assert seen(e, CORRIDOR, NOON, 7) != morning
    assert seen(e, CORRIDOR, NOON + 2, 7, "Chirag") == morning
    assert seen(e, CORRIDOR, NOON + 4, 7) == morning       # and it stays


def test_a_stranger_still_gets_a_new_number_after_a_long_absence():
    e = engine()
    first = seen(e, ROOM, NOON - HOUR, 1)
    look(e, ROOM, 1, RED)
    seen(e, ROOM, NOON, 2)
    look(e, ROOM, 2, RED)                 # even looking exactly the same
    assert seen(e, ROOM, NOON + 1, 2) != first


def test_two_recognised_people_keep_two_numbers():
    e = engine()
    chirag = seen(e, ROOM, NOON - HOUR, 1, "Chirag")
    ravi = seen(e, ROOM, NOON - HOUR, 2, "Ravi", u=0.8)
    assert seen(e, CORRIDOR, NOON, 5, "Ravi") == ravi
    assert seen(e, CORRIDOR, NOON, 6, "Chirag", u=0.8) == chirag
    assert chirag != ravi


def test_one_name_on_two_people_at_once_is_not_given_one_number():
    """The recogniser calling two visible tracks by one name is its mistake;
    putting one number on both would hide it."""
    e = engine()
    morning = seen(e, ROOM, NOON - HOUR, 1, "Chirag")
    out = send(e, CORRIDOR, NOON, person(5, 0.3, 0.6, identity="Chirag"),
               person(6, 0.7, 0.6, identity="Chirag"))
    assert sorted(out.values()).count(morning) == 1


def test_a_new_day_is_a_new_number():
    e = engine()
    today = seen(e, ROOM, NOON, 1, "Chirag")
    for i in range(5):
        seen(e, CORRIDOR, NOON + 60 + i, 50 + i)
    assert seen(e, ROOM, NOON + 26 * HOUR, 2, "Chirag") != today


def test_the_days_numbers_survive_a_restart(tmp_path):
    path = str(tmp_path / "numbers.json")
    e = HandoffEngine(numbers_path=path)
    e.set_config(HandoffConfig.from_rows(ROWS))
    for i in range(6):
        seen(e, CORRIDOR, NOON - HOUR + i, 50 + i)
    chirag = seen(e, ROOM, NOON - HOUR + 10, 1, "Chirag")
    assert chirag > 1

    restarted = HandoffEngine(numbers_path=path)            # nothing in memory at all
    restarted.set_config(HandoffConfig.from_rows(ROWS))
    stranger = seen(restarted, CORRIDOR, NOON, 9)
    assert stranger != chirag                               # never handed out while reserved
    assert seen(restarted, ROOM, NOON + 1, 3, "Chirag") == chirag


def test_an_unreadable_numbers_file_is_ignored(tmp_path):
    path = tmp_path / "numbers.json"
    path.write_text("{ nonsense")
    e = HandoffEngine(numbers_path=str(path))
    e.set_config(HandoffConfig.from_rows(ROWS))
    assert seen(e, ROOM, NOON, 1, "Chirag") == 1
