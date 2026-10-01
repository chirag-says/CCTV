"""
Cross-camera zone tracking: one person, two cameras facing each other.

The scenario is the COE corridor. `left` sees your face (so it recognises you);
`right` sees your back from far away (so it never can). The feature works when
`right` marks you as you the moment `left` knows who you are — driven purely by
where you are standing on the floor.

These tests pin the safety rules as hard as the happy path, because the
dangerous failure here is not "no association", it is "confidently wrong
identity propagated everywhere and treated as corroborated".

No database, no cameras, no models.
"""

import threading

import pytest

from app.vision.zone_tracker import (
    Assignment,
    TrackObservation,
    ZoneTracker,
    get_zone_registry,
)


def obs(track_id, x, y, sigma=0.2, **kw) -> TrackObservation:
    return TrackObservation(track_id=track_id, x=x, y=y, sigma=sigma, **kw)


def known(track_id, x, y, person_id, name, conf=0.9, sigma=0.2) -> TrackObservation:
    """An observation where THIS camera recognised the person itself."""
    return TrackObservation(
        track_id=track_id, x=x, y=y, sigma=sigma,
        person_id=person_id, person_name=name, confidence=conf,
        identity_confirmed=True,
    )


@pytest.fixture
def zone() -> ZoneTracker:
    return ZoneTracker("COE", member_ttl_sec=2.0, identity_ttl_sec=10.0)


# ── Association ──────────────────────────────────────────────────────────────

def test_same_floor_position_becomes_one_person(zone):
    zone.publish("left", [obs(1, 2.0, 5.0)], now=100.0)
    zone.publish("right", [obs(7, 2.1, 5.05)], now=100.0)

    a = zone.publish("left", [obs(1, 2.0, 5.0)], now=100.1)[1]
    b = zone.publish("right", [obs(7, 2.1, 5.05)], now=100.1)[7]

    assert a.zone_person_id == b.zone_person_id
    assert a.camera_count == 2
    assert a.is_multi_camera


def test_distant_positions_stay_separate(zone):
    zone.publish("left", [obs(1, 1.0, 1.0)], now=100.0)
    zone.publish("right", [obs(7, 3.5, 9.0)], now=100.0)

    a = zone.publish("left", [obs(1, 1.0, 1.0)], now=100.1)[1]
    b = zone.publish("right", [obs(7, 3.5, 9.0)], now=100.1)[7]
    assert a.zone_person_id != b.zone_person_id
    assert a.camera_count == 1


def test_two_tracks_in_one_camera_never_merge(zone):
    """
    Within-camera distinctness is authoritative. That camera's tracker had the
    pixels; a homography with half a metre of error does not get to overrule it
    and fuse two people standing close together.
    """
    result = zone.publish("left", [obs(1, 2.0, 5.0), obs(2, 2.05, 5.05)], now=100.0)
    assert result[1].zone_person_id != result[2].zone_person_id
    # And it must stay that way as they keep standing next to each other.
    result = zone.publish("left", [obs(1, 2.0, 5.0), obs(2, 2.05, 5.05)], now=100.2)
    assert result[1].zone_person_id != result[2].zone_person_id


def test_two_people_pair_up_correctly_across_cameras(zone):
    zone.publish("left", [obs(1, 1.0, 2.0), obs(2, 3.0, 8.0)], now=100.0)
    right = zone.publish("right", [obs(9, 3.1, 8.1), obs(8, 1.1, 2.1)], now=100.0)
    left = zone.publish("left", [obs(1, 1.0, 2.0), obs(2, 3.0, 8.0)], now=100.1)

    assert left[1].zone_person_id == right[8].zone_person_id
    assert left[2].zone_person_id == right[9].zone_person_id


def test_track_keeps_its_zone_person_while_walking(zone):
    ids = set()
    for step in range(10):
        y = 1.0 + step * 0.4
        zone.publish("right", [obs(7, 2.0, y + 0.05)], now=100.0 + step * 0.1)
        result = zone.publish("left", [obs(1, 2.0, y)], now=100.0 + step * 0.1)
        ids.add(result[1].zone_person_id)
    assert len(ids) == 1, "zone person id churned while the person simply walked"


# ── Fusion ───────────────────────────────────────────────────────────────────

def test_position_is_dominated_by_the_confident_camera(zone):
    """
    The near camera has a small sigma, the far one a large one. Inverse-variance
    fusion must land close to the near camera's estimate; a plain mean would sit
    halfway and inherit the far camera's error.
    """
    zone.publish("left", [obs(1, 2.0, 5.0, sigma=0.2)], now=100.0)
    result = zone.publish("right", [obs(7, 4.0, 5.0, sigma=2.0)], now=100.0)
    fused = result[7]
    assert fused.x == pytest.approx(2.0, abs=0.15)
    assert fused.sigma < 0.2, "fusing two observations must not increase uncertainty"


# ── Identity propagation: the happy path ─────────────────────────────────────

def test_recognised_identity_reaches_the_camera_that_cannot_see_the_face(zone):
    """The feature, in one test."""
    zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.0)
    right = zone.publish("right", [obs(7, 2.1, 5.05)], now=100.0)

    assert right[7].inherited_person_id == "p-chirag"
    assert right[7].inherited_person_name == "Chirag"
    assert right[7].inherited_from == "left"
    assert right[7].has_inherited_identity


def test_the_recognising_camera_does_not_inherit_from_itself(zone):
    zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.0)
    left = zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.1)
    assert not left[1].has_inherited_identity


def test_stranger_label_propagates_too(zone):
    zone.publish(
        "left",
        [obs(1, 2.0, 5.0, unknown_id="u-123", unknown_label="Unknown-abc123")],
        now=100.0,
    )
    right = zone.publish("right", [obs(7, 2.1, 5.05)], now=100.0)
    assert right[7].inherited_unknown_id == "u-123"
    assert right[7].inherited_unknown_label == "Unknown-abc123"


def test_identity_survives_the_face_turning_away(zone):
    """
    Being the only camera with a face is the normal state in an opposed pair,
    so identity must persist across frames where nobody currently recognises.
    """
    zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.0)
    zone.publish("right", [obs(7, 2.1, 5.05)], now=100.0)
    # Now nobody sees a face at all for several seconds.
    for step in range(1, 6):
        t = 100.0 + step * 0.5
        zone.publish("left", [obs(1, 2.0, 5.0)], now=t)
        right = zone.publish("right", [obs(7, 2.1, 5.05)], now=t)
    assert right[7].inherited_person_id == "p-chirag"


def test_identity_expires_eventually():
    zone = ZoneTracker("COE", member_ttl_sec=60.0, identity_ttl_sec=2.0)
    zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.0)
    zone.publish("right", [obs(7, 2.1, 5.05)], now=100.0)
    for step in range(1, 12):
        t = 100.0 + step * 1.0
        zone.publish("left", [obs(1, 2.0, 5.0)], now=t)
        right = zone.publish("right", [obs(7, 2.1, 5.05)], now=t)
    assert right[7].inherited_person_id is None, "a stale identity rode along forever"


# ── Identity propagation: the safety rules ───────────────────────────────────

def test_inherited_identity_never_propagates_onward(zone):
    """
    The laundering guard. `right` inherits from `left`. A third camera must NOT
    then treat `right` as a source, or one guess becomes zone-wide certainty
    that no later evidence can dislodge.
    """
    zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.0)
    right = zone.publish("right", [obs(7, 2.1, 5.05)], now=100.0)
    assert right[7].inherited_from == "left"

    # `right` reports back what it now displays -- but as an OBSERVATION it has
    # no local identity, which is what the pipeline must send. The zone's
    # identity source must still be `left`, never `right`.
    zone.publish("right", [obs(7, 2.1, 5.05)], now=100.1)
    third = zone.publish("third", [obs(3, 2.05, 5.02)], now=100.1)
    assert third[3].inherited_from == "left"


def test_local_recognition_is_never_overridden(zone):
    """
    Geometry may say WHICH TRACK; it may never say WHO. A camera that
    recognised someone itself keeps its own answer.
    """
    zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.0)
    right = zone.publish(
        "right", [known(7, 2.1, 5.05, "p-arjun", "Arjun")], now=100.0
    )
    assert right[7].inherited_person_id is None
    assert not right[7].has_inherited_identity


def test_identity_conflict_is_recorded_not_hidden(zone):
    zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.0)
    zone.publish("right", [known(7, 2.1, 5.05, "p-arjun", "Arjun")], now=100.0)
    zone.publish("right", [known(7, 2.1, 5.05, "p-arjun", "Arjun")], now=100.1)
    assert zone.stats()["conflicts"] > 0


def test_inheritance_only_fills_a_void(zone):
    """A track already carrying its own unknown id keeps it."""
    zone.publish(
        "left", [obs(1, 2.0, 5.0, unknown_id="u-left", unknown_label="Unknown-left")],
        now=100.0,
    )
    right = zone.publish(
        "right", [obs(7, 2.1, 5.05, unknown_id="u-right", unknown_label="Unknown-right")],
        now=100.0,
    )
    assert right[7].inherited_unknown_id is None


def test_enrolled_identity_beats_a_stranger_label(zone):
    zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.0)
    right = zone.publish("right", [obs(7, 2.1, 5.05)], now=100.0)
    assert right[7].inherited_person_id == "p-chirag"
    assert right[7].inherited_unknown_id is None


# ── Housekeeping ─────────────────────────────────────────────────────────────

def test_stale_members_are_dropped(zone):
    zone.publish("left", [obs(1, 2.0, 5.0)], now=100.0)
    zone.publish("right", [obs(7, 2.1, 5.05)], now=100.0)
    assert zone.stats()["multi_camera"] == 1

    # `left` goes quiet well past the member TTL.
    result = zone.publish("right", [obs(7, 2.1, 5.05)], now=110.0)
    assert result[7].camera_count == 1


def test_people_are_removed_when_everyone_stops_seeing_them(zone):
    zone.publish("left", [obs(1, 2.0, 5.0)], now=100.0)
    assert zone.stats()["people"] == 1
    zone.publish("left", [], now=110.0)
    assert zone.stats()["people"] == 0


def test_no_track_belongs_to_two_zone_people(zone):
    """A membership left behind in a previous person would double-count."""
    for step in range(6):
        zone.publish("left", [obs(1, 1.0 + step * 0.9, 5.0)], now=100.0 + step * 0.1)
    assert zone.stats()["people"] == 1


def test_snapshot_reports_the_zone(zone):
    zone.publish("left", [known(1, 2.0, 5.0, "p-chirag", "Chirag")], now=100.0)
    zone.publish("right", [obs(7, 2.1, 5.05)], now=100.0)
    snap = zone.snapshot()
    assert len(snap) == 1
    assert snap[0]["camera_count"] == 2
    assert snap[0]["person_name"] == "Chirag"
    assert snap[0]["identified_by"] == "left"
    assert sorted(snap[0]["cameras"]) == ["left", "right"]


def test_empty_publish_is_safe(zone):
    assert zone.publish("left", [], now=100.0) == {}


# ── Threading ────────────────────────────────────────────────────────────────

def test_concurrent_publishes_do_not_corrupt_the_zone():
    zone = ZoneTracker("COE")
    errors = []
    barrier = threading.Barrier(4)

    def camera(cam_id, offset):
        try:
            barrier.wait(timeout=10)
            for step in range(50):
                zone.publish(
                    cam_id,
                    [obs(1, 2.0 + offset, 5.0 + offset), obs(2, 8.0 + offset, 9.0)],
                    now=200.0 + step * 0.02,
                )
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [
        threading.Thread(target=camera, args=(f"cam{i}", i * 0.05)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    stats = zone.stats()
    # Four cameras, two real people. Association may be imperfect under a race,
    # but the structure must stay sane rather than exploding into one person
    # per camera per frame.
    assert 2 <= stats["people"] <= 8


# ── Registry ─────────────────────────────────────────────────────────────────

def test_zone_registry_is_a_singleton_per_zone():
    reg = get_zone_registry()
    assert reg is get_zone_registry()
    assert reg.zone("COE") is reg.zone("COE")
    assert reg.zone("COE") is not reg.zone("Front Gate")
