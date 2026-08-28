"""
Step 1.5 -- unidentified occupants get a real lifecycle.

Person tracking (Step 1) made "a tracked body we never identified" the ordinary
case rather than a rarity. Before this, such a track was given a synthetic
`anon:<camera>:<track>` identity and pushed through the identity path. Two
things were wrong with that:

  * it embedded the ByteTrack id, so a recreated track minted a brand-new
    "person", and PersonTracker._cooldowns -- which had no time-based expiry --
    kept every one of them forever;
  * it sat in `_active_tracks` beside real identities, where attendance,
    presence and journeys could reach it.

These occupants are now counted in a separate lane, and never named.
"""

import time

import pytest

from app.vision.tracker import PersonTracker, UnidentifiedState


def _tracker(**kw):
    t = PersonTracker(**kw)
    # Pin the thresholds so the tests do not track config edits.
    t._unidentified_entry_sec = 5.0
    t._unidentified_exit_sec = 15.0
    return t


class TestLifecycle:
    def test_first_sighting_does_not_confirm(self):
        t = _tracker()
        assert t.on_unidentified(1, "cam-a", centroid=(10, 10), timestamp=1000.0) is None
        assert t.unidentified_count == 0, "not present until it has persisted"

    def test_confirms_after_entry_threshold(self):
        seen = []
        t = _tracker(on_unidentified_entry=seen.append)

        t.on_unidentified(1, "cam-a", centroid=(10, 10), timestamp=1000.0)
        assert t.on_unidentified(1, "cam-a", timestamp=1003.0) is None, "too soon"
        event = t.on_unidentified(1, "cam-a", timestamp=1006.0)

        assert event is not None
        assert t.unidentified_count == 1
        assert len(seen) == 1 and seen[0] is event

    def test_confirms_only_once(self):
        seen = []
        t = _tracker(on_unidentified_entry=seen.append)
        t.on_unidentified(1, "cam-a", timestamp=1000.0)
        t.on_unidentified(1, "cam-a", timestamp=1006.0)
        for i in range(5):
            t.on_unidentified(1, "cam-a", timestamp=1010.0 + i)
        assert len(seen) == 1

    def test_expiry_emits_exit_for_confirmed(self):
        exits = []
        t = _tracker(on_unidentified_exit=exits.append)
        t.on_unidentified(1, "cam-a", timestamp=1000.0)
        t.on_unidentified(1, "cam-a", timestamp=1006.0)

        assert t.expire_unidentified(1010.0) == [], "still recently seen"
        events = t.expire_unidentified(1030.0)

        assert len(events) == 1
        assert events[0]["subtype"] == "unidentified_exit"
        assert t.unidentified_count == 0
        assert exits == events

    def test_unconfirmed_track_expires_silently(self):
        """A body glimpsed for two seconds is not an occupancy record."""
        exits = []
        t = _tracker(on_unidentified_exit=exits.append)
        t.on_unidentified(1, "cam-a", timestamp=1000.0)
        assert t.expire_unidentified(1030.0) == []
        assert exits == []
        assert len(t._unidentified) == 0, "but it must still be dropped"


class TestCannotPolluteIdentityData:
    def test_never_enters_the_identity_lane(self):
        t = _tracker()
        for ts in (1000.0, 1006.0, 1012.0):
            t.on_unidentified(1, "cam-a", centroid=(5, 5), timestamp=ts)

        assert t._active_tracks == {}, "attendance/presence must not see it"
        assert t._cooldowns == {}
        assert t.get_person_centroids() == {}
        assert t.get_person_names() == {}

    def test_event_type_is_never_entry_or_exit(self):
        """
        Footfall, attendance and journeys all filter on event_type in
        ('entry','exit'). An unidentified body must not be countable there.
        """
        t = _tracker()
        t.on_unidentified(1, "cam-a", timestamp=1000.0)
        entry = t.on_unidentified(1, "cam-a", timestamp=1006.0)
        exits = t.expire_unidentified(1030.0)

        for event in (entry, exits[0]):
            assert event["event_type"] == "detection"
            assert event["subtype"] in ("unidentified_entry", "unidentified_exit")
            assert event["unidentified"] is True
            assert "person_id" not in event, "it has no identity, by definition"

    def test_centroid_keys_cannot_collide_with_person_ids(self):
        t = _tracker()
        t.on_unidentified(7, "cam-a", centroid=(40, 50), timestamp=1000.0)
        centroids = t.get_unidentified_centroids()
        assert centroids == {"track:7": (40, 50)}


class TestBounded:
    def test_track_map_is_capped(self):
        t = _tracker()
        t._unidentified_max = 10
        for i in range(50):
            t.on_unidentified(i, "cam-a", centroid=(i, i), timestamp=1000.0 + i)
        assert len(t._unidentified) <= 10

    def test_eviction_drops_the_stalest(self):
        t = _tracker()
        t._unidentified_max = 2
        t.on_unidentified(1, "cam-a", timestamp=1000.0)
        t.on_unidentified(2, "cam-a", timestamp=1005.0)
        t.on_unidentified(3, "cam-a", timestamp=1010.0)  # evicts track 1

        assert 1 not in t._unidentified
        assert {2, 3} <= set(t._unidentified)

    def test_reset_clears_the_lane(self):
        t = _tracker()
        t.on_unidentified(1, "cam-a", timestamp=1000.0)
        t.reset()
        assert t._unidentified == {}


class TestCooldownLeak:
    """
    The latent bug Step 1.5 fixes. `_cooldowns` was only ever pruned when that
    exact person_id was seen again, so any id that never returned stayed
    forever.
    """

    def test_expired_cooldowns_are_pruned(self):
        t = _tracker()
        t._cooldowns["gone-forever"] = 500.0
        t._cooldowns["still-cooling"] = 2000.0

        t.check_exits(1000.0)

        assert "gone-forever" not in t._cooldowns
        assert "still-cooling" in t._cooldowns, "an unexpired cooldown must survive"

    def test_pruning_does_not_disturb_active_tracks(self):
        t = _tracker()
        t.on_detection("p1", "Rahul", "cam-a", timestamp=1000.0)
        t._cooldowns["stale"] = 1.0
        t.check_exits(1000.5)
        assert "p1" in t._active_tracks
        assert "stale" not in t._cooldowns


class TestStats:
    def test_stats_separate_identified_from_not(self):
        t = _tracker()
        t.on_detection("p1", "Rahul", "cam-a", timestamp=1000.0)
        t.on_unidentified(9, "cam-a", timestamp=1000.0)
        t.on_unidentified(9, "cam-a", timestamp=1006.0)

        stats = t.stats  # a property, not a method
        assert stats["unidentified_present"] == 1
        assert stats["unidentified_tracked"] == 1
        assert stats["total_unidentified"] == 1
        assert stats["active_tracks"] == 1, "the named person, alone"


class TestLaneAssignment:
    """
    One human must occupy exactly one lane at a time. The three cases, in the
    order the pipeline checks them.
    """

    def test_known_and_unknown_ids_share_the_identified_lane(self):
        """
        An UnknownRegistry id is an identity: stable across restarts and
        cameras. It belongs with named people, not with anonymous bodies.
        """
        t = _tracker()
        t.on_detection("p1", "Rahul", "cam-a", timestamp=1000.0)
        t.on_detection("u-abc123", "Unknown-abc123", "cam-a", timestamp=1000.0)

        assert set(t._active_tracks) == {"p1", "u-abc123"}
        assert t._unidentified == {}

    def test_a_stranger_who_turns_away_is_not_double_counted(self):
        """
        The regression this guards: a stranger gets an unknown id from their
        face, turns around, and -- if the track had forgotten that id -- would
        ALSO be counted as an anonymous body for the same visit.
        """
        t = _tracker()
        # Face visible: counted under the stable unknown id.
        for ts in (1000.0, 1002.0, 1004.0):
            t.on_detection("u-abc123", "Unknown-abc123", "cam-a", timestamp=ts)
        # Turns away. The pipeline keeps refreshing the SAME id because the
        # track still carries it, so on_unidentified is never reached.
        for ts in (1006.0, 1008.0, 1010.0):
            t.on_detection("u-abc123", "Unknown-abc123", "cam-a", timestamp=ts)

        assert t.unidentified_count == 0, "must not also be an anonymous body"
        assert t.get_occupancy() == 1
