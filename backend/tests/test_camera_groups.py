"""
camera_group(): the zone key that bunches monitor feeds and scopes
cross-camera reasoning. location wins when set (migration 013 backfills the
existing fleet); the name-token fallback covers cameras registered without
one. No DB required.
"""

from app.services.camera_service import camera_group


def test_location_is_the_zone_when_set():
    assert camera_group("coe left corridor", "COE") == "COE"
    assert camera_group("MainGate", "Main Gate") == "Main Gate"
    assert camera_group("anything", "Remote (Browser)") == "Remote (Browser)"


def test_name_token_fallback_groups_siblings():
    # A camera added without a location still lands next to its siblings.
    assert camera_group("coe robotic", "") == "Coe"
    assert camera_group("coe left corridor", "") == "Coe"
    assert camera_group("Coe Corridor", None) == "Coe"
    assert camera_group("frontgate inside", "") == "Frontgate"


def test_whitespace_location_falls_through():
    assert camera_group("coe", "   ") == "Coe"


def test_degenerate_names():
    assert camera_group("", "") == "Ungrouped"
    assert camera_group(None, None) == "Ungrouped"
