"""
Learned camera topology: what two cameras teach about each other by watching
people walk.

The scenario is the COE corridor as measured: two cameras face each other, so a
person walking away from one walks towards the other, and their foot point
climbs one picture while it descends the other.

Every assertion is about the property that makes learning from unlabelled
traffic safe -- the truth repeats and coincidence does not -- rather than about
a particular number. No cameras, no models, no database.
"""

import json

import pytest

from app.vision.topology import Topology, TransitionModel, ViewMap


def facing(s):
    """Foot and height of someone a fraction `s` along the corridor, as the
    near camera and the far camera each see them."""
    here = ((0.5, 0.25 + 0.6 * s), 0.10 + 0.30 * s)
    there = ((0.5, 0.85 - 0.6 * s), 0.40 - 0.30 * s)
    return here, there


def teach(view, walks, steps=40, lateral=0.0):
    for walk in range(walks):
        for i in range(steps):
            (src, _h), (dst, height) = facing(i / (steps - 1))
            view.add((src[0] + lateral, src[1]), dst, walk=("walk", walk), dst_height=height)


# ── View maps ────────────────────────────────────────────────────────────────

def test_a_view_map_learns_where_a_person_appears_in_the_other_camera():
    view = ViewMap()
    teach(view, walks=6)
    assert view.ready
    (src, _), (dst, height) = facing(0.5)
    u, v, spread, n, tall = view.predict(src)
    assert (u, v) == pytest.approx(dst, abs=0.03)
    assert tall == pytest.approx(height, abs=0.04)
    assert spread < 0.03 and n >= view.k_min


def test_it_says_nothing_about_places_nobody_has_walked():
    view = ViewMap()
    teach(view, walks=6)
    assert view.predict((0.1, 0.1)) is None


def test_one_walk_is_not_enough_however_long():
    """A single pair of tracks may be two strangers who walked at once."""
    view = ViewMap()
    teach(view, walks=1, steps=400)
    assert not view.ready
    assert view.predict(facing(0.5)[0][0]) is None


def test_someone_standing_still_cannot_become_the_consensus():
    view = ViewMap()
    for _ in range(500):
        view.add((0.5, 0.5), (0.2, 0.2), walk="loiterer")
    assert view.samples <= view.per_walk


def test_walks_that_disagree_are_not_a_shared_view():
    """Two cameras that merely had people walking at the same time: every walk
    pairs the same place here with a different place there."""
    view = ViewMap()
    for walk in range(12):
        for i in range(40):
            src = (0.5, 0.25 + 0.6 * i / 39)
            # A different stranger each time, somewhere else each time.
            dst = (0.15 + 0.06 * walk, 0.2 + ((i * 7 + walk * 13) % 40) / 60)
            view.add(src, dst, walk=walk)
    assert view.samples >= view.ready_samples and view.walks >= view.ready_walks
    assert view.agreement < view.ready_agreement
    assert not view.ready
    assert view.predict((0.5, 0.5)) is None


def test_a_few_coincidences_do_not_spoil_a_real_view():
    view = ViewMap()
    teach(view, walks=8)
    for i in range(40):      # one walk that was really two different people
        view.add((0.5, 0.25 + 0.6 * i / 39), (0.9, 0.9), walk="coincidence")
    assert view.ready
    (src, _), (dst, _h) = facing(0.4)
    u, v, *_ = view.predict(src)
    assert (u, v) == pytest.approx(dst, abs=0.03)


def test_a_view_map_survives_a_restart():
    view = ViewMap()
    teach(view, walks=6)
    restored = ViewMap()
    restored.load_dict(json.loads(json.dumps(view.to_dict())))
    assert restored.ready and restored.samples == view.samples
    src = facing(0.3)[0][0]
    # Stored to four decimals: a ten-thousandth of the frame, far below a pixel.
    assert restored.predict(src)[:2] == pytest.approx(view.predict(src)[:2], abs=1e-3)
    # New walks after a restart must count as new, not collide with old ids.
    before = restored.walks
    restored.add((0.9, 0.9), (0.1, 0.1), walk="after restart")
    assert restored.walks == before + 1


# ── Transitions ──────────────────────────────────────────────────────────────

DOOR, ARRIVAL = (0.1, 0.9), (0.9, 0.9)


def test_a_route_is_the_delay_that_keeps_repeating():
    model = TransitionModel()
    for episode in range(6):
        model.vote(DOOR, ARRIVAL, gap=3.0 + 0.2 * (episode % 3), when=100.0 + 30 * episode)
    low, high, episodes = model.window(DOOR, ARRIVAL)
    assert low == pytest.approx(3.0) and high == pytest.approx(3.4)
    assert episodes == 6


def test_a_group_walking_through_together_is_one_piece_of_evidence():
    """Five friends through a door at once is one occasion, not five."""
    model = TransitionModel()
    for person in range(8):
        model.vote(DOOR, ARRIVAL, gap=3.0, when=100.0 + 0.3 * person)
    assert model.window(DOOR, ARRIVAL) is None


def test_coincidences_spread_over_time_are_not_a_route():
    model = TransitionModel()
    for i in range(40):
        model.vote(DOOR, ARRIVAL, gap=-1.5 + (i * 0.79) % 31, when=100.0 + 20 * i)
    assert model.window(DOOR, ARRIVAL) is None


def test_a_real_route_stands_out_of_the_coincidences():
    model = TransitionModel()
    for i in range(10):
        model.vote(DOOR, ARRIVAL, gap=5.0 + (i * 2.3) % 24, when=50.0 + 17 * i)
    for episode in range(14):
        model.vote(DOOR, ARRIVAL, gap=3.0 + 0.1 * (episode % 4), when=300.0 + 25 * episode)
    low, high, _ = model.window(DOOR, ARRIVAL)
    assert 2.9 <= low <= high <= 3.5


def test_a_route_belongs_to_its_doorway():
    model = TransitionModel()
    for episode in range(6):
        model.vote(DOOR, ARRIVAL, gap=3.0, when=100.0 + 30 * episode)
    assert model.window((0.9, 0.2), ARRIVAL) is None      # left somewhere else
    assert model.window(DOOR, (0.2, 0.2)) is None         # arrived somewhere else
    assert model.routes()[0]["episodes"] == 6


# ── The whole network ────────────────────────────────────────────────────────

def test_topology_round_trips_through_its_file(tmp_path):
    topology = Topology()
    for walk in range(6):
        for i in range(40):
            (a, ha), (b, hb) = facing(i / 39)
            topology.add_correspondence("left", a, "right", b, ("w", walk), ha, hb)
    for episode in range(6):
        topology.add_transition("centre", DOOR, "left", ARRIVAL, 2.0, 100.0 + 30 * episode)
    path = str(tmp_path / "nested" / "topology.json")
    assert topology.dirty and topology.save(path) and not topology.dirty

    restored = Topology()
    assert restored.load(path)
    assert restored.view_if_ready("left", "right") is not None
    assert restored.view_if_ready("right", "left") is not None     # learned both ways
    assert restored.transition("centre", "left").window(DOOR, ARRIVAL) is not None
    described = restored.describe()
    assert described["overlaps"][0]["ready"] and len(described["routes"]) == 1


def test_a_damaged_file_means_relearning_not_a_crash(tmp_path):
    path = tmp_path / "topology.json"
    path.write_text("{ not json")
    topology = Topology()
    assert topology.load(str(path)) is False
    assert topology.load(str(tmp_path / "missing.json")) is False
    assert topology.describe() == {"overlaps": [], "routes": []}


def test_forgetting_a_camera_drops_only_what_involved_it():
    topology = Topology()
    topology.add_correspondence("left", (0.5, 0.5), "right", (0.5, 0.5), "w")
    topology.add_correspondence("left", (0.5, 0.5), "centre", (0.5, 0.5), "w")
    topology.add_transition("centre", DOOR, "room", ARRIVAL, 2.0, 100.0)
    assert topology.forget(["right"]) == 2            # both directions of left<->right
    assert topology.view("left", "centre").samples == 1
    assert topology.transition("centre", "room").votes == 1
