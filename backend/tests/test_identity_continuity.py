"""
Identity continuity scenario harness.

Deterministic, headless regression suite for the identity pipeline. No camera,
no GPU, no database, no network. Runs the REAL ByteTracker and the REAL
IdentityLedger against synthetic detection sequences.

Purpose: give Phase 1 a before/after scorecard that does not depend on running
a live camera and reading logs.

Run directly:
    cd backend && python tests/test_identity_continuity.py

Or under pytest once it is installed:
    pytest tests/test_identity_continuity.py -v

Each scenario declares what SHOULD happen. Milestone 1 is observe-only, so the
recovery scenarios are expected to FAIL on identity today -- what they assert
now is that the ledger correctly *identifies the opportunity*. Milestone 2 will
flip `expect_identity_preserved` on.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from app.vision.bytetrack import ByteTracker
from app.vision.identity import IdentityLedger

PERSON = ("p-teacher-a", "Teacher A")
OTHER = ("p-teacher-b", "Teacher B")


def box(x, y, w=100, h=100):
    return np.array([[x, y, x + w, y + h]], dtype=np.float32)


def boxes(*specs):
    return np.array([[x, y, x + w, y + h] for x, y, w, h in specs], dtype=np.float32)


def scores(n):
    return np.ones(n, dtype=np.float32)


EMPTY_B = np.empty((0, 4), dtype=np.float32)
EMPTY_S = np.empty(0, dtype=np.float32)


class Rig:
    """A tracker + ledger pair, driven frame by frame."""

    def __init__(self):
        self.tracker = ByteTracker()
        self.ledger = IdentityLedger()
        self.opportunities = []
        # Opportunities raised by the MOST RECENT step only. The first frame of
        # every scenario necessarily creates a track (with no predecessor), so
        # the cumulative list is never empty and cannot tell us whether the
        # tracker held across the event under test.
        self.last_opps = []

    def step(self, b=EMPTY_B, s=None, assign=None):
        if s is None:
            s = scores(len(b))
        tracks = self.tracker.update(b, s)
        opps = self.ledger.observe(tracks)
        self.last_opps = opps
        self.opportunities.extend(opps)
        if assign:
            for t in tracks:
                if t.person_id is None:
                    t.person_id, t.person_name = assign
                    t.recognition_confidence = 0.93
                    t.recognition_locked = True
                    t.record_distance(0.31)
                    t.record_distance(0.29)
        return tracks


# ── Scenarios ────────────────────────────────────────────────────────────

def scenario_brief_occlusion():
    """Person briefly occluded (3-frame gap), reappears close to where they left."""
    r = Rig()
    for i in range(6):
        r.step(box(300 + i * 6, 200), assign=PERSON)
    for _ in range(3):
        r.step()
    tracks = r.step(box(345, 205))
    return r, tracks, "brief occlusion, small displacement"


def scenario_fast_motion_same_frame():
    """Head turn / fast motion: track breaks and a detection appears same frame."""
    r = Rig()
    for i in range(6):
        r.step(box(300 + i * 10, 200), assign=PERSON)
    tracks = r.step(box(620, 260))
    return r, tracks, "fast motion, same-frame recreation"


def scenario_long_gap():
    """Longer occlusion (12 frames), moderate displacement."""
    r = Rig()
    for i in range(8):
        r.step(box(300 + i * 5, 200), assign=PERSON)
    for _ in range(12):
        r.step()
    tracks = r.step(box(480, 240))
    return r, tracks, "12-frame gap, moderate displacement"


def scenario_genuinely_new_person():
    """A different person appears far away. Must NOT be recoverable."""
    r = Rig()
    for i in range(6):
        r.step(box(200, 200), assign=PERSON)
    for _ in range(3):
        r.step()
    tracks = r.step(box(1500, 900))
    return r, tracks, "genuinely new person, far away"


def scenario_two_people_crossing():
    """Two identified people cross. Neither may take the other's identity."""
    r = Rig()
    for i in range(6):
        t = r.tracker.update(boxes((200 + i * 20, 300, 100, 100),
                                   (900 - i * 20, 300, 100, 100)), scores(2))
        r.ledger.observe(t)
        for tr in t:
            if tr.person_id is None:
                who = PERSON if tr.bbox[0] < 550 else OTHER
                tr.person_id, tr.person_name = who
                tr.recognition_locked = True
    t = r.tracker.update(boxes((520, 300, 100, 100), (560, 300, 100, 100)), scores(2))
    opps = r.ledger.observe(t)
    r.last_opps = opps
    r.opportunities.extend(opps)
    return r, t, "two people crossing"


# ── Runner ───────────────────────────────────────────────────────────────

# Expectations, corrected after the first run revealed two wrong assumptions:
#
#  - "brief occlusion" does NOT break the track at all. ByteTrack's Step 3
#    matches it natively, so no recreation happens and there is no opportunity
#    to observe. The correct expectation is `tracker_held=True`.
#  - "two people crossing" DOES produce an opportunity, and that is the point:
#    it must be flagged AMBIGUOUS (>1 eligible candidate) so Milestone 2
#    refuses to assign. Recovering here would be an identity swap.
#
# (scenario, tracker_held, expect_opportunity, expect_ambiguous)
SCENARIOS = [
    (scenario_brief_occlusion,       True,  False, False),
    (scenario_fast_motion_same_frame, False, True,  False),
    (scenario_long_gap,               False, True,  False),
    (scenario_genuinely_new_person,   False, False, False),
    (scenario_two_people_crossing,    False, True,  True),
]


def run_all(verbose=True):
    rows = []
    for fn, held, want_opp, want_amb in SCENARIOS:
        r, tracks, label = fn()
        identity_kept = any(t.person_id == PERSON[0] for t in tracks)
        opp = r.last_opps[-1] if r.last_opps else None
        got_opp = bool(opp and opp.recoverable)
        n_elig = opp.n_eligible if opp else 0
        ambiguous = n_elig > 1
        best = opp.best if (opp and opp.best) else None
        tracker_held = not r.last_opps

        ok = (tracker_held == held) and (got_opp == want_opp) and (ambiguous == want_amb)

        rows.append({
            "label": label, "held": tracker_held, "want_held": held,
            "opp": got_opp, "want_opp": want_opp,
            "amb": ambiguous, "want_amb": want_amb,
            "n_elig": n_elig, "identity_kept": identity_kept,
            "iou": best.iou if best else 0.0,
            "dist": best.centroid_distance if best else float("nan"),
            "gap": best.frames_since_retired if best else -1,
            "ok": ok,
        })

    if verbose:
        print("=" * 100)
        print("IDENTITY CONTINUITY SCORECARD    (Milestone 1 — ledger is OBSERVE-ONLY)")
        print("=" * 100)
        print(f"{'scenario':<38} {'held':>5} {'opp':>5} {'amb':>5} {'#elig':>6} "
              f"{'ident':>6} {'IoU':>7} {'dist':>7} {'gap':>5}  ")
        print("-" * 100)
        for x in rows:
            print(f"{x['label']:<38} {str(x['held']):>5} {str(x['opp']):>5} "
                  f"{str(x['amb']):>5} {x['n_elig']:>6} {str(x['identity_kept']):>6} "
                  f"{x['iou']:>7.3f} {x['dist']:>7.0f} {x['gap']:>5}  "
                  f"{'PASS' if x['ok'] else 'FAIL'}")
        print("-" * 100)

        rec = [x for x in rows if x["want_opp"] and not x["want_amb"]]
        kept = sum(1 for x in rec if x["identity_kept"])
        iou_gate = sum(1 for x in rec if x["iou"] > 0.30)
        print(f"  recoverable scenarios detected  : "
              f"{sum(1 for x in rec if x['opp'])}/{len(rec)}")
        print(f"  identity actually preserved     : {kept}/{len(rec)}   "
              f"<- Milestone 2 target: {len(rec)}/{len(rec)}")
        print(f"  a pure IoU>0.30 gate would catch: {iou_gate}/{len(rec)}   "
              f"<- geometry alone recovers NOTHING")
        amb = [x for x in rows if x["want_amb"]]
        print(f"  ambiguous cases correctly flagged: "
              f"{sum(1 for x in amb if x['amb'])}/{len(amb)}   "
              f"<- these must NEVER be auto-assigned")
        print("=" * 100)

    return rows


# ── pytest entry points ──────────────────────────────────────────────────

def test_ledger_detects_recoverable_opportunities():
    for x in run_all(verbose=False):
        if x["want_opp"]:
            assert x["opp"], f"ledger missed opportunity: {x['label']}"


def test_ledger_rejects_implausible_candidates():
    for x in run_all(verbose=False):
        if not x["want_opp"]:
            assert not x["opp"], f"ledger proposed a bad match: {x['label']}"


def test_ambiguous_candidates_are_flagged():
    """Crossing people must produce >1 eligible candidate so Milestone 2 refuses."""
    for x in run_all(verbose=False):
        if x["want_amb"]:
            assert x["amb"], (
                f"{x['label']}: only {x['n_elig']} eligible candidate — "
                f"Milestone 2 would auto-assign and swap identities"
            )


def test_no_identity_swap_when_people_cross():
    r, tracks, _ = scenario_two_people_crossing()
    ids = [t.person_id for t in tracks if t.person_id]
    assert len(ids) == len(set(ids)), f"identity duplicated across tracks: {ids}"


if __name__ == "__main__":
    rows = run_all()
    sys.exit(0 if all(x["ok"] for x in rows) else 1)
