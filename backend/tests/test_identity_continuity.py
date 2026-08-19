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


# ── Milestone 2: propose -> verify -> restore ────────────────────────────

DIM = 512


def _unit(v):
    return v / np.linalg.norm(v)


def at_cosine(base, cos_target, rng):
    """
    Return a unit vector at EXACTLY `cos_target` cosine similarity to `base`.

    Additive Gaussian noise is useless for this in high dimensions: at 512-d a
    randn vector scaled by 0.3 already has norm ~6.8, so it swamps the unit
    signal and the result is effectively random. Instead, project out the base
    direction and recombine at the exact angle we want.
    """
    n = rng.randn(len(base))
    n -= base * float(np.dot(n, base))      # orthogonalise against base
    n = _unit(n)
    return base * cos_target + n * float(np.sqrt(1.0 - cos_target ** 2))


def _gallery():
    """
    Synthetic 3-person gallery:
      A  the target
      B  clearly different (stage-1 reject case)
      C  a near-twin of A (stage-2 MARGIN reject case)
    """
    from app.vision.recognizer import FaceRecognizer
    rng = np.random.RandomState(42)

    base_a = _unit(rng.randn(DIM))
    base_b = _unit(rng.randn(DIM))
    base_c = at_cosine(base_a, 0.85, rng)     # near-twin of A

    rows = []
    for i in range(8):                        # A: 8 tight multi-pose embeddings
        rows.append({"encoding_id": f"a{i}", "person_id": "p-a", "person_name": "Teacher A",
                     "encoding": at_cosine(base_a, 0.98, rng), "quality": 0.9})
    for i in range(4):
        rows.append({"encoding_id": f"b{i}", "person_id": "p-b", "person_name": "Teacher B",
                     "encoding": at_cosine(base_b, 0.98, rng), "quality": 0.9})
    for i in range(4):
        rows.append({"encoding_id": f"c{i}", "person_id": "p-c", "person_name": "Teacher C",
                     "encoding": at_cosine(base_c, 0.98, rng), "quality": 0.9})

    r = FaceRecognizer()
    r.load_encodings(rows)
    return r, base_a, base_b, base_c


def run_verification_matrix(verbose=True):
    """Exercise FaceRecognizer.verify_against_person across the decision space."""
    from app.config import settings
    rng = np.random.RandomState(7)
    rec, base_a, base_b, base_c = _gallery()

    # Equidistant between A and its near-twin C: close enough to pass the
    # distance gate, but with almost no margin -> must fail at stage 2.
    between_a_and_c = _unit(base_a + base_c)

    # NOTE on "hard angle, twin in gallery": this is EXPECTED to be declined.
    # Teacher C sits at cosine 0.85 from Teacher A, so a hard-angle capture of A
    # is genuinely close to both. Declining to restore a *locked* identity there
    # is the swap protection doing its job -- the track simply falls through to
    # ordinary recognition, which applies its own margin check. The cost is a
    # lower recovery rate in galleries containing similar-looking people, which
    # is the correct trade for an attendance record.
    cases = [
        ("same person, clean",             at_cosine(base_a, 0.97, rng), "p-a", True),
        ("same person, hard angle (no twin)", at_cosine(base_b, 0.70, rng), "p-b", True),
        ("hard angle, twin in gallery",    at_cosine(base_a, 0.70, rng), "p-a", False),
        ("same person, past threshold",    at_cosine(base_a, 0.30, rng), "p-a", False),
        ("WRONG person proposed",          at_cosine(base_b, 0.97, rng), "p-a", False),
        ("near-twin, ambiguous margin",    between_a_and_c,              "p-a", False),
        ("stranger proposed as A",         _unit(rng.randn(DIM)),        "p-a", False),
        ("person not in gallery",          base_a.copy(),                "p-zz", False),
    ]

    rows = []
    for label, enc, pid, want in cases:
        res = rec.verify_against_person(enc, pid)
        rows.append({
            "label": label, "want": want, "got": res.matched,
            "dist": res.distance, "margin": res.margin,
            "reason": res.rejection_reason or "-",
            "ok": res.matched == want,
        })

    if verbose:
        print()
        print("=" * 100)
        print("VERIFICATION MATRIX  (recogniser is the authority; ledger only proposes)")
        print(f"  max_distance={settings.IDENTITY_VERIFY_MAX_DISTANCE}  "
              f"min_margin={settings.IDENTITY_VERIFY_MIN_MARGIN}")
        print("=" * 100)
        print(f"{'case':<32} {'want':>6} {'got':>6} {'dist':>7} {'margin':>8}  {'reason':<34}")
        print("-" * 100)
        for x in rows:
            print(f"{x['label']:<32} {str(x['want']):>6} {str(x['got']):>6} "
                  f"{x['dist']:>7.4f} {x['margin']:>8.4f}  {x['reason']:<34}"
                  f"{'PASS' if x['ok'] else 'FAIL'}")
        print("=" * 100)
    return rows


# ── End-to-end: propose -> verify -> restore ─────────────────────────────

class StubPipeline:
    """
    Minimal stand-in exposing exactly what _maybe_recover_identity touches.

    Avoids constructing a real VisionPipeline (which would load SCRFD, YOLO,
    DeepFace and friends) while still exercising the REAL method under test.
    """
    camera_id = "test-cam"

    def __init__(self, ledger, recognizer):
        self.identity_ledger = ledger
        self.recognizer = recognizer
        self._recovered_track_ids = set()
        self._recognition_lock_confidence = 0.85

    _maybe_recover_identity = None  # bound below


def _bind_real_method():
    from app.vision.pipeline import VisionPipeline
    StubPipeline._maybe_recover_identity = VisionPipeline._maybe_recover_identity


def run_end_to_end(verbose=True):
    """
    Full chain on a track ByteTrack recreated: the ledger proposes, the real
    recogniser verifies, and the full recognition state is restored.
    """
    from app.vision.identity import IdentityLedger, RetiredIdentity
    _bind_real_method()

    rng = np.random.RandomState(11)
    rec, base_a, base_b, _ = _gallery()
    ledger = IdentityLedger()
    pipe = StubPipeline(ledger, rec)

    class FakeTrack:
        def __init__(self, tid):
            self.track_id = tid
            self.bbox = np.array([300, 200, 400, 300], np.float32)
            self.person_id = None
            self.person_name = None
            self.recognition_confidence = 0.0
            self.recognition_time = 0.0
            self.recognition_locked = False
            self.identity_fail_streak = 0
            self._distance_history = []
            self.tracklet_len = 0

    retired = RetiredIdentity(
        track_id=1, person_id="p-a", person_name="Teacher A",
        bbox=np.array([280, 195, 380, 295], np.float32),
        retired_frame=1, retired_at=0.0,
        recognition_confidence=0.94, recognition_locked=True,
        identity_fail_streak=2, distance_history=[0.31, 0.28, 0.33],
    )

    rows = []
    for label, enc, want in [
        ("correct person reappears", at_cosine(base_a, 0.97, rng), True),
        ("different person appears", at_cosine(base_b, 0.97, rng), False),
    ]:
        ledger._retired = [retired]
        ledger._frame_index = 2
        track = FakeTrack(99)
        from app.vision.identity import RecoveryCandidate, RecoveryOpportunity
        cand = RecoveryCandidate(retired=retired, iou=0.0, centroid_distance=25.0,
                                 frames_since_retired=1, seconds_since_retired=0.1)
        ledger._proposals = {99: RecoveryOpportunity(
            track_id=99, bbox=track.bbox, best=cand, n_eligible=1, n_retired_total=1)}
        pipe._recovered_track_ids.clear()

        restored = pipe._maybe_recover_identity(track, enc)
        rows.append({
            "label": label, "want": want, "restored": restored,
            "person": track.person_name,
            "locked": track.recognition_locked,
            "history": len(track._distance_history),
            "streak": track.identity_fail_streak,
            "ok": restored == want and (
                (track.person_id == "p-a" and len(track._distance_history) == 3)
                if want else track.person_id is None
            ),
        })

    if verbose:
        print()
        print("=" * 100)
        print("END-TO-END  ledger proposes -> recogniser verifies -> state restored")
        print("=" * 100)
        print(f"{'case':<30} {'want':>6} {'restored':>9} {'person':>12} "
              f"{'locked':>7} {'hist':>5} {'streak':>7}")
        print("-" * 100)
        for x in rows:
            print(f"{x['label']:<30} {str(x['want']):>6} {str(x['restored']):>9} "
                  f"{str(x['person']):>12} {str(x['locked']):>7} {x['history']:>5} "
                  f"{x['streak']:>7}   {'PASS' if x['ok'] else 'FAIL'}")
        print("=" * 100)
    return rows


# ── pytest entry points ──────────────────────────────────────────────────

def test_end_to_end_recovery():
    for x in run_end_to_end(verbose=False):
        assert x["ok"], f"{x['label']}: restored={x['restored']} want={x['want']}"


def test_restore_brings_back_temporal_state():
    """
    The whole point of restoring: hysteresis and temporal rescue need the
    distance history and fail streak back, not just the name. ByteTrack's own
    carry-over copied only person_id/person_name.
    """
    rows = run_end_to_end(verbose=False)
    good = [x for x in rows if x["want"]][0]
    assert good["history"] == 3, "distance history was not restored"
    assert good["streak"] == 2, "identity fail streak was not restored"
    assert good["locked"] is True, "lock was not re-earned on a strong verification"

def test_verification_matrix():
    for x in run_verification_matrix(verbose=False):
        assert x["ok"], f"{x['label']}: wanted matched={x['want']}, got {x['got']} ({x['reason']})"


def test_verification_rejects_wrong_person_cheaply():
    """A clearly wrong proposal must fail at stage 1, before the full search."""
    rec, base_a, base_b, _ = _gallery()
    res = rec.verify_against_person(_unit(base_b), "p-a")
    assert not res.matched
    assert res.rejection_reason == "verify_distance_exceeds_threshold", res.rejection_reason


def test_verification_rejects_near_twin_on_margin():
    """
    The stage-2 path must actually be exercised: a vector equidistant between
    two similar people passes the distance gate but must fail on margin.
    Without this, a near-twin would silently inherit a locked identity.
    """
    rec, base_a, _, base_c = _gallery()
    res = rec.verify_against_person(_unit(base_a + base_c), "p-a")
    assert not res.matched, "near-twin was verified — identity swap risk"
    assert res.rejection_reason == "verify_ambiguous_margin", res.rejection_reason

def test_recognizer_reports_what_it_loaded():
    """
    PREVENTS: the double gallery load at camera start. CameraManager loads the
    cached gallery into a new pipeline, then the pipeline's own safety-net
    refresh compared the DB count against a counter it kept itself -- which was
    still 0 -- and reloaded the identical gallery. Callers must be able to ask
    the recognizer what it already has.
    """
    rec, _, _, _ = _gallery()
    assert rec.source_count == 16, rec.source_count

    # A row the recognizer cannot use must still count against the source, or a
    # DB-count comparison never converges and reloads on every single check.
    rec.load_encodings([
        {"encoding_id": "bad", "person_id": "p-x", "person_name": "X",
         "encoding": np.ones(8, dtype=np.float32), "quality": 1.0},
    ])
    assert rec.known_count == 0, "unusable encoding was loaded"
    assert rec.source_count == 1, "unusable row lost from the source count"


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
    vrows = run_verification_matrix()
    erows = run_end_to_end()
    ok = all(x["ok"] for x in rows + vrows + erows)
    print()
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)
