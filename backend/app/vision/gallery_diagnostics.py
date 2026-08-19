"""
Gallery Diagnostics — measure the quality of the enrolled face gallery.

WHY THIS EXISTS
---------------
Recognition failures in this system are reported as thresholds:

    security 1 (distance=0.5323) vs security 3 (distance=0.5476), margin=0.0153
    Recovery rejected: verify_ambiguous_margin

Read as a threshold problem, the fix looks like "lower FACE_MATCH_MIN_MARGIN".
That is the wrong fix. A margin of 0.0153 between two DIFFERENT people means the
gallery cannot separate them — the embeddings enrolled for those two people are
nearly identical. Loosening the margin does not create the missing information,
it just converts an honest rejection into a silent identity swap.

So before touching any threshold, measure the gallery. This module answers:

  * Are one person's own embeddings consistent with each other?
  * Who is each person most confusable with, and by how much?
  * Which enrolled embeddings would be matched to the WRONG person?
  * Which embeddings are near-duplicates that cost search time and add no
    coverage?
  * Who needs re-enrollment, and why?

METHOD
------
Leave-one-out (LOO) replay. Every enrolled embedding is treated as if it were a
live query and pushed through the same decision rules `FaceRecognizer` uses:
best-distance-per-person, then a margin check against the runner-up person. The
embedding itself is excluded from its own person's candidates — without that,
every query matches itself at distance 0 and every margin looks perfect.

This is a proxy, not live accuracy: enrolled embeddings are cleaner than camera
frames, so real-world margins are WORSE than what this reports. A person flagged
here is definitely a problem; a person that looks clean here can still fail
live. Treat the numbers as an upper bound on quality.

Both decision gates are replayed, because they fail differently:

  recognition gate   FACE_MATCH_BALANCED / FACE_MATCH_MIN_MARGIN
                     -> failure means the person is not recognised at all
  verification gate  IDENTITY_VERIFY_MAX_DISTANCE / IDENTITY_VERIFY_MIN_MARGIN
                     -> failure means identity RECOVERY is refused, which is
                        what `verify_ambiguous_margin` in the logs actually is

COST
----
O(N^2) in the number of enrolled embeddings, held as one float32 matrix. At the
current scale (~10^2 embeddings) that is microseconds and kilobytes. It stays
comfortable into the low tens of thousands; it is not built for millions, and it
runs offline, so this is deliberate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Risk bands, worst first. Ordering matters: reports sort by it.
RISK_HIGH = "high"
RISK_MEDIUM = "medium"
RISK_GOOD = "good"

_RISK_ORDER = {RISK_HIGH: 0, RISK_MEDIUM: 1, RISK_GOOD: 2}


@dataclass
class Thresholds:
    """
    The decision rules being replayed.

    Defaults mirror `settings` so the report describes the system as actually
    configured, but every value is overridable — the point of the tool is to ask
    "what would happen if this were 0.55?" without restarting a camera.
    """
    match_distance: float          # FACE_MATCH_BALANCED
    match_margin: float            # FACE_MATCH_MIN_MARGIN
    verify_distance: float         # IDENTITY_VERIFY_MAX_DISTANCE
    verify_margin: float           # IDENTITY_VERIFY_MIN_MARGIN
    duplicate_distance: float      # SUGGESTION_DUPLICATE_THRESHOLD
    min_embeddings: int = 5        # below this, coverage is thin
    target_embeddings: int = 10    # what a healthy person looks like

    @classmethod
    def from_settings(cls, **overrides) -> "Thresholds":
        from app.config import settings

        base = dict(
            match_distance=settings.FACE_MATCH_BALANCED,
            match_margin=settings.FACE_MATCH_MIN_MARGIN,
            verify_distance=settings.IDENTITY_VERIFY_MAX_DISTANCE,
            verify_margin=settings.IDENTITY_VERIFY_MIN_MARGIN,
            duplicate_distance=settings.SUGGESTION_DUPLICATE_THRESHOLD,
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)


@dataclass
class EmbeddingProbe:
    """One enrolled embedding replayed as a query."""
    encoding_id: str
    quality: float

    own_distance: float                     # LOO best distance to same person
    other_distance: float                   # best distance to any other person
    other_person_id: Optional[str]
    other_person_name: Optional[str]

    @property
    def measurable(self) -> bool:
        """
        False when this person has only ONE embedding: removing it leaves no
        same-person candidate, so `own_distance` is infinite and every derived
        number would be meaningless. Such a probe is excluded from rates rather
        than counted as a failure — the gallery is untestable here, not wrong.
        """
        return bool(np.isfinite(self.own_distance))

    @property
    def margin(self) -> float:
        """Runner-up gap, exactly as FaceRecognizer computes it."""
        if not self.measurable:
            return float("nan")
        return self.other_distance - self.own_distance

    @property
    def is_misidentified(self) -> bool:
        """
        Another person is a CLOSER match than this embedding's own person.
        The most severe finding available: this is an identity swap that the
        margin check is the only thing standing between you and.
        """
        return self.measurable and self.other_distance < self.own_distance

    def would_recognise(self, t: Thresholds) -> bool:
        return (
            self.measurable
            and self.own_distance <= t.match_distance
            and self.margin >= t.match_margin
        )

    def would_verify(self, t: Thresholds) -> bool:
        return (
            self.measurable
            and self.own_distance <= t.verify_distance
            and self.margin >= t.verify_margin
        )

    def rejection_reason(self, t: Thresholds) -> Optional[str]:
        """Mirrors MatchResult.rejection_reason so report and logs use one vocabulary."""
        if not self.measurable:
            return "not_measurable_single_embedding"
        if self.is_misidentified:
            return "would_match_wrong_person"
        if self.own_distance > t.match_distance:
            return "distance_exceeds_threshold"
        if self.margin < t.match_margin:
            return "ambiguous_margin"
        return None


@dataclass
class PersonDiagnostics:
    """Gallery health for one enrolled person."""
    person_id: str
    person_name: str
    embedding_count: int
    probes: List[EmbeddingProbe] = field(default_factory=list)

    # Consistency of this person with themselves (pairwise, within person).
    intra_mean: Optional[float] = None
    intra_max: Optional[float] = None

    # Closest OTHER person in the gallery, by best-pair distance.
    nearest_person_id: Optional[str] = None
    nearest_person_name: Optional[str] = None
    nearest_distance: Optional[float] = None

    quality_mean: float = 0.0
    quality_min: float = 0.0
    duplicate_pairs: int = 0

    risk: str = RISK_GOOD
    reasons: List[str] = field(default_factory=list)
    recommendation: str = ""

    # ── Derived counts ────────────────────────────────────────────────────

    @property
    def measurable_probes(self) -> List[EmbeddingProbe]:
        return [p for p in self.probes if p.measurable]

    @property
    def worst_margin(self) -> Optional[float]:
        """The margin that will fail first. A single bad embedding sets this."""
        return min((p.margin for p in self.measurable_probes), default=None)

    @property
    def median_margin(self) -> Optional[float]:
        measurable = self.measurable_probes
        if not measurable:
            return None
        return float(np.median([p.margin for p in measurable]))

    @property
    def misidentified_count(self) -> int:
        return sum(1 for p in self.probes if p.is_misidentified)

    def recognisable_count(self, t: Thresholds) -> int:
        return sum(1 for p in self.probes if p.would_recognise(t))

    def verifiable_count(self, t: Thresholds) -> int:
        return sum(1 for p in self.probes if p.would_verify(t))

    def recognise_rate(self, t: Thresholds) -> Optional[float]:
        measurable = self.measurable_probes
        if not measurable:
            return None
        return self.recognisable_count(t) / len(measurable)

    def verify_rate(self, t: Thresholds) -> Optional[float]:
        """
        Share of embeddings that clear the STRICTER recovery gate. A low value
        here is the direct, measurable cause of `verify_ambiguous_margin` in the
        logs: recovery is being refused because the gallery cannot support it.
        """
        measurable = self.measurable_probes
        if not measurable:
            return None
        return self.verifiable_count(t) / len(measurable)

    @property
    def contaminated_encodings(self) -> List[EmbeddingProbe]:
        """Embeddings that match someone else better than their own person."""
        return [p for p in self.probes if p.is_misidentified]


@dataclass
class ConfusablePair:
    """Two people whose galleries overlap, worst-pair first."""
    a_id: str
    a_name: str
    b_id: str
    b_name: str
    min_distance: float          # closest embeddings across the two people
    centroid_distance: float     # how far apart they are on average

    def severity(self, t: Thresholds) -> str:
        if self.min_distance < t.match_margin:
            return RISK_HIGH
        if self.min_distance < t.verify_margin * 2:
            return RISK_MEDIUM
        return RISK_GOOD


@dataclass
class NameCollision:
    """
    Several person records enrolled under the same name.

    Almost always one human enrolled more than once — a re-enrollment that
    created a new record instead of adding to the existing one. It is worth
    separating from ordinary confusability because the fix is the opposite:
    these records must be MERGED, not given more captures. Asking the recogniser
    to keep two records of the same person apart is asking it to do something
    impossible, and no threshold change can help.

    Matching is on the exact name after case/whitespace normalisation. Typos
    ("secutiry 3" vs "security 3") are deliberately not fuzzy-matched — guessing
    at intent would merge genuinely different people. Those still show up in the
    confusable-pairs list.
    """
    name: str
    person_ids: List[str]
    embedding_counts: List[int]
    min_distance: float           # closest pair of embeddings across the records
    likely_same_person: bool      # close enough that they would match each other


@dataclass
class GalleryReport:
    thresholds: Thresholds
    persons: List[PersonDiagnostics]
    confusable_pairs: List[ConfusablePair]
    total_embeddings: int
    skipped_embeddings: int
    name_collisions: List[NameCollision] = field(default_factory=list)

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def high_risk(self) -> List[PersonDiagnostics]:
        return [p for p in self.persons if p.risk == RISK_HIGH]

    @property
    def reenrollment_candidates(self) -> List[PersonDiagnostics]:
        """High and medium risk, worst first — the actionable worklist."""
        return [p for p in self.persons if p.risk in (RISK_HIGH, RISK_MEDIUM)]

    @property
    def total_misidentified(self) -> int:
        return sum(p.misidentified_count for p in self.persons)

    @property
    def _measurable(self) -> List[EmbeddingProbe]:
        return [p for person in self.persons for p in person.measurable_probes]

    @property
    def unmeasurable_count(self) -> int:
        """Embeddings belonging to single-embedding persons — untestable by LOO."""
        return sum(1 for person in self.persons for p in person.probes if not p.measurable)

    def overall_recognise_rate(self) -> Optional[float]:
        probes = self._measurable
        if not probes:
            return None
        return sum(1 for p in probes if p.would_recognise(self.thresholds)) / len(probes)

    def overall_verify_rate(self) -> Optional[float]:
        probes = self._measurable
        if not probes:
            return None
        return sum(1 for p in probes if p.would_verify(self.thresholds)) / len(probes)


# ── Input normalisation ───────────────────────────────────────────────────

def _coerce_encoding(raw, dim: int) -> Optional[np.ndarray]:
    """Accept ndarray or the DB's serialized bytes, return an L2-normalised vector."""
    if raw is None:
        return None

    if isinstance(raw, (bytes, bytearray)):
        from app.vision.recognizer import deserialize_encoding
        arr = deserialize_encoding(bytes(raw))
    elif isinstance(raw, np.ndarray):
        arr = raw
    else:
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except Exception:
            return None

    if not isinstance(arr, np.ndarray) or arr.shape != (dim,):
        return None

    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return None
    return (arr / norm).astype(np.float32)


# ── Analysis ──────────────────────────────────────────────────────────────

def analyse_gallery(
    encodings: Sequence[dict],
    thresholds: Optional[Thresholds] = None,
    embedding_dim: int = 512,
    max_confusable_pairs: int = 20,
) -> GalleryReport:
    """
    Analyse an enrolled gallery.

    Args:
        encodings: rows as produced by `PersonService.get_all_encodings()` —
            dicts with person_id, person_name, encoding, encoding_id, quality.
            Passing that function's output directly is the intended use, so the
            report describes exactly what the recogniser would load.
        thresholds: decision rules to replay. Defaults to live `settings`.
        embedding_dim: expected vector length; rows of any other shape are
            counted as skipped rather than silently dropped.
        max_confusable_pairs: cap on the global confusable-pair list.

    Returns:
        GalleryReport. Persons are sorted worst-risk first.
    """
    if thresholds is None:
        thresholds = Thresholds.from_settings()

    vectors: List[np.ndarray] = []
    person_ids: List[str] = []
    names: Dict[str, str] = {}
    encoding_ids: List[str] = []
    qualities: List[float] = []
    skipped = 0

    for row in encodings:
        vec = _coerce_encoding(row.get("encoding"), embedding_dim)
        pid = row.get("person_id")
        if vec is None or not pid:
            skipped += 1
            continue
        vectors.append(vec)
        person_ids.append(pid)
        names.setdefault(pid, row.get("person_name") or "Unknown")
        encoding_ids.append(row.get("encoding_id") or "")
        qualities.append(float(row.get("quality") or 1.0))

    if not vectors:
        return GalleryReport(
            thresholds=thresholds, persons=[], confusable_pairs=[],
            total_embeddings=0, skipped_embeddings=skipped,
        )

    matrix = np.vstack(vectors)                       # (N, D), rows unit-norm
    pid_array = np.asarray(person_ids)
    n = len(vectors)

    # Cosine distance, every embedding against every embedding. Unit-norm rows
    # make the dot product the cosine similarity directly.
    distances = 1.0 - (matrix @ matrix.T)
    np.fill_diagonal(distances, np.inf)               # leave-one-out

    # Group row indices by person once; every metric below is a mask over this.
    groups: Dict[str, np.ndarray] = {
        pid: np.flatnonzero(pid_array == pid) for pid in dict.fromkeys(person_ids)
    }

    centroids = {
        pid: _unit(matrix[idx].mean(axis=0)) for pid, idx in groups.items()
    }

    persons: List[PersonDiagnostics] = []
    for pid, idx in groups.items():
        same = np.zeros(n, dtype=bool)
        same[idx] = True

        diag = PersonDiagnostics(
            person_id=pid,
            person_name=names[pid],
            embedding_count=len(idx),
            quality_mean=float(np.mean([qualities[i] for i in idx])),
            quality_min=float(np.min([qualities[i] for i in idx])),
        )

        # ── Intra-person spread: is this person consistent with themselves? ──
        if len(idx) >= 2:
            intra = distances[np.ix_(idx, idx)]
            finite = intra[np.isfinite(intra)]
            diag.intra_mean = float(finite.mean())
            diag.intra_max = float(finite.max())
            # Pairs so close they carry no new information (each counted once).
            diag.duplicate_pairs = int(
                np.count_nonzero(finite < thresholds.duplicate_distance) // 2
            )

        # ── LOO replay: every embedding as a live query ─────────────────────
        for i in idx:
            row = distances[i]

            own = float(row[same].min()) if len(idx) >= 2 else float("inf")

            other_mask = ~same
            if other_mask.any():
                j = int(np.flatnonzero(other_mask)[np.argmin(row[other_mask])])
                other, other_pid = float(row[j]), str(pid_array[j])
                other_name = names[other_pid]
            else:
                other, other_pid, other_name = float("inf"), None, None

            diag.probes.append(EmbeddingProbe(
                encoding_id=encoding_ids[i],
                quality=qualities[i],
                own_distance=own,
                other_distance=other,
                other_person_id=other_pid,
                other_person_name=other_name,
            ))

        # ── Nearest other person, by closest embedding pair ─────────────────
        # Best-pair (not centroid) distance, because best-per-person is exactly
        # how FaceRecognizer scores a candidate.
        nearest = min(
            (p for p in diag.probes if p.other_person_id),
            key=lambda p: p.other_distance,
            default=None,
        )
        if nearest is not None and np.isfinite(nearest.other_distance):
            diag.nearest_person_id = nearest.other_person_id
            diag.nearest_person_name = nearest.other_person_name
            diag.nearest_distance = nearest.other_distance

        persons.append(diag)

    # Scored only after every person exists: a split identity is a property of
    # the gallery as a whole, and it changes the advice given to each record in
    # the group from "re-enroll" to "merge".
    collisions = _name_collisions(persons, groups, distances, thresholds)
    by_person = {pid: c for c in collisions for pid in c.person_ids}
    for diag in persons:
        _score_risk(diag, thresholds, by_person.get(diag.person_id))

    persons.sort(key=lambda p: (_RISK_ORDER[p.risk], p.worst_margin if p.worst_margin is not None else 9.99))

    return GalleryReport(
        thresholds=thresholds,
        persons=persons,
        confusable_pairs=_confusable_pairs(
            groups, distances, centroids, names, max_confusable_pairs
        ),
        total_embeddings=n,
        skipped_embeddings=skipped,
        name_collisions=collisions,
    )


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _confusable_pairs(
    groups: Dict[str, np.ndarray],
    distances: np.ndarray,
    centroids: Dict[str, np.ndarray],
    names: Dict[str, str],
    limit: int,
) -> List[ConfusablePair]:
    """Every person-pair ranked by their closest cross-person embedding pair."""
    pids = list(groups.keys())
    pairs: List[ConfusablePair] = []

    for a in range(len(pids)):
        for b in range(a + 1, len(pids)):
            pa, pb = pids[a], pids[b]
            block = distances[np.ix_(groups[pa], groups[pb])]
            if block.size == 0:
                continue
            pairs.append(ConfusablePair(
                a_id=pa, a_name=names[pa],
                b_id=pb, b_name=names[pb],
                min_distance=float(block.min()),
                centroid_distance=float(1.0 - np.dot(centroids[pa], centroids[pb])),
            ))

    pairs.sort(key=lambda p: p.min_distance)
    return pairs[:limit]


def _normalise_name(name: str) -> str:
    return " ".join((name or "").split()).casefold()


def _name_collisions(
    persons: List[PersonDiagnostics],
    groups: Dict[str, np.ndarray],
    distances: np.ndarray,
    t: Thresholds,
) -> List[NameCollision]:
    """Find names owned by more than one person record, worst (closest) first."""
    by_name: Dict[str, List[PersonDiagnostics]] = {}
    for p in persons:
        by_name.setdefault(_normalise_name(p.person_name), []).append(p)

    collisions: List[NameCollision] = []
    for members in by_name.values():
        if len(members) < 2:
            continue

        closest = float("inf")
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                block = distances[
                    np.ix_(groups[members[a].person_id], groups[members[b].person_id])
                ]
                if block.size:
                    closest = min(closest, float(block.min()))

        collisions.append(NameCollision(
            name=members[0].person_name,
            person_ids=[p.person_id for p in members],
            embedding_counts=[p.embedding_count for p in members],
            min_distance=closest,
            likely_same_person=closest <= t.match_distance,
        ))

    collisions.sort(key=lambda c: c.min_distance)
    return collisions


def _score_risk(
    diag: PersonDiagnostics,
    t: Thresholds,
    collision: Optional[NameCollision] = None,
):
    """
    Assign a risk band and a specific, actionable recommendation.

    Ordering is by consequence, not by magnitude: an embedding that matches the
    WRONG person is worse than one that matches nobody, because the first is a
    silent identity swap and the second is a visible failure.
    """
    reasons: List[str] = []
    actions: List[str] = []
    risk = RISK_GOOD

    def escalate(level: str):
        nonlocal risk
        if _RISK_ORDER[level] < _RISK_ORDER[risk]:
            risk = level

    # ── Split identity: this person is enrolled more than once ──────────
    # Checked first, and stated first, because it invalidates everything
    # below it. Every other number for this person is measured against a
    # record that should not exist as a separate identity.
    if collision is not None:
        others = len(collision.person_ids) - 1
        reasons.insert(0, (
            f"enrolled {len(collision.person_ids)} times under the name "
            f"'{collision.name}' (records hold {'/'.join(str(c) for c in collision.embedding_counts)} "
            f"embeddings; closest pair {collision.min_distance:.4f})"
        ))
        if collision.likely_same_person:
            actions.insert(0, (
                f"MERGE these {len(collision.person_ids)} records into one person -- "
                f"the recogniser is being asked to tell someone apart from themselves, "
                f"which no threshold can do"
            ))
        else:
            actions.insert(0, (
                f"give these {len(collision.person_ids)} records distinct names, or merge "
                f"them if they are the same person -- {others} other record(s) share this name"
            ))
        escalate(RISK_HIGH)

    # ── Cannot be measured ──────────────────────────────────────────────
    if diag.embedding_count < 2:
        reasons.append("only 1 embedding -- self-consistency cannot be measured")
        actions.append(f"enroll {t.target_embeddings - diag.embedding_count} more captures")
        escalate(RISK_HIGH)

    # ── Contamination: someone else's face enrolled under this person ────
    contaminated = diag.contaminated_encodings
    if contaminated:
        worst = min(contaminated, key=lambda p: p.other_distance)
        reasons.append(
            f"{len(contaminated)} embedding(s) match another person better than "
            f"this one (closest: {worst.other_person_name} at {worst.other_distance:.4f})"
        )
        actions.append(
            "review and remove the contaminated embedding(s) -- likely a wrong-person "
            "capture enrolled under this identity"
        )
        escalate(RISK_HIGH)

    # ── Separation from the nearest other person ────────────────────────
    worst_margin = diag.worst_margin
    if worst_margin is not None and np.isfinite(worst_margin):
        if worst_margin < t.match_margin:
            reasons.append(
                f"worst margin {worst_margin:.4f} < recognition minimum {t.match_margin} "
                f"-- rejected as ambiguous vs {diag.nearest_person_name}"
            )
            actions.append(
                f"add captures that separate this person from {diag.nearest_person_name} "
                f"(different pose, lighting, distance)"
            )
            escalate(RISK_HIGH)
        elif worst_margin < t.verify_margin:
            reasons.append(
                f"worst margin {worst_margin:.4f} clears recognition but fails the "
                f"stricter recovery gate ({t.verify_margin}) -- identity recovery will be refused"
            )
            actions.append("add varied captures to widen separation")
            escalate(RISK_MEDIUM)

    # ── Self-consistency: do this person's own embeddings agree? ─────────
    unmatched = [
        p for p in diag.probes
        if np.isfinite(p.own_distance) and p.own_distance > t.match_distance
    ]
    if unmatched:
        reasons.append(
            f"{len(unmatched)} embedding(s) do not match any other embedding of the "
            f"same person (distance > {t.match_distance}) -- possible mislabel or "
            f"extreme pose/lighting outlier"
        )
        actions.append("inspect the outlier embedding(s); re-capture under normal conditions")
        escalate(RISK_MEDIUM)

    # ── Coverage ────────────────────────────────────────────────────────
    if 2 <= diag.embedding_count < t.min_embeddings:
        reasons.append(f"only {diag.embedding_count} embeddings (target {t.target_embeddings})")
        actions.append(f"enroll {t.target_embeddings - diag.embedding_count} more varied captures")
        escalate(RISK_MEDIUM)

    # ── Redundancy: cost without coverage ───────────────────────────────
    if diag.duplicate_pairs:
        reasons.append(
            f"{diag.duplicate_pairs} near-duplicate pair(s) below {t.duplicate_distance} "
            f"-- search cost without added coverage"
        )
        actions.append("prune duplicates and replace with varied poses")
        # Not an escalation on its own: wasteful, not wrong.

    # ── Quality ─────────────────────────────────────────────────────────
    if diag.quality_mean < 0.5:
        reasons.append(f"mean embedding quality {diag.quality_mean:.2f} is low")
        actions.append("re-capture in better lighting / closer to the camera")
        escalate(RISK_MEDIUM)

    diag.risk = risk
    diag.reasons = reasons
    diag.recommendation = "; ".join(actions) if actions else "no action needed"


# ── Reporting ─────────────────────────────────────────────────────────────

_RISK_LABEL = {RISK_HIGH: "HIGH", RISK_MEDIUM: "MEDIUM", RISK_GOOD: "GOOD"}


def _fmt(value: Optional[float], places: int = 4, width: int = 0) -> str:
    text = "n/a" if value is None or not np.isfinite(value) else f"{value:.{places}f}"
    return text.rjust(width) if width else text


def format_report(report: GalleryReport, verbose: bool = False) -> str:
    """Render a GalleryReport as plain text. ASCII only -- this runs in consoles."""
    t = report.thresholds
    out: List[str] = []
    add = out.append

    add("=" * 100)
    add("FACE GALLERY DIAGNOSTICS")
    add("=" * 100)
    add(
        f"{report.total_embeddings} embeddings across {report.person_count} persons"
        + (f"  ({report.skipped_embeddings} unusable, skipped)" if report.skipped_embeddings else "")
    )
    add(
        f"Recognition gate: distance <= {t.match_distance}, margin >= {t.match_margin}   |   "
        f"Recovery gate: distance <= {t.verify_distance}, margin >= {t.verify_margin}"
    )

    if not report.persons:
        add("")
        add("Gallery is empty -- nothing to analyse.")
        return "\n".join(out)

    rec, ver = report.overall_recognise_rate(), report.overall_verify_rate()
    if rec is None:
        add(
            "Leave-one-out replay: not possible -- every person has a single "
            "embedding, so there is nothing to match them against."
        )
    else:
        add(
            f"Leave-one-out replay: {rec * 100:.1f}% would be recognised, "
            f"{ver * 100:.1f}% would pass recovery verification"
            + (
                f"  ({report.unmeasurable_count} embedding(s) excluded -- single-embedding persons)"
                if report.unmeasurable_count else ""
            )
        )
    if report.total_misidentified:
        add(
            f"*** {report.total_misidentified} embedding(s) match the WRONG person -- "
            f"these are identity swaps waiting on the margin check ***"
        )
    add("")

    # ── Split identities: fix these before reading anything else ────────
    if report.name_collisions:
        add("SPLIT IDENTITIES -- FIX THESE FIRST")
        add("-" * 100)
        add(
            "  The same name is enrolled as several separate person records. Until these are"
        )
        add(
            "  merged, the recogniser is being asked to distinguish a person from themselves,"
        )
        add(
            "  and every margin below is measured against a record that should not exist."
        )
        add("")
        for c in report.name_collisions:
            counts = ", ".join(
                f"{pid[:8]} ({n} emb)"
                for pid, n in zip(c.person_ids, c.embedding_counts)
            )
            verdict = (
                "same person -- MERGE"
                if c.likely_same_person
                else "far apart in embedding space -- verify before merging"
            )
            add(f"  '{c.name}' x{len(c.person_ids)}  closest={c.min_distance:.4f}  [{verdict}]")
            add(f"        {counts}")
        add("")

    # ── Per-person table ────────────────────────────────────────────────
    add(
        f"{'Person':<22}{'Emb':>5}{'Nearest neighbour':>22}{'NN dist':>10}"
        f"{'Worst mgn':>11}{'Intra max':>11}{'Recog':>8}{'Recov':>8}  Risk"
    )
    add("-" * 100)
    for p in report.persons:
        rec_rate, ver_rate = p.recognise_rate(t), p.verify_rate(t)
        add(
            f"{p.person_name[:21]:<22}"
            f"{p.embedding_count:>5}"
            f"{(p.nearest_person_name or '-')[:21]:>22}"
            f"{_fmt(p.nearest_distance, 4, 10)}"
            f"{_fmt(p.worst_margin, 4, 11)}"
            f"{_fmt(p.intra_max, 4, 11)}"
            f"{('n/a' if rec_rate is None else f'{rec_rate * 100:.0f}%'):>8}"
            f"{('n/a' if ver_rate is None else f'{ver_rate * 100:.0f}%'):>8}"
            f"  {_RISK_LABEL[p.risk]}"
        )
    add("")

    # ── Most confusable identities ──────────────────────────────────────
    if report.confusable_pairs:
        add("MOST CONFUSABLE IDENTITIES (closest embedding pair across two people)")
        add("-" * 100)
        for pair in report.confusable_pairs:
            flag = "  <-- below recognition margin" if pair.min_distance < t.match_margin else ""
            add(
                f"  {pair.a_name[:24]:<26} vs {pair.b_name[:24]:<26} "
                f"closest={pair.min_distance:.4f}  centroid={pair.centroid_distance:.4f}"
                f"{flag}"
            )
        add("")

    # ── Worklist ────────────────────────────────────────────────────────
    candidates = report.reenrollment_candidates
    if candidates:
        add("RE-ENROLLMENT CANDIDATES (worst first)")
        add("-" * 100)
        for p in candidates:
            add(f"  [{_RISK_LABEL[p.risk]}] {p.person_name}  ({p.embedding_count} embeddings)")
            for reason in p.reasons:
                add(f"        - {reason}")
            add(f"        -> {p.recommendation}")
            if verbose:
                for probe in p.probes:
                    reason = probe.rejection_reason(t)
                    add(
                        f"           {(probe.encoding_id or '?')[:8]:<10} "
                        f"own={_fmt(probe.own_distance)} "
                        f"other={_fmt(probe.other_distance)} "
                        f"({probe.other_person_name or '-'}) "
                        f"margin={_fmt(probe.margin)} "
                        f"q={probe.quality:.2f}"
                        + (f"  [{reason}]" if reason else "")
                    )
            add("")
    else:
        add("No re-enrollment needed -- every person is separable at the configured thresholds.")

    return "\n".join(out)


def report_to_dict(report: GalleryReport) -> dict:
    """JSON-serialisable form, for an API endpoint or for diffing runs over time."""
    t = report.thresholds
    return {
        "thresholds": {
            "match_distance": t.match_distance,
            "match_margin": t.match_margin,
            "verify_distance": t.verify_distance,
            "verify_margin": t.verify_margin,
            "duplicate_distance": t.duplicate_distance,
        },
        "summary": {
            "persons": report.person_count,
            "embeddings": report.total_embeddings,
            "skipped": report.skipped_embeddings,
            "unmeasurable": report.unmeasurable_count,
            "misidentified": report.total_misidentified,
            "recognise_rate": report.overall_recognise_rate(),
            "verify_rate": report.overall_verify_rate(),
            "high_risk_persons": len(report.high_risk),
            "split_identities": len(report.name_collisions),
        },
        "name_collisions": [
            {
                "name": c.name,
                "person_ids": c.person_ids,
                "embedding_counts": c.embedding_counts,
                "min_distance": c.min_distance,
                "likely_same_person": c.likely_same_person,
            }
            for c in report.name_collisions
        ],
        "persons": [
            {
                "person_id": p.person_id,
                "person_name": p.person_name,
                "embedding_count": p.embedding_count,
                "intra_mean": p.intra_mean,
                "intra_max": p.intra_max,
                "nearest_person_id": p.nearest_person_id,
                "nearest_person_name": p.nearest_person_name,
                "nearest_distance": p.nearest_distance,
                "worst_margin": p.worst_margin,
                "median_margin": p.median_margin,
                "misidentified_count": p.misidentified_count,
                "duplicate_pairs": p.duplicate_pairs,
                "quality_mean": p.quality_mean,
                "quality_min": p.quality_min,
                "recognise_rate": p.recognise_rate(t),
                "verify_rate": p.verify_rate(t),
                "risk": p.risk,
                "reasons": p.reasons,
                "recommendation": p.recommendation,
                "contaminated_encoding_ids": [
                    c.encoding_id for c in p.contaminated_encodings
                ],
            }
            for p in report.persons
        ],
        "confusable_pairs": [
            {
                "a_id": c.a_id, "a_name": c.a_name,
                "b_id": c.b_id, "b_name": c.b_name,
                "min_distance": c.min_distance,
                "centroid_distance": c.centroid_distance,
                "severity": c.severity(t),
            }
            for c in report.confusable_pairs
        ],
    }
