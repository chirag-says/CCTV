"""
Tests for the Gallery Optimizer's member-level approval path.

Two layers:

  * Unit tests — pose estimation and bucketing, outlier maths, coverage
    projection. No database.

  * End-to-end tests — a real person, a real cluster of unknown faces with
    real pose metadata, run through preview → approve → rollback against the
    live database. These are the tests that actually prove approval adds
    individual embeddings rather than one averaged centroid, and that a
    rejection leaves the gallery untouched.

The E2E fixtures create their own rows under a recognisable prefix and delete
them afterwards, including on failure. They skip cleanly if no database is
reachable.
"""

import os
from uuid import uuid4

import numpy as np
import pytest

from app.services.clustering_service import centroid_distances, outlier_flags
from app.services.embedding_suggestions import classify_pose_bucket
from app.vision.quality import FaceQualityGate


# ── Helpers ──────────────────────────────────────────────────────

def landmarks_for(nose_offset, eye_span=44.0, cx=100.0, cy=100.0, roll_dy=0.0):
    """
    Synthesise InsightFace-style 5-point landmarks with a controlled yaw.

    estimate_pose() derives yaw from how far the nose sits from the midpoint
    between the eyes, in units of eye separation. An offset of 0.5 is ~45deg.
    """
    return np.array([
        [cx - eye_span / 2, cy],
        [cx + eye_span / 2, cy + roll_dy],
        [cx + nose_offset * eye_span, cy + 22.0],
        [cx - 13.0, cy + 44.0],
        [cx + 13.0, cy + 44.0],
    ], dtype=np.float32)


def unit(v):
    return v / np.linalg.norm(v)


def spread_embeddings(base, n, alpha, seed=7):
    """
    n unit vectors arranged around `base`, each displaced along its own
    orthogonal direction so they are mutually distinct but all close to base.
    """
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        d = rng.randn(len(base)).astype(np.float32)
        d -= np.dot(d, base) * base          # orthogonalise against base
        out.append(unit(base + alpha * unit(d)))
    return out


# ── Phase A: pose estimation and bucketing ───────────────────────

class TestPoseEstimation:
    """Real landmarks must produce real, distinguishable poses."""

    @pytest.mark.parametrize("offset,expected", [
        (-0.85, "extreme_right"),
        (-0.40, "right"),
        (-0.20, "right"),
        (0.0,   "frontal"),
        (0.10,  "frontal"),
        (0.40,  "left"),
        (0.85,  "extreme_left"),
    ])
    def test_yaw_buckets_are_distinct(self, offset, expected):
        yaw, _, _ = FaceQualityGate.estimate_pose(landmarks_for(offset), face_width=100)
        assert classify_pose_bucket(yaw) == expected

    def test_all_five_buckets_reachable(self):
        """
        The whole point of the fix: five different views must land in five
        different buckets. Before landmarks were plumbed through, every one
        of these returned yaw=0 and bucketed as "frontal".
        """
        offsets = [-0.85, -0.40, 0.0, 0.40, 0.85]
        buckets = {
            classify_pose_bucket(
                FaceQualityGate.estimate_pose(landmarks_for(o), 100)[0]
            )
            for o in offsets
        }
        assert buckets == {
            "extreme_right", "right", "frontal", "left", "extreme_left",
        }

    def test_yaw_is_monotonic_in_offset(self):
        yaws = [
            FaceQualityGate.estimate_pose(landmarks_for(o), 100)[0]
            for o in [-0.8, -0.4, -0.1, 0.0, 0.1, 0.4, 0.8]
        ]
        assert yaws == sorted(yaws)

    def test_roll_detected_from_eye_line(self):
        _, _, roll = FaceQualityGate.estimate_pose(
            landmarks_for(0.0, roll_dy=22.0), 100,
        )
        assert roll > 15.0

    def test_no_landmarks_returns_zeros(self):
        """
        estimate_pose() cannot know the pose without landmarks. It returns
        zeros — which is exactly why callers must not persist that value.
        See test_missing_landmarks_yields_no_metadata.
        """
        assert FaceQualityGate.estimate_pose(None) == (0.0, 0.0, 0.0)
        assert FaceQualityGate.estimate_pose(np.zeros((3, 2))) == (0.0, 0.0, 0.0)


class TestPoseMetadataIsNeverFabricated:
    """A missing pose must stay missing, not become 'frontal'."""

    def test_missing_landmarks_yields_no_metadata(self):
        from app.vision.pipeline import VisionPipeline

        pipeline = object.__new__(VisionPipeline)  # no camera/model needed
        meta = VisionPipeline._pose_metadata(
            pipeline, frame=np.zeros((80, 80, 3), np.uint8),
            location=(0, 60, 60, 0), landmarks=None,
        )
        assert meta == {}
        assert "pose_bucket" not in meta
        assert "yaw" not in meta


# ── Phase D: outlier detection ───────────────────────────────────

class TestOutlierMaths:

    def test_one_wrong_face_is_flagged(self):
        rng = np.random.RandomState(3)
        base = unit(rng.randn(512).astype(np.float32))
        members = spread_embeddings(base, 9, alpha=0.15)
        members.append(unit(rng.randn(512).astype(np.float32)))  # a different person

        centroid = np.mean(members, axis=0)
        flags, threshold = outlier_flags(
            centroid_distances(members, centroid), outlier_factor=1.5,
        )

        assert flags[-1] is True or flags[-1] == True   # the intruder
        assert sum(flags) == 1                          # and only the intruder
        assert threshold > 0

    def test_consistent_cluster_flags_nothing(self):
        rng = np.random.RandomState(11)
        base = unit(rng.randn(512).astype(np.float32))
        members = spread_embeddings(base, 8, alpha=0.12)
        centroid = np.mean(members, axis=0)

        flags, _ = outlier_flags(centroid_distances(members, centroid), 1.5)
        assert sum(flags) == 0

    def test_two_member_cluster_flags_nothing(self):
        """With two members there is no majority to be an outlier from."""
        flags, threshold = outlier_flags([0.1, 0.9], outlier_factor=1.5)
        assert flags == [False, False]
        assert threshold == 0.0

    def test_unusable_vector_is_maximally_distant(self):
        base = unit(np.random.RandomState(5).randn(512).astype(np.float32))
        d = centroid_distances([base, np.zeros(512, np.float32)], base)
        assert d[0] == pytest.approx(0.0, abs=1e-5)
        assert d[1] == 1.0


# ── Phase E: coverage projection ─────────────────────────────────

class TestCoverageProjection:

    def test_multiple_poses_project_more_than_one(self, monkeypatch):
        from app.services import gallery_health

        class FakeProfile:
            coverage_percentage = 30.0
            covered_buckets = ["pose_frontal", "appearance_plain", "quality_medium"]
            health_score = 0.5
            quality_score = 0.6
            diversity_score = 0.5
            recognition_score = 0.4
            redundancy_penalty = 0.0

        monkeypatch.setattr(
            gallery_health, "compute_person_profile",
            lambda person_id, db=None: FakeProfile(),
        )

        one = gallery_health.compute_coverage_delta(
            "p", new_pose_bucket="left", new_quality=0.9, db=object(),
        )
        many = gallery_health.compute_coverage_delta(
            "p", new_pose_buckets=["left", "extreme_left", "right"],
            new_quality=0.9, db=object(),
        )

        assert many.projected_coverage > one.projected_coverage
        assert many.delta > one.delta

    def test_null_pose_adds_no_pose_coverage(self, monkeypatch):
        from app.services import gallery_health

        class FakeProfile:
            coverage_percentage = 30.0
            covered_buckets = ["pose_frontal", "appearance_plain", "quality_medium"]
            health_score = 0.5
            quality_score = 0.6
            diversity_score = 0.5
            recognition_score = 0.4
            redundancy_penalty = 0.0

        monkeypatch.setattr(
            gallery_health, "compute_person_profile",
            lambda person_id, db=None: FakeProfile(),
        )

        d = gallery_health.compute_coverage_delta(
            "p", new_pose_buckets=[None, None], new_quality=0.6, db=object(),
        )
        assert d.fills_bucket is None


# ── Snapshot path resolution ─────────────────────────────────────

class TestSnapshotPathResolution:
    """
    save_snapshot() stores "/snapshots/<sub>/<name>.jpg" while SNAPSHOT_DIR is
    "./snapshots". Joining the two doubles the segment, and on Windows the
    leading slash does not make the path absolute (no drive letter), so the
    usual is_absolute() guard misses it. That combination silently broke every
    VLM call: _prepare_image returned None and the reviewer answered
    image_load_failed in 0.0s.
    """

    def test_stored_url_resolves_to_the_real_file(self, tmp_path, monkeypatch):
        from app.config import settings
        from app.utils.image_utils import resolve_snapshot_path

        snaps = tmp_path / "snapshots"
        (snaps / "unknowns").mkdir(parents=True)
        target = snaps / "unknowns" / "unknown_abc.jpg"
        target.write_bytes(b"jpeg")

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings, "SNAPSHOT_DIR", "./snapshots")

        resolved = resolve_snapshot_path("/snapshots/unknowns/unknown_abc.jpg")
        assert resolved is not None, "the doubled-prefix bug is back"
        assert os.path.isfile(resolved)
        assert os.path.samefile(resolved, target)

    def test_absolute_path_passes_through(self, tmp_path, monkeypatch):
        from app.config import settings
        from app.utils.image_utils import resolve_snapshot_path

        f = tmp_path / "face.jpg"
        f.write_bytes(b"jpeg")
        monkeypatch.setattr(settings, "SNAPSHOT_DIR", "./snapshots")
        assert os.path.samefile(resolve_snapshot_path(str(f)), f)

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        from app.config import settings
        from app.utils.image_utils import resolve_snapshot_path

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings, "SNAPSHOT_DIR", "./snapshots")
        assert resolve_snapshot_path("/snapshots/unknowns/nope.jpg") is None
        assert resolve_snapshot_path("") is None
        assert resolve_snapshot_path(None) is None


# ── End-to-end against the real database ─────────────────────────

def _db_available():
    try:
        from app.db.session import SessionLocal
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("select 1"))
        db.close()
        return True
    except Exception:
        return False


pytestmark_db = pytest.mark.skipif(
    not _db_available(), reason="no database reachable",
)


@pytest.fixture
def fixture_world():
    """
    A person with one frontal gallery encoding, and a cluster of unknown faces:
      3 diverse real views  (left, extreme_left, right)
      1 near-duplicate of the person's existing frontal encoding
      1 legacy member with no pose metadata
      1 outlier — a different person entirely

    Yields a dict of ids; deletes everything afterwards.
    """
    from sqlalchemy import text
    from app.db.session import SessionLocal
    from app.db.models import (
        Camera, FaceEncoding, IdentityReviewCandidate, Person,
        UnknownCluster, UnknownFace,
    )
    from app.vision.recognizer import serialize_encoding

    tag = f"pytest-gs-{uuid4().hex[:8]}"
    db = SessionLocal()
    ids = {"tag": tag, "face_ids": {}}

    try:
        rng = np.random.RandomState(101)
        base = unit(rng.randn(512).astype(np.float32))

        cam = Camera(id=str(uuid4()), name=tag, location=tag,
                     stream_url="0", camera_type="webcam", is_active=False)
        db.add(cam)

        person = Person(id=str(uuid4()), name=f"{tag} Person",
                        role="student", person_type="student",
                        department="CSE", is_active=True)
        db.add(person)
        db.flush()

        # Existing gallery: a single frontal encoding.
        gallery_enc = FaceEncoding(
            id=str(uuid4()), person_id=person.id,
            encoding_data=serialize_encoding(base),
            source_image=f"/snapshots/{tag}_gallery.jpg",
            quality=0.80, yaw=2.0, pitch=0.0, roll=0.0,
            det_score=0.95, blur_score=180.0, lighting="normal",
            pose_bucket="frontal", model_version="arcface_r50", version=1,
        )
        db.add(gallery_enc)

        cluster = UnknownCluster(
            id=str(uuid4()), status="active", member_count=0,
            embedding_centroid=serialize_encoding(base),
        )
        db.add(cluster)
        db.flush()

        # Members far enough from `base` to clear the duplicate filter
        # (SUGGESTION_DUPLICATE_THRESHOLD is a cosine distance) but still
        # clearly the same person.
        diverse = spread_embeddings(base, 3, alpha=0.75, seed=21)
        specs = [
            ("left",          diverse[0], 40.0,  0.78),
            ("extreme_left",  diverse[1], 70.0,  0.72),
            ("right",         diverse[2], -38.0, 0.75),
            # Near-duplicate of the gallery frontal -> must be rejected.
            ("frontal_dup",   unit(base + 0.02 * unit(rng.randn(512).astype(np.float32))),
                              1.0, 0.79),
        ]

        for name, enc, yaw, quality in specs:
            fid = str(uuid4())
            ids["face_ids"][name] = fid
            db.add(UnknownFace(
                id=fid, camera_id=cam.id, cluster_id=cluster.id,
                snapshot_url=f"/snapshots/{tag}_{name}.jpg",
                face_encoding=serialize_encoding(enc),
                occurrence=2, status="pending",
                yaw=yaw, pitch=1.0, roll=0.0,
                pose_bucket=classify_pose_bucket(yaw),
                quality=quality, blur_score=150.0, det_score=0.9,
                lighting="normal",
            ))

        # Legacy row: embedding but NO pose metadata (pre-migration-010 state).
        legacy_id = str(uuid4())
        ids["face_ids"]["legacy"] = legacy_id
        db.add(UnknownFace(
            id=legacy_id, camera_id=cam.id, cluster_id=cluster.id,
            snapshot_url=f"/snapshots/{tag}_legacy.jpg",
            face_encoding=serialize_encoding(spread_embeddings(base, 1, 0.6, seed=99)[0]),
            occurrence=1, status="pending",
        ))

        # Outlier: a different person that clustering swept in.
        outlier_id = str(uuid4())
        ids["face_ids"]["outlier"] = outlier_id
        db.add(UnknownFace(
            id=outlier_id, camera_id=cam.id, cluster_id=cluster.id,
            snapshot_url=f"/snapshots/{tag}_outlier.jpg",
            face_encoding=serialize_encoding(unit(rng.randn(512).astype(np.float32))),
            occurrence=1, status="pending",
            yaw=-70.0, pitch=0.0, roll=0.0, pose_bucket="extreme_right",
            quality=0.70, blur_score=140.0, det_score=0.88, lighting="normal",
        ))

        cluster.member_count = 6

        candidate = IdentityReviewCandidate(
            id=str(uuid4()), cluster_id=cluster.id, person_id=person.id,
            embedding_similarity=0.78, image_quality_score=0.75,
            overall_score=0.72, recommended_action="review",
            status="pending", review_priority="high",
        )
        db.add(candidate)
        db.commit()

        ids.update(
            camera_id=cam.id, person_id=person.id, cluster_id=cluster.id,
            candidate_id=candidate.id, gallery_encoding_id=gallery_enc.id,
        )
        db.close()

        yield ids

    finally:
        db2 = SessionLocal()
        try:
            for stmt in (
                "delete from gallery_updates where cluster_id = :c",
                "delete from identity_review_activity where cluster_id = :c",
                "delete from identity_review_candidates where cluster_id = :c",
                "delete from unknown_faces where cluster_id = :c",
                "delete from unknown_clusters where id = :c",
                "delete from suggested_encodings where person_id = :p",
                "delete from face_encodings where person_id = :p",
                "delete from persons where id = :p",
                "delete from cameras where id = :cam",
            ):
                db2.execute(text(stmt), {
                    "c": ids.get("cluster_id"),
                    "p": ids.get("person_id"),
                    "cam": ids.get("camera_id"),
                })
            db2.commit()
        finally:
            db2.close()


@pytestmark_db
class TestApprovalEndToEnd:

    def test_preview_reports_per_member_verdicts(self, fixture_world):
        from app.services.identity_review_service import IdentityReviewService

        r = IdentityReviewService.preview_candidate(fixture_world["candidate_id"])
        by_id = {m["face_id"]: m for m in r["members"]}
        f = fixture_world["face_ids"]

        assert r["total_members"] == 6

        # The intruder is flagged and deselected by default.
        assert by_id[f["outlier"]]["is_outlier"] is True
        assert by_id[f["outlier"]]["selected"] is False
        assert by_id[f["outlier"]]["status"] == "outlier_excluded"

        # Real diverse views are added as INDIVIDUAL embeddings.
        for name in ("left", "extreme_left", "right"):
            assert by_id[f[name]]["status"] == "accepted", by_id[f[name]]["reason"]

        # The near-duplicate of the existing frontal is rejected, not appended.
        assert by_id[f["frontal_dup"]]["status"] == "redundant"

        # The pre-migration row is skipped rather than assumed frontal.
        assert by_id[f["legacy"]]["status"] == "no_pose_metadata"

        assert r["accepted_count"] == 3
        assert r["applied"] is False

    def test_preview_changes_nothing(self, fixture_world):
        from app.db.session import SessionLocal
        from app.db.models import FaceEncoding, IdentityReviewCandidate
        from app.services.identity_review_service import IdentityReviewService

        db = SessionLocal()
        before = db.query(FaceEncoding).filter(
            FaceEncoding.person_id == fixture_world["person_id"]).count()
        db.close()

        IdentityReviewService.preview_candidate(fixture_world["candidate_id"])

        db = SessionLocal()
        after = db.query(FaceEncoding).filter(
            FaceEncoding.person_id == fixture_world["person_id"]).count()
        status = db.query(IdentityReviewCandidate).filter(
            IdentityReviewCandidate.id == fixture_world["candidate_id"]).first().status
        db.close()

        assert after == before == 1
        assert status == "pending"

    def test_coverage_delta_is_measured(self, fixture_world):
        from app.services.identity_review_service import IdentityReviewService

        r = IdentityReviewService.preview_candidate(fixture_world["candidate_id"])

        assert r["coverage_after"] > r["coverage_before"]
        assert r["coverage_delta"] == pytest.approx(
            round(r["coverage_after"] - r["coverage_before"], 1), abs=0.11)
        # The gain is named, not just numeric.
        assert "pose_left" in r["buckets_gained"]
        assert "pose_extreme_left" in r["buckets_gained"]
        assert "pose_right" in r["buckets_gained"]

    def test_approve_adds_individual_embeddings_not_a_centroid(self, fixture_world):
        from app.db.session import SessionLocal
        from app.db.models import FaceEncoding, GalleryUpdate, UnknownFace
        from app.services.identity_review_service import IdentityReviewService
        from app.vision.recognizer import deserialize_encoding

        result = IdentityReviewService.approve_candidate(
            candidate_id=fixture_world["candidate_id"], actor_id="pytest-admin",
        )
        assert result["status"] == "approved"
        assert result["accepted_count"] == 3

        db = SessionLocal()
        try:
            encs = db.query(FaceEncoding).filter(
                FaceEncoding.person_id == fixture_world["person_id"]).all()
            assert len(encs) == 4  # 1 original + 3 added

            # Each addition carries a REAL pose, and they differ.
            added = [e for e in encs if e.id != fixture_world["gallery_encoding_id"]]
            assert {e.pose_bucket for e in added} == {"left", "extreme_left", "right"}

            # Each stored vector equals one member's own embedding.
            faces = {
                f.id: deserialize_encoding(f.face_encoding)
                for f in db.query(UnknownFace).filter(
                    UnknownFace.cluster_id == fixture_world["cluster_id"]).all()
            }
            updates = db.query(GalleryUpdate).filter(
                GalleryUpdate.cluster_id == fixture_world["cluster_id"]).all()
            assert len(updates) == 3

            for u in updates:
                assert u.source_face_id in faces
                assert u.encoding_id is not None
                assert u.selection_reason
                stored = deserialize_encoding(u.encoding_data)
                member = faces[u.source_face_id]
                cos = float(np.dot(unit(stored), unit(member)))
                assert cos == pytest.approx(1.0, abs=1e-4), (
                    "gallery got an averaged vector, not the member's own"
                )
        finally:
            db.close()

    def test_approve_writes_a_complete_audit_trail(self, fixture_world):
        from app.db.session import SessionLocal
        from app.db.models import (
            EncodingVersion, GalleryUpdate, IdentityReviewActivity,
            IdentityReviewCandidate, UnknownCluster, UnknownFace,
        )
        from app.services.identity_review_service import IdentityReviewService

        db = SessionLocal()
        ver_before = db.query(EncodingVersion).filter(EncodingVersion.id == 1).first()
        ver_before = ver_before.version if ver_before else 0
        db.close()

        IdentityReviewService.approve_candidate(
            candidate_id=fixture_world["candidate_id"], actor_id="pytest-admin",
        )

        db = SessionLocal()
        try:
            act = db.query(IdentityReviewActivity).filter(
                IdentityReviewActivity.cluster_id == fixture_world["cluster_id"],
                IdentityReviewActivity.action == "approved",
            ).first()
            assert act is not None
            d = act.details
            assert act.actor == "pytest-admin"
            assert d["person_id"] == fixture_world["person_id"]
            assert d["gallery_version"] == 1
            assert len(d["accepted"]) == 3
            assert all(a["reason"] for a in d["accepted"])       # WHY it was selected
            assert any(r["status"] == "redundant" for r in d["rejected"])
            assert d["coverage_after"] > d["coverage_before"]

            for u in db.query(GalleryUpdate).filter(
                    GalleryUpdate.cluster_id == fixture_world["cluster_id"]).all():
                assert u.status == "applied"
                assert u.gallery_version == 1
                assert u.applied_at is not None

            cand = db.query(IdentityReviewCandidate).filter(
                IdentityReviewCandidate.id == fixture_world["candidate_id"]).first()
            assert cand.status == "approved"
            assert cand.reviewed_by == "pytest-admin"
            assert cand.reviewed_at is not None

            cluster = db.query(UnknownCluster).filter(
                UnknownCluster.id == fixture_world["cluster_id"]).first()
            assert cluster.status == "reviewed"

            # Selected members resolve; the excluded outlier stays pending.
            outlier = db.query(UnknownFace).filter(
                UnknownFace.id == fixture_world["face_ids"]["outlier"]).first()
            assert outlier.status == "pending"
            left = db.query(UnknownFace).filter(
                UnknownFace.id == fixture_world["face_ids"]["left"]).first()
            assert left.status == "enrolled"

            # Pipeline cache signal for the enlarged gallery.
            ver_after = db.query(EncodingVersion).filter(
                EncodingVersion.id == 1).first().version
            assert ver_after > ver_before
        finally:
            db.close()

    def test_reviewer_can_exclude_a_member(self, fixture_world):
        from app.db.session import SessionLocal
        from app.db.models import FaceEncoding
        from app.services.identity_review_service import IdentityReviewService

        f = fixture_world["face_ids"]
        # Reviewer keeps only two of the three diverse views.
        IdentityReviewService.approve_candidate(
            candidate_id=fixture_world["candidate_id"],
            actor_id="pytest-admin",
            selected_face_ids=[f["left"], f["right"]],
        )

        db = SessionLocal()
        try:
            added = [
                e for e in db.query(FaceEncoding).filter(
                    FaceEncoding.person_id == fixture_world["person_id"]).all()
                if e.id != fixture_world["gallery_encoding_id"]
            ]
            assert {e.pose_bucket for e in added} == {"left", "right"}
            assert "extreme_left" not in {e.pose_bucket for e in added}
        finally:
            db.close()

    def test_reviewer_can_include_a_flagged_outlier(self, fixture_world):
        """
        Exclusion is a default, not a lock. If the human says the flagged face
        really is this person, it goes in — that is the human being in charge.
        """
        from app.db.session import SessionLocal
        from app.db.models import GalleryUpdate
        from app.services.identity_review_service import IdentityReviewService

        f = fixture_world["face_ids"]
        IdentityReviewService.approve_candidate(
            candidate_id=fixture_world["candidate_id"],
            actor_id="pytest-admin",
            selected_face_ids=[f["left"], f["outlier"]],
        )

        db = SessionLocal()
        try:
            sources = {
                u.source_face_id for u in db.query(GalleryUpdate).filter(
                    GalleryUpdate.cluster_id == fixture_world["cluster_id"]).all()
            }
            assert f["outlier"] in sources
        finally:
            db.close()

    def test_flagged_outlier_is_still_gated(self, fixture_world):
        """Including a flagged face overrides the outlier default, not the gates."""
        from app.services.identity_review_service import IdentityReviewService

        f = fixture_world["face_ids"]
        r = IdentityReviewService.preview_candidate(
            fixture_world["candidate_id"],
            selected_face_ids=[f["outlier"]],
        )
        m = next(m for m in r["members"] if m["face_id"] == f["outlier"])
        assert m["is_outlier"] is True          # still reported as suspicious
        assert m["selected"] is True
        assert m["status"] != "outlier_excluded"

    def test_rejection_never_touches_the_gallery(self, fixture_world):
        from app.db.session import SessionLocal
        from app.db.models import (
            FaceEncoding, GalleryUpdate, IdentityReviewCandidate, UnknownFace,
        )
        from app.services.identity_review_service import IdentityReviewService

        IdentityReviewService.reject_candidate(
            candidate_id=fixture_world["candidate_id"], actor_id="pytest-admin",
        )

        db = SessionLocal()
        try:
            assert db.query(FaceEncoding).filter(
                FaceEncoding.person_id == fixture_world["person_id"]).count() == 1
            assert db.query(GalleryUpdate).filter(
                GalleryUpdate.cluster_id == fixture_world["cluster_id"]).count() == 0
            assert db.query(IdentityReviewCandidate).filter(
                IdentityReviewCandidate.id == fixture_world["candidate_id"]
            ).first().status == "rejected"
            # Members stay pending — a rejected match resolves nothing.
            assert db.query(UnknownFace).filter(
                UnknownFace.cluster_id == fixture_world["cluster_id"],
                UnknownFace.status == "pending",
            ).count() == 6
        finally:
            db.close()

    def test_double_approval_is_refused(self, fixture_world):
        from app.services.identity_review_service import IdentityReviewService

        IdentityReviewService.approve_candidate(
            candidate_id=fixture_world["candidate_id"], actor_id="pytest-admin",
        )
        with pytest.raises(ValueError, match="already approved"):
            IdentityReviewService.approve_candidate(
                candidate_id=fixture_world["candidate_id"], actor_id="pytest-admin",
            )

    def test_rollback_removes_the_added_encoding(self, fixture_world):
        from app.db.session import SessionLocal
        from app.db.models import FaceEncoding, GalleryUpdate, UnknownFace
        from app.services.identity_review_service import IdentityReviewService

        IdentityReviewService.approve_candidate(
            candidate_id=fixture_world["candidate_id"], actor_id="pytest-admin",
        )

        db = SessionLocal()
        updates = db.query(GalleryUpdate).filter(
            GalleryUpdate.cluster_id == fixture_world["cluster_id"]).all()
        target = updates[0]
        target_id, enc_id, src_id = target.id, target.encoding_id, target.source_face_id
        db.close()

        IdentityReviewService.rollback_gallery_update(target_id)

        db = SessionLocal()
        try:
            assert db.query(FaceEncoding).filter(FaceEncoding.id == enc_id).first() is None
            assert db.query(FaceEncoding).filter(
                FaceEncoding.person_id == fixture_world["person_id"]).count() == 3
            assert db.query(GalleryUpdate).filter(
                GalleryUpdate.id == target_id).first().status == "rolled_back"
            assert db.query(UnknownFace).filter(
                UnknownFace.id == src_id).first().status == "pending"
        finally:
            db.close()

    def test_capacity_cap_is_enforced(self, fixture_world, monkeypatch):
        from app.db.session import SessionLocal
        from app.db.models import FaceEncoding
        from app.services import gallery_selection
        from app.services.identity_review_service import IdentityReviewService

        # Cap at 2: the existing frontal plus exactly one addition.
        monkeypatch.setattr(gallery_selection.settings, "MAX_ENCODINGS_PER_PERSON", 2)

        r = IdentityReviewService.approve_candidate(
            candidate_id=fixture_world["candidate_id"], actor_id="pytest-admin",
        )
        assert r["accepted_count"] == 1
        assert any(m["status"] == "gallery_full" for m in r["members"])

        # The survivor is a rare pose, not another frontal — the cap protects
        # the additions that actually improve coverage.
        db = SessionLocal()
        try:
            added = [
                e for e in db.query(FaceEncoding).filter(
                    FaceEncoding.person_id == fixture_world["person_id"]).all()
                if e.id != fixture_world["gallery_encoding_id"]
            ]
            assert len(added) == 1
            assert added[0].pose_bucket == "extreme_left"
        finally:
            db.close()

    def test_accepted_members_gate_later_members(self, fixture_world):
        """
        The sequential guarantee: member N is judged against a gallery that
        already contains members 1..N-1.

        Two near-identical crops of the same new pose must not both be added.
        Evaluated independently they would both look like "a pose the gallery
        lacks"; that is how you end up with 500 copies of one angle.
        """
        from sqlalchemy import text
        from app.db.session import SessionLocal
        from app.services.identity_review_service import IdentityReviewService

        # Make the extreme_left member a near-twin of the left member, and
        # relabel it so both claim the same (uncovered) bucket.
        db = SessionLocal()
        row = db.execute(
            text("select face_encoding from unknown_faces where id = :i"),
            {"i": fixture_world["face_ids"]["left"]},
        ).first()
        db.execute(
            text("""update unknown_faces
                       set face_encoding = :e, pose_bucket = 'left', yaw = 38.0
                     where id = :i"""),
            {"e": row[0], "i": fixture_world["face_ids"]["extreme_left"]},
        )
        db.commit()
        db.close()

        r = IdentityReviewService.preview_candidate(fixture_world["candidate_id"])
        twins = [
            m for m in r["members"]
            if m["face_id"] in (fixture_world["face_ids"]["left"],
                                fixture_world["face_ids"]["extreme_left"])
        ]
        statuses = sorted(m["status"] for m in twins)
        assert statuses == ["accepted", "redundant"], statuses

        loser = next(m for m in twins if m["status"] == "redundant")
        assert "similar" in loser["reason"].lower()

    def test_low_quality_member_is_rejected(self, fixture_world):
        from sqlalchemy import text
        from app.db.session import SessionLocal
        from app.services.identity_review_service import IdentityReviewService

        db = SessionLocal()
        db.execute(
            text("update unknown_faces set blur_score = 5.0 where id = :i"),
            {"i": fixture_world["face_ids"]["extreme_left"]},
        )
        db.commit()
        db.close()

        r = IdentityReviewService.preview_candidate(fixture_world["candidate_id"])
        m = next(m for m in r["members"]
                 if m["face_id"] == fixture_world["face_ids"]["extreme_left"])
        assert m["status"] == "low_quality"
        assert "blur" in m["reason"].lower()
