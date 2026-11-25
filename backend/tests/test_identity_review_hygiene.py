"""
AI Identity Review: clusters that follow their faces, and a verifier that
admits when it is not there.

Both were found broken on 2026-11-20:

  * The first retention run removed twelve thousand old unknown faces and left
    61 "active" clusters with no faces in them, plus three review candidates
    whose only evidence was an image that had just been deleted. The page
    showed broken pictures for matches nobody could check.
  * Ollama was installed but not running. Every cycle the worker sent each
    ambiguous match to it, every call was refused, and the page gave no sign
    that the "AI" had not answered in weeks.

SQLite in memory; no models, no Ollama.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import (
    Base, Camera, IdentityReviewCandidate, Person, UnknownCluster, UnknownFace, UnknownFaceView,
)
from app.services import clustering_service, unknown_service, vlm_service
from app.services.clustering_service import ClusteringService
from app.services.unknown_service import UnknownService
from app.services.vlm_service import get_vlm_status

NOW = datetime.now(timezone.utc)


@pytest.fixture
def db(monkeypatch, tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[
        Camera.__table__, Person.__table__, UnknownFace.__table__, UnknownFaceView.__table__,
        UnknownCluster.__table__, IdentityReviewCandidate.__table__,
    ])
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(clustering_service, "SessionLocal", Session)
    monkeypatch.setattr(unknown_service, "SessionLocal", Session)
    monkeypatch.setattr(unknown_service.settings, "SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    with Session() as session:
        session.add(Camera(id="cam", name="coe", location="COE", stream_url="x", camera_type="rtsp"))
        session.add(Person(id="p1", name="Ravi", role="employee"))
        session.commit()
    return Session


def cluster(Session, cluster_id, faces, status="active", shows=None, candidates=("pending",)):
    """A cluster with `faces` = [(face id, days since seen, times seen, status)]."""
    with Session() as session:
        for face_id, days_ago, seen, face_status in faces:
            session.add(UnknownFace(
                id=face_id, camera_id="cam", status=face_status, occurrence=seen, cluster_id=cluster_id,
                first_seen=NOW - timedelta(days=days_ago), last_seen=NOW - timedelta(days=days_ago),
                snapshot_url=f"/snapshots/unknowns/unknown_{face_id}.jpg",
            ))
        shown = shows or (faces[0][0] if faces else "gone")
        session.add(UnknownCluster(
            id=cluster_id, status=status, member_count=99,
            representative_face_id=shown,
            representative_snapshot=f"/snapshots/unknowns/unknown_{shown}.jpg",
        ))
        for i, candidate_status in enumerate(candidates):
            session.add(IdentityReviewCandidate(
                id=f"{cluster_id}-c{i}", cluster_id=cluster_id, person_id="p1", status=candidate_status,
            ))
        session.commit()


def state(Session):
    with Session() as session:
        return (
            {c.id: (c.member_count, c.representative_face_id) for c in session.query(UnknownCluster)},
            sorted(c.id for c in session.query(IdentityReviewCandidate)),
        )


# ── Clusters follow their faces ──────────────────────────────────────────────

def test_a_cluster_with_no_faces_left_is_removed_with_its_candidates(db):
    cluster(db, "hollow", faces=[], candidates=("pending", "rejected"))
    cluster(db, "alive", faces=[("a", 1, 3, "pending"), ("b", 1, 1, "pending")])
    stats = ClusteringService.reconcile()
    clusters, candidates = state(db)
    assert list(clusters) == ["alive"]
    assert candidates == ["alive-c0"]
    assert stats["clusters_removed"] == 1 and stats["candidates_removed"] == 2


def test_faces_that_were_enrolled_or_dismissed_no_longer_hold_a_cluster_open(db):
    """Nothing left to review is nothing to show, whatever the rows say."""
    cluster(db, "done", faces=[("a", 1, 3, "enrolled"), ("b", 1, 1, "dismissed")])
    ClusteringService.reconcile()
    assert state(db) == ({}, [])


def test_a_cluster_that_lost_some_faces_is_corrected_not_removed(db):
    cluster(db, "partial", faces=[("a", 1, 2, "pending"), ("b", 1, 9, "pending")], shows="deleted-face")
    stats = ClusteringService.reconcile()
    clusters, candidates = state(db)
    assert clusters == {"partial": (2, "b")}          # recounted; now shows its most-seen face
    assert candidates == ["partial-c0"]
    assert stats["clusters_repaired"] == 1
    with db() as session:
        assert session.get(UnknownCluster, "partial").representative_snapshot.endswith("unknown_b.jpg")


def test_reviewed_clusters_are_history_and_are_left_alone(db):
    cluster(db, "history", faces=[], status="reviewed", candidates=("approved",))
    ClusteringService.reconcile()
    clusters, candidates = state(db)
    assert list(clusters) == ["history"] and candidates == ["history-c0"]


def test_a_healthy_cluster_is_not_touched(db):
    cluster(db, "fine", faces=[("a", 1, 5, "pending"), ("b", 1, 1, "pending")])
    with db() as session:
        session.get(UnknownCluster, "fine").member_count = 2
        session.commit()
    assert ClusteringService.reconcile() == {"clusters_removed": 0, "candidates_removed": 0, "clusters_repaired": 0}


def test_removing_old_faces_takes_their_clusters_with_them(db):
    """The exact sequence that broke the page: retention ran, clusters stayed."""
    cluster(db, "old", faces=[("a", 40, 1, "pending"), ("b", 30, 1, "pending")])
    cluster(db, "mixed", faces=[("c", 40, 1, "pending"), ("d", 1, 4, "pending")], shows="c")
    assert UnknownService.purge_expired(15)["faces"] == 3
    clusters, candidates = state(db)
    assert clusters == {"mixed": (1, "d")}
    assert candidates == ["mixed-c0"]


# ── The verifier says whether it is there ────────────────────────────────────

@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.setitem(vlm_service._status_cache, "value", None)
    monkeypatch.setattr(vlm_service.settings, "IDENTITY_REVIEW_VLM_PROVIDER", "ollama")
    monkeypatch.setattr(vlm_service.settings, "OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setattr(vlm_service.settings, "OLLAMA_MODEL", "qwen2.5vl:7b")

    def answer(models=None, error=None):
        import httpx

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"models": [{"name": m} for m in models]}

        def fake_get(url, timeout=None):
            if error:
                raise error
            assert url == "http://localhost:11434/api/tags" and timeout <= 3
            return Response()

        monkeypatch.setattr(httpx, "get", fake_get)
        monkeypatch.setitem(vlm_service._status_cache, "value", None)

    return answer


def test_a_running_ollama_with_the_model_is_available(verifier):
    verifier(models=["qwen2.5vl:7b", "something-else:latest"])
    status = get_vlm_status()
    assert status["available"] is True and status["model"] == "qwen2.5vl:7b"


def test_ollama_not_running_is_reported_with_what_to_do(verifier):
    verifier(error=ConnectionRefusedError("actively refused"))
    status = get_vlm_status()
    assert status["available"] is False
    assert "not running" in status["detail"] and "ollama serve" in status["detail"]


def test_ollama_running_without_the_model_says_which_model_to_get(verifier):
    verifier(models=["llama3:latest"])
    status = get_vlm_status()
    assert status["available"] is False
    assert "ollama pull qwen2.5vl:7b" in status["detail"]


def test_the_check_is_not_repeated_on_every_request(verifier, monkeypatch):
    verifier(error=ConnectionRefusedError())
    assert get_vlm_status()["available"] is False
    verifier(models=["qwen2.5vl:7b"])                  # clears the cache: Ollama has come up
    assert get_vlm_status()["available"] is True
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed again")))
    assert get_vlm_status()["available"] is True       # served from the cache, no second probe


def test_a_switched_off_verifier_says_so(verifier, monkeypatch):
    monkeypatch.setattr(vlm_service.settings, "IDENTITY_REVIEW_VLM_PROVIDER", "disabled")
    status = get_vlm_status()
    assert (status["provider"], status["available"]) == ("disabled", False)
    assert "switched off" in status["detail"]
