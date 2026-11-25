"""
The Unknown Faces queue: that new faces reach it at all, that it can be read a
page at a time in a chosen order, and that old faces are removed for good.

Three separate things went wrong or were missing here:

  * From the day cameras moved into processes (2026-09-10) not one new unknown
    face was saved. The report from the camera process carried no label, the
    parent took it for an ordinary event, and it failed with
    "Failed to persist event: 'event_type'". The page looked broken; the page
    was fine, nothing was arriving.
  * The page loaded fifty faces and called that the total.
  * Nothing ever left the queue except by hand.

SQLite in memory; no cameras, no models.
"""

import os
import queue
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Camera, UnknownFace, UnknownFaceView
from app.services import unknown_service
from app.services.unknown_service import UnknownService
from app.vision.camera_process import as_unknown_face_event
from app.vision.process_manager import ProcessCameraManager

NOW = datetime.now(timezone.utc)


@pytest.fixture
def db(monkeypatch, tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine, tables=[Camera.__table__, UnknownFace.__table__, UnknownFaceView.__table__])
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(unknown_service, "SessionLocal", Session)
    monkeypatch.setattr(unknown_service.settings, "SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    with Session() as session:
        session.add(Camera(id="cam-a", name="coe", location="COE", stream_url="x", camera_type="rtsp"))
        session.add(Camera(id="cam-b", name="gate", location="Gate", stream_url="y", camera_type="rtsp"))
        session.commit()
    return Session


def add(Session, face_id, *, days_ago=0.0, seen=1, status="pending", camera="cam-a", first_days_ago=None):
    last = NOW - timedelta(days=days_ago)
    first = NOW - timedelta(days=first_days_ago if first_days_ago is not None else days_ago)
    with Session() as session:
        session.add(UnknownFace(
            id=face_id, camera_id=camera, status=status, occurrence=seen,
            first_seen=first, last_seen=last, created_at=first,
            snapshot_url=f"/snapshots/unknowns/unknown_{face_id}.jpg",
            full_frame=f"/snapshots/frames/fullframe_{face_id}.jpg",
        ))
        session.commit()


def ids(page):
    return [face.id for face, _camera in page["rows"]]


# ── New faces reach the queue ────────────────────────────────────────────────

def pump(events):
    """Run the parent's event pump over `events`; returns (to on_event, to on_unknown)."""
    manager = ProcessCameraManager.__new__(ProcessCameraManager)
    manager._running = threading.Event()
    manager._running.set()
    manager._event_q = queue.Queue()
    manager._beats = {}
    seen_event, seen_unknown = [], []
    manager._on_event, manager._on_unknown = seen_event.append, seen_unknown.append
    for event in events:
        manager._event_q.put(event)
    manager._event_q.put(None)
    manager._pump_events()
    return seen_event, seen_unknown


def test_an_unknown_face_from_a_camera_process_reaches_the_unknown_handler():
    report = {"unknown_id": "u1", "camera_id": "cam-a", "encoding": [0.1], "timestamp": "2026-11-20T07:00:00+00:00"}
    to_event, to_unknown = pump([as_unknown_face_event(report)])
    assert to_event == []
    assert to_unknown[0]["unknown_id"] == "u1"
    assert report.get("event_type") is None            # the pipeline's dict is not altered


def test_repeat_sightings_and_learned_views_are_routed_the_same_way():
    sighting = as_unknown_face_event({"unknown_id": "u1", "camera_id": "cam-a", "sighting_only": True})
    views = as_unknown_face_event({"views_only": True, "views": [("u1", [0.1])], "camera_id": "cam-a"})
    to_event, to_unknown = pump([sighting, views])
    assert to_event == [] and len(to_unknown) == 2


def test_an_unlabelled_report_is_still_recognised_by_its_shape():
    """The label was forgotten once and cost two months of strangers. The
    parent no longer depends on it alone."""
    to_event, to_unknown = pump([
        {"unknown_id": "u1", "camera_id": "cam-a"},
        {"views_only": True, "views": [], "camera_id": "cam-a"},
    ])
    assert to_event == [] and len(to_unknown) == 2


def test_ordinary_events_are_not_mistaken_for_unknown_faces():
    entry = {"event_type": "entry", "person_id": "p1", "camera_id": "cam-a"}
    # A detection event may mention a stranger's id; it is still an event.
    stranger = {"event_type": "detection", "unknown_id": "u1", "camera_id": "cam-a"}
    to_event, to_unknown = pump([entry, stranger])
    assert to_event == [entry, stranger] and to_unknown == []


# ── Sorting and paging ───────────────────────────────────────────────────────

def test_the_newest_seen_face_is_first_by_default(db):
    add(db, "old", days_ago=5)
    add(db, "new", days_ago=0.1)
    add(db, "mid", days_ago=2)
    assert ids(UnknownService.list_page()) == ["new", "mid", "old"]


@pytest.mark.parametrize("sort, order, expected", [
    ("last_seen", "asc", ["c", "b", "a"]),
    ("occurrence", "desc", ["b", "c", "a"]),
    ("occurrence", "asc", ["a", "c", "b"]),
    ("first_seen", "desc", ["b", "a", "c"]),
])
def test_the_queue_can_be_sorted(db, sort, order, expected):
    add(db, "a", days_ago=1, seen=1, first_days_ago=3)
    add(db, "b", days_ago=2, seen=9, first_days_ago=2)
    add(db, "c", days_ago=3, seen=4, first_days_ago=8)
    assert ids(UnknownService.list_page(sort=sort, order=order)) == expected


def test_an_unknown_sort_falls_back_instead_of_reaching_the_database(db):
    add(db, "a", days_ago=1)
    add(db, "b", days_ago=0.5)
    assert ids(UnknownService.list_page(sort="id; drop table unknown_faces")) == ["b", "a"]


def test_pages_cover_every_face_exactly_once(db):
    for i in range(53):
        # Many identical timestamps: without a tiebreak a face lands on two pages or none.
        add(db, f"f{i:02d}", days_ago=1 if i % 2 else 2)
    seen = []
    for page in range(6):
        seen += ids(UnknownService.list_page(limit=10, offset=page * 10))
    assert len(seen) == 53 and len(set(seen)) == 53


def test_totals_describe_the_whole_queue_not_the_page(db):
    for i in range(30):
        add(db, f"f{i:02d}", days_ago=i * 0.1, seen=2)
    add(db, "done", status="enrolled", seen=50)
    page = UnknownService.list_page(limit=10)
    assert len(page["rows"]) == 10
    assert page["total"] == 30
    assert page["total_sightings"] == 60


def test_each_face_comes_with_its_cameras_name(db):
    add(db, "a", camera="cam-b")
    (face, camera_name), = UnknownService.list_page()["rows"]
    assert (face.id, camera_name) == ("a", "gate")


def test_the_queue_can_be_narrowed_to_one_camera_or_widened_to_every_status(db):
    add(db, "a", camera="cam-a")
    add(db, "b", camera="cam-b")
    add(db, "c", status="dismissed")
    assert ids(UnknownService.list_page(camera_id="cam-b")) == ["b"]
    assert len(UnknownService.list_page(status="all")["rows"]) == 3


# ── Removal after a fixed age ────────────────────────────────────────────────

def make_files(tmp_snapshots, face_id):
    paths = [
        os.path.join(tmp_snapshots, "unknowns", f"unknown_{face_id}.jpg"),
        os.path.join(tmp_snapshots, "unknowns", f"context_{face_id}.jpg"),
        os.path.join(tmp_snapshots, "frames", f"fullframe_{face_id}.jpg"),
    ]
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(b"jpeg")
    return paths


def test_faces_not_seen_for_fifteen_days_are_removed_with_their_images(db):
    snapshots = unknown_service.settings.SNAPSHOT_DIR
    add(db, "old", days_ago=16)
    add(db, "recent", days_ago=14)
    old_files, recent_files = make_files(snapshots, "old"), make_files(snapshots, "recent")
    with db() as session:
        session.add(UnknownFaceView(id="v1", unknown_face_id="old", encoding=b"x"))
        session.commit()

    assert UnknownService.count_expired(15) == 1
    assert UnknownService.purge_expired(15) == {"faces": 1, "files": 3}

    assert ids(UnknownService.list_page(status="all")) == ["recent"]
    assert not any(os.path.exists(p) for p in old_files)
    assert all(os.path.exists(p) for p in recent_files)
    with db() as session:
        assert session.query(UnknownFaceView).count() == 0


def test_age_is_counted_from_when_a_face_was_last_seen(db):
    """A regular first seen months ago and again yesterday is not old."""
    add(db, "regular", days_ago=1, first_days_ago=200, seen=40)
    assert UnknownService.purge_expired(15)["faces"] == 0


def test_enrolled_faces_are_never_removed(db):
    """Their snapshot is the person's avatar and the source of their first encoding."""
    snapshots = unknown_service.settings.SNAPSHOT_DIR
    add(db, "person", days_ago=300, status="enrolled")
    files = make_files(snapshots, "person")
    add(db, "dismissed", days_ago=300, status="dismissed")
    assert UnknownService.purge_expired(15)["faces"] == 1
    assert ids(UnknownService.list_page(status="all")) == ["person"]
    assert all(os.path.exists(p) for p in files)


def test_removal_is_off_at_zero_days(db):
    add(db, "old", days_ago=400)
    assert UnknownService.count_expired(0) == 0
    assert UnknownService.purge_expired(0) == {"faces": 0, "files": 0}
    assert len(UnknownService.list_page()["rows"]) == 1


def test_many_expired_faces_are_all_removed(db):
    for i in range(45):
        add(db, f"f{i:02d}", days_ago=20)
    assert UnknownService.purge_expired(15, batch=10)["faces"] == 45
    assert UnknownService.list_page(status="all")["total"] == 0


def test_a_stored_path_cannot_delete_anything_outside_the_snapshot_folder(db, tmp_path):
    outside = tmp_path / "important.txt"
    outside.write_text("keep me")
    with db() as session:
        session.add(UnknownFace(
            id="evil", camera_id="cam-a", status="pending", occurrence=1,
            first_seen=NOW - timedelta(days=30), last_seen=NOW - timedelta(days=30),
            snapshot_url="/snapshots/../important.txt", full_frame=str(outside),
        ))
        session.commit()
    assert UnknownService.purge_expired(15)["faces"] == 1
    assert outside.read_text() == "keep me"


# ── A tracked stranger actually gets saved ───────────────────────────────────
#
# Routing was not the only break. The pipeline created a row only on the call
# that MINTED the identity, and only if the track was already confirmed as a
# stranger -- but minting happens on the first unmatched frame and
# confirmation needs several, a second apart. So the one call allowed to save
# always arrived unconfirmed, and every later call reported a "repeat
# sighting" of an id that had no row.

import numpy as np

from app.vision.camera_process import CONTROL, _drain_results
from app.vision.pipeline import VisionPipeline
from app.vision.unknown_registry import UnknownRegistry


class _Stub:
    """Just enough of a pipeline for _handle_unknown_face."""

    camera_id = "cam-a"

    def __init__(self, cooldown=30.0):
        self.unknown_registry = UnknownRegistry(
            match_threshold=0.62, max_entries=100, sighting_cooldown=cooldown, max_views=5,
        )
        self.reports = []
        self.on_unknown = self.reports.append
        self.tracker = type("T", (), {"on_detection": staticmethod(lambda **kw: None)})()
        self.detector = type("D", (), {
            "extract_face_crop": staticmethod(lambda frame, location, padding=0: frame[:4, :4]),
        })()

    def _extract_context_crop(self, frame, location):
        return frame[:8, :8]

    def _pose_metadata(self, frame, location, landmarks):
        return {}

    def see(self, face, confirmed):
        frame = np.zeros((32, 32, 3), np.uint8)
        return VisionPipeline._handle_unknown_face(
            self, frame, frame, (2, 12, 12, 2), face, centroid=(7, 7), track_id=1, confirmed=confirmed,
        )

    @property
    def saves(self):
        return [r for r in self.reports if "face_crop" in r]


def a_face(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_a_stranger_is_saved_once_confirmed_even_though_first_seen_unconfirmed():
    pipe, face = _Stub(), a_face()
    uid = pipe.see(face, confirmed=False)             # first unmatched frame: identity minted
    assert pipe.reports == []                         # ... but not yet believed
    pipe.see(face, confirmed=False)
    assert pipe.reports == []
    assert pipe.see(face, confirmed=True) == uid      # the vote has gone through
    assert len(pipe.saves) == 1
    assert pipe.saves[0]["unknown_id"] == uid and pipe.saves[0]["encoding"] is face


def test_the_save_is_not_repeated_on_every_frame_while_waiting_for_the_answer():
    pipe, face = _Stub(), a_face()
    pipe.see(face, confirmed=False)
    for _ in range(50):
        pipe.see(face, confirmed=True)
    assert len(pipe.saves) == 1


def test_once_the_row_exists_later_frames_are_sightings_not_saves():
    pipe, face = _Stub(cooldown=0.0), a_face()
    uid = pipe.see(face, confirmed=False)
    pipe.see(face, confirmed=True)
    pipe.unknown_registry.mark_persisted(uid, uid)    # the consumer's answer
    pipe.reports.clear()
    pipe.see(face, confirmed=True)
    assert pipe.saves == []
    assert pipe.reports[-1] == {**pipe.reports[-1], "unknown_id": uid, "sighting_only": True}


def test_an_unconfirmed_stranger_is_never_saved():
    pipe, face = _Stub(cooldown=0.0), a_face()
    for _ in range(20):
        pipe.see(face, confirmed=False)
    assert pipe.reports == []


def test_the_parent_can_tell_a_camera_process_its_stranger_was_saved():
    inbox, cameras, told = queue.Queue(), {"cam-a": queue.Queue()}, []
    inbox.put(("cam-a", 7, "an inference result"))
    inbox.put((CONTROL, "cam-a", {"unknown_persisted": ["local", "canonical"]}))
    inbox.put(None)
    _drain_results(inbox, cameras, threading.Event(), on_control=lambda cam, payload: told.append((cam, payload)))
    assert told == [("cam-a", {"unknown_persisted": ["local", "canonical"]})]
    assert cameras["cam-a"].get_nowait() == ("cam-a", 7, "an inference result")   # results still flow
    assert cameras["cam-a"].empty()                                                # and control never reaches a camera


def test_a_control_message_goes_to_the_process_that_holds_the_camera():
    manager = ProcessCameraManager.__new__(ProcessCameraManager)
    manager._proc_specs = {0: [{"camera_id": "cam-a"}], 1: [{"camera_id": "cam-b"}]}
    manager._result_queues = {0: queue.Queue(), 1: queue.Queue()}
    assert manager.send_control("cam-b", {"unknown_persisted": ["x", "y"]})
    assert manager._result_queues[0].empty()
    assert manager._result_queues[1].get_nowait() == (CONTROL, "cam-b", {"unknown_persisted": ["x", "y"]})
    assert manager.send_control("not-running", {}) is False
