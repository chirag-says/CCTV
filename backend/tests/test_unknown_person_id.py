"""
Step 3 -- an unenrolled person's history becomes queryable.

The data was already being written; it just went into a JSON blob nobody could
index. These tests pin the two things that must hold for the column to be
trustworthy, plus the boot-time guard that stops a missed migration from
silently destroying the event log.
"""

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine

from app.db.models import DetectionEvent


class TestModel:
    def test_column_exists_and_is_nullable(self):
        col = DetectionEvent.__table__.columns["unknown_person_id"]
        assert col.nullable, "known people have no unknown id, and vice versa"
        assert col.type.length == 36, "holds an unknown_faces.id"

    def test_column_is_not_a_foreign_key(self):
        """
        It points at unknown_faces.id, but that row can be enrolled or
        dismissed. Neither should cascade into or block the permanent event
        log -- the id stays a meaningful historical record either way.
        """
        col = DetectionEvent.__table__.columns["unknown_person_id"]
        assert not col.foreign_keys

    def test_journey_index_exists(self):
        """'Where has this stranger been?' must not be a sequential scan."""
        indexes = {
            idx.name: [c.name for c in idx.columns]
            for idx in DetectionEvent.__table__.indexes
        }
        assert "ix_events_unknown_created" in indexes
        assert indexes["ix_events_unknown_created"] == [
            "unknown_person_id",
            "created_at",
        ]

    def test_camera_time_index_exists(self):
        """'Who was on this camera between T1 and T2?' -- the incident query."""
        indexes = {
            idx.name: [c.name for c in idx.columns]
            for idx in DetectionEvent.__table__.indexes
        }
        assert "ix_events_camera_created" in indexes


class TestWritePath:
    """
    main.py's FK guard is the only thing that routes a stranger's id. These
    mirror its branches without standing up a database.
    """

    @staticmethod
    def _route(person_id, is_known):
        """The shape of the guard in _process_pipeline_event."""
        metadata = {}
        unknown_person_id = None
        if person_id and not is_known:
            if str(person_id).startswith("anon:"):
                metadata["anonymous"] = True
            else:
                unknown_person_id = person_id
                metadata["unknown_person_id"] = person_id
            person_id = None
        return person_id, unknown_person_id, metadata

    def test_known_person_keeps_the_fk_and_gets_no_unknown_id(self):
        pid, uid, meta = self._route("p-1", is_known=True)
        assert pid == "p-1"
        assert uid is None
        assert meta == {}

    def test_stranger_moves_to_the_column(self):
        pid, uid, meta = self._route("u-abc", is_known=False)
        assert pid is None, "no persons row exists, so the FK must stay empty"
        assert uid == "u-abc"

    def test_stranger_is_also_still_written_to_metadata(self):
        """
        Deliberate duplication. Existing readers -- campus_summary's
        _person_key and the analytics export -- read the JSON, and every row
        written before migration 011 only has it there.
        """
        _pid, uid, meta = self._route("u-abc", is_known=False)
        assert meta["unknown_person_id"] == uid

    def test_the_two_id_fields_are_mutually_exclusive(self):
        for person_id, is_known in (("p-1", True), ("u-abc", False)):
            pid, uid, _meta = self._route(person_id, is_known)
            assert not (pid and uid), "a subject is enrolled or it is not"


class TestSchemaVerification:
    """
    The guard against a hand-applied migration being missed. Without it, a
    model column the database lacks turns every INSERT into an error --
    for detection_events, silent and total loss of the event log.
    """

    def _engine_missing_a_column(self):
        engine = create_engine("sqlite://")
        md = MetaData()
        Table(
            "detection_events",
            md,
            Column("id", String(36), primary_key=True),
            # unknown_person_id deliberately absent: migration 011 not applied
        )
        md.create_all(engine)
        return engine

    def test_detects_a_missing_column(self, monkeypatch):
        from app.db import session as sess

        engine = self._engine_missing_a_column()
        monkeypatch.setattr(sess, "engine", engine)

        missing = sess.verify_schema()
        assert "detection_events" in missing
        assert "unknown_person_id" in missing["detection_events"]

    def test_reports_nothing_when_the_schema_is_current(self, monkeypatch):
        from app.db import session as sess

        # Only detection_events is built here -- the full metadata cannot be
        # created on SQLite (IncidentReview uses a Postgres sequence), and
        # tables that are simply absent are not drift: create_all() makes them.
        engine = create_engine("sqlite://")
        md = MetaData()
        Table(
            "detection_events",
            md,
            *[
                Column(c.name, String(64), primary_key=c.primary_key)
                for c in DetectionEvent.__table__.columns
            ],
        )
        md.create_all(engine)
        monkeypatch.setattr(sess, "engine", engine)

        assert sess.verify_schema() == {}

    def test_never_raises_on_an_unreachable_database(self, monkeypatch):
        """
        Schema drift in one table must not stop the cameras. It should be
        impossible to miss in the logs, and that is all.
        """
        from app.db import session as sess

        class _Broken:
            def __getattr__(self, name):
                raise RuntimeError("database is down")

        monkeypatch.setattr(sess, "engine", _Broken())
        assert sess.verify_schema() == {}

    def test_ignores_tables_that_do_not_exist_yet(self, monkeypatch):
        """create_all() will make those; they are not drift."""
        from app.db import session as sess

        engine = create_engine("sqlite://")  # completely empty
        monkeypatch.setattr(sess, "engine", engine)
        assert sess.verify_schema() == {}
