"""
Unknown Face Service — database access for unenrolled identities.

Exists so the vision layer can refresh its UnknownRegistry without embedding
SQLAlchemy queries in the pipeline, matching how PersonService.get_all_encodings()
feeds the face gallery.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.db.models import UnknownFace, UnknownFaceView
from app.vision.recognizer import deserialize_encoding

logger = logging.getLogger(__name__)


class UnknownService:
    """Read/write helpers for the unknown_faces table."""

    @staticmethod
    def add_views(unknown_id: str, views: List[Any]) -> int:
        """
        Persist additional face views for an unknown identity (migration 012).

        These are what make an identity re-identifiable on a DIFFERENT camera
        or after a restart. Before this existed the registry learned views in
        memory and dropped them, so cross-camera re-identification measured
        0.0%.

        Bounded by UNKNOWN_REGISTRY_MAX_VIEWS: past that the identity already
        covers enough poses, and every extra row costs a line in the search
        matrix of every camera, forever.

        Silently skips an identity whose row is gone (enrolled or dismissed
        between the sighting and this write) -- the FK would reject it, and it
        is not an error.
        """
        if not views:
            return 0

        from app.vision.recognizer import serialize_encoding

        max_views = getattr(settings, "UNKNOWN_REGISTRY_MAX_VIEWS", 5)
        db: Session = SessionLocal()
        try:
            exists = (
                db.query(UnknownFace.id)
                .filter(UnknownFace.id == unknown_id)
                .first()
            )
            if not exists:
                return 0

            have = (
                db.query(func.count(UnknownFaceView.id))
                .filter(UnknownFaceView.unknown_face_id == unknown_id)
                .scalar()
            ) or 0
            room = max_views - have
            if room <= 0:
                return 0

            written = 0
            for view in views[:room]:
                blob = serialize_encoding(view)
                if blob is None:
                    continue
                db.add(UnknownFaceView(
                    id=str(uuid4()),
                    unknown_face_id=unknown_id,
                    encoding=blob,
                ))
                written += 1

            if written:
                db.commit()
            return written
        except Exception as e:
            db.rollback()
            logger.warning(f"Failed to store unknown views for {unknown_id}: {e}")
            return 0
        finally:
            db.close()

    @staticmethod
    def get_pending_encodings(limit: int = None) -> List[Dict[str, Any]]:
        """
        Load pending unknown identities together with ALL of their face views.

        Ordered by last_seen DESC and capped, so a long-lived deployment loads
        the recently active people rather than everyone who ever walked past.

        THE CAP IS LOAD-BEARING, AND WAS TOO SMALL. It defaulted to 500 against
        9,624 pending identities -- each camera could see 5% of the pool, and
        the shortfall was self-reinforcing: a stranger outside the resident set
        cannot match, so a new identity is minted, so the pool grows further
        past the cap. It is now settings.UNKNOWN_REGISTRY_LOAD_LIMIT.

        Rows without an encoding (body-only detections) are skipped: they
        cannot be matched against a face embedding.
        """
        if limit is None:
            limit = getattr(settings, "UNKNOWN_REGISTRY_LOAD_LIMIT", 5000)

        db: Session = SessionLocal()
        try:
            rows = (
                db.query(UnknownFace)
                .filter(
                    UnknownFace.status == "pending",
                    UnknownFace.face_encoding.isnot(None),
                )
                .order_by(UnknownFace.last_seen.desc())
                .limit(limit)
                .all()
            )
            if not rows:
                return []

            ids = [r.id for r in rows]

            # One query for every view of every identity, not one per identity:
            # this runs on a refresh timer with thousands of ids.
            views_by_id: Dict[str, List[Any]] = {}
            for face_id, blob in (
                db.query(UnknownFaceView.unknown_face_id, UnknownFaceView.encoding)
                .filter(UnknownFaceView.unknown_face_id.in_(ids))
                .all()
            ):
                arr = deserialize_encoding(blob)
                if arr is not None:
                    views_by_id.setdefault(face_id, []).append(arr)

            out: List[Dict[str, Any]] = []
            for r in rows:
                encodings = views_by_id.get(r.id)
                if not encodings:
                    # No views row yet (created before migration 012 seeded, or
                    # seeding skipped it). Fall back to the canonical first view.
                    arr = deserialize_encoding(r.face_encoding)
                    if arr is None:
                        continue
                    encodings = [arr]
                out.append({
                    "id": r.id,
                    "encodings": encodings,
                    "occurrence": r.occurrence or 1,
                    "first_seen": r.first_seen.timestamp() if r.first_seen else 0.0,
                    "last_seen": r.last_seen.timestamp() if r.last_seen else 0.0,
                })
            return out
        finally:
            db.close()

    @staticmethod
    def record_sighting(unknown_id: str, camera_id: str, timestamp) -> bool:
        """
        Bump occurrence/last_seen for an already-known unknown identity.

        Used for repeat sightings, which carry no snapshots — the row already
        has them. Returns False if the row is gone (enrolled or dismissed
        between the sighting and this write), which is not an error.

        An ARCHIVED row is reactivated rather than dropped. Archiving means
        "not seen for a long time", so a fresh sighting disproves it. Without
        this, a returning visitor whose row had aged out would keep resolving
        to an archived id in the pipeline's registry and silently lose every
        subsequent sighting.
        """
        db: Session = SessionLocal()
        try:
            row = db.query(UnknownFace).filter(UnknownFace.id == unknown_id).first()
            if row is None:
                return False
            if row.status == "archived":
                logger.info(
                    f"Unknown {unknown_id[:8]}… reappeared after archiving; "
                    f"restored to pending"
                )
                row.status = "pending"
            elif row.status != "pending":
                # Enrolled or dismissed while we were queued; drop silently.
                return False
            row.occurrence = (row.occurrence or 0) + 1
            row.last_seen = timestamp
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to record unknown sighting {unknown_id[:8]}…: {e}")
            return False
        finally:
            db.close()

    # ── Growth control ────────────────────────────────────────────────────
    #
    # The in-memory UnknownRegistry is bounded twice over: get_pending_encodings
    # caps its query and orders by last_seen DESC, and the registry itself holds
    # an LRU of UNKNOWN_REGISTRY_MAX_ENTRIES. So RAM and per-face search cost are
    # already flat no matter how big this table gets.
    #
    # The table is not. Rows only leave `pending` when a human enrolls or
    # dismisses them, so every stranger who ever walked past a camera stays in
    # the review queue forever. That is a review-workload and disk problem, not
    # a recognition one, and archiving is the right shape of fix: it is
    # reversible (see record_sighting), keeps the snapshot for audit, and drops
    # the row out of the pending query the registry loads from.

    # ── The review queue, a page at a time ────────────────────────────────

    # What the queue can be ordered by. A whitelist, because the value comes
    # from a query string and goes into ORDER BY.
    SORT_COLUMNS = {
        "last_seen": UnknownFace.last_seen,
        "first_seen": UnknownFace.first_seen,
        "occurrence": UnknownFace.occurrence,
        "created_at": UnknownFace.created_at,
    }

    @staticmethod
    def list_page(
        status: Optional[str] = "pending",
        sort: str = "last_seen",
        order: str = "desc",
        limit: int = 24,
        offset: int = 0,
        camera_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        One page of the queue, with the totals for the WHOLE filter.

        The totals are computed here rather than from the page: the page used
        to count whatever it had loaded, so with twelve thousand faces pending
        it reported "50 pending" and summed the sightings of those fifty.
        """
        from app.db.models import Camera

        column = UnknownService.SORT_COLUMNS.get(sort, UnknownFace.last_seen)
        direction = column.asc() if order == "asc" else column.desc()
        db: Session = SessionLocal()
        try:
            query = db.query(UnknownFace, Camera.name).outerjoin(
                Camera, Camera.id == UnknownFace.camera_id
            )
            if status and status != "all":
                query = query.filter(UnknownFace.status == status)
            if camera_id:
                query = query.filter(UnknownFace.camera_id == camera_id)

            total, sightings = query.with_entities(
                func.count(UnknownFace.id), func.coalesce(func.sum(UnknownFace.occurrence), 0)
            ).one()
            # The id is a tiebreak: many rows share a timestamp or a count, and
            # without it a face can appear on two pages or on none.
            rows = (
                query.order_by(direction, UnknownFace.id.asc())
                .offset(max(0, int(offset)))
                .limit(max(1, int(limit)))
                .all()
            )
            return {
                "rows": rows,
                "total": int(total or 0),
                "total_sightings": int(sightings or 0),
            }
        finally:
            db.close()

    # ── Removal after a fixed age ─────────────────────────────────────────
    #
    # Different from archive_stale() below on purpose. Archiving hides a face
    # and keeps it; this DELETES the row, its extra views and its image files,
    # for good. Enrolled rows are never touched: their snapshot is the
    # person's avatar and the source image of their first face encoding.

    @staticmethod
    def _expired_query(db: Session, days: int):
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return db.query(UnknownFace).filter(
            UnknownFace.status != "enrolled",
            UnknownFace.last_seen < cutoff,
        )

    @staticmethod
    def count_expired(days: int) -> int:
        """How many unknown faces purge_expired(days) would remove right now."""
        if days <= 0:
            return 0
        db: Session = SessionLocal()
        try:
            return int(UnknownService._expired_query(db, days).count())
        finally:
            db.close()

    @staticmethod
    def _snapshot_files(unknown_id: str, urls: List[Optional[str]]) -> List[str]:
        """Image files that belong to one unknown face, inside SNAPSHOT_DIR only."""
        import os

        root = os.path.abspath(settings.SNAPSHOT_DIR)
        candidates = [
            os.path.join(root, "unknowns", f"unknown_{unknown_id}.jpg"),
            os.path.join(root, "unknowns", f"context_{unknown_id}.jpg"),
            os.path.join(root, "frames", f"fullframe_{unknown_id}.jpg"),
        ]
        for url in urls:
            if url and url.startswith("/snapshots/"):
                candidates.append(os.path.join(root, *url[len("/snapshots/"):].split("/")))
        files = []
        for path in candidates:
            path = os.path.abspath(path)
            # A stored URL is data, not a trusted path: nothing outside the
            # snapshot folder is ever deleted on its say-so.
            if path.startswith(root + os.sep) and path not in files:
                files.append(path)
        return files

    @staticmethod
    def purge_expired(days: int, batch: int = 500) -> Dict[str, int]:
        """
        Permanently remove unknown faces not seen for `days` days.

        Keyed on last_seen, like archiving: someone who keeps coming back is
        never removed, however long ago they were first seen.

        Rows go first and files after, per batch. If the process dies in
        between, the leftovers are orphan image files nobody links to -- the
        other order would leave rows on the page pointing at missing images.
        """
        import os

        removed = {"faces": 0, "files": 0}
        if days <= 0:
            return removed
        while True:
            db: Session = SessionLocal()
            try:
                rows = (
                    UnknownService._expired_query(db, days)
                    .with_entities(UnknownFace.id, UnknownFace.snapshot_url, UnknownFace.full_frame)
                    .limit(batch)
                    .all()
                )
                if not rows:
                    break
                ids = [r[0] for r in rows]
                # Explicit, not left to ON DELETE CASCADE: the constraint
                # exists only where migration 012 was applied by hand.
                db.query(UnknownFaceView).filter(
                    UnknownFaceView.unknown_face_id.in_(ids)
                ).delete(synchronize_session=False)
                db.query(UnknownFace).filter(UnknownFace.id.in_(ids)).delete(
                    synchronize_session=False
                )
                db.commit()
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to remove expired unknown faces: {e}")
                break
            finally:
                db.close()

            removed["faces"] += len(rows)
            for unknown_id, snapshot_url, full_frame in rows:
                for path in UnknownService._snapshot_files(unknown_id, [snapshot_url, full_frame]):
                    try:
                        os.remove(path)
                        removed["files"] += 1
                    except FileNotFoundError:
                        pass
                    except OSError as e:
                        logger.warning(f"Could not delete {path}: {e}")
                try:
                    # Or the next sighting of a look-alike is "merged" into a
                    # row that no longer exists, and disappears.
                    from app.services.unknown_dedup import get_dedup_index
                    get_dedup_index().drop(unknown_id)
                except Exception:
                    pass
        if removed["faces"]:
            logger.info(
                f"Removed {removed['faces']} unknown faces not seen in {days} days "
                f"({removed['files']} image files)"
            )
            try:
                # The review clusters built from those faces must not outlive them.
                from app.services.clustering_service import ClusteringService
                ClusteringService.reconcile()
            except Exception as e:
                logger.warning(f"Cluster reconcile after removal failed: {e}")
        return removed

    @staticmethod
    def start_retention(interval_sec: float = 6 * 3600.0, first_delay_sec: float = 120.0) -> None:
        """Remove expired unknown faces now and then, for as long as the server runs.

        Reads UNKNOWN_FACE_RETENTION_DAYS each time, so 0 (off) is honoured
        without a restart mattering. Started once from the app's lifespan.
        """
        import threading
        import time

        def loop():
            time.sleep(first_delay_sec)
            while True:
                try:
                    days = int(getattr(settings, "UNKNOWN_FACE_RETENTION_DAYS", 0) or 0)
                    if days > 0:
                        UnknownService.purge_expired(days)
                except Exception as e:
                    logger.error(f"Unknown-face retention pass failed: {e}")
                time.sleep(interval_sec)

        threading.Thread(target=loop, name="unknown-retention", daemon=True).start()

    @staticmethod
    def count_by_status() -> Dict[str, int]:
        """Row counts per status — how full the review queue actually is."""
        db: Session = SessionLocal()
        try:
            rows = (
                db.query(UnknownFace.status, func.count(UnknownFace.id))
                .group_by(UnknownFace.status)
                .all()
            )
            return {status or "unknown": int(count) for status, count in rows}
        finally:
            db.close()

    @staticmethod
    def count_stale_pending(unseen_days: int) -> int:
        """How many pending rows archive_stale() would touch right now."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=unseen_days)
        db: Session = SessionLocal()
        try:
            return (
                db.query(UnknownFace)
                .filter(
                    UnknownFace.status == "pending",
                    UnknownFace.last_seen < cutoff,
                )
                .count()
            )
        finally:
            db.close()

    @staticmethod
    def archive_stale(unseen_days: int) -> int:
        """
        Move pending unknowns not seen for `unseen_days` to `archived`.

        Keyed on last_seen, not first_seen: a visitor who has been coming back
        for a year is active and must stay in the queue, while one sighting six
        months ago is not worth a reviewer's attention.

        Nothing is deleted. Archived rows remain queryable (`?status=archived`)
        and are restored automatically if the person is seen again.

        Returns the number of rows archived.
        """
        if unseen_days <= 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=unseen_days)
        db: Session = SessionLocal()
        try:
            archived = (
                db.query(UnknownFace)
                .filter(
                    UnknownFace.status == "pending",
                    UnknownFace.last_seen < cutoff,
                )
                .update({"status": "archived"}, synchronize_session=False)
            )
            db.commit()
            if archived:
                logger.info(
                    f"Archived {archived} unknown identities not seen in "
                    f"{unseen_days} days"
                )
            return int(archived)
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to archive stale unknowns: {e}")
            return 0
        finally:
            db.close()
