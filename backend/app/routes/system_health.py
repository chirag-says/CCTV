"""
System Health Dashboard API.

Single endpoint returning all pipeline metrics for the observability dashboard:
  - Recognition: known/unknown/gray-zone rates, latency, gallery coverage
  - Gallery: embeddings, pose histogram, health, duplicates
  - VLM: queue length, reviewed/approved/rejected, cache hit rate
  - Pipeline: FPS, GPU/CPU, SCRFD/ArcFace/JPEG timing
  - Tracking: creations, recoveries, carry-over, ID switches
  - Unknowns: created/hour, matched, maturation-filtered
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system-health", tags=["System Health"])


def _safe_get(func, default=None):
    """Call func, return default on any error."""
    try:
        return func()
    except Exception:
        return default


@router.get("")
async def get_system_health():
    """
    Returns a comprehensive system health snapshot.
    All metrics are collected from in-memory profilers and DB queries.
    """

    # ── Pipeline profiler stats ──────────────────────────────────
    pipeline_stats = _get_pipeline_stats()

    # ── Recognition stats ────────────────────────────────────────
    recognition = _get_recognition_stats(pipeline_stats)

    # ── Gallery stats ────────────────────────────────────────────
    gallery = _get_gallery_stats()

    # ── VLM stats ────────────────────────────────────────────────
    vlm = _get_vlm_stats()

    # ── Pipeline timing ──────────────────────────────────────────
    pipeline_timing = _get_pipeline_timing(pipeline_stats)

    # ── Tracking stats ───────────────────────────────────────────
    tracking = _get_tracking_stats(pipeline_stats)

    # ── Unknown stats ────────────────────────────────────────────
    unknowns = _get_unknown_stats()

    # ── Snapshot worker stats ─────────────────────────────────────
    snapshot_worker = _get_snapshot_worker_stats()

    # ── Shadow gallery stats ─────────────────────────────────────
    shadow = _get_shadow_gallery_stats()

    return {
        "recognition": recognition,
        "gallery": gallery,
        "vlm": vlm,
        "pipeline": pipeline_timing,
        "tracking": tracking,
        "unknowns": unknowns,
        "snapshot_worker": snapshot_worker,
        "shadow_gallery": shadow,
        "timestamp": time.time(),
    }


@router.post("/shadow-gallery/init")
async def init_shadow_gallery():
    """
    Auto-generate a validation set from the current gallery.
    Holds out recent encodings' source images as test data.
    """
    try:
        from app.services.shadow_gallery import ShadowGallery, get_shadow_gallery
        manifest = ShadowGallery.create_validation_set()
        # Start the shadow gallery if it's not running
        sg = get_shadow_gallery()
        if not sg._running:
            sg.start()
        return {
            "success": True,
            "persons": len(manifest),
            "images": sum(len(p["images"]) for p in manifest.values()),
            "message": f"Validation set created with {len(manifest)} persons",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/shadow-gallery/run")
async def run_shadow_validation():
    """
    Trigger an immediate shadow gallery validation run.
    Non-blocking: dispatches to a background thread and returns immediately.
    Results will appear in the shadow_gallery stats.
    """
    import threading
    try:
        from app.services.shadow_gallery import get_shadow_gallery
        sg = get_shadow_gallery()

        def _run():
            try:
                version = sg._get_gallery_version()
                score = sg.run_validation(version)
                if score:
                    sg._history.append(score)
                    if len(sg._history) > sg._max_history:
                        sg._history = sg._history[-sg._max_history:]
                    sg._save_history()
                    sg._last_gallery_version = version
                    logger.info(
                        f"🔮 Shadow Gallery validation complete: "
                        f"accuracy={score.accuracy * 100:.1f}% "
                        f"({score.true_positives}/{score.total_tests} correct)"
                    )
            except Exception as e:
                logger.error(f"Shadow Gallery background validation failed: {e}")

        t = threading.Thread(target=_run, name="ShadowGalleryManualRun", daemon=True)
        t.start()
        return {
            "success": True,
            "message": "Validation started in background. Check /api/system-health for results.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def _get_pipeline_stats() -> dict:
    """Get raw stats from the pipeline profiler."""
    try:
        from app.core.profiler import PerformanceProfiler
        profiler = PerformanceProfiler.get_instance()
        if profiler and profiler.enabled:
            return profiler.get_health_stats()
    except Exception as e:
        logger.debug(f"Failed to get pipeline stats: {e}")
    return {}


def _get_recognition_stats(ps: dict) -> dict:
    """Recognition accuracy metrics."""
    rec = ps.get("recognition", {})
    total = rec.get("total_recognitions", 0)
    matched = rec.get("matched", 0)
    unknown = rec.get("unknown", 0)
    gray = rec.get("gray_zone", 0)

    return {
        "known_pct": round(matched / total * 100, 1) if total else 0,
        "unknown_pct": round(unknown / total * 100, 1) if total else 0,
        "gray_zone_pct": round(gray / total * 100, 1) if total else 0,
        "total_recognitions": total,
        "latency_ms": round(rec.get("avg_latency_ms", 0), 2),
        "cache_hit_pct": round(rec.get("cache_hit_pct", 0), 1),
        "cache_hits": rec.get("cache_hits", 0),
        "cache_saves": rec.get("cache_saves", 0),
    }


def _get_gallery_stats() -> dict:
    """Gallery embedding statistics."""
    try:
        from app.db.session import SessionLocal
        from app.db.models import FaceEncoding, Person, SuggestedEncoding

        db = SessionLocal()
        try:
            total_embeddings = db.query(FaceEncoding).count()
            active_persons = db.query(Person).filter(Person.is_active == True).count()
            avg_per_person = round(total_embeddings / active_persons, 1) if active_persons else 0

            # Suggestions stats
            pending_suggestions = db.query(SuggestedEncoding).filter(
                SuggestedEncoding.status == "pending"
            ).count()
            approved_suggestions = db.query(SuggestedEncoding).filter(
                SuggestedEncoding.status == "approved"
            ).count()

            # Pose histogram (from suggested_encodings metadata)
            pose_histogram = {"frontal": 0, "left": 0, "right": 0,
                              "up": 0, "down": 0, "extreme": 0}
            try:
                from sqlalchemy import func
                pose_counts = (
                    db.query(FaceEncoding.pose_bucket, func.count())
                    .filter(FaceEncoding.pose_bucket.isnot(None))
                    .group_by(FaceEncoding.pose_bucket)
                    .all()
                )
                for bucket, count in pose_counts:
                    if bucket in pose_histogram:
                        pose_histogram[bucket] = count
            except Exception:
                pass  # pose_bucket column may not exist

            return {
                "total_embeddings": total_embeddings,
                "active_persons": active_persons,
                "avg_per_person": avg_per_person,
                "pending_suggestions": pending_suggestions,
                "approved_suggestions": approved_suggestions,
                "pose_histogram": pose_histogram,
            }
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"Failed to get gallery stats: {e}")
        return {
            "total_embeddings": 0, "active_persons": 0,
            "avg_per_person": 0, "pending_suggestions": 0,
            "approved_suggestions": 0, "pose_histogram": {},
        }


def _get_vlm_stats() -> dict:
    """VLM / Gallery Optimizer statistics."""
    try:
        from app.db.session import SessionLocal
        from app.db.models import IdentityReviewCandidate

        db = SessionLocal()
        try:
            pending = db.query(IdentityReviewCandidate).filter(
                IdentityReviewCandidate.status == "pending"
            ).count()
            approved = db.query(IdentityReviewCandidate).filter(
                IdentityReviewCandidate.status == "approved"
            ).count()
            rejected = db.query(IdentityReviewCandidate).filter(
                IdentityReviewCandidate.status == "rejected"
            ).count()
            total_reviewed = approved + rejected

            # Worker stats
            worker_stats = {}
            try:
                from app.services.identity_review_worker import get_identity_review_worker
                worker = get_identity_review_worker()
                worker_stats = worker.stats
            except Exception:
                pass

            # VLM cache stats
            vlm_cache_stats = {}
            try:
                from app.services.vlm_service import get_vlm_cache
                cache = get_vlm_cache()
                vlm_cache_stats = cache.stats if hasattr(cache, 'stats') else {}
            except Exception:
                pass

            return {
                "queue_length": pending,
                "reviewed_total": total_reviewed,
                "approved": approved,
                "rejected": rejected,
                "merge_rate_pct": round(
                    approved / total_reviewed * 100, 1
                ) if total_reviewed else 0,
                "worker_running": worker_stats.get("running", False),
                "worker_cycles": worker_stats.get("run_count", 0),
                "worker_failures": worker_stats.get("consecutive_failures", 0),
                "cache_stats": vlm_cache_stats,
            }
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"Failed to get VLM stats: {e}")
        return {
            "queue_length": 0, "reviewed_total": 0, "approved": 0,
            "rejected": 0, "merge_rate_pct": 0, "worker_running": False,
            "worker_cycles": 0, "worker_failures": 0, "cache_stats": {},
        }


def _get_pipeline_timing(ps: dict) -> dict:
    """Pipeline performance metrics."""
    timing = ps.get("timing", {})
    return {
        "fps": round(timing.get("fps", 0), 1),
        "scrfd_ms": round(timing.get("scrfd_ms", 0), 1),
        "arcface_ms": round(timing.get("arcface_ms", 0), 1),
        "jpeg_ms": round(timing.get("jpeg_ms", 0), 1),
        "snapshot_ms": round(timing.get("snapshot_ms", 0), 1),
        "snapshot_queue": timing.get("snapshot_queue", 0),
        "gpu": ps.get("gpu", {}),
        "cpu": ps.get("cpu", {}),
    }


def _get_tracking_stats(ps: dict) -> dict:
    """Tracking and identity recovery metrics."""
    tracking = ps.get("tracking", {})
    recovery = ps.get("recovery", {})

    return {
        "active_tracks": tracking.get("active_tracks", 0),
        "total_entries": tracking.get("total_entries", 0),
        "total_exits": tracking.get("total_exits", 0),
        "identity_swaps": recovery.get("identity_swaps", 0),
        "recovery_attempted": recovery.get("attempted", 0),
        "recovery_verified": recovery.get("verified", 0),
        "carryover_pct": round(
            recovery.get("verified", 0) / recovery.get("attempted", 1) * 100, 1
        ) if recovery.get("attempted", 0) else 0,
        "carryover_rejections": recovery.get("rejections", {}),
    }


def _get_unknown_stats() -> dict:
    """Unknown face statistics."""
    try:
        from app.db.session import SessionLocal
        from app.db.models import UnknownFace
        from sqlalchemy import func
        from datetime import datetime, timezone, timedelta

        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            one_hour_ago = now - timedelta(hours=1)
            one_day_ago = now - timedelta(hours=24)

            created_last_hour = db.query(UnknownFace).filter(
                UnknownFace.created_at >= one_hour_ago
            ).count()

            matched_last_24h = db.query(UnknownFace).filter(
                UnknownFace.status == "enrolled",
                UnknownFace.created_at >= one_day_ago,
            ).count()

            total_pending = db.query(UnknownFace).filter(
                UnknownFace.status == "pending"
            ).count()

            total_enrolled = db.query(UnknownFace).filter(
                UnknownFace.status == "enrolled"
            ).count()

            total = db.query(UnknownFace).count()

            return {
                "created_last_hour": created_last_hour,
                "matched_last_24h": matched_last_24h,
                "total_pending": total_pending,
                "total_enrolled": total_enrolled,
                "total": total,
            }
        finally:
            db.close()
    except Exception as e:
        logger.debug(f"Failed to get unknown stats: {e}")
        return {
            "created_last_hour": 0, "matched_last_24h": 0,
            "total_pending": 0, "total_enrolled": 0, "total": 0,
        }


def _get_snapshot_worker_stats() -> dict:
    """Snapshot worker queue and throughput stats."""
    try:
        from app.services.snapshot_worker import get_snapshot_worker
        worker = get_snapshot_worker()
        return worker.stats
    except Exception as e:
        logger.debug(f"Failed to get snapshot worker stats: {e}")
        return {
            "queue_size": 0, "max_queue": 0, "fill_pct": 0,
            "processed": 0, "dropped_normal": 0, "dropped_debug": 0,
            "avg_write_ms": 0, "max_write_ms": 0, "running": False,
        }


def _get_shadow_gallery_stats() -> dict:
    """Shadow gallery validation stats."""
    try:
        from app.services.shadow_gallery import get_shadow_gallery
        sg = get_shadow_gallery()
        return sg.stats
    except Exception as e:
        logger.debug(f"Failed to get shadow gallery stats: {e}")
        return {
            "running": False, "interval_minutes": 0,
            "total_validations": 0, "latest": None,
            "accuracy_trend": None, "has_manifest": False,
        }
