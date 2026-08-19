"""
Identity Review Worker — Background daemon for the Gallery Optimizer.

Runs in a dedicated thread, completely isolated from the realtime pipeline.
Periodically:
  1. Clusters unknown faces
  2. Nearest candidate search (Top-K persons per cluster)
  3. Priority scoring (similarity + gallery gain + pose rarity + quality + cluster size)
  4. Three-tier decision:
       High  (>= TIER_HIGH): skip VLM, create candidate directly
       Medium (TIER_LOW .. TIER_HIGH): VLM verification (budget-capped, fairness-enforced)
       Low   (< TIER_LOW): reject immediately
  5. VLM cache lookup before calling Qwen (cluster_id + person_id + gallery_version)
  6. Creates enriched review candidates for human approval
  7. Prints structured telemetry block per cycle

Fail-safe: every phase is wrapped in try/except. Worker failures never
affect the live detection pipeline.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# ── Priority Scoring ────────────────────────────────────────────────

# Weights for the composite priority score.
# Objective: spend VLM time on candidates that most improve the gallery,
# not merely the ones with the highest embedding similarity.
_PRIORITY_WEIGHTS = {
    "similarity":           0.45,
    "gallery_improvement":  0.25,
    "pose_rarity":          0.15,
    "image_quality":        0.10,
    "cluster_size":         0.05,
}

# Pose buckets ordered by rarity (higher score = rarer = more valuable).
# Frontal is the most common and contributes least incremental value.
_POSE_RARITY_SCORES: Dict[str, float] = {
    "frontal":       0.10,
    "left":          0.50,
    "right":         0.50,
    "extreme_left":  0.90,
    "extreme_right": 0.90,
    "up":            0.70,
    "down":          0.70,
}

_GALLERY_IMPROVEMENT_SCORES: Dict[str, float] = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.3,
    "none":   0.0,
}


def compute_priority_score(
    similarity: float,
    gallery_improvement_value: str,
    pose_bucket: str,
    image_quality: float,
    cluster_size: int,
) -> float:
    """
    Compute a composite priority score for ranking candidates.

    Returns a float in [0, 1] — higher is more valuable to review.

    Formula:
        priority = 0.45 * similarity
                 + 0.25 * gallery_improvement
                 + 0.15 * pose_rarity
                 + 0.10 * image_quality
                 + 0.05 * cluster_size_norm
    """
    w = _PRIORITY_WEIGHTS

    # Normalise cluster size: 3->0.1, 10->0.5, 20+->1.0
    cluster_norm = min(1.0, max(0.0, (cluster_size - 1) / 19.0))

    gallery_score = _GALLERY_IMPROVEMENT_SCORES.get(
        gallery_improvement_value, 0.0,
    )
    pose_score = _POSE_RARITY_SCORES.get(pose_bucket, 0.30)

    priority = (
        w["similarity"]          * max(0.0, min(1.0, similarity))
        + w["gallery_improvement"] * gallery_score
        + w["pose_rarity"]         * pose_score
        + w["image_quality"]       * max(0.0, min(1.0, image_quality))
        + w["cluster_size"]        * cluster_norm
    )
    return round(max(0.0, min(1.0, priority)), 4)


def _compute_review_priority(
    similarity: float,
    gallery_improvement_value: str,
    cluster_size: int,
    vlm_agreement: Optional[float] = None,
) -> str:
    """
    Assign review priority based on multiple signals.

    Priority rules:
      Critical: similarity >= 0.80 AND improvement = "high" AND cluster_size >= 5
      High:     similarity >= 0.65 AND improvement in {"high", "medium"}
      Medium:   similarity >= 0.55 AND improvement != "none"
      Low:      everything else
    """
    if (similarity >= 0.80
            and gallery_improvement_value == "high"
            and cluster_size >= 5):
        return "critical"

    if (similarity >= 0.65
            and gallery_improvement_value in ("high", "medium")):
        return "high"

    if similarity >= 0.55 and gallery_improvement_value != "none":
        return "medium"

    return "low"


class IdentityReviewWorker:
    """
    Background daemon that runs the Gallery Optimizer pipeline periodically.

    Started from main.py lifespan if ENABLE_IDENTITY_REVIEW is True.
    Runs on its own thread with its own DB sessions.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10
        self._run_count = 0

    def start(self):
        """Start the background worker thread."""
        if self._running:
            logger.warning("Identity review worker already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="gallery-optimizer-worker",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        logger.info(
            f"✅ Gallery Optimizer Worker started "
            f"(interval={settings.IDENTITY_REVIEW_INTERVAL_SEC}s, "
            f"vlm={settings.IDENTITY_REVIEW_VLM_PROVIDER})"
        )

    def stop(self):
        """Stop the background worker gracefully."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._running = False
        logger.info("🛑 Gallery Optimizer Worker stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "running": self._running,
            "run_count": self._run_count,
            "consecutive_failures": self._consecutive_failures,
        }

    def _worker_loop(self):
        """Main worker loop — runs periodically until stopped."""
        logger.info("Gallery Optimizer Worker loop started")

        # Initial delay to let the system warm up
        self._stop_event.wait(timeout=10)

        while not self._stop_event.is_set():
            try:
                self._run_one_cycle()
                self._consecutive_failures = 0
                self._run_count += 1
            except Exception as e:
                self._consecutive_failures += 1
                logger.error(
                    f"Gallery Optimizer cycle failed "
                    f"(failure #{self._consecutive_failures}): {e}",
                    exc_info=True,
                )

                if self._consecutive_failures >= self._max_consecutive_failures:
                    logger.error(
                        f"Gallery Optimizer: {self._consecutive_failures} "
                        f"consecutive failures, backing off to 5 minutes"
                    )

            # Wait for next cycle (or until stopped)
            interval = settings.IDENTITY_REVIEW_INTERVAL_SEC
            if self._consecutive_failures >= self._max_consecutive_failures:
                interval = max(interval, 300)  # At least 5 min on repeated failures

            self._stop_event.wait(timeout=interval)

        logger.info("Gallery Optimizer Worker loop exiting")

    def _run_one_cycle(self):
        """Execute one full review cycle."""
        cycle_start = time.time()
        logger.info(f"Gallery Optimizer cycle #{self._run_count + 1} starting...")

        # Phase 1: Clustering
        cluster_stats = self._phase_clustering()

        # Phase 2: Candidate search with three-tier pipeline
        candidate_stats = self._phase_candidate_search()

        elapsed = time.time() - cycle_start

        # ── Telemetry block ─────────────────────────────────────
        logger.info(
            "\n"
            "==================================================\n"
            "  Identity Review Cycle #%d\n"
            "==================================================\n"
            "  Clusters scanned     : %d\n"
            "  Low  (< %.2f)        : %d\n"
            "  Medium (%.2f - %.2f) : %d\n"
            "  High (>= %.2f)       : %d\n"
            "--------------------------------------------------\n"
            "  VLM budget           : %d\n"
            "  Sent to VLM          : %d\n"
            "  VLM cache hits       : %d\n"
            "  VLM accepted         : %d\n"
            "  VLM rejected         : %d\n"
            "  Avg VLM time         : %.1f s\n"
            "--------------------------------------------------\n"
            "  Candidates created   : %d\n"
            "  Per-person capped    : %d\n"
            "  Elapsed              : %.1f s\n"
            "==================================================",
            self._run_count + 1,
            candidate_stats.get("clusters_scanned", 0),
            settings.IDENTITY_REVIEW_TIER_LOW,
            candidate_stats.get("tier_low", 0),
            settings.IDENTITY_REVIEW_TIER_LOW,
            settings.IDENTITY_REVIEW_TIER_HIGH,
            candidate_stats.get("tier_medium", 0),
            settings.IDENTITY_REVIEW_TIER_HIGH,
            candidate_stats.get("tier_high", 0),
            candidate_stats.get("vlm_budget", 0),
            candidate_stats.get("vlm_calls", 0),
            candidate_stats.get("vlm_cached", 0),
            candidate_stats.get("vlm_accepted", 0),
            candidate_stats.get("vlm_rejected", 0),
            candidate_stats.get("avg_vlm_time", 0.0),
            candidate_stats.get("candidates_created", 0),
            candidate_stats.get("person_fairness_capped", 0),
            elapsed,
        )

    def _phase_clustering(self) -> dict:
        """Phase 1: Run clustering on unknown faces."""
        try:
            from app.services.clustering_service import ClusteringService
            stats = ClusteringService.run_clustering()
            logger.info(
                f"[Phase 1/Clustering] "
                f"{stats.get('faces_with_encoding', 0)} faces -> "
                f"{stats.get('clusters_created', 0)} new clusters, "
                f"{stats.get('clusters_updated', 0)} updated"
            )
            return stats
        except Exception as e:
            logger.error(f"[Phase 1/Clustering] Failed: {e}", exc_info=True)
            return {}

    def _confidence_gate(self, similarity: float) -> str:
        """
        Three-tier confidence gate with safety-valve review window.

        Flow:
          1. similarity < REVIEW_MIN  -> "reject"        (below safety valve)
          2. similarity > REVIEW_MAX  -> "auto_queue"     (above safety valve)
          3. similarity >= TIER_HIGH  -> "skip_vlm_queue" (strong match)
          4. similarity >= TIER_LOW   -> "call_vlm"       (ambiguous, VLM decides)
          5. otherwise                -> "reject"         (below tier)

        Returns one of: "reject", "auto_queue", "skip_vlm_queue", "call_vlm"
        """
        review_min = settings.IDENTITY_REVIEW_REVIEW_MIN
        review_max = settings.IDENTITY_REVIEW_REVIEW_MAX
        tier_high = settings.IDENTITY_REVIEW_TIER_HIGH
        tier_low = settings.IDENTITY_REVIEW_TIER_LOW

        # Safety valve
        if similarity < review_min:
            return "reject"
        if similarity > review_max:
            return "auto_queue"

        # Three-tier within the review window
        if similarity >= tier_high:
            return "skip_vlm_queue"
        elif similarity >= tier_low:
            return "call_vlm"
        else:
            return "reject"

    def _detect_outliers(
        self,
        members: List[Tuple],
        centroid: np.ndarray,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Detect embedding outliers in a cluster.

        Members whose cosine distance from the centroid exceeds
        GALLERY_OPT_OUTLIER_FACTOR * median_distance are flagged as outliers.

        Args:
            members: List of (UnknownFace, embedding) tuples
            centroid: Cluster centroid embedding (normalized)

        Returns:
            (inliers, outliers) -- split member lists
        """
        if len(members) < 3:
            return members, []

        # Compute distances from centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return members, []
        centroid_n = centroid / centroid_norm

        distances = []
        for face, enc in members:
            enc_norm = np.linalg.norm(enc)
            if enc_norm > 0:
                dist = 1.0 - float(np.dot(centroid_n, enc / enc_norm))
            else:
                dist = 1.0
            distances.append(dist)

        median_dist = float(np.median(distances))
        threshold = median_dist * settings.GALLERY_OPT_OUTLIER_FACTOR

        inliers = []
        outliers = []
        for (face, enc), dist in zip(members, distances):
            if dist > threshold and threshold > 0:
                outliers.append((face, enc))
            else:
                inliers.append((face, enc))

        if outliers:
            logger.info(
                f"Cluster outlier detection: {len(outliers)} outliers / "
                f"{len(members)} members (threshold={threshold:.4f})"
            )

        return inliers, outliers

    def _phase_candidate_search(self) -> dict:
        """
        Phase 2: Three-tier candidate search with priority scoring,
        per-person fairness, adaptive VLM budget, and full telemetry.

        Pipeline:
          1. Fetch active clusters
          2. For each cluster -> find nearest known persons
          3. Apply safety-valve review window
          4. Compute priority score per (cluster, person) pair
          5. Sort all pairs by priority DESC
          6. Apply per-person fairness cap
          7. Apply three-tier gate:
               High   -> create candidate directly
               Medium -> check VLM cache, then call VLM if budget allows
               Low    -> reject
          8. Create enriched candidate records
        """
        stats = {
            "clusters_scanned": 0,
            "tier_high": 0,
            "tier_medium": 0,
            "tier_low": 0,
            "vlm_budget": 0,
            "vlm_calls": 0,
            "vlm_cached": 0,
            "vlm_accepted": 0,
            "vlm_rejected": 0,
            "vlm_total_time": 0.0,
            "avg_vlm_time": 0.0,
            "candidates_created": 0,
            "person_fairness_capped": 0,
        }

        try:
            from app.db.session import SessionLocal
            from app.db.models import UnknownCluster, UnknownFace, FaceEncoding
            from app.services.identity_review_service import IdentityReviewService
            from app.services.vlm_service import (
                get_vlm_reviewer, get_vlm_cache,
                select_best_gallery_image,
            )
            from app.services.gallery_health import (
                compute_coverage_delta, compute_person_profile,
            )
            from app.services.embedding_suggestions import classify_pose_bucket
            from app.vision.recognizer import deserialize_encoding

            db = SessionLocal()
            try:
                # ── Step 1: Fetch active clusters ────────────────────
                clusters = (
                    db.query(UnknownCluster)
                    .filter(
                        UnknownCluster.status == "active",
                        UnknownCluster.member_count >= settings.IDENTITY_REVIEW_MIN_CLUSTER_SIZE,
                    )
                    .order_by(UnknownCluster.last_seen.desc())
                    .all()
                )

                if not clusters:
                    logger.debug("[Phase 2/Search] No active clusters to process")
                    return stats

                stats["clusters_scanned"] = len(clusters)

                vlm = get_vlm_reviewer()
                vlm_cache = get_vlm_cache()
                gallery_version = self._get_gallery_version(db)

                # ── Step 2-4: Collect all (cluster, candidate) pairs with priority ──
                scored_pairs: List[dict] = []

                for cluster in clusters:
                    if self._stop_event.is_set():
                        break

                    try:
                        centroid = deserialize_encoding(cluster.embedding_centroid)
                        if centroid is None:
                            continue

                        # Find nearest known persons
                        candidates = IdentityReviewService.find_candidates_for_cluster(
                            cluster_id=cluster.id,
                            centroid=centroid,
                        )

                        if not candidates:
                            continue

                        for cand in candidates:
                            similarity = cand["similarity"]

                            # Safety-valve gate
                            gate = self._confidence_gate(similarity)
                            if gate == "reject":
                                stats["tier_low"] += 1
                                continue

                            # Compute gallery improvement for priority scoring
                            unknown_pose = "frontal"  # Default
                            try:
                                member_faces = (
                                    db.query(UnknownFace)
                                    .filter(UnknownFace.cluster_id == cluster.id)
                                    .limit(1)
                                    .all()
                                )
                                if member_faces:
                                    pass  # Pose metadata not stored on unknowns
                            except Exception:
                                pass

                            coverage_delta = compute_coverage_delta(
                                person_id=cand["person_id"],
                                new_pose_bucket=unknown_pose,
                                new_quality=cand.get("quality", 0.5) or 0.5,
                                db=db,
                            )

                            # Priority score
                            priority = compute_priority_score(
                                similarity=similarity,
                                gallery_improvement_value=coverage_delta.improvement_value,
                                pose_bucket=unknown_pose,
                                image_quality=cand.get("quality", 0.5) or 0.5,
                                cluster_size=cluster.member_count or 1,
                            )

                            # Triage
                            if gate in ("auto_queue", "skip_vlm_queue"):
                                tier = "high"
                                stats["tier_high"] += 1
                            else:
                                tier = "medium"
                                stats["tier_medium"] += 1

                            scored_pairs.append({
                                "cluster": cluster,
                                "cand": cand,
                                "centroid": centroid,
                                "similarity": similarity,
                                "gate": gate,
                                "tier": tier,
                                "priority": priority,
                                "coverage_delta": coverage_delta,
                                "unknown_pose": unknown_pose,
                            })

                    except Exception as e:
                        logger.error(
                            f"[Phase 2/Search] Cluster {cluster.id[:8]}... "
                            f"scoring failed: {e}"
                        )
                        continue

                if not scored_pairs:
                    logger.info(
                        f"[Phase 2/Triage] {stats['clusters_scanned']} clusters "
                        f"scanned, no candidates passed safety valve"
                    )
                    return stats

                # ── Step 5: Sort by priority DESC ────────────────────
                scored_pairs.sort(key=lambda p: p["priority"], reverse=True)

                logger.info(
                    f"[Phase 2/Triage] {stats['clusters_scanned']} clusters -> "
                    f"{len(scored_pairs)} candidates "
                    f"(high={stats['tier_high']}, "
                    f"medium={stats['tier_medium']}, "
                    f"low={stats['tier_low']})"
                )

                # ── Step 6: Per-person fairness cap ──────────────────
                person_cluster_count: Dict[str, int] = defaultdict(int)
                max_per_person = settings.IDENTITY_REVIEW_MAX_CLUSTERS_PER_PERSON

                # Adaptive VLM budget: min(configured, actual medium-tier count)
                medium_count = stats["tier_medium"]
                vlm_budget = min(
                    settings.IDENTITY_REVIEW_VLM_MAX_PER_CYCLE,
                    medium_count,
                )
                stats["vlm_budget"] = vlm_budget
                vlm_calls_this_cycle = 0

                # ── Step 7-8: Process candidates ─────────────────────
                for pair in scored_pairs:
                    if self._stop_event.is_set():
                        break

                    cluster = pair["cluster"]
                    cand = pair["cand"]
                    centroid = pair["centroid"]
                    similarity = pair["similarity"]
                    gate = pair["gate"]
                    coverage_delta = pair["coverage_delta"]
                    unknown_pose = pair["unknown_pose"]
                    person_id = cand["person_id"]

                    # Per-person fairness check
                    if person_cluster_count[person_id] >= max_per_person:
                        stats["person_fairness_capped"] += 1
                        continue
                    person_cluster_count[person_id] += 1

                    # ── VLM review ───────────────────────────────────
                    vlm_confidence = None
                    vlm_explanation = ""
                    vlm_detected_pose = ""
                    vlm_quality_assessment = ""

                    if gate == "call_vlm":
                        # Check VLM cache first (respects cluster_id + person_id + gallery_version)
                        if cluster.representative_snapshot and cand.get("source_image"):
                            cached_result = vlm_cache.get(
                                cluster.id, person_id, gallery_version,
                            )
                            if cached_result:
                                vlm_confidence = cached_result.agreement
                                vlm_explanation = cached_result.explanation
                                vlm_detected_pose = cached_result.detected_pose
                                vlm_quality_assessment = cached_result.quality
                                stats["vlm_cached"] += 1

                                # Apply cached decision
                                if vlm_confidence is not None and vlm_confidence < 0.5:
                                    stats["vlm_rejected"] += 1
                                    continue  # Skip -- VLM previously rejected
                                else:
                                    stats["vlm_accepted"] += 1
                            elif vlm_calls_this_cycle < vlm_budget:
                                # Fresh VLM call
                                try:
                                    best_image = select_best_gallery_image(
                                        person_id, centroid,
                                    )
                                    if best_image:
                                        vlm_start = time.time()
                                        vlm_result = vlm.compare_faces(
                                            cluster.representative_snapshot,
                                            best_image,
                                        )
                                        vlm_elapsed = time.time() - vlm_start
                                        stats["vlm_total_time"] += vlm_elapsed
                                        vlm_calls_this_cycle += 1
                                        stats["vlm_calls"] += 1

                                        if vlm_result.error is None:
                                            vlm_confidence = vlm_result.agreement
                                            vlm_explanation = vlm_result.explanation
                                            vlm_detected_pose = vlm_result.detected_pose
                                            vlm_quality_assessment = vlm_result.quality
                                            # Cache this result for future cycles
                                            vlm_cache.put(
                                                cluster.id, person_id,
                                                gallery_version, vlm_result,
                                            )

                                            # VLM rejection
                                            if vlm_confidence is not None and vlm_confidence < 0.5:
                                                stats["vlm_rejected"] += 1
                                                continue
                                            else:
                                                stats["vlm_accepted"] += 1
                                        else:
                                            logger.debug(
                                                f"VLM returned error: {vlm_result.error}"
                                            )
                                except Exception as e:
                                    logger.debug(f"VLM review failed: {e}")
                            else:
                                # Budget exhausted -- skip this candidate
                                continue

                    # gate == "skip_vlm_queue" or "auto_queue" -- no VLM needed

                    # ── Review priority ──────────────────────────────
                    review_priority = _compute_review_priority(
                        similarity=similarity,
                        gallery_improvement_value=coverage_delta.improvement_value,
                        cluster_size=cluster.member_count or 1,
                        vlm_agreement=vlm_confidence,
                    )

                    # ── Create candidate record ──────────────────────
                    result = IdentityReviewService.create_candidate(
                        cluster_id=cluster.id,
                        person_id=person_id,
                        embedding_similarity=similarity,
                        vlm_confidence=vlm_confidence,
                        vlm_explanation=vlm_explanation,
                        image_quality_score=cand.get("quality", 0.5) or 0.5,
                        gallery_improvement_value=coverage_delta.improvement_value,
                        gallery_improvement_reason=(
                            coverage_delta.fills_bucket or ""
                        ),
                        coverage_before=coverage_delta.current_coverage,
                        coverage_after=coverage_delta.projected_coverage,
                        vlm_detected_pose=vlm_detected_pose,
                        vlm_quality_assessment=vlm_quality_assessment,
                        review_priority=review_priority,
                    )

                    if result:
                        stats["candidates_created"] += 1

                # Compute average VLM time
                if stats["vlm_calls"] > 0:
                    stats["avg_vlm_time"] = round(
                        stats["vlm_total_time"] / stats["vlm_calls"], 1
                    )

            finally:
                db.close()

        except Exception as e:
            logger.error(f"[Phase 2/Search] Failed: {e}", exc_info=True)

        return stats

    @staticmethod
    def _get_gallery_version(db) -> int:
        """Get the current gallery encoding version for cache keys."""
        try:
            from app.db.models import EncodingVersion
            ver = db.query(EncodingVersion).filter(EncodingVersion.id == 1).first()
            return ver.version if ver else 0
        except Exception:
            return 0


# Singleton instance
_worker: Optional[IdentityReviewWorker] = None


def get_identity_review_worker() -> IdentityReviewWorker:
    """Get or create the singleton worker instance."""
    global _worker
    if _worker is None:
        _worker = IdentityReviewWorker()
    return _worker
