"""
Clustering Service — Groups unknown faces into identity clusters.

Uses agglomerative clustering with cosine distance on face embeddings.
Runs in the background worker thread, never in the realtime pipeline.

Pipeline:
  1. Load pending unknown face embeddings from DB
  2. Run agglomerative clustering (scipy)
  3. Create/update UnknownCluster rows
  4. Link unknown_faces.cluster_id to clusters
"""

import logging
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from app.config import settings
from app.db.session import SessionLocal
from app.db.models import UnknownFace, UnknownCluster
from app.vision.recognizer import deserialize_encoding, serialize_encoding

logger = logging.getLogger(__name__)


def centroid_distances(
    embeddings: List[np.ndarray],
    centroid: np.ndarray,
) -> List[float]:
    """
    Cosine distance of each embedding from a cluster centroid.

    A member that no longer looks like the rest of its cluster shows up here
    as a large distance. Returns 1.0 (maximum distance) for unusable vectors
    rather than dropping them, so the returned list always lines up with the
    input list index-for-index.
    """
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return [1.0] * len(embeddings)
    centroid_n = centroid / centroid_norm

    distances = []
    for enc in embeddings:
        enc_norm = np.linalg.norm(enc) if enc is not None else 0.0
        if enc_norm > 0:
            distances.append(1.0 - float(np.dot(centroid_n, enc / enc_norm)))
        else:
            distances.append(1.0)
    return distances


def outlier_flags(
    distances: List[float],
    outlier_factor: float,
    min_members: int = 3,
) -> Tuple[List[bool], float]:
    """
    Flag cluster members whose distance from the centroid is anomalous.

    A member is an outlier when its distance exceeds
    `outlier_factor * median_distance`. The median (not the mean) is the
    reference precisely because one badly-wrong face should not drag the
    threshold out to cover itself.

    Below `min_members` there is no meaningful median, so nothing is flagged —
    a 2-member cluster cannot tell you which of the two is the odd one out.

    Returns (flags, threshold). threshold is 0.0 when no check was performed.
    """
    if len(distances) < min_members:
        return [False] * len(distances), 0.0

    median_dist = float(np.median(distances))
    threshold = median_dist * outlier_factor
    if threshold <= 0:
        return [False] * len(distances), 0.0

    return [d > threshold for d in distances], threshold


def compute_cluster_groups(
    face_data: List[Tuple[Any, np.ndarray]],
    threshold: float,
) -> Dict[int, List[Tuple[Any, np.ndarray]]]:
    """
    Agglomerative-cluster pre-normalised embeddings into label -> members.

    Pure computation, no database access — which is what lets the backlog
    diagnostic measure exactly what a real cycle would produce without
    writing a single row.
    """
    if len(face_data) < 2:
        return {}

    embeddings = np.array([fd[1] for fd in face_data], dtype=np.float32)
    distances = pdist(embeddings, metric="cosine")
    linkage_matrix = linkage(distances, method="average")
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    groups: Dict[int, List[Tuple[Any, np.ndarray]]] = {}
    for member, label in zip(face_data, labels):
        groups.setdefault(int(label), []).append(member)
    return groups


class ClusteringService:
    """
    Groups unknown faces into identity clusters using agglomerative clustering.

    Designed to run periodically in a background thread. Each run:
    1. Loads all pending unknown faces with embeddings
    2. Computes pairwise cosine distances
    3. Runs agglomerative clustering with a configurable threshold
    4. Creates/updates cluster records in the database
    """

    @staticmethod
    def run_clustering() -> Dict[str, Any]:
        """
        Execute one clustering pass over pending unknown faces.

        Returns stats dict with counts of clusters created/updated/etc.
        """
        stats = {
            "faces_loaded": 0,
            "faces_with_encoding": 0,
            "clusters_found": 0,
            "clusters_created": 0,
            "clusters_updated": 0,
            "singletons_skipped": 0,
        }

        db = SessionLocal()
        try:
            # 1. Load pending unknown faces with embeddings
            faces = (
                db.query(UnknownFace)
                .filter(
                    UnknownFace.status == "pending",
                    UnknownFace.face_encoding.isnot(None),
                )
                .order_by(UnknownFace.last_seen.desc())
                .limit(2000)  # Cap to prevent OOM
                .all()
            )

            stats["faces_loaded"] = len(faces)

            if len(faces) < 2:
                logger.info(
                    f"Clustering: only {len(faces)} pending faces, "
                    f"need ≥2 to cluster"
                )
                return stats

            # 2. Deserialize embeddings
            face_data: List[Tuple[UnknownFace, np.ndarray]] = []
            for face in faces:
                enc = deserialize_encoding(face.face_encoding)
                if enc is not None and len(enc) > 0:
                    # Normalize for cosine distance
                    norm = np.linalg.norm(enc)
                    if norm > 0:
                        face_data.append((face, enc / norm))

            stats["faces_with_encoding"] = len(face_data)

            if len(face_data) < 2:
                return stats

            # 3-5. Cluster and group by label
            cluster_groups = compute_cluster_groups(
                face_data, settings.IDENTITY_REVIEW_CLUSTER_THRESHOLD,
            )

            stats["clusters_found"] = len(cluster_groups)

            # 6. Create/update cluster records
            min_size = settings.IDENTITY_REVIEW_MIN_CLUSTER_SIZE
            for label, members in cluster_groups.items():
                if len(members) < min_size:
                    stats["singletons_skipped"] += 1
                    continue

                # Check if these faces already belong to an existing cluster
                existing_cluster_ids = set()
                for face, _ in members:
                    if face.cluster_id:
                        existing_cluster_ids.add(face.cluster_id)

                cluster = None
                if existing_cluster_ids:
                    # Use the largest existing cluster
                    for cid in existing_cluster_ids:
                        existing = (
                            db.query(UnknownCluster)
                            .filter(UnknownCluster.id == cid)
                            .first()
                        )
                        if existing and existing.status == "active":
                            cluster = existing
                            break

                if cluster:
                    # Update existing cluster
                    cluster.member_count = len(members)
                    cluster.last_seen = max(
                        (f.last_seen for f, _ in members if f.last_seen),
                        default=datetime.now(timezone.utc),
                    )
                    cluster.updated_at = datetime.now(timezone.utc)

                    # Update centroid
                    centroid = np.mean([enc for _, enc in members], axis=0)
                    cluster.embedding_centroid = serialize_encoding(centroid)

                    # Update representative (best quality / most seen)
                    best_face = max(members, key=lambda m: m[0].occurrence or 1)[0]
                    cluster.representative_snapshot = best_face.snapshot_url
                    cluster.representative_face_id = best_face.id

                    stats["clusters_updated"] += 1
                else:
                    # Create new cluster
                    centroid = np.mean([enc for _, enc in members], axis=0)
                    best_face = max(members, key=lambda m: m[0].occurrence or 1)[0]

                    cluster = UnknownCluster(
                        id=str(uuid4()),
                        representative_snapshot=best_face.snapshot_url,
                        representative_face_id=best_face.id,
                        embedding_centroid=serialize_encoding(centroid),
                        member_count=len(members),
                        first_seen=min(
                            (f.first_seen for f, _ in members if f.first_seen),
                            default=datetime.now(timezone.utc),
                        ),
                        last_seen=max(
                            (f.last_seen for f, _ in members if f.last_seen),
                            default=datetime.now(timezone.utc),
                        ),
                        status="active",
                    )
                    db.add(cluster)
                    stats["clusters_created"] += 1

                # Link all member faces to this cluster
                for face, _ in members:
                    face.cluster_id = cluster.id

            db.commit()

            logger.info(
                f"Clustering complete: {stats['faces_with_encoding']} faces → "
                f"{stats['clusters_found']} raw groups → "
                f"{stats['clusters_created']} new + {stats['clusters_updated']} updated clusters "
                f"({stats['singletons_skipped']} singletons skipped, "
                f"min_size={min_size})"
            )

            return stats

        except Exception as e:
            db.rollback()
            logger.error(f"Clustering failed: {e}", exc_info=True)
            return stats
        finally:
            db.close()

    @staticmethod
    def reconcile() -> Dict[str, int]:
        """
        Bring active clusters back in line with the faces that still exist.

        A cluster is a claim about a set of unknown faces. Faces leave -- they
        are enrolled, dismissed, or removed after UNKNOWN_FACE_RETENTION_DAYS --
        and nothing told the cluster. The first retention run (2026-11-20) left
        61 "active" clusters with no faces at all and three review candidates
        whose only evidence was an image that had just been deleted: the
        review page showed broken pictures for matches nobody could check.

        So, for every active cluster:
          * no pending face left -> the cluster and its candidates are removed.
            There is nothing to look at, so there is nothing to review.
          * some left            -> its count is corrected, and if the face
            it showed is gone it shows the best one that remains.

        Reviewed and merged clusters are left alone: they are history.
        """
        from collections import defaultdict

        from app.db.models import IdentityReviewCandidate

        stats = {"clusters_removed": 0, "candidates_removed": 0, "clusters_repaired": 0}
        db = SessionLocal()
        try:
            clusters = db.query(UnknownCluster).filter(UnknownCluster.status == "active").all()
            if not clusters:
                return stats
            members = defaultdict(list)
            for face in (
                db.query(UnknownFace)
                .filter(
                    UnknownFace.status == "pending",
                    UnknownFace.cluster_id.in_([c.id for c in clusters]),
                )
                .all()
            ):
                members[face.cluster_id].append(face)

            for cluster in clusters:
                faces = members.get(cluster.id, [])
                if not faces:
                    # Explicit, not left to ON DELETE CASCADE: the constraint
                    # only exists where the migration was applied by hand.
                    stats["candidates_removed"] += (
                        db.query(IdentityReviewCandidate)
                        .filter(IdentityReviewCandidate.cluster_id == cluster.id)
                        .delete(synchronize_session=False)
                    )
                    db.delete(cluster)
                    stats["clusters_removed"] += 1
                    continue
                repaired = False
                if cluster.member_count != len(faces):
                    cluster.member_count = len(faces)
                    repaired = True
                if cluster.representative_face_id not in {f.id for f in faces}:
                    best = max(faces, key=lambda f: f.occurrence or 1)
                    cluster.representative_face_id = best.id
                    cluster.representative_snapshot = best.snapshot_url
                    repaired = True
                if repaired:
                    cluster.updated_at = datetime.now(timezone.utc)
                    stats["clusters_repaired"] += 1
            db.commit()
            if any(stats.values()):
                logger.info(
                    f"Cluster reconcile: removed {stats['clusters_removed']} empty clusters "
                    f"({stats['candidates_removed']} candidates), repaired {stats['clusters_repaired']}"
                )
            return stats
        except Exception as e:
            db.rollback()
            logger.error(f"Cluster reconcile failed: {e}", exc_info=True)
            return stats
        finally:
            db.close()

    @staticmethod
    def get_cluster_members(cluster_id: str) -> List[Dict[str, Any]]:
        """Get all unknown faces in a cluster."""
        db = SessionLocal()
        try:
            faces = (
                db.query(UnknownFace)
                .filter(UnknownFace.cluster_id == cluster_id)
                .order_by(UnknownFace.last_seen.desc())
                .all()
            )
            return [
                {
                    "id": f.id,
                    "snapshot_url": f.snapshot_url,
                    "full_frame": f.full_frame,
                    "occurrence": f.occurrence,
                    "first_seen": f.first_seen.isoformat() if f.first_seen else None,
                    "last_seen": f.last_seen.isoformat() if f.last_seen else None,
                    "camera_id": f.camera_id,
                }
                for f in faces
            ]
        finally:
            db.close()

    @staticmethod
    def get_cluster_centroid(cluster_id: str) -> Optional[np.ndarray]:
        """Get the centroid embedding for a cluster."""
        db = SessionLocal()
        try:
            cluster = (
                db.query(UnknownCluster)
                .filter(UnknownCluster.id == cluster_id)
                .first()
            )
            if cluster and cluster.embedding_centroid:
                return deserialize_encoding(cluster.embedding_centroid)
            return None
        finally:
            db.close()
