"""
Embedding Migration: Recompute all face embeddings with correct color order.

Background:
    InsightFace expects BGR input — it does swapRB=True internally in both
    SCRFD and ArcFace. The pipeline was converting BGR→RGB before calling
    app.get(), which double-swapped the channels. All stored embeddings were
    computed on color-inverted images.

    This script recomputes embeddings from the original source images with
    correct BGR input. If source images are missing, the person is flagged
    for manual re-enrollment.

Usage:
    # Dry run (report only, no changes):
    python migrate_embeddings.py --dry-run

    # Full migration:
    python migrate_embeddings.py

    # Verify migration:
    python migrate_embeddings.py --verify
"""

import os
import sys
import argparse
import logging
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.db.session import get_session
from app.db.models import FaceEncoding, Person
from app.vision.recognizer import serialize_encoding

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_insightface_app():
    """Load InsightFace with correct BGR input (no color conversion)."""
    from insightface.app import FaceAnalysis

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    app = FaceAnalysis(
        name=settings.INSIGHTFACE_MODEL_NAME,
        providers=providers,
    )
    app.prepare(
        ctx_id=0,
        det_size=(settings.INSIGHTFACE_DET_SIZE, settings.INSIGHTFACE_DET_SIZE),
        det_thresh=settings.INSIGHTFACE_DET_THRESH,
    )

    # Verify CUDA
    for model_key in ['detection', 'recognition']:
        model = app.models.get(model_key)
        if model and hasattr(model, 'session'):
            active = model.session.get_providers()
            logger.info(f"  {model_key} providers: {active}")

    return app


def find_source_image(source_image_path: str) -> str | None:
    """Try to locate the source image on disk."""
    if not source_image_path:
        return None

    # Try as-is
    if os.path.exists(source_image_path):
        return source_image_path

    # Try relative to common locations
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots"),
    ]

    basename = os.path.basename(source_image_path)
    for search_dir in search_dirs:
        candidate = os.path.join(search_dir, basename)
        if os.path.exists(candidate):
            return candidate

        # Walk subdirectories
        for root, dirs, files in os.walk(search_dir):
            if basename in files:
                return os.path.join(root, basename)

    return None


def migrate_encoding(app, encoding_row: FaceEncoding, dry_run: bool = False) -> dict:
    """
    Recompute a single encoding from its source image.

    Returns dict with status:
        - "migrated": successfully recomputed
        - "missing_image": source image not found
        - "no_face": no face detected in source image
        - "error": unexpected error
    """
    result = {
        "encoding_id": encoding_row.id,
        "person_id": encoding_row.person_id,
        "source_image": encoding_row.source_image,
        "status": None,
    }

    # Find source image
    image_path = find_source_image(encoding_row.source_image)
    if image_path is None:
        result["status"] = "missing_image"
        return result

    try:
        # Read image (BGR from OpenCV)
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            result["status"] = "error"
            result["detail"] = f"cv2.imread failed for {image_path}"
            return result

        # Run InsightFace with correct BGR input — NO color conversion
        faces = app.get(image_bgr)

        if not faces:
            result["status"] = "no_face"
            return result

        # Use the first face (enrollment images should have exactly one)
        face = faces[0]
        if face.embedding is None:
            result["status"] = "no_face"
            result["detail"] = "Face detected but no embedding generated"
            return result

        new_embedding = face.embedding

        if not dry_run:
            # Store the V2 embedding
            encoding_row.encoding_data_v2 = serialize_encoding(new_embedding)
            encoding_row.embedding_version = 2

        # Compute cosine distance between old and new for reporting
        from app.vision.recognizer import deserialize_encoding
        old_embedding = deserialize_encoding(encoding_row.encoding_data)
        if old_embedding is not None:
            old_norm = old_embedding / (np.linalg.norm(old_embedding) + 1e-10)
            new_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-10)
            cosine_sim = float(np.dot(old_norm, new_norm))
            result["old_vs_new_similarity"] = round(cosine_sim, 4)

        result["status"] = "migrated"
        return result

    except Exception as e:
        result["status"] = "error"
        result["detail"] = str(e)
        return result


def run_migration(dry_run: bool = False, verify: bool = False):
    """Run the full embedding migration."""
    logger.info("=" * 70)
    logger.info("Embedding Migration: V1 (color-swapped) → V2 (correct BGR)")
    logger.info(f"  Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
    logger.info("=" * 70)

    # Load InsightFace
    logger.info("Loading InsightFace model...")
    app = get_insightface_app()

    # Query all encodings
    session = next(get_session())
    try:
        if verify:
            # Only check V2 encodings
            encodings = session.query(FaceEncoding).filter(
                FaceEncoding.embedding_version == 2
            ).all()
            logger.info(f"Verifying {len(encodings)} V2 encodings...")
        else:
            # Migrate all V1 encodings
            encodings = session.query(FaceEncoding).filter(
                FaceEncoding.embedding_version < 2
            ).all()
            logger.info(f"Found {len(encodings)} V1 encodings to migrate")

        # Stats
        stats = {"migrated": 0, "missing_image": 0, "no_face": 0, "error": 0}
        missing_persons = set()
        similarities = []

        for i, enc in enumerate(encodings):
            result = migrate_encoding(app, enc, dry_run=dry_run)
            stats[result["status"]] += 1

            if result["status"] == "missing_image":
                missing_persons.add(result["person_id"])

            if "old_vs_new_similarity" in result:
                similarities.append(result["old_vs_new_similarity"])

            # Progress
            if (i + 1) % 10 == 0 or i == len(encodings) - 1:
                logger.info(f"  Progress: {i+1}/{len(encodings)} "
                           f"(migrated={stats['migrated']}, "
                           f"missing={stats['missing_image']}, "
                           f"errors={stats['error']})")

        if not dry_run and not verify:
            session.commit()
            logger.info("Migration committed to database.")
        elif dry_run:
            session.rollback()
            logger.info("Dry run complete — no changes made.")

        # Report
        logger.info("\n" + "=" * 70)
        logger.info("Migration Report")
        logger.info("=" * 70)
        logger.info(f"  Total encodings:     {len(encodings)}")
        logger.info(f"  Successfully migrated: {stats['migrated']}")
        logger.info(f"  Missing source image:  {stats['missing_image']}")
        logger.info(f"  No face detected:      {stats['no_face']}")
        logger.info(f"  Errors:                {stats['error']}")

        if similarities:
            logger.info(f"\n  Old↔New embedding similarity:")
            logger.info(f"    Mean:   {np.mean(similarities):.4f}")
            logger.info(f"    Median: {np.median(similarities):.4f}")
            logger.info(f"    Min:    {np.min(similarities):.4f}")
            logger.info(f"    Max:    {np.max(similarities):.4f}")
            logger.info(f"    (If mean ≈ 1.0, the color fix had minimal impact.)")
            logger.info(f"    (If mean < 0.9, the double-swap was significantly hurting accuracy.)")

        if missing_persons:
            logger.info(f"\n  ⚠️  {len(missing_persons)} person(s) need re-enrollment:")
            for pid in missing_persons:
                person = session.query(Person).filter_by(id=pid).first()
                name = person.name if person else "Unknown"
                logger.info(f"    - {name} (id={pid[:12]}...)")

    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate face embeddings to V2 (correct color order)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no changes")
    parser.add_argument("--verify", action="store_true", help="Verify existing V2 embeddings")
    args = parser.parse_args()

    run_migration(dry_run=args.dry_run, verify=args.verify)
