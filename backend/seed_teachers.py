"""
Seed script — Register teachers from the Teachers/ folder into the database.

For each teacher:
  1. Creates a Person record (name, role/designation, department)
  2. Detects face in EACH photo using SCRFD (insightface)
  3. Generates ArcFace 512-d embedding per photo
  4. Stores multiple encodings in face_encodings table (multi-embedding recognition)
  5. Copies best photo as avatar snapshot
  6. Refreshes the recognition cache in all active pipelines
"""

import sys
import os

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import cv2
import numpy as np
import shutil
import glob
import re

sys.path.insert(0, ".")

from uuid import uuid4
from datetime import datetime, timezone

from app.db.session import SessionLocal
from app.db.models import Person, FaceEncoding
from app.config import settings

# ── Teacher metadata ────────────────────────────────────────────
# The Teachers/ folder has files named like "dileep (1).jpeg", "farhana (2).jpeg", etc.
# We group by first-name prefix and assign metadata here.
#
# Every enrolled teacher is CSE: that is the actual roster today, and the
# department-wise attendance rollup is only honest if it reflects it.
# Spreading these names across ECE/Mechanical/Civil made the dashboard look
# fuller, but it reported one-person departments that do not exist.
# These are the only faces available in Teachers/, so the roster is
# genuinely small -- no placeholder people are invented, because a
# fabricated teacher with no photo can never be recognised and would show up
# as a permanent, unexplained absence in every report.
#
# Chirag was first enrolled as a CSE student; his photos were copied from
# Student/ into Teachers/ when he moved onto the faculty roster, so this
# seeder can rebuild him from scratch on a clean database.
#
# `person_type` is what attendance and presence filter on; `role` is the
# human-facing designation. See app/migrations/009_college_features.sql.
TEACHERS = [
    {
        "prefix": "dileep",
        "name": "Dileep",
        "role": "hod",
        "person_type": "faculty",
        "department": "CSE",
        "email": "dileep@atria.edu",
    },
    {
        "prefix": "chirag",
        "name": "Chirag",
        "role": "assistant_professor",
        "person_type": "faculty",
        "department": "CSE",
        "email": "chirag@atria.edu",
    },
    {
        "prefix": "farhana",
        "name": "Farhana",
        "role": "assistant_professor",
        "person_type": "faculty",
        "department": "CSE",
        "email": "farhana@atria.edu",
    },
    {
        "prefix": "geetha",
        "name": "Geetha",
        "role": "associate_professor",
        "person_type": "faculty",
        "department": "CSE",
        "email": "geetha@atria.edu",
    },
    {
        "prefix": "niranjani",
        "name": "Niranjani",
        "role": "assistant_professor",
        "person_type": "faculty",
        "department": "CSE",
        "email": "niranjani@atria.edu",
    },
    {
        "prefix": "reena",
        "name": "Reena",
        "role": "professor",
        "person_type": "faculty",
        "department": "CSE",
        "email": "reena@atria.edu",
    },
    {
        "prefix": "rithvik",
        "name": "Rithvik",
        "role": "assistant_professor",
        "person_type": "faculty",
        "department": "CSE",
        "email": "rithvik@atria.edu",
    },
]

# Resolve Teachers directory relative to this script → ../Teachers
TEACHERS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Teachers")
)
if not os.path.isdir(TEACHERS_DIR):
    TEACHERS_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "Teachers")
    )
if not os.path.isdir(TEACHERS_DIR):
    TEACHERS_DIR = r"c:\Users\chira\Downloads\CCTV-COLLEGE\CCTV\Teachers"


def load_insightface():
    """Load the shared insightface app for face detection + embedding."""
    from app.vision.detector import _get_insightface_app
    app = _get_insightface_app()
    if app is None:
        print("ERROR: Could not load insightface model.")
        sys.exit(1)
    return app


def find_teacher_images(prefix: str) -> list:
    """
    Find all images for a teacher by filename prefix.
    Matches patterns like 'dileep (1).jpeg', 'dileep (2).jpeg', etc.
    """
    pattern = os.path.join(TEACHERS_DIR, f"{prefix} (*).jpeg")
    files = sorted(glob.glob(pattern))
    if not files:
        # Try without spaces, or other extensions
        pattern = os.path.join(TEACHERS_DIR, f"{prefix}*.jpeg")
        files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(TEACHERS_DIR, f"{prefix}*.jpg")
        files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(TEACHERS_DIR, f"{prefix}*.png")
        files = sorted(glob.glob(pattern))
    return files


def seed_teachers():
    print("=" * 60)
    print("  SentinelAI — Teacher Seeder (Multi-Embedding)")
    print("=" * 60)
    print(f"  Teachers dir : {TEACHERS_DIR}")
    print(f"  Teachers     : {len(TEACHERS)}")
    print()

    if not os.path.isdir(TEACHERS_DIR):
        print(f"  ❌ Teachers directory not found: {TEACHERS_DIR}")
        sys.exit(1)

    # List actual files in directory
    all_files = os.listdir(TEACHERS_DIR)
    print(f"  Files found  : {len(all_files)}")
    print()

    # Load insightface
    print("Loading insightface (SCRFD + ArcFace)...")
    app = load_insightface()
    print("  ✅ insightface loaded\n")

    db = SessionLocal()
    persons_created = 0
    encodings_created = 0
    skipped = 0
    errors = 0

    try:
        for teacher in TEACHERS:
            prefix = teacher["prefix"]
            images = find_teacher_images(prefix)

            print(f"── {teacher['name']} ({teacher['role']}) ──")
            print(f"   Images found: {len(images)}")

            if not images:
                print(f"  ❌ No images found for prefix '{prefix}'")
                errors += 1
                continue

            # Check if person already exists
            existing = (
                db.query(Person)
                .filter(Person.name == teacher["name"], Person.is_active == True)
                .first()
            )
            if existing:
                # Keep the (expensive) face encodings, but refresh the roster
                # metadata. Re-running the seeder after a department or
                # designation change should not require deleting and
                # re-enrolling the person.
                changes = []
                for field in ("role", "person_type", "department", "email"):
                    new_value = teacher.get(field)
                    if new_value and getattr(existing, field, None) != new_value:
                        changes.append(
                            f"{field}: {getattr(existing, field, None)} → {new_value}"
                        )
                        setattr(existing, field, new_value)
                if changes:
                    existing.updated_at = datetime.now(timezone.utc)
                    db.commit()
                    print(f"  ♻️  Already exists — updated {', '.join(changes)}")
                else:
                    print(f"  ⏭️  Already exists (id={existing.id[:12]}...), skipping")
                skipped += 1
                continue

            # Create Person record first
            person_id = str(uuid4())
            person = Person(
                id=person_id,
                name=teacher["name"],
                role=teacher["role"],
                person_type=teacher.get("person_type", "faculty"),
                department=teacher.get("department", "CSE"),
                email=teacher.get("email"),
                avatar_url=None,  # Will be set after processing images
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(person)
            db.flush()

            # Process each image → create one FaceEncoding per image
            best_quality = -1
            best_image = None
            enc_count = 0

            for img_path in images:
                filename = os.path.basename(img_path)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"     ❌ Failed to read: {filename}")
                    continue

                # Detect faces (BGR — InsightFace handles internally)
                faces = app.get(image)

                if not faces:
                    print(f"     ❌ No face detected in {filename}")
                    continue

                if len(faces) > 1:
                    print(f"     ⚠️  {len(faces)} faces in {filename}, using largest")
                    faces = sorted(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                        reverse=True,
                    )

                face = faces[0]

                if face.embedding is None:
                    print(f"     ❌ No embedding for {filename}")
                    continue

                encoding = face.embedding  # 512-d numpy array

                # Quality assessment
                from app.vision.recognizer import serialize_encoding
                from app.vision.quality import FaceQualityGate

                quality_gate = FaceQualityGate(min_blur=30.0, min_size=40)
                bbox_i = face.bbox.astype(int)
                face_crop = image[max(0, bbox_i[1]):bbox_i[3], max(0, bbox_i[0]):bbox_i[2]]
                landmarks = face.kps if hasattr(face, 'kps') and face.kps is not None else None
                qr = quality_gate.assess(
                    face_crop=face_crop,
                    face_location=(bbox_i[1], bbox_i[2], bbox_i[3], bbox_i[0]),
                    det_score=float(face.det_score) if hasattr(face, 'det_score') else 1.0,
                    landmarks=landmarks,
                )

                # Save face crop image to snapshots/faces/ for frontend display
                enc_id = str(uuid4())
                crop_filename = f"teacher_{teacher['prefix']}_{enc_count + 1}.jpg"
                crop_dir = os.path.join(settings.SNAPSHOT_DIR, "faces")
                os.makedirs(crop_dir, exist_ok=True)
                crop_path = os.path.join(crop_dir, crop_filename)
                cv2.imwrite(crop_path, face_crop)
                source_image_url = f"/snapshots/faces/{crop_filename}"

                # Create FaceEncoding record
                enc = FaceEncoding(
                    id=enc_id,
                    person_id=person_id,
                    encoding_data=serialize_encoding(encoding),
                    source_image=source_image_url,
                    quality=qr.quality_score,
                    yaw=qr.metrics.get('yaw'),
                    pitch=qr.metrics.get('pitch'),
                    roll=qr.metrics.get('roll'),
                    det_score=qr.metrics.get('det_score'),
                    blur_score=qr.metrics.get('blur_score'),
                    lighting=qr.metrics.get('lighting'),
                    model_version="arcface_r50",
                    created_at=datetime.now(timezone.utc),
                )
                db.add(enc)
                enc_count += 1

                norm = np.linalg.norm(encoding)
                print(f"     ✅ {filename}: quality={qr.quality_score:.2f}, "
                      f"norm={norm:.2f}, "
                      f"yaw={qr.metrics.get('yaw', 0):.1f}°, "
                      f"det={qr.metrics.get('det_score', 0):.2f}")

                # Track best quality image for avatar
                if qr.quality_score > best_quality:
                    best_quality = qr.quality_score
                    best_image = img_path

            # Save avatar from best quality image
            if best_image is not None:
                avatar_filename = f"teacher_{teacher['name'].replace(' ', '_')}.jpg"
                avatar_dir = os.path.join(settings.SNAPSHOT_DIR, "avatars")
                os.makedirs(avatar_dir, exist_ok=True)
                avatar_path = os.path.join(avatar_dir, avatar_filename)
                shutil.copy2(best_image, avatar_path)
                avatar_url = f"/snapshots/avatars/{avatar_filename}"
                person.avatar_url = avatar_url
                print(f"   📷 Avatar: {avatar_url} (quality={best_quality:.2f})")

            if enc_count > 0:
                print(f"   ✅ Created {teacher['name']}: {enc_count} embeddings → id={person_id[:12]}...")
                persons_created += 1
                encodings_created += enc_count
            else:
                print(f"   ❌ No valid embeddings for {teacher['name']}, removing person record")
                db.delete(person)
                errors += 1

        db.commit()
        print()
        print("=" * 60)
        print(f"  RESULTS")
        print(f"    Persons created   : {persons_created}")
        print(f"    Embeddings stored  : {encodings_created}")
        print(f"    Skipped (existing) : {skipped}")
        print(f"    Errors             : {errors}")
        print("=" * 60)

        # Refresh encoding cache in active pipelines
        if encodings_created > 0:
            try:
                from app.vision.camera_worker import camera_manager
                from app.services.person_service import PersonService

                all_encodings = PersonService.get_all_encodings()
                camera_manager.load_encodings_all(all_encodings)
                print(f"\n  🔄 Refreshed {len(all_encodings)} total encodings in recognition cache")
            except Exception as e:
                print(f"\n  ⚠️  Could not refresh cache (server may not be running): {e}")
                print("     Encodings will be loaded automatically when a camera starts.")

    except Exception as e:
        db.rollback()
        print(f"\n  ❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_teachers()
