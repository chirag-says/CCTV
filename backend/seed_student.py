"""
Seed script — Register students from the Student/ folder into the database.

For each student:
  1. Creates a Person record (name, role, department)
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

# ── Student metadata ────────────────────────────────────────────
# The Student/ folder has files named like "abhitha (1).jpeg", "abhitha (2).jpeg", etc.
# Chirag is seeded by seed_teachers.py -- he is CSE faculty, not a student.
# We group by first-name prefix and assign metadata here.
STUDENTS = [
    {
        "prefix": "abhitha",
        "name": "Abhitha",
        "role": "student",
        "department": "CSE",
        "email": "abhitha@atria.edu",
    },
]

# Resolve Student directory relative to this script → ../Student
STUDENTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Student")
)
if not os.path.isdir(STUDENTS_DIR):
    STUDENTS_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "Student")
    )
if not os.path.isdir(STUDENTS_DIR):
    STUDENTS_DIR = r"c:\Users\chira\Downloads\CCTV-COLLEGE\CCTV\Student"


def load_insightface():
    """Load the shared insightface app for face detection + embedding."""
    from app.vision.detector import _get_insightface_app
    app = _get_insightface_app()
    if app is None:
        print("ERROR: Could not load insightface model.")
        sys.exit(1)
    return app


def find_student_images(prefix: str) -> list:
    """
    Find all images for a student by filename prefix.
    Matches patterns like 'abhitha (1).jpeg', 'abhitha (2).jpeg', etc.
    """
    pattern = os.path.join(STUDENTS_DIR, f"{prefix} (*).jpeg")
    files = sorted(glob.glob(pattern))
    if not files:
        # Try without spaces, or other extensions
        pattern = os.path.join(STUDENTS_DIR, f"{prefix}*.jpeg")
        files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(STUDENTS_DIR, f"{prefix}*.jpg")
        files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(STUDENTS_DIR, f"{prefix}*.png")
        files = sorted(glob.glob(pattern))
    return files


def seed_students():
    print("=" * 60)
    print("  SentinelAI — Student Seeder (Multi-Embedding)")
    print("=" * 60)
    print(f"  Students dir : {STUDENTS_DIR}")
    print(f"  Students     : {len(STUDENTS)}")
    print()

    if not os.path.isdir(STUDENTS_DIR):
        print(f"  ❌ Students directory not found: {STUDENTS_DIR}")
        sys.exit(1)

    # List actual files in directory
    all_files = os.listdir(STUDENTS_DIR)
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
        for student in STUDENTS:
            prefix = student["prefix"]
            images = find_student_images(prefix)

            print(f"── {student['name']} ({student['role']}) ──")
            print(f"   Images found: {len(images)}")

            if not images:
                print(f"  ❌ No images found for prefix '{prefix}'")
                errors += 1
                continue

            # Check if person already exists
            existing = (
                db.query(Person)
                .filter(Person.name == student["name"], Person.is_active == True)
                .first()
            )
            if existing:
                print(f"  ⏭️  Already exists (id={existing.id[:12]}...), skipping")
                skipped += 1
                continue

            # Create Person record first
            person_id = str(uuid4())
            person = Person(
                id=person_id,
                name=student["name"],
                role=student["role"],
                person_type=student.get("person_type", "student"),
                department=student.get("department", "CSE"),
                email=student.get("email"),
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
                crop_filename = f"student_{student['prefix']}_{enc_count + 1}.jpg"
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
                avatar_filename = f"student_{student['name'].replace(' ', '_')}.jpg"
                avatar_dir = os.path.join(settings.SNAPSHOT_DIR, "avatars")
                os.makedirs(avatar_dir, exist_ok=True)
                avatar_path = os.path.join(avatar_dir, avatar_filename)
                shutil.copy2(best_image, avatar_path)
                avatar_url = f"/snapshots/avatars/{avatar_filename}"
                person.avatar_url = avatar_url
                print(f"   📷 Avatar: {avatar_url} (quality={best_quality:.2f})")

            if enc_count > 0:
                print(f"   ✅ Created {student['name']}: {enc_count} embeddings → id={person_id[:12]}...")
                persons_created += 1
                encodings_created += enc_count
            else:
                print(f"   ❌ No valid embeddings for {student['name']}, removing person record")
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
    seed_students()
