"""
Seed script — Register teachers from the Teachers/ folder into the database.

For each teacher:
  1. Creates a Person record (name, role/designation, department)
  2. Detects face in the photo using SCRFD (insightface)
  3. Generates ArcFace 512-d embedding
  4. Stores the encoding in face_encodings table
  5. Copies the photo as avatar snapshot
  6. Refreshes the recognition cache in all active pipelines
"""

import sys
import os
import cv2
import numpy as np
import shutil

sys.path.insert(0, ".")

from uuid import uuid4
from datetime import datetime, timezone

from app.db.session import SessionLocal
from app.db.models import Person, FaceEncoding
from app.config import settings

# ── Teacher data from the Atria IT CSE department page ──────────────
# Matched by filename → name/designation from the screenshots
TEACHERS = [
    {
        "filename": "DeviKannan.png",
        "name": "Dr. Devi Kannan",
        "role": "hod",           # HOD + Professor
        "department": "CSE",
        "email": "devikannan@atria.edu",
    },
    {
        "filename": "VijaySwaroop.png",
        "name": "A. Vijay Swaroop",
        "role": "assistant_professor",
        "department": "CSE",
        "email": "vijayswaroop@atria.edu",
    },
    {
        "filename": "Keerthi.png",
        "name": "Keerthi K S",
        "role": "assistant_professor",
        "department": "CSE",
        "email": "keerthiks@atria.edu",
    },
    {
        "filename": "Pavithra.png",
        "name": "Pavithra S",
        "role": "assistant_professor",
        "department": "CSE",
        "email": "pavithras@atria.edu",
    },
    {
        "filename": "Veena.png",
        "name": "Veena S",
        "role": "assistant_professor",
        "department": "CSE",
        "email": "veenas@atria.edu",
    },
    {
        "filename": "Vijaylaxmi.png",
        "name": "V. Vijayalakshmi",
        "role": "assistant_professor",
        "department": "CSE",
        "email": "vijayalakshmi@atria.edu",
    },
]

TEACHERS_DIR = os.path.join(os.path.dirname(__file__), "..", "Teachers")
TEACHERS_DIR = os.path.normpath(TEACHERS_DIR)

# Fallback: if Teachers dir is at project root level
if not os.path.isdir(TEACHERS_DIR):
    TEACHERS_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "Teachers")
    )
if not os.path.isdir(TEACHERS_DIR):
    TEACHERS_DIR = r"c:\Users\lokes\OneDrive\Desktop\CCTV\CCTV\Teachers"


def load_insightface():
    """Load the shared insightface app for face detection + embedding."""
    from app.vision.detector import _get_insightface_app
    app = _get_insightface_app()
    if app is None:
        print("ERROR: Could not load insightface model.")
        sys.exit(1)
    return app


def seed_teachers():
    print("=" * 60)
    print("  SentinelAI — Teacher Seeder")
    print("=" * 60)
    print(f"  Teachers dir: {TEACHERS_DIR}")
    print(f"  Teachers to seed: {len(TEACHERS)}")
    print()

    # Load insightface
    print("Loading insightface (SCRFD + ArcFace)...")
    app = load_insightface()
    print("  ✅ insightface loaded\n")

    db = SessionLocal()
    created = 0
    skipped = 0
    errors = 0

    try:
        for teacher in TEACHERS:
            filename = teacher["filename"]
            filepath = os.path.join(TEACHERS_DIR, filename)

            print(f"── {teacher['name']} ({teacher['role']}) ──")

            if not os.path.exists(filepath):
                print(f"  ❌ Photo not found: {filepath}")
                errors += 1
                continue

            # Check if already exists
            existing = (
                db.query(Person)
                .filter(Person.name == teacher["name"], Person.is_active == True)
                .first()
            )
            if existing:
                print(f"  ⏭️  Already exists (id={existing.id[:12]}...), skipping")
                skipped += 1
                continue

            # Read and process image
            image = cv2.imread(filepath)
            if image is None:
                print(f"  ❌ Failed to read image: {filepath}")
                errors += 1
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect face and get embedding
            faces = app.get(rgb_image)

            if not faces:
                print(f"  ❌ No face detected in {filename}")
                errors += 1
                continue

            if len(faces) > 1:
                print(f"  ⚠️  Multiple faces ({len(faces)}) detected, using largest")
                # Use the face with the largest bounding box
                faces = sorted(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    reverse=True,
                )

            face = faces[0]

            if face.embedding is None:
                print(f"  ❌ No embedding generated for {filename}")
                errors += 1
                continue

            encoding = face.embedding  # 512-d numpy array
            print(f"  📐 Embedding: shape={encoding.shape}, norm={np.linalg.norm(encoding):.2f}")

            # Save avatar snapshot
            avatar_filename = f"teacher_{teacher['name'].replace(' ', '_').replace('.', '')}.jpg"
            avatar_dir = os.path.join(settings.SNAPSHOT_DIR, "avatars")
            os.makedirs(avatar_dir, exist_ok=True)
            avatar_path = os.path.join(avatar_dir, avatar_filename)
            cv2.imwrite(avatar_path, image)
            avatar_url = f"/snapshots/avatars/{avatar_filename}"

            # Create Person record
            person_id = str(uuid4())
            person = Person(
                id=person_id,
                name=teacher["name"],
                role=teacher["role"],
                department=teacher.get("department", "CSE"),
                email=teacher.get("email"),
                avatar_url=avatar_url,
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(person)
            db.flush()  # Get person into session before FK reference

            # Create FaceEncoding record
            from app.vision.recognizer import serialize_encoding
            from app.vision.quality import FaceQualityGate

            # Assess quality (relaxed for studio portraits)
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

            enc = FaceEncoding(
                id=str(uuid4()),
                person_id=person_id,
                encoding_data=serialize_encoding(encoding),  # raw float32 bytes
                source_image=filename,
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

            print(f"  ✅ Created: {teacher['name']} → id={person_id[:12]}...")
            print(f"     Role: {teacher['role']}, Dept: {teacher.get('department', 'CSE')}")
            print(f"     Avatar: {avatar_url}")
            created += 1

        db.commit()
        print()
        print("=" * 60)
        print(f"  RESULTS: {created} created, {skipped} skipped, {errors} errors")
        print("=" * 60)

        # Refresh encoding cache in active pipelines
        if created > 0:
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
