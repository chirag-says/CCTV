"""
Seed script — Register a student into the database.
"""
import sys, os, cv2, numpy as np, shutil
sys.path.insert(0, ".")

from uuid import uuid4
from datetime import datetime, timezone
from app.db.session import SessionLocal
from app.db.models import Person, FaceEncoding
from app.config import settings

STUDENTS = [
    {
        "filename": "JitinRaju.png",
        "name": "Jitin Raju",
        "role": "student",
        "department": "CSE",
        "email": "jitinraju@atria.edu",
    },
]

STUDENTS_DIR = r"c:\Users\lokes\OneDrive\Desktop\CCTV\CCTV\Student"


def load_insightface():
    from app.vision.detector import _get_insightface_app
    app = _get_insightface_app()
    if app is None:
        print("ERROR: Could not load insightface model.")
        sys.exit(1)
    return app


def seed_students():
    print("=" * 60)
    print("  SentinelAI — Student Seeder")
    print("=" * 60)

    print("Loading insightface (SCRFD + ArcFace)...")
    app = load_insightface()
    print("  ✅ insightface loaded\n")

    db = SessionLocal()
    created = 0

    try:
        for student in STUDENTS:
            filepath = os.path.join(STUDENTS_DIR, student["filename"])
            print(f"── {student['name']} ({student['role']}) ──")

            if not os.path.exists(filepath):
                print(f"  ❌ Photo not found: {filepath}")
                continue

            # Check if already exists
            existing = (
                db.query(Person)
                .filter(Person.name == student["name"], Person.is_active == True)
                .first()
            )
            if existing:
                print(f"  ⏭️  Already exists (id={existing.id[:12]}...), skipping")
                continue

            image = cv2.imread(filepath)
            if image is None:
                print(f"  ❌ Failed to read image")
                continue

            # Pass BGR directly — InsightFace handles BGR→RGB internally
            faces = app.get(image)

            if not faces:
                print(f"  ❌ No face detected")
                continue

            if len(faces) > 1:
                print(f"  ⚠️  Multiple faces ({len(faces)}), using largest")
                faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)

            face = faces[0]
            if face.embedding is None:
                print(f"  ❌ No embedding generated")
                continue

            encoding = face.embedding
            print(f"  📐 Embedding: shape={encoding.shape}, norm={np.linalg.norm(encoding):.2f}")

            # Save avatar
            avatar_filename = f"student_{student['name'].replace(' ', '_')}.jpg"
            avatar_dir = os.path.join(settings.SNAPSHOT_DIR, "avatars")
            os.makedirs(avatar_dir, exist_ok=True)
            avatar_path = os.path.join(avatar_dir, avatar_filename)
            cv2.imwrite(avatar_path, image)
            avatar_url = f"/snapshots/avatars/{avatar_filename}"

            # Create Person
            person_id = str(uuid4())
            person = Person(
                id=person_id,
                name=student["name"],
                role=student["role"],
                department=student.get("department", "CSE"),
                email=student.get("email"),
                avatar_url=avatar_url,
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(person)
            db.flush()

            # Create FaceEncoding
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
                source_image=student["filename"],
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

            print(f"  ✅ Created: {student['name']} → id={person_id[:12]}...")
            print(f"     Role: {student['role']}, Dept: {student.get('department', 'CSE')}")
            print(f"     Avatar: {avatar_url}")
            created += 1

        db.commit()
        print(f"\n{'='*60}")
        print(f"  RESULTS: {created} student(s) created")
        print(f"{'='*60}")

    except Exception as e:
        db.rollback()
        print(f"\n  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    seed_students()
