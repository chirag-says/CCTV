"""Seed Mukesh Rao into the SentinelAI database with face encodings."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from app.services.person_service import PersonService
from app.vision.camera_worker import camera_manager

IMAGE_PATH = r"c:\Users\lokes\OneDrive\Desktop\CCTV\CCTV\Student\MukeshRao.png"

print("=" * 50)
print("  Seeding: Mukesh Rao")
print("=" * 50)

# 1. Load image
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"ERROR: Could not load {IMAGE_PATH}")
    sys.exit(1)
print(f"Image loaded: {img.shape}")

# 2. Extract face embedding with InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = app.get(rgb)
print(f"Faces detected: {len(faces)}")

if not faces:
    print("ERROR: No face found in image!")
    sys.exit(1)

face = faces[0]
embedding = face.normed_embedding  # 512-d ArcFace embedding
print(f"Embedding shape: {embedding.shape}, det_score: {face.det_score:.3f}")

# 3. Create person in DB
try:
    person = PersonService.create_person({
        "name": "Mukesh Rao",
        "role": "student",
        "department": "Computer Science",
    })
    person_id = person["id"]
    print(f"Created person: {person['name']} (ID: {person_id})")
except Exception as e:
    if "Failed" in str(e):
        # May already exist - try to find
        result = PersonService.list_persons(search="Mukesh Rao")
        if result["data"]:
            person_id = result["data"][0]["id"]
            print(f"Person already exists: {person_id}")
        else:
            raise
    else:
        raise

# 4. Store face encoding
enc = PersonService.add_face_encoding(
    person_id=person_id,
    encoding=embedding,
    source_image="MukeshRao.png",
    quality=float(face.det_score),
)
print(f"Encoding stored: {enc['id']}")

# 5. Reload encodings into active camera pipelines
encodings = PersonService.get_all_encodings()
camera_manager.load_encodings_all(encodings)
print(f"Reloaded {len(encodings)} encodings into active pipelines")

print()
print("Done! Mukesh Rao is now registered in the system.")
