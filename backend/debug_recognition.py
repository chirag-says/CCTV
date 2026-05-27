"""
Debug script — Test face recognition for Pavithra against stored encodings.
Simulates what the pipeline does.
"""
import sys
sys.path.insert(0, ".")

import pickle
import numpy as np
from app.db.session import SessionLocal
from app.db.models import Person, FaceEncoding
from app.vision.recognizer import FaceRecognizer
from app.services.person_service import PersonService
from app.config import settings

print("=" * 60)
print("  Face Recognition Debug")
print("=" * 60)

# 1. Load all encodings
all_encodings = PersonService.get_all_encodings()
print(f"\n✅ Total encodings from DB: {len(all_encodings)}")
for enc in all_encodings:
    print(f"   {enc['person_name']}: shape={enc['encoding'].shape}, "
          f"norm={np.linalg.norm(enc['encoding']):.4f}")

# 2. Create recognizer and load encodings
recognizer = FaceRecognizer(tolerance=settings.FACE_MATCH_TOLERANCE)
recognizer.load_encodings(all_encodings)
print(f"\n✅ Recognizer loaded: {recognizer.known_count} known faces")
print(f"   Tolerance: {settings.FACE_MATCH_TOLERANCE}")

# 3. Test Pavithra's stored encoding against itself (should match perfectly)
print("\n" + "=" * 60)
print("  Test 1: Pavithra's stored encoding vs all known faces")
print("=" * 60)

db = SessionLocal()
pavithra_person = db.query(Person).filter(Person.name == "Pavithra S").first()
if pavithra_person:
    pavithra_enc = db.query(FaceEncoding).filter(
        FaceEncoding.person_id == pavithra_person.id
    ).first()
    if pavithra_enc:
        pavithra_embedding = pickle.loads(pavithra_enc.encoding_data)
        print(f"  Pavithra embedding: shape={pavithra_embedding.shape}, "
              f"norm={np.linalg.norm(pavithra_embedding):.4f}")
        
        result = recognizer.recognize_face(pavithra_embedding)
        print(f"\n  Result: matched={result.matched}")
        print(f"  Person: {result.person_name}")
        print(f"  Distance: {result.distance:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")
        
        # Also show distance to ALL known faces
        print(f"\n  Distances to ALL known faces:")
        norm_q = pavithra_embedding / np.linalg.norm(pavithra_embedding)
        for enc in all_encodings:
            norm_k = enc['encoding'] / np.linalg.norm(enc['encoding'])
            sim = np.dot(norm_q, norm_k)
            dist = 1.0 - sim
            marker = "✅" if dist <= settings.FACE_MATCH_TOLERANCE else "❌"
            print(f"    {marker} {enc['person_name']:20s}  distance={dist:.4f}  "
                  f"similarity={sim:.4f}  (threshold={settings.FACE_MATCH_TOLERANCE})")
else:
    print("  ❌ Pavithra not found in database!")
db.close()

# 4. Try loading Pavithra's image and generating a live embedding
print("\n" + "=" * 60)
print("  Test 2: Fresh encoding from Pavithra's photo")
print("=" * 60)

import cv2
import os

teachers_dir = r"c:\Users\lokes\OneDrive\Desktop\CCTV\CCTV\Teachers"
pavithra_photo = os.path.join(teachers_dir, "Pavithra.png")

if os.path.exists(pavithra_photo):
    image = cv2.imread(pavithra_photo)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    from app.vision.detector import _get_insightface_app
    app = _get_insightface_app()
    
    faces = app.get(rgb)
    print(f"  Faces detected in photo: {len(faces)}")
    
    if faces:
        live_embedding = faces[0].embedding
        print(f"  Live embedding: shape={live_embedding.shape}, "
              f"norm={np.linalg.norm(live_embedding):.4f}")
        
        result = recognizer.recognize_face(live_embedding)
        print(f"\n  Result: matched={result.matched}")
        print(f"  Person: {result.person_name}")
        print(f"  Distance: {result.distance:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")
        
        # Show distance to all
        norm_q = live_embedding / np.linalg.norm(live_embedding)
        print(f"\n  Distances to ALL known faces:")
        for enc in all_encodings:
            norm_k = enc['encoding'] / np.linalg.norm(enc['encoding'])
            sim = np.dot(norm_q, norm_k)
            dist = 1.0 - sim
            marker = "✅" if dist <= settings.FACE_MATCH_TOLERANCE else "❌"
            print(f"    {marker} {enc['person_name']:20s}  distance={dist:.4f}  "
                  f"similarity={sim:.4f}")
else:
    print(f"  ❌ Photo not found: {pavithra_photo}")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
