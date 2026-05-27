"""Debug script to check face encoding state in the database."""
import sys
import pickle
import numpy as np

sys.path.insert(0, ".")
from app.db.session import SessionLocal
from app.db.models import Person, FaceEncoding

db = SessionLocal()

# Check persons
persons = db.query(Person).filter(Person.is_active == True).all()
print(f"Active persons: {len(persons)}")
for p in persons:
    print(f"  Person: name={p.name}, role={p.role}, id={p.id[:12]}...")

# Check encodings
encs = db.query(FaceEncoding).all()
print(f"\nTotal face encodings in DB: {len(encs)}")

for e in encs:
    try:
        arr = pickle.loads(e.encoding_data)
        print(f"  Encoding: person_id={e.person_id[:12]}..., shape={arr.shape}, dtype={arr.dtype}, norm={np.linalg.norm(arr):.4f}")
    except Exception as ex:
        print(f"  Encoding: person_id={e.person_id[:12]}..., ERROR decoding: {ex}")

# Check what get_all_encodings returns
from app.services.person_service import PersonService
all_enc = PersonService.get_all_encodings()
print(f"\nPersonService.get_all_encodings() returned: {len(all_enc)} encodings")
for ae in all_enc:
    enc = ae["encoding"]
    print(f"  name={ae['person_name']}, person_id={ae['person_id'][:12]}..., shape={enc.shape}, norm={np.linalg.norm(enc):.4f}")

# Simulate recognition
if len(all_enc) > 0:
    from app.vision.recognizer import FaceRecognizer
    recognizer = FaceRecognizer()
    recognizer.load_encodings(all_enc)
    print(f"\nRecognizer loaded {recognizer.known_count} encodings")
    
    # Try matching each encoding against the cache (self-matching test)
    for ae in all_enc:
        result = recognizer.recognize_face(ae["encoding"])
        status = "MATCH" if result.matched else "NO MATCH"
        print(f"  Self-test {ae['person_name']}: {status}, distance={result.distance}, confidence={result.confidence}")

db.close()
print("\nDone.")
