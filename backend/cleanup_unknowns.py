"""
Cleanup script — Deduplicate existing unknown faces in the database.
Compares all pending unknown face encodings and merges duplicates by keeping
the one with the best snapshot and summing occurrence counts.
"""
import sys, pickle
import numpy as np
sys.path.insert(0, ".")

from app.db.session import SessionLocal
from app.db.models import UnknownFace
from app.config import settings

def cleanup_unknown_faces():
    db = SessionLocal()
    
    try:
        # Get all pending faces with encodings
        pending = (
            db.query(UnknownFace)
            .filter(UnknownFace.status == "pending")
            .order_by(UnknownFace.first_seen.asc())
            .all()
        )
        
        print(f"Total pending unknown faces: {len(pending)}")
        
        # Separate into faces with and without encodings
        with_enc = []
        without_enc = []
        for face in pending:
            if face.face_encoding:
                try:
                    enc = pickle.loads(face.face_encoding)
                    norm = np.linalg.norm(enc)
                    if norm > 0:
                        with_enc.append((face, enc / norm))
                    else:
                        without_enc.append(face)
                except:
                    without_enc.append(face)
            else:
                without_enc.append(face)
        
        print(f"  With face encodings: {len(with_enc)}")
        print(f"  Body-only (no encoding): {len(without_enc)}")
        
        # Delete body-only unknowns (low value, can't be enrolled)
        for face in without_enc:
            db.delete(face)
        print(f"  Deleted {len(without_enc)} body-only entries")
        
        # Cluster faces by similarity
        threshold = settings.UNKNOWN_SIMILARITY_THRESHOLD  # 0.70
        used = set()
        clusters = []
        
        for i, (face_i, enc_i) in enumerate(with_enc):
            if face_i.id in used:
                continue
            
            cluster = [face_i]
            used.add(face_i.id)
            
            for j, (face_j, enc_j) in enumerate(with_enc):
                if face_j.id in used:
                    continue
                
                similarity = float(np.dot(enc_i, enc_j))
                distance = 1.0 - similarity
                
                if distance < threshold:
                    cluster.append(face_j)
                    used.add(face_j.id)
            
            clusters.append(cluster)
        
        print(f"\n  Found {len(clusters)} unique persons from {len(with_enc)} face entries")
        
        # For each cluster, keep the first entry and merge occurrence counts
        kept = 0
        deleted = 0
        for cluster in clusters:
            # Keep the first (earliest) entry
            primary = cluster[0]
            total_occ = sum(f.occurrence for f in cluster)
            primary.occurrence = total_occ
            
            # Update last_seen to the most recent
            latest = max(f.last_seen for f in cluster if f.last_seen)
            primary.last_seen = latest
            
            # Delete the rest
            for dup in cluster[1:]:
                db.delete(dup)
                deleted += 1
            
            kept += 1
        
        db.commit()
        print(f"\n  ✅ Kept {kept} unique entries, deleted {deleted} duplicates")
        print(f"  Total remaining pending: {db.query(UnknownFace).filter(UnknownFace.status == 'pending').count()}")
        
    except Exception as e:
        db.rollback()
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    cleanup_unknown_faces()
