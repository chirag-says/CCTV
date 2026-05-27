"""Quick test of insightface face detection."""
import cv2
import numpy as np
import sys

# Test with a frame from the video
video_path = r"training\videos\raw\Security _Gate_Inside_20260408193138_20260408193445.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    # Try other video
    import glob
    vids = glob.glob("training/videos/raw/*.mp4")
    if vids:
        video_path = vids[0]
        cap = cv2.VideoCapture(video_path)
        print(f"Using: {video_path}")

# Skip to a frame with people
cap.set(cv2.CAP_PROP_POS_FRAMES, 125)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read frame")
    sys.exit(1)

print(f"Frame shape: {frame.shape}")
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Test insightface
try:
    from insightface.app import FaceAnalysis
    print(f"InsightFace imported OK")
    
    app = FaceAnalysis(name="buffalo_l")
    print("FaceAnalysis created")
    
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
    print("Model prepared")
    
    # Scale down like the detector does
    scale = 0.5
    small = cv2.resize(rgb, (0, 0), fx=scale, fy=scale)
    print(f"Small frame: {small.shape}")
    
    faces = app.get(small)
    print(f"Faces detected: {len(faces)}")
    
    for i, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        print(f"  Face {i}: bbox={bbox}, det_score={face.det_score:.3f}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
