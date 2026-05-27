"""
Quick plate model test — tests if the new model can detect plates on the uploaded video.
Run: py quick_plate_test.py
"""
import cv2, os, sys
from ultralytics import YOLO

# Find the detet number plate.mp4 (~23MB file)
uploads = 'uploads/videos'
candidates = []
for f in os.listdir(uploads):
    if f.endswith('.mp4'):
        sz = os.path.getsize(os.path.join(uploads, f))
        if abs(sz - 23344322) < 5000:  # ~23MB = detet number plate.mp4
            candidates.append((f, os.path.getmtime(os.path.join(uploads, f))))

if not candidates:
    print("❌ Could not find the test video in uploads/videos/")
    sys.exit(1)

# Pick the most recent one
candidates.sort(key=lambda x: x[1], reverse=True)
video = os.path.join(uploads, candidates[0][0])
print(f"Video: {video}")

# Load plate model
model = YOLO('models/license-plate-finetune-v1l.pt')
print(f"Model classes: {model.names}")

# Extract frames from different parts of the video
cap = cv2.VideoCapture(video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w}x{h}, {total} frames")

test_frames = [total//4, total//2, total*3//4]
found_any = False

for frame_pos in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    if not ret:
        continue
    
    print(f"\n── Frame {frame_pos}/{total} ──")
    
    # Test at various confidence levels
    for conf in [0.25, 0.15, 0.10, 0.05, 0.01]:
        results = model(frame, conf=conf, verbose=False)
        count = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
        if count > 0:
            found_any = True
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        c = float(box.conf[0])
                        x1,y1,x2,y2 = [int(x) for x in box.xyxy[0].tolist()]
                        cls = int(box.cls[0])
                        print(f"  ✅ conf_thresh={conf}: plate at ({x1},{y1},{x2},{y2}) "
                              f"actual_conf={c:.3f} class={model.names[cls]}")
            break  # Found at this confidence, no need to go lower
        elif conf == 0.01:
            print(f"  ❌ NO plates found even at conf=0.01")

    # Also test with larger inference size (1280)
    results_1280 = model(frame, conf=0.10, imgsz=1280, verbose=False)
    count_1280 = sum(len(r.boxes) if r.boxes is not None else 0 for r in results_1280)
    if count_1280 > 0:
        print(f"  📐 With imgsz=1280: {count_1280} plates detected!")
        for r in results_1280:
            if r.boxes is not None:
                for box in r.boxes:
                    c = float(box.conf[0])
                    x1,y1,x2,y2 = [int(x) for x in box.xyxy[0].tolist()]
                    print(f"     plate at ({x1},{y1},{x2},{y2}) conf={c:.3f}")
    else:
        print(f"  📐 With imgsz=1280: still 0 plates")

cap.release()

if not found_any:
    print("\n⚠️  The model detected ZERO plates on this video.")
    print("    This model may not work well with Indian plates or this camera angle.")
    print("    Recommendation: Try a different model or train a custom one.")
else:
    print("\n✅ Model CAN detect plates — the issue is in the pipeline logic.")
