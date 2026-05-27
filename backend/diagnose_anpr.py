"""
ANPR Diagnostic Script — Tests every stage of plate detection pipeline.
Run: py diagnose_anpr.py
"""
import sys, os, types, warnings, logging, time
import cv2
import numpy as np

# Patch for torchvision compatibility
from torchvision.transforms import functional as F
sys.modules['torchvision.transforms.functional_tensor'] = types.ModuleType('torchvision.transforms.functional_tensor')
sys.modules['torchvision.transforms.functional_tensor'].rgb_to_grayscale = F.rgb_to_grayscale
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG, format='%(name)s | %(message)s')
logging.getLogger("ultralytics").setLevel(logging.WARNING)

VIDEO = r'C:\Users\lokes\OneDrive\Desktop\CCTV\CCTV\detet number plate.mp4'

# ── Check video file exists ──────────────────────────────────────────
if not os.path.exists(VIDEO):
    # Try alternate locations
    for alt in [
        r'C:\Users\lokes\OneDrive\Desktop\CCTV\detet number plate.mp4',
        r'C:\Users\lokes\OneDrive\Desktop\detet number plate.mp4',
        os.path.join(os.path.dirname(__file__), 'uploads', 'videos'),
    ]:
        if os.path.isdir(alt):
            for f in os.listdir(alt):
                if 'number' in f.lower() or 'plate' in f.lower():
                    VIDEO = os.path.join(alt, f)
                    break
        elif os.path.exists(alt):
            VIDEO = alt
            break

print("=" * 70)
print("  ANPR DIAGNOSTIC — Step-by-Step Pipeline Test")
print("=" * 70)

# ── Step 1: Check model file ─────────────────────────────────────────
model_path = "models/license-plate-finetune-v1l.pt"
print(f"\n📁 Step 1: Model file check")
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"   ✅ Found: {model_path} ({size_mb:.1f} MB)")
else:
    print(f"   ❌ NOT FOUND: {model_path}")
    # Check what's actually in models/
    if os.path.exists("models"):
        print(f"   📂 models/ contains: {os.listdir('models')}")
    sys.exit(1)

# ── Step 2: Load model ───────────────────────────────────────────────
print(f"\n🔧 Step 2: Loading YOLO plate model...")
from ultralytics import YOLO
plate_model = YOLO(model_path)
print(f"   ✅ Model loaded")
print(f"   Classes: {plate_model.names}")
print(f"   Num classes: {len(plate_model.names)}")
print(f"   Task: {plate_model.task}")

# ── Step 3: Load vehicle model ───────────────────────────────────────
print(f"\n🔧 Step 3: Loading YOLO vehicle model (yolov8m.pt)...")
vehicle_model = YOLO("yolov8m.pt")
print(f"   ✅ Vehicle model loaded")

# ── Step 4: Open video ───────────────────────────────────────────────
print(f"\n📹 Step 4: Opening video: {VIDEO}")
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"   ❌ Cannot open video!")
    # Try to find the video in uploads
    uploads_dir = os.path.join("uploads", "videos")
    if os.path.exists(uploads_dir):
        files = os.listdir(uploads_dir)
        print(f"   📂 uploads/videos/ contains: {files}")
        for f in files:
            if f.endswith('.mp4'):
                VIDEO = os.path.join(uploads_dir, f)
                print(f"   🔄 Trying: {VIDEO}")
                cap = cv2.VideoCapture(VIDEO)
                if cap.isOpened():
                    break
    if not cap.isOpened():
        sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"   ✅ {w}x{h} @ {fps:.1f}fps, {total} frames ({total/fps:.1f}s)")

# ── Step 5: Test on every 5th frame ──────────────────────────────────
print(f"\n🔍 Step 5: Processing frames...")
print(f"   Pipeline: Full frame → vehicle detection → vehicle crop → plate detection → OCR")

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
frame_count = 0
total_vehicles = 0
total_plate_dets = 0
total_ocr_reads = 0

# Also try plate detection on full frame (not just vehicle crops)
total_full_frame_dets = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 5 != 0:
        continue

    print(f"\n  ── Frame {frame_count}/{total} ──")

    # Step 5a: Detect vehicles
    veh_results = vehicle_model(frame, conf=0.25, classes=list(VEHICLE_CLASSES.keys()), verbose=False)
    vehicles = []
    for r in veh_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
            vtype = VEHICLE_CLASSES.get(cls_id, "?")
            vehicles.append((x1, y1, x2, y2, vtype, conf))
    
    print(f"  Stage 1 - Vehicles: {len(vehicles)}")
    total_vehicles += len(vehicles)

    # Step 5b: Try plate detection on FULL FRAME first
    full_results = plate_model(frame, conf=0.15, verbose=False)
    full_frame_plates = 0
    for r in full_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            full_frame_plates += 1
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
            cls_name = plate_model.names.get(cls_id, f"cls{cls_id}")
            print(f"  🔲 FULL-FRAME plate det: class={cls_name}({cls_id}) conf={conf:.3f} box=({x1},{y1},{x2},{y2})")
    
    total_full_frame_dets += full_frame_plates
    if full_frame_plates == 0:
        print(f"  ⚠️  No plates detected on full frame (even at conf=0.15)")

    # Step 5c: Try plate detection on each vehicle crop
    for vx1, vy1, vx2, vy2, vtype, vconf in vehicles:
        crop = frame[vy1:vy2, vx1:vx2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            print(f"  Vehicle {vtype} ({vx1},{vy1},{vx2},{vy2}): crop too small, skipping")
            continue
        
        crop_h, crop_w = crop.shape[:2]
        crop_area = crop_h * crop_w
        
        plate_results = plate_model(crop, conf=0.15, verbose=False)
        
        for r in plate_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                total_plate_dets += 1
                pconf = float(box.conf[0])
                cls_id = int(box.cls[0])
                px1, py1, px2, py2 = [int(c) for c in box.xyxy[0].tolist()]
                pw = px2 - px1
                ph = py2 - py1
                parea = pw * ph
                ratio = parea / crop_area if crop_area > 0 else 0
                cls_name = plate_model.names.get(cls_id, f"cls{cls_id}")
                
                status = "✅"
                notes = ""
                if ratio > 0.80:
                    status = "❌ REJECTED (>80% of vehicle)"
                    notes = f" ratio={ratio*100:.0f}%"
                elif ratio > 0.40:
                    notes = f" ratio={ratio*100:.0f}% (WOULD HAVE BEEN REJECTED before fix)"
                elif pw < 15 or ph < 8:
                    status = "❌ REJECTED (too small)"
                
                print(f"  {status} Plate in {vtype}: class={cls_name}({cls_id}) conf={pconf:.3f} "
                      f"size={pw}x{ph} in {crop_w}x{crop_h} crop{notes}")
                
                # Try OCR if plate looks valid
                if ratio <= 0.80 and pw >= 15 and ph >= 8:
                    plate_crop = crop[py1:py2, px1:px2]
                    if plate_crop.size > 0:
                        try:
                            import easyocr
                            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                            ocr_results = reader.readtext(plate_crop)
                            if ocr_results:
                                total_ocr_reads += 1
                                texts = [r[1] for r in ocr_results]
                                confs = [r[2] for r in ocr_results]
                                merged = ''.join(t.upper().replace(' ', '') for t in texts)
                                print(f"    📝 OCR: '{merged}' (fragments: {texts}, confs: {[f'{c:.2f}' for c in confs]})")
                            else:
                                print(f"    📝 OCR: (no text detected)")
                        except Exception as e:
                            print(f"    📝 OCR error: {e}")

cap.release()

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  DIAGNOSTIC SUMMARY")
print(f"{'='*70}")
print(f"  Video: {VIDEO}")
print(f"  Frames processed: {frame_count // 5} (every 5th of {total})")
print(f"  Total vehicles detected (Stage 1): {total_vehicles}")
print(f"  Full-frame plate detections: {total_full_frame_dets}")
print(f"  In-crop plate detections (Stage 2): {total_plate_dets}")
print(f"  OCR readings (Stage 3): {total_ocr_reads}")
print(f"{'='*70}")

if total_full_frame_dets == 0 and total_plate_dets == 0:
    print(f"\n  ⚠️  CONCLUSION: Model detects ZERO plates.")
    print(f"     The model may not be suitable for this video content.")
    print(f"     Try testing with lower confidence: plate_model(frame, conf=0.05)")
    print(f"     Or try on a different test image with clearly visible plates.")
