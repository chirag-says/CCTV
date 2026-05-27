"""
Fast ANPR test on 4K clear footage.
Tests immediate-emit pipeline (EasyOCR only, no TrOCR).
Target: <400ms per processed frame.
"""
import sys, types, warnings, logging, time
from torchvision.transforms import functional as F
sys.modules['torchvision.transforms.functional_tensor'] = types.ModuleType('torchvision.transforms.functional_tensor')
sys.modules['torchvision.transforms.functional_tensor'].rgb_to_grayscale = F.rgb_to_grayscale
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

import cv2
from app.vision.anpr import PlateRecognizer

VIDEO = r'C:\Users\lokes\OneDrive\Desktop\CCTV\CCTV\14598470_2160_3840_30fps.mp4'

events = []
def on_plate(event):
    events.append(event)
    meta = event['metadata']
    print(f"  ✅ PLATE: {meta['plate']} | conf={meta['confidence']} | {meta['vehicle_type']}")

anpr = PlateRecognizer(
    yolo_model_path="models/license-plate-finetune-v1l.pt",
    frame_interval=5,
    plate_cooldown_sec=30.0,  # 30s = each vehicle emits only once
    on_plate=on_plate,
)

print("="*60)
print("  Fast ANPR Test: 4K Clear Footage")
print("="*60)

cap = cv2.VideoCapture(VIDEO)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

start = time.time()
frame_count = 0
frame_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    
    t0 = time.time()
    anpr._process(frame, "gate_cam_4k")
    dt = time.time() - t0
    frame_times.append(dt * 1000)
    
    if frame_count % 50 == 0:
        elapsed = time.time() - start
        avg_ms = sum(frame_times[-50:]) / min(50, len(frame_times[-50:]))
        print(f"  Frame {frame_count}/{total_frames} | "
              f"avg={avg_ms:.0f}ms/frame | "
              f"{len(events)} plates")

cap.release()
elapsed = time.time() - start

# Stats
processed_times = [t for t in frame_times if t > 5]  # only frames that were actually processed
print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  Frames: {frame_count}/{total_frames}")
print(f"  Wall time: {elapsed:.1f}s")
print(f"  Avg frame time: {sum(frame_times)/len(frame_times):.0f}ms")
if processed_times:
    print(f"  Avg PROCESSED frame: {sum(processed_times)/len(processed_times):.0f}ms")
    print(f"  Max processed frame: {max(processed_times):.0f}ms")
    print(f"  Frames < 500ms: {sum(1 for t in processed_times if t < 500)}/{len(processed_times)}")
print(f"  Plates detected: {len(events)}")

for i, e in enumerate(events):
    meta = e['metadata']
    print(f"\n  #{i+1}: {meta['plate']} (conf={meta['confidence']}, {meta['vehicle_type']})")
