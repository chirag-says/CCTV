# ANPR Production Redesign — Part 1: Analysis & Detection Strategy

## 1. ROOT CAUSE ANALYSIS (Deep Dive)

### 1.1 Architectural Flaw: No Proper Tracker Coupling

The current `PlateRecognizer` has a homegrown IoU-based vehicle tracker (`_tracked_vehicles`) that is **completely decoupled** from `TrafficMonitor`'s detection. Two separate YOLO models run independently:
- `TrafficMonitor` → Roboflow/RF-DETR/YOLO for vehicle counting
- `PlateRecognizer` → yolov8m.pt for vehicle detection (again)

**Impact**: Double GPU load, inconsistent vehicle IDs between modules, wasted VRAM.

**Fix**: ANPR must consume vehicle detections FROM TrafficMonitor (the `predetected_vehicles` path exists but is underutilized).

### 1.2 Architectural Flaw: OCR Runs on Every Detection Frame

Current flow per frame:
```
frame → vehicle detect → plate detect → ESRGAN enhance → EasyOCR → emit
```

There is NO accumulation buffer that works reliably. The consensus voting system (`_add_vote`) exists in code but **is never called** — the main `_process()` method emits events directly in both Strategy A and Strategy B, bypassing `_add_vote()` entirely (lines 905-928 and 1001-1024).

**Impact**: Every single OCR read is emitted as an event. No temporal smoothing actually happens.

### 1.3 Architectural Flaw: ESRGAN is Catastrophically Slow

`RealESRGAN_x4plus` with `RRDBNet(num_block=23)` is a 16.7M parameter model. Running 4x upscale on every plate crop:
- ~200-500ms per crop on RTX 5070
- Blocks the single ANPR worker thread
- Loads the full model even when plates are already >200px wide

**Impact**: At 5 vehicles per frame, ESRGAN alone costs 1-2.5 seconds per frame.

### 1.4 Model Flaw: Plate Detector Quality

Current model: `license-plate-finetune-v1l.pt` (51MB, YOLOv11-L)
- Trained on `keremberke/license-plate-object-detection` — a dataset of ~250 images of mostly European/American plates
- Single class detection — cannot distinguish front vs rear plate
- Indian plates are narrower and have different aspect ratios
- Model misses >50% of plates at CCTV angles

### 1.5 Model Flaw: Vehicle Detection via COCO

Using yolov8m.pt with COCO classes {2:car, 3:motorcycle, 5:bus, 7:truck}:
- **No auto-rickshaw class** — India's most common campus vehicle
- **No tempo/LCV class**
- COCO "motorcycle" ≠ Indian two-wheelers (scooters, mopeds)
- COCO was trained on Western traffic — poor on Indian road conditions

### 1.6 OCR Pipeline Flaw: EasyOCR is Not Designed for Plates

EasyOCR is a general-purpose scene text reader. For license plates:
- It fragments Indian plates into 2-4 pieces ("KA", "53", "MK", "0449")
- Fragment merging via `_merge_ocr_fragments()` is fragile
- No character-level confidence for position-aware correction
- Slow (~150-300ms per crop on GPU)

### 1.7 OCR Pipeline Flaw: Over-Aggressive Indian Plate Correction

`_fix_indian_plate()` rejects ANY plate that doesn't match `^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$`. Problems:
- Rejects valid partial reads that could contribute to consensus
- The `_normalize_plate()` fallback accepts anything ≥6 chars — contradicting the strict filter
- OCR correction tables (`OCR_CHAR_CORRECTIONS`) can create phantom plates from noise

### 1.8 Deployment Flaw: Thread Pool Size = 1

```python
self._executor = ThreadPoolExecutor(max_workers=1)
```

With 150 cameras, a single worker means ANPR frames queue up and get stale. By the time a frame is processed, the vehicle may have left.

---

## 2. VEHICLE DETECTION REDESIGN

### 2.1 Model Selection: YOLOv8m COCO + Fine-tuned Indian Vehicle Head

**Recommendation**: Two-tier approach:

| Tier | Model | Purpose | Why |
|------|-------|---------|-----|
| Primary | **YOLOv8m.pt (COCO)** | General vehicle detection | Already loaded, good baseline, detects car/bus/truck/motorcycle |
| Secondary | **Fine-tuned YOLOv8s** | Indian-specific classes | Auto-rickshaw, tempo, scooter — classes COCO misses |

**Why NOT switch to YOLOv9/RT-DETR/RF-DETR for vehicles?**
- COCO yolov8m already detects standard vehicles well
- The problem isn't detection architecture — it's missing Indian vehicle classes
- RF-DETR is already loaded for TrafficMonitor — adding another transformer model wastes VRAM
- YOLOv8 has the best ecosystem for custom training on Windows

### 2.2 Eliminating Duplicate Detection

**Critical change**: Remove vehicle detection from `PlateRecognizer` entirely.

```
TrafficMonitor (already runs per frame)
  └→ vehicles[] + persons[]
       └→ Pass vehicles[] to PlateRecognizer via predetected_vehicles
            └→ PlateRecognizer only does plate detection + OCR
```

This path already exists (`predetected_vehicles` parameter) but isn't wired in `_process_frame_traffic()`.

### 2.3 Training Strategy for Indian Vehicles

**Dataset**: Combine these sources:
1. **Roboflow Indian Vehicles Dataset** (already integrated — mine the predictions as pseudo-labels)
2. **IDD (Indian Driving Dataset)** — 10K annotated frames from Indian roads
3. **Custom gate footage** — 500-1000 manually labeled frames from YOUR cameras

**Classes** (7 total):
```
0: car
1: motorcycle (includes scooters)
2: auto_rickshaw
3: bus
4: truck
5: bicycle
6: tempo_lcv
```

**Training recipe**:
```python
model = YOLO("yolov8s.pt")  # Start from COCO pretrained
model.train(
    data="indian_vehicles.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,     # Synthetic augmentation
    degrees=15,          # Handle camera angles
    perspective=0.001,   # CCTV perspective distortion
    flipud=0.0,          # Never flip vertically (plates become unreadable)
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,          # Handle varying lighting
)
```

### 2.4 Handling Auto-Rickshaws

Auto-rickshaws are the hardest Indian vehicle class because:
- Three-wheeled — unique silhouette
- Often have open sides — confuses bbox regression
- Passengers visible inside — can trigger person detection overlap

**Solution**: Dedicated class in fine-tuned model + aspect ratio post-filter (auto-rickshaws have characteristic ~1:1 aspect ratio from overhead CCTV).

---

## 3. PLATE DETECTION REDESIGN

### 3.1 Current Model Assessment

| Model | Size | Dataset | Indian Plate Accuracy | Verdict |
|-------|------|---------|----------------------|---------|
| `license_plate_detector.pt` | 6MB | Unknown | ~20% recall | ❌ Unusable |
| `license-plate-finetune-v1l.pt` | 51MB | keremberke (250 imgs) | ~40% recall | ❌ Weak |

Both models fail because they were trained on small Western plate datasets.

### 3.2 Recommended Model: YOLOv8m Fine-tuned on Indian Plates

**Best pretrained starting point**: Use these Roboflow/HuggingFace datasets:

1. **`Indian-Number-Plates` on Roboflow** — 1500+ Indian plate images with tight bounding boxes
2. **`ANPR-Indian-License-Plates` on Roboflow** — 800+ images, various angles
3. **`keremberke/license-plate-object-detection`** (already using — keep for diversity)
4. **Custom gate footage plates** — 200-500 manually labeled from YOUR cameras

**Why YOLOv8m, not YOLOv8s or YOLOv11-L?**
- YOLOv8s (22MB) is too small for small-object plate detection at CCTV distances
- YOLOv11-L (51MB) is overkill — plates are a single-class detection problem
- YOLOv8m (52MB) has the best speed/accuracy tradeoff for this task

### 3.3 Training Recipe for Plate Detector

```python
model = YOLO("yolov8m.pt")
model.train(
    data="indian_plates.yaml",
    epochs=150,
    imgsz=1280,          # CRITICAL: high-res input for small plates
    batch=8,             # Smaller batch due to 1280 imgsz
    patience=25,
    lr0=0.001,           # Lower LR for fine-tuning
    lrf=0.01,
    mosaic=1.0,
    mixup=0.0,           # No mixup — plates need precise localization
    degrees=10,
    perspective=0.001,
    scale=0.5,           # Aggressive scale augmentation for distance variation
    shear=5,
    flipud=0.0,
    fliplr=0.0,          # NEVER flip plates — text becomes mirrored
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,           # Strong brightness augmentation for day/night
    erasing=0.3,         # Random erasing for partial occlusion robustness
)
```

### 3.4 Full-Frame Detection Strategy (Keep and Improve)

Strategy A (full-frame at imgsz=1280) is correct and should remain primary. Improvements:

1. **Use imgsz=1280 always** — plates are tiny in full frame
2. **Lower confidence to 0.15** for detection, filter by plate geometry post-hoc
3. **Add plate geometry validation**:
   ```python
   # Indian plates: aspect ratio 2.0-6.0 (width/height)
   # Minimum absolute size: 30x10 pixels in original frame
   aspect = plate_w / plate_h
   if not (1.5 < aspect < 7.0):
       continue  # Not a plate shape
   ```

### 3.5 Perspective Correction for Tilted Plates

Add a perspective rectification step BEFORE OCR:

```python
def rectify_plate(plate_crop):
    """Correct perspective distortion on detected plate."""
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return plate_crop
    
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]
    
    # Only correct if angle is significant but not extreme
    if 2 < abs(angle) < 45:
        h, w = plate_crop.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(plate_crop, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return plate_crop
```

### 3.6 Corner-Based Detection (Advanced)

For severely distorted plates, add **keypoint detection** as a secondary approach:
- Train a model to detect the 4 corners of the plate
- Apply homography to warp the plate to a frontal view
- This is how commercial ANPR systems handle oblique angles

**Implementation**: Use YOLOv8-pose with 4 keypoints (one per plate corner). Train on the same Indian plate dataset with corner annotations.

This is a Phase 2 improvement — get the basic pipeline working first.

---

## 4. CAMERA POSITIONING GUIDANCE

### 4.1 Why Overhead CCTV Fails for ANPR

Overhead cameras at gates cause:
- **Extreme foreshortening** — plates appear as thin horizontal lines
- **Perspective distortion** — characters are trapezoidal, not rectangular
- **Motion blur** — plates sweep through the frame quickly
- **Occlusion** — vehicle hood/roof blocks the plate

### 4.2 Ideal ANPR Camera Setup

| Parameter | Recommended | Your Current | Impact |
|-----------|-------------|-------------|--------|
| Mounting angle | 15-30° from horizontal | ~60-80° (overhead) | **Critical** |
| Height | 1.0-1.5m (plate level) | 3-5m (ceiling) | High |
| Distance | 5-15m from capture point | 2-3m | Medium |
| Focal length | 12-25mm (narrow FoV) | Wide-angle | High |
| Shutter speed | 1/500s minimum | Auto | High |
| IR illumination | 850nm IR LED array | None | Critical at night |
| Resolution on plate | ≥120px width | ~40-60px | **Critical** |

### 4.3 Practical Recommendation for Your Setup

Since you can't reposition existing CCTV cameras, dedicate **2-4 cameras specifically for ANPR**:
- Mount at gate barrier level (1.0-1.5m height)
- Angle 15-25° to face oncoming plates
- Use IP cameras with varifocal 8-25mm lens
- Enable IR cut filter with external 850nm IR illuminator
- Set shutter priority to 1/500s or faster
- Resolution: 1080p minimum, 4K preferred

**Cost**: ~₹5,000-8,000 per dedicated ANPR camera. For a college with gates, you need 2 per gate (entry + exit).
