# ANPR Production Redesign — Part 3: Architecture, Dataset, Testing & Migration

## 9. PRODUCTION ARCHITECTURE

### 9.1 Modular File Structure (Redesigned)

```
app/vision/
├── anpr/
│   ├── __init__.py              # Public API: ANPREngine
│   ├── plate_detector.py        # YOLO plate detection (single responsibility)
│   ├── plate_ocr.py             # PaddleOCR + EasyOCR dual reader
│   ├── plate_preprocessor.py    # CLAHE, sharpen, deskew, glare removal
│   ├── plate_corrector.py       # Indian plate format correction
│   ├── consensus.py             # VehicleTrackBuffer + majority voting
│   ├── plate_validator.py       # Plate format validation + state codes
│   └── config.py                # ANPR-specific constants
├── traffic.py                   # TrafficMonitor (vehicle + person detection)
├── pipeline.py                  # Main orchestrator
├── bytetrack.py                 # Tracker (shared)
└── ...
```

**Current `anpr.py` is 1748 lines** — a monolith containing detection, OCR, preprocessing, correction, tracking, voting, and visualization. Split it.

### 9.2 New `ANPREngine` API

```python
class ANPREngine:
    """
    Production ANPR pipeline.
    
    Consumes pre-detected vehicles from TrafficMonitor.
    Runs plate detection + OCR + consensus voting.
    Emits high-confidence plate events only after multi-frame agreement.
    """
    
    def __init__(self, config: ANPRConfig, on_plate: Callable):
        self.detector = PlateDetector(config.plate_model_path)
        self.ocr = PlateOCR(config.ocr_engines)
        self.preprocessor = PlatePreprocessor()
        self.corrector = IndianPlateCorrector()
        self.consensus = VehicleTrackBuffer(config.consensus)
        self.on_plate = on_plate
    
    def process_frame(self, frame: np.ndarray, camera_id: str,
                      vehicles: List[dict], vehicle_track_ids: List[int]):
        """
        Process a frame with pre-detected vehicles.
        
        Args:
            frame: Full resolution frame
            camera_id: Camera identifier
            vehicles: Vehicle detections from TrafficMonitor
            vehicle_track_ids: ByteTrack IDs for each vehicle
        """
        # 1. Detect plates (full-frame primary, per-crop fallback)
        plates = self.detector.detect(frame, vehicles)
        
        # 2. For each plate: preprocess → OCR → correct → add to consensus
        for plate_det in plates:
            crop = plate_det['crop']
            
            # Skip blurry crops
            if self.preprocessor.is_blurry(crop):
                continue
            
            # Preprocess
            color_img, gray_img = self.preprocessor.enhance(crop)
            
            # OCR (dual engine)
            text, confidence = self.ocr.read(color_img, gray_img)
            if not text:
                continue
            
            # Indian plate correction
            corrected = self.corrector.correct(text, confidence)
            if not corrected:
                continue
            
            # Add to consensus buffer
            track_id = plate_det.get('vehicle_track_id', -1)
            self.consensus.add_reading(
                track_id=track_id,
                bbox=plate_det['vehicle_box'],
                plate_text=corrected,
                ocr_conf=confidence,
                vehicle_type=plate_det.get('vehicle_type', 'vehicle'),
                camera_id=camera_id,
            )
        
        # 3. Flush expired tracks → emit consensus events
        events = self.consensus.flush_expired()
        for event in events:
            self.on_plate(event)
```

### 9.3 Monolith vs Microservices

**Recommendation: Modular monolith** (NOT microservices).

Why:
- Your system runs on a single machine with one GPU
- Microservices add network latency between detection → OCR → consensus
- GPU sharing across processes is complex (CUDA contexts)
- A single FastAPI process with ThreadPoolExecutor is correct for your scale

**When to move to microservices**: Only if you deploy on multiple machines (e.g., separate GPU server for inference, separate app server for FastAPI).

### 9.4 Fault Tolerance

```python
class CameraWorker:
    """Resilient camera connection with exponential backoff."""
    
    MAX_RETRIES = 10
    BASE_DELAY = 2.0  # seconds
    MAX_DELAY = 120.0  # 2 minutes max
    
    async def connect_with_retry(self, stream_url: str):
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                cap = cv2.VideoCapture(stream_url)
                if cap.isOpened():
                    logger.info(f"Connected to {stream_url}")
                    retries = 0  # Reset on success
                    yield cap
                else:
                    raise ConnectionError("Stream not available")
            except Exception as e:
                retries += 1
                delay = min(self.BASE_DELAY * (2 ** retries), self.MAX_DELAY)
                logger.warning(f"Camera reconnect attempt {retries}/{self.MAX_RETRIES}, "
                             f"waiting {delay:.0f}s: {e}")
                await asyncio.sleep(delay)
            finally:
                if cap:
                    cap.release()
```

### 9.5 Logging & Metrics Strategy

```python
# Structured logging for ANPR events
import structlog

anpr_logger = structlog.get_logger("anpr")

# Every plate event:
anpr_logger.info("plate_detected",
    camera_id=camera_id,
    plate=plate_text,
    confidence=0.87,
    consensus_votes=5,
    agreement=0.80,
    vehicle_type="car",
    inference_ms=45,
    ocr_engine="paddleocr",
)

# Metrics to track:
# - plates_detected_total (counter)
# - plates_rejected_total (counter, by reason: blur/low_conf/no_consensus)
# - ocr_latency_ms (histogram)
# - detection_latency_ms (histogram)
# - consensus_agreement_ratio (histogram)
# - active_vehicle_tracks (gauge)
# - frames_processed_total (counter, by camera)
```

---

## 10. DATASET STRATEGY

### 10.1 How Much Data You Need

| Component | Minimum | Recommended | Source |
|-----------|---------|-------------|--------|
| Vehicle detection (Indian) | 1,000 images | 3,000-5,000 | IDD + Roboflow + custom |
| Plate detection (Indian) | 500 images | 2,000-3,000 | Roboflow + custom gate footage |
| OCR training data | Not needed | Not needed | Use pretrained PaddleOCR |
| Custom gate validation set | 100 images | 300-500 | Manual labeling from YOUR cameras |

### 10.2 Labeling Guidelines

**Vehicle labeling rules**:
- Label the full visible body of the vehicle (not just the visible portion)
- Include partially occluded vehicles if >50% visible
- Auto-rickshaws: label the entire three-wheeler including canopy
- Motorcycles with riders: bbox should include the rider
- Don't label parked vehicles in the background (only vehicles on the road)

**Plate labeling rules**:
- Label the plate region TIGHTLY — no extra margin
- Include both front and rear plates
- Label partially visible plates if >60% of characters are visible
- Don't label completely unreadable plates (motion blur, too far)

**Tools**: Use [Roboflow](https://roboflow.com) or [CVAT](https://cvat.ai) for labeling. Roboflow has a free tier and auto-exports to YOLO format.

### 10.3 Avoiding Label Noise

The current `gate_vehicles_v2.pt` failed because of noisy auto-labels. Rules:

1. **Never use auto-labels without human verification** — review every bounding box
2. **Use consistent class definitions** — write a labeling guide document
3. **Inter-annotator agreement**: Have 2 people label the same 50 images, compare IoU
4. **Discard ambiguous samples** — if you can't tell if it's a car or auto, remove it
5. **Validate before training**: Run `yolo val` on 10% held-out set with ground truth

### 10.4 Augmentation Strategy

```yaml
# In YOLO training config:
augmentation:
  hsv_h: 0.015        # Hue variation (different lighting)
  hsv_s: 0.5          # Saturation variation (rain/fog)
  hsv_v: 0.4          # Brightness variation (day/night)
  degrees: 15          # Rotation (camera tilt)
  translate: 0.1       # Translation
  scale: 0.5           # Scale variation (near/far vehicles)
  shear: 5             # Shear (perspective effect)
  perspective: 0.001   # Perspective distortion
  flipud: 0.0          # NEVER flip vertically
  fliplr: 0.5          # Horizontal flip (mirror is valid)
  mosaic: 1.0          # Mosaic augmentation
  mixup: 0.1           # MixUp (light)
  copy_paste: 0.1      # Copy-paste augmentation
  erasing: 0.3         # Random erasing (occlusion simulation)
```

### 10.5 Synthetic Data for Plates

Generate synthetic Indian plates to augment training data:

```python
def generate_synthetic_plate():
    """Generate a random Indian license plate image."""
    import random
    from PIL import Image, ImageDraw, ImageFont
    
    state = random.choice(list(INDIAN_STATE_CODES))
    district = f"{random.randint(1, 99):02d}"
    series = ''.join(random.choices('ABCDEFGHJKLMNPQRSTUVWXYZ', k=random.randint(1,2)))
    number = f"{random.randint(1, 9999):04d}"
    
    plate_text = f"{state}{district}{series}{number}"
    
    # Create plate image with realistic styling
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 60)  # Use a plate-like font
    draw.text((20, 15), plate_text, fill='black', font=font)
    
    # Add realistic noise: blur, perspective, brightness variation
    # ... (augmentation code)
    
    return img, plate_text
```

---

## 11. TESTING STRATEGY

### 11.1 Benchmark Methodology

Create a test set of 200-500 frames from YOUR cameras with ground truth:

```python
# test_anpr_benchmark.py
import json

# Ground truth format:
# {"frame_path": "test_frames/001.jpg", "plates": ["KA53MK0449", "MH12AB1234"]}

def benchmark_anpr(engine, test_set):
    results = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "ocr_correct": 0,
        "ocr_partial": 0,  # >70% character match
        "ocr_wrong": 0,
        "latencies_ms": [],
    }
    
    for sample in test_set:
        frame = cv2.imread(sample["frame_path"])
        gt_plates = set(sample["plates"])
        
        start = time.monotonic()
        detected_plates = engine.detect_and_read(frame)
        latency = (time.monotonic() - start) * 1000
        results["latencies_ms"].append(latency)
        
        detected_set = set(p["text"] for p in detected_plates)
        
        # Plate detection metrics
        for gt in gt_plates:
            if any(plate_similarity(gt, d) > 0.8 for d in detected_set):
                results["true_positives"] += 1
                # Check OCR accuracy
                best_match = max(detected_set, key=lambda d: plate_similarity(gt, d))
                sim = plate_similarity(gt, best_match)
                if sim == 1.0:
                    results["ocr_correct"] += 1
                elif sim > 0.7:
                    results["ocr_partial"] += 1
                else:
                    results["ocr_wrong"] += 1
            else:
                results["false_negatives"] += 1
        
        for d in detected_set:
            if not any(plate_similarity(gt, d) > 0.8 for gt in gt_plates):
                results["false_positives"] += 1
    
    # Compute metrics
    precision = results["true_positives"] / max(1, results["true_positives"] + results["false_positives"])
    recall = results["true_positives"] / max(1, results["true_positives"] + results["false_negatives"])
    f1 = 2 * precision * recall / max(0.001, precision + recall)
    ocr_accuracy = results["ocr_correct"] / max(1, results["true_positives"])
    
    print(f"Detection Precision: {precision:.1%}")
    print(f"Detection Recall:    {recall:.1%}")
    print(f"Detection F1:        {f1:.1%}")
    print(f"OCR Exact Match:     {ocr_accuracy:.1%}")
    print(f"OCR Partial Match:   {results['ocr_partial']/max(1,results['true_positives']):.1%}")
    print(f"Avg Latency:         {sum(results['latencies_ms'])/len(results['latencies_ms']):.0f}ms")
    print(f"P95 Latency:         {sorted(results['latencies_ms'])[int(len(results['latencies_ms'])*0.95)]:.0f}ms")
```

### 11.2 Target Metrics

| Metric | Current (estimated) | Target | Commercial ANPR |
|--------|-------------------|--------|-----------------|
| Plate Detection Recall | ~30-40% | >85% | >95% |
| Plate Detection Precision | ~50-60% | >90% | >98% |
| OCR Exact Match | ~10-20% | >70% | >90% |
| OCR Partial Match (>70% chars) | ~30-40% | >85% | >95% |
| Latency per frame | ~1-3s | <150ms | <50ms |
| False positive rate | ~20-30% | <5% | <1% |

---

## 12. MIGRATION PLAN (Phased)

### Phase 1: Quick Wins (1-2 days) — HIGHEST IMPACT

1. **Wire `predetected_vehicles` from TrafficMonitor → ANPR** (remove duplicate vehicle detection)
2. **Remove ESRGAN from realtime** (replace with cv2 CLAHE + bilateral filter)
3. **Remove TrOCR and Qwen2-VL from realtime** (keep for dashboard verify only)
4. **Wire consensus voting** (make `_process()` call `_add_vote()` instead of emitting directly)
5. **Increase ANPR worker threads** from 1 to 2-3

**Expected improvement**: 3-5x faster inference, ~20% better accuracy from consensus.

### Phase 2: OCR Upgrade (1-2 days)

1. **Install PaddleOCR** (`pip install paddlepaddle-gpu paddleocr`)
2. **Replace EasyOCR as primary** with PaddleOCR
3. **Keep EasyOCR as secondary** voter
4. **Implement dual-OCR voting**

**Expected improvement**: 2-3x faster OCR, ~15% better OCR accuracy.

### Phase 3: Plate Detection Model (3-5 days)

1. **Collect 500+ plate images from YOUR gate cameras**
2. **Combine with Roboflow Indian plate datasets** (2000+ images)
3. **Fine-tune YOLOv8m at imgsz=1280** (150 epochs)
4. **Validate on held-out test set**
5. **Deploy new model**

**Expected improvement**: 2-3x better plate detection recall.

### Phase 4: Preprocessing & Correction (1-2 days)

1. **Add blur detection** — skip blurry frames
2. **Add glare removal**
3. **Add plate deskewing** (rotation correction)
4. **Improve `_fix_indian_plate()` with probabilistic correction**

**Expected improvement**: ~10-15% better OCR on challenging conditions.

### Phase 5: Architecture Refactor (2-3 days)

1. **Split `anpr.py` into modular package**
2. **Implement `VehicleTrackBuffer`** (proper consensus)
3. **Use ByteTrack for vehicle tracking**
4. **Add structured logging and metrics**

### Phase 6: Camera Optimization (Ongoing)

1. **Deploy 2-4 dedicated ANPR cameras at gates**
2. **Configure shutter speed, IR illumination**
3. **Test and iterate on camera positioning**

---

## 13. PRODUCTION CHECKLIST

```
□ ESRGAN removed from realtime pipeline
□ TrOCR removed from realtime pipeline  
□ Qwen2-VL removed from realtime pipeline
□ PaddleOCR installed and integrated as primary OCR
□ Consensus voting system wired into _process()
□ Vehicle detections shared from TrafficMonitor (no duplicate YOLO)
□ Plate detector fine-tuned on Indian plate dataset (≥2000 images)
□ Blur detection gate before OCR
□ CLAHE + bilateral filter preprocessing (replaces ESRGAN)
□ Plate deskewing enabled
□ anpr.py split into modular package
□ ByteTrack used for vehicle tracking
□ VehicleTrackBuffer implemented with proper cooldown
□ Worker threads increased to 2-3
□ TensorRT FP16 export for YOLO models
□ Structured logging for all ANPR events
□ Benchmark test suite with 200+ ground truth frames
□ Dedicated ANPR cameras deployed at gates
□ VRAM usage validated < 10GB total
□ Latency validated < 150ms per frame
□ Plate detection recall validated > 85%
□ OCR exact match validated > 70%
```

---

## 14. HIDDEN FLAWS YOU MAY NOT HAVE NOTICED

1. **`_process()` never calls `_add_vote()`** — consensus voting is dead code
2. **ESRGAN downloads weights from GitHub on first run** — will fail in air-gapped deployment
3. **`_plate_cooldowns` dict grows unbounded** — memory leak over days
4. **ThreadPoolExecutor(max_workers=1)** — single-threaded bottleneck
5. **`_fix_indian_plate()` returns None for valid partial reads** — kills consensus
6. **Roboflow API key is hardcoded in `.env`** — rate-limited at 1000 calls/month on free tier
7. **No GPU memory cleanup** — ESRGAN + TrOCR + Qwen2-VL all stay resident
8. **Frame copies** — `frame.copy()` at line 741 creates unnecessary memory pressure
9. **No queue depth monitoring** — can't tell when ANPR is falling behind
10. **Plate model runs at default imgsz=640** in `_detect_plates_in_crop()` — should be 1280 for small plates
