# ANPR Production Redesign — Part 2: OCR, Consensus, Preprocessing & Performance

## 5. OCR PIPELINE REDESIGN

### 5.1 Current OCR Stack Assessment

| Engine | Speed | Indian Plate Accuracy | VRAM | Verdict |
|--------|-------|----------------------|------|---------|
| EasyOCR | ~200ms/crop | 30-50% (fragments text) | ~500MB | Replace |
| TrOCR (base-printed) | ~500ms/crop | 20-35% (not trained on plates) | ~1GB | Remove |
| TrOCR (plate-finetuned) | ~500ms/crop | 40-55% | ~1GB | Remove |
| Qwen2-VL-2B | ~2-3s/crop | 60-75% | ~4GB | Remove from realtime |
| **PaddleOCR** | ~30-50ms/crop | 55-70% | ~200MB | **✅ Use this** |

### 5.2 Recommended Stack: PaddleOCR + Custom CTC Head

**Primary OCR**: PaddleOCR v4 (PP-OCRv4)
- 10x faster than EasyOCR
- Better at structured text (plates are structured)
- Built-in text detection + recognition
- Lightweight — ~200MB VRAM

**Installation**:
```bash
pip install paddlepaddle-gpu paddleocr
```

**Usage**:
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,    # Auto-rotate
    lang='en',
    use_gpu=True,
    det_db_thresh=0.3,
    rec_batch_num=6,       # Batch multiple plates
    max_text_length=15,    # Indian plates are max 13 chars
)

results = ocr.ocr(plate_image, cls=True)
```

**Secondary OCR (consensus voter)**: EasyOCR — keep as a second opinion, NOT primary.

**Remove entirely**: TrOCR, Qwen2-VL from the realtime pipeline. They're too slow and their marginal accuracy gain doesn't justify 10-50x latency increase.

### 5.3 OCR Batching Strategy

Instead of running OCR one plate at a time:
```python
# CURRENT (slow):
for plate in plates:
    text = ocr.readtext(plate)  # 200ms each

# REDESIGNED (fast):
plate_batch = [plate1, plate2, plate3, plate4]
results = ocr.ocr_batch(plate_batch)  # ~80ms total for 4 plates
```

PaddleOCR supports native batching. EasyOCR does not.

### 5.4 Character-Level Post-Processing

Replace `_fix_indian_plate()` with a probabilistic correction system:

```python
def correct_indian_plate(raw_text: str, char_confidences: list) -> tuple:
    """
    Probabilistic Indian plate correction using character confidences.
    
    Indian format: SS DD LLL DDDD
    - SS: state code (known set of 36 valid codes)
    - DD: district (01-99)
    - LLL: series (1-3 uppercase letters)
    - DDDD: number (1-4 digits)
    
    Returns: (corrected_text, confidence_score)
    """
    # Step 1: Strip non-alphanumeric
    chars = []
    confs = []
    for ch, cf in zip(raw_text, char_confidences):
        if ch.isalnum():
            chars.append(ch.upper())
            confs.append(cf)
    
    if len(chars) < 8:
        return None, 0.0
    
    # Step 2: Position-aware correction
    # Positions 0-1 MUST be letters → if digit, use confusion matrix
    # Positions 2-3 MUST be digits → if letter, use confusion matrix
    # etc.
    
    # Step 3: Validate state code against INDIAN_STATE_CODES
    # Step 4: Score = product of corrected character confidences
    
    return corrected, score
```

**Key improvement**: Use per-character confidence from OCR to decide whether to correct. Don't blindly swap characters — only swap when the OCR confidence for that character is LOW.

### 5.5 Dual-OCR Voting

Run both PaddleOCR and EasyOCR on each plate, then pick the best:

```python
def dual_ocr_read(plate_crop):
    paddle_text, paddle_conf = paddle_ocr(plate_crop)
    easy_text, easy_conf = easyocr_read(plate_crop)
    
    # If both agree → high confidence
    if plate_similarity(paddle_text, easy_text) > 0.8:
        return paddle_text, max(paddle_conf, easy_conf) * 1.2
    
    # If they disagree → return the one with better Indian format match
    p_valid = is_valid_indian_plate(paddle_text)
    e_valid = is_valid_indian_plate(easy_text)
    
    if p_valid and not e_valid:
        return paddle_text, paddle_conf
    elif e_valid and not p_valid:
        return easy_text, easy_conf
    else:
        return (paddle_text, paddle_conf) if paddle_conf > easy_conf else (easy_text, easy_conf)
```

---

## 6. MULTI-FRAME CONSENSUS SYSTEM (Complete Redesign)

### 6.1 Why Current Consensus Doesn't Work

The `_add_vote()` and `_emit_consensus()` methods exist but are **dead code**. The `_process()` method emits events directly at lines 905-928 (Strategy A) and 1001-1024 (Strategy B) without ever calling `_add_vote()`.

### 6.2 New Architecture: VehicleTrackBuffer

```python
class VehicleTrackBuffer:
    """
    Accumulates OCR readings for each tracked vehicle across frames.
    Only emits a plate event when consensus is reached or vehicle exits.
    """
    
    def __init__(self):
        self.tracks: Dict[int, VehicleTrack] = {}
        self.min_votes = 3          # Minimum reads before emitting
        self.max_votes = 8          # Stop accumulating after this
        self.agreement_threshold = 0.6  # 60% votes must agree
        self.max_age_sec = 8.0      # Expire tracks after 8 seconds
        self.cooldown_sec = 60.0    # Don't re-emit same plate within 60s
        self.emitted_plates: Dict[str, float] = {}  # plate → timestamp
    
    def add_reading(self, track_id: int, bbox: tuple, 
                    plate_text: str, ocr_conf: float,
                    vehicle_type: str, camera_id: str):
        """Add one OCR reading to a vehicle's accumulation buffer."""
        
        if track_id not in self.tracks:
            self.tracks[track_id] = VehicleTrack(
                track_id=track_id,
                bbox=bbox,
                vehicle_type=vehicle_type,
                camera_id=camera_id,
            )
        
        track = self.tracks[track_id]
        track.add_vote(plate_text, ocr_conf)
        track.update_bbox(bbox)
        
        # Check if ready to emit
        if not track.emitted and len(track.votes) >= self.min_votes:
            result = track.get_consensus()
            if result and result.agreement >= self.agreement_threshold:
                if not self._is_cooldown(result.plate):
                    self._emit(track, result)
    
    def flush_expired(self):
        """Expire old tracks and emit partial results if useful."""
        now = time.time()
        expired = [tid for tid, t in self.tracks.items() 
                   if now - t.last_seen > self.max_age_sec]
        for tid in expired:
            track = self.tracks[tid]
            if not track.emitted and len(track.votes) >= 2:
                result = track.get_consensus()
                if result and result.agreement >= 0.5:
                    if not self._is_cooldown(result.plate):
                        self._emit(track, result)
            del self.tracks[tid]


class VehicleTrack:
    """Single vehicle being tracked for plate reading."""
    
    def __init__(self, track_id, bbox, vehicle_type, camera_id):
        self.track_id = track_id
        self.bbox = bbox
        self.vehicle_type = vehicle_type
        self.camera_id = camera_id
        self.votes: List[Tuple[str, float]] = []  # (plate_text, confidence)
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.emitted = False
    
    def add_vote(self, plate_text: str, confidence: float):
        self.votes.append((plate_text, confidence))
        self.last_seen = time.time()
    
    def get_consensus(self) -> Optional[ConsensusResult]:
        """Compute majority vote with fuzzy matching."""
        if not self.votes:
            return None
        
        # Group similar readings (Levenshtein > 0.7)
        groups = []
        for plate, conf in self.votes:
            matched = False
            for group in groups:
                if plate_similarity(plate, group[0][0]) > 0.7:
                    group.append((plate, conf))
                    matched = True
                    break
            if not matched:
                groups.append([(plate, conf)])
        
        # Find largest group
        best_group = max(groups, key=len)
        
        # Pick the most common spelling in the group
        from collections import Counter
        texts = [p for p, c in best_group]
        winner = Counter(texts).most_common(1)[0][0]
        
        avg_conf = sum(c for _, c in best_group) / len(best_group)
        agreement = len(best_group) / len(self.votes)
        
        return ConsensusResult(
            plate=winner,
            confidence=avg_conf * agreement,
            agreement=agreement,
            total_votes=len(self.votes),
            winner_votes=len(best_group),
        )
```

### 6.3 Integration with ByteTrack

Use ByteTrack (already in your codebase) for vehicle tracking instead of the homegrown IoU matcher:

```python
# In the ANPR pipeline:
from app.vision.bytetrack import ByteTracker

vehicle_tracker = ByteTracker(
    track_thresh=0.4,
    track_buffer=30,   # Keep tracks for 30 frames (1 second at 30fps)
    match_thresh=0.8,
)

# Each frame:
bboxes = np.array([[v['box'] for v in vehicles]])
scores = np.array([v['confidence'] for v in vehicles])
tracks = vehicle_tracker.update(bboxes, scores)

for track in tracks:
    track_id = track.track_id  # Stable across frames
    # Use this track_id as the key for VehicleTrackBuffer
```

### 6.4 Event Emission Logic

```
Vehicle detected
  → Plate detected in vehicle bbox
    → OCR plate text
      → Add to VehicleTrackBuffer[track_id]
        → IF votes >= 3 AND agreement >= 60%:
            → Check cooldown (fuzzy match against last 60s)
              → IF not duplicate: EMIT event
              → Mark track as emitted
        → IF vehicle exits frame (track expired):
            → IF votes >= 2 AND agreement >= 50%:
              → EMIT with lower confidence flag
```

---

## 7. IMAGE PREPROCESSING REDESIGN

### 7.1 ESRGAN Assessment: Remove from Realtime

ESRGAN (RealESRGAN_x4plus) is **inappropriate for realtime ANPR**:
- 200-500ms per plate crop
- 16.7M parameters
- Hallucinates texture that can corrupt OCR
- VRAM hungry

**Replace with**: Lightweight upscaling pipeline:

```python
def preprocess_plate(plate_crop, target_width=300):
    """Fast preprocessing pipeline for plate crops."""
    h, w = plate_crop.shape[:2]
    
    # 1. Resize to target width (maintain aspect ratio)
    if w < target_width:
        scale = target_width / w
        plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE (adaptive contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 4))
    enhanced = clahe.apply(gray)
    
    # 4. Bilateral filter (denoise while preserving edges)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 5. Unsharp masking (sharpen text edges)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)
    
    # 6. Return both color (for PaddleOCR) and grayscale (for EasyOCR)
    h2, w2 = plate_crop.shape[:2]
    color_resized = cv2.resize(plate_crop, (w2, h2))
    
    return color_resized, sharpened
```

**Total time**: ~5ms vs 200-500ms for ESRGAN. **40-100x faster**.

### 7.2 When to Use ESRGAN

Keep ESRGAN as an **offline verification tool** only:
- Dashboard "Verify Plate" button → runs ESRGAN + Qwen2-VL on stored crop
- Nightly batch processing of low-confidence plates
- NOT in the realtime pipeline

### 7.3 Glare Removal

```python
def remove_glare(image):
    """Remove specular highlights from plate images."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Detect glare: high value, low saturation
    glare_mask = cv2.threshold(v, 220, 255, cv2.THRESH_BINARY)[1]
    glare_mask = cv2.bitwise_and(glare_mask, 
                                 cv2.threshold(s, 30, 255, cv2.THRESH_BINARY_INV)[1])
    
    # Inpaint glare regions
    result = cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)
    return result
```

### 7.4 Motion Blur Detection and Handling

```python
def detect_blur(image, threshold=100):
    """Detect if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

# In the pipeline:
is_blurry, blur_score = detect_blur(plate_crop)
if is_blurry:
    # Skip this frame's OCR — wait for a sharper frame
    # The consensus system will naturally prefer sharp-frame readings
    logger.debug(f"Skipping blurry plate (score={blur_score:.0f})")
    continue
```

---

## 8. PERFORMANCE OPTIMIZATION

### 8.1 VRAM Budget for RTX 5070 (12GB)

| Model | VRAM | Status |
|-------|------|--------|
| InsightFace (buffalo_l) | ~1.5GB | Keep |
| ByteTrack | ~0MB (CPU) | Keep |
| YOLOv8m (vehicles - TrafficMonitor) | ~800MB | Keep |
| RF-DETR Large | ~2GB | Keep (or replace with shared YOLO) |
| YOLOv8m (plate detector) | ~800MB | Keep |
| PaddleOCR PP-OCRv4 | ~200MB | New |
| EasyOCR | ~500MB | Keep (secondary) |
| ESRGAN | ~800MB | **REMOVE from realtime** |
| TrOCR | ~1GB | **REMOVE** |
| Qwen2-VL-2B | ~4GB | **REMOVE from realtime** |
| **Total current** | **~12.6GB** | **Over budget!** |
| **Total redesigned** | **~5.8GB** | **Safe margin** |

### 8.2 Multi-Camera Scaling Architecture

```
┌─────────────────────────────────────────────┐
│              Camera Workers (N)              │
│  cam_1 ──→ frame_queue_1 ──┐                │
│  cam_2 ──→ frame_queue_2 ──┤                │
│  cam_3 ──→ frame_queue_3 ──┤                │
│  ...                       │                │
│  cam_N ──→ frame_queue_N ──┤                │
│                            ▼                │
│         ┌──────────────────────┐            │
│         │  Inference Workers   │            │
│         │  (GPU-bound, 2-4)    │            │
│         │                      │            │
│         │  Worker 1: Face mode │            │
│         │  Worker 2: Traffic   │            │
│         │  Worker 3: ANPR      │            │
│         │  Worker 4: ANPR      │            │
│         └──────────┬───────────┘            │
│                    ▼                        │
│         ┌──────────────────────┐            │
│         │  Event Queue (Redis) │            │
│         └──────────┬───────────┘            │
│                    ▼                        │
│         ┌──────────────────────┐            │
│         │  FastAPI WebSocket   │            │
│         └──────────────────────┘            │
└─────────────────────────────────────────────┘
```

### 8.3 Inference Scheduling

For 150 cameras on one GPU:
- **Face mode cameras**: Process every 3rd frame at 640px → ~15ms/frame → 50 cameras at 10fps
- **Traffic cameras**: Process every 5th frame at 640px → ~20ms/frame → 30 cameras at 6fps  
- **ANPR cameras**: Process every 3rd frame at 1280px → ~35ms/frame → 10 cameras at 10fps

**Total**: 90 cameras at acceptable FPS. For 150, you need a second GPU or reduce processing frequency.

### 8.4 Model Quantization

Export models to TensorRT FP16 for 2x speed:

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.export(format="engine", half=True, device=0, imgsz=1280)
# Use the exported .engine file instead of .pt
```

### 8.5 Async Pipeline Design

```python
import asyncio
from asyncio import Queue

class ANPRPipelineAsync:
    def __init__(self):
        self.frame_queue = Queue(maxsize=100)
        self.result_queue = Queue(maxsize=100)
        self.track_buffer = VehicleTrackBuffer()
    
    async def frame_producer(self, camera_id, stream_url):
        """Capture frames and push to queue."""
        cap = cv2.VideoCapture(stream_url)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(1)
                continue
            frame_count += 1
            if frame_count % 3 == 0:  # Every 3rd frame
                await self.frame_queue.put((camera_id, frame))
    
    async def inference_worker(self):
        """GPU worker — processes frames from queue."""
        while True:
            camera_id, frame = await self.frame_queue.get()
            # Run detection + OCR
            results = await asyncio.to_thread(
                self._process_frame, frame, camera_id
            )
            if results:
                await self.result_queue.put(results)
    
    async def event_emitter(self):
        """Emit consensus events."""
        while True:
            self.track_buffer.flush_expired()
            await asyncio.sleep(0.5)
```
