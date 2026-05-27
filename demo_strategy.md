# SentinelAI — ANPR Pipeline Overhaul Plan
## Agreed & Finalized: 2026-05-23

> **Status:** READY TO IMPLEMENT (starting 2026-05-24)

---

## Architecture Decision — FINAL

```
┌─────────────────────────────────────────────────────────────┐
│                    ANPR PIPELINE v3                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Frame                                                      │
│    │                                                        │
│    ▼                                                        │
│  ① Vehicle Detection ─── YOLOv8m (COCO pretrained)          │
│    │                     Classes: car, motorcycle, bus,      │
│    │                     truck. Autos detected as "car".     │
│    │                     REPLACES: gate_vehicles_v2.pt       │
│    ▼                                                        │
│  ② Plate Detection ───── New pretrained plate YOLO model    │
│    │                     Find tight plate bounding box       │
│    │                     REPLACES: license_plate_detector.pt │
│    ▼                                                        │
│  ③ Plate Enhancement ─── Real-ESRGAN 4x (KEEP AS-IS)       │
│    │                     ~60x20px → ~240x80px               │
│    ▼                                                        │
│  ④ Plate OCR ──────────── EasyOCR + TrOCR (parallel)       │
│    │                      Both read the enhanced plate       │
│    │                      Results feed into consensus voter  │
│    ▼                                                        │
│  ⑤ Consensus Voting ──── Across 5-10 frames per vehicle    │
│    │                     Majority vote picks final plate     │
│    │                     REPLACES: single-frame read         │
│    ▼                                                        │
│  ⑥ Indian Plate Fix ──── _fix_indian_plate() (KEEP AS-IS)  │
│    │                     Position-aware OCR correction       │
│    ▼                                                        │
│  ⑦ Event Emission ────── Whitelist check → dashboard event  │
│                                                             │
│  ⑧ Qwen2-VL ──────────── Dashboard "Verify Plate" button   │
│                          NOT in real-time pipeline           │
│                          Admin clicks to re-read with VLM    │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Changes Summary

| # | Component | Current | New | Action |
|---|-----------|---------|-----|--------|
| 1 | Vehicle Detection | `gate_vehicles_v2.pt` (custom, broken) | `yolov8m.pt` (COCO pretrained) | **REPLACE** |
| 2 | Plate Detection | `license_plate_detector.pt` (6MB, weak) | New pretrained plate YOLO | **REPLACE** |
| 3 | Plate Enhancement | Real-ESRGAN 4x | Same | **KEEP** ✅ |
| 4 | Plate OCR | EasyOCR + TrOCR + Qwen2-VL (all in real-time) | EasyOCR + TrOCR (parallel, consensus) | **SIMPLIFY** |
| 5 | Plate Correction | `_fix_indian_plate()` | Same | **KEEP** ✅ |
| 6 | Qwen2-VL | In real-time pipeline (slow, 2-3s) | Dashboard "Verify" button only | **MOVE** |

---

## Implementation Tasks (Day 1)

### Task 1: Replace Vehicle Detection Model
**File:** `anpr.py` → `_load_vehicle_model()` and `_detect_vehicles()`

- [ ] Change model path from `gate_vehicles_v2.pt` to `yolov8m.pt`
- [ ] Update class mappings: COCO uses `{2: car, 3: motorcycle, 5: bus, 7: truck}`
- [ ] Update `traffic.py` class constants to match COCO IDs
- [ ] Remove dependency on custom class names
- [ ] Test: run on gate footage, verify cars/bikes detected correctly

### Task 2: Find & Integrate Plate Detection Model
**File:** `anpr.py` → `_detect_plates_in_crop()`

- [ ] Search Roboflow/GitHub for best pretrained YOLOv8 plate detector
- [ ] Download `.pt` weights, test on gate footage ESRGAN crops
- [ ] If accuracy < 80%: fine-tune yolov8n on Roboflow Indian plate dataset (~20 min)
- [ ] Replace `license_plate_detector.pt` with new model
- [ ] Test: verify tight plate crops on both test videos

### Task 3: Implement Consensus Voting
**File:** `anpr.py` → new `_consensus_vote()` method

- [ ] Add per-vehicle-track plate reading buffer (dict: track_id → list of readings)
- [ ] After 5-10 frames of the same vehicle, majority vote on plate text
- [ ] Emit plate event only after consensus is reached
- [ ] Handle edge case: vehicle leaves frame before 5 reads → emit best available
- [ ] Test: verify same vehicle gets ONE final plate, not 50 duplicate events

### Task 4: Remove Qwen2-VL from Real-Time Pipeline
**File:** `anpr.py` → `_direct_plate_ocr()` and `_ocr_region()`

- [ ] Remove Qwen2-VL calls from `_direct_plate_ocr()`
- [ ] Remove Qwen2-VL calls from `_ocr_region()`
- [ ] Keep `_qwen_read_plate()` method available for API/dashboard use
- [ ] Don't remove Qwen2-VL loading — just don't call it in real-time
- [ ] Add API endpoint: `POST /api/verify-plate` → takes image, returns Qwen2-VL reading

### Task 5: Clean Up & Test
- [ ] Remove unused TrOCR CRAFT fragmentation logic (we'll feed TrOCR the full plate crop now)
- [ ] Test full pipeline on Video 1 (red car KA53MK0449)
- [ ] Test full pipeline on Video 2 (white Dzire KA02MG2926)
- [ ] Measure FPS — target: >5 FPS on gate footage
- [ ] Measure accuracy — target: >80% plate text accuracy

---

## Files That Need Changes

| File | Changes |
|------|---------|
| `backend/app/vision/anpr.py` | Major — vehicle model swap, plate model swap, consensus voting, Qwen2-VL removal from real-time |
| `backend/app/vision/traffic.py` | Class ID constants update for COCO |
| `backend/models/` | Add new plate detector `.pt`, keep `yolov8m.pt` |
| `backend/app/api/` | Add `/api/verify-plate` endpoint for Qwen2-VL |

---

## Testing Checklist

| Test | Pass Criteria |
|------|--------------|
| Vehicle detection on gate footage | Detects cars, bikes, buses. No false positives on trees/walls |
| Plate detection on vehicle crops | Finds plate in >80% of vehicle crops |
| OCR on enhanced plates | Reads at least 7/10 characters correctly per frame |
| Consensus voting | Same vehicle → 1 final plate event (not 50 duplicates) |
| End-to-end Video 1 | Reads `KA53MK0449` or within 1-2 chars |
| End-to-end Video 2 | Reads `KA02MG2926` or within 1-2 chars |
| FPS | >5 FPS sustained on 1080p gate footage |

---

## Models Inventory (after changes)

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| `yolov8m.pt` | Vehicle detection (COCO) | ~50MB | Ultralytics (downloaded ✅) |
| `plate_detector.pt` | Plate localization | ~6-12MB | TBD — Roboflow/GitHub |
| `RealESRGAN_x4plus.pth` | Plate super-resolution | ~64MB | Already have ✅ |
| `DunnBC22/trocr-base-printed_license_plates_ocr` | Plate text (transformer) | ~900MB | HuggingFace (cached ✅) |
| `Qwen/Qwen2-VL-2B-Instruct` | Manual verify only | ~4GB | HuggingFace (cached ✅) |

---

## What We're NOT Changing
- Real-ESRGAN enhancement pipeline
- `_fix_indian_plate()` correction logic
- Indian plate format validation
- Face recognition system (InsightFace)
- Frontend dashboard UI
