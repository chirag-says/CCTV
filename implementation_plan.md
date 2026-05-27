# 🎯 AI CCTV Surveillance System — Complete Training & Deployment Plan

## College Profile
- **Type:** Engineering college, 5000+ students
- **Gate:** 1 common gate, 2-3 security guards
- **Hostels:** Separate boys & girls hostels with curfew
- **Vehicles:** Only student/teacher bikes & cars allowed
- **Cameras:** 500+ (starting with front gate, then campus-wide)
- **Deployment:** Production — real-world use

---

## 📊 Feature List — What Needs Training vs What Doesn't

### ❌ No Training Required (Pretrained / Config-Based)

These work out of the box with your existing SCRFD + ArcFace + ByteTrack pipeline:

| # | Feature | Method | Phase |
|---|---------|--------|-------|
| 1 | **Face recognition at gate** | ArcFace embeddings — register faces | Phase 1 |
| 2 | **Unauthorized entry alerts** | Unknown face → cosine similarity < threshold | Phase 1 |
| 3 | **Teacher arrival time tracking** | Face recognized + timestamp logged | Phase 1 |
| 4 | **Teacher punctuality reports** | Entry time vs expected schedule | Phase 1 |
| 5 | **Entry/exit people counter** (time-based) | ByteTrack direction counting + hourly stats | Phase 1 |
| 6 | **Vendor/delivery overstay** (>20 min) | ByteTrack dwell time at gate zone | Phase 1 |
| 7 | **Loitering / bunking detection** | ByteTrack dwell time in corridors | Phase 2 |
| 8 | **Restricted area intrusion** | Face recognition + geofence zones per camera | Phase 2 |
| 9 | **Hostel curfew violation** | Face detection + time-of-day rules | Phase 3 |
| 10 | **Gender zone monitoring** (boys/girls hostel) | Face recognition → gender/identity match | Phase 3 |
| 11 | **Teacher movement tracking** | Face recognition across multiple cameras | Phase 3 |

---

### ✅ Custom Training Required

These need labeled data from your footage + model training on your RTX 5070:

| # | Feature | Model | Labels Needed | Training Time | Phase |
|---|---------|-------|---------------|---------------|-------|
| 12 | **ID card wearing detection** | YOLOv8 | ~800 frames (wearing / not wearing) | ~1.5 hrs | Phase 1 |
| 13 | **Vehicle classification** (bike/car/auto/bus) | YOLOv8 | ~500 frames | ~1 hr | Phase 1 |
| 14 | **License plate reading** (Indian plates) | EasyOCR fine-tune | ~200 plate images | ~30 min | Phase 1 |
| 15 | **Smoking / vaping detection** | YOLOv8 | ~500 frames | ~1 hr | Phase 2 |
| 16 | **Fight / scuffle detection** | SlowFast / I3D action classifier | ~400 video clips | ~3 hrs | Phase 2 |
| 17 | **Crowd gathering alert** | YOLOv8 + density threshold | ~300 frames | ~1 hr | Phase 2 |
| 18 | **Unattended object detection** | YOLOv8 + static object tracker | ~300 frames | ~1 hr | Phase 2 |
| 19 | **Parking violation detection** | YOLOv8 + zone mapping | ~400 frames | ~1 hr | Phase 2 |
| 20 | **Emergency detection** (fire/stampede/collapse) | YOLOv8 + action classifier | ~500 frames/clips | ~2 hrs | Phase 3 |

---

## 🏗️ Phased Deployment Plan

### Phase 1 — Front Gate Intelligence (Week 1-2)

> **Goal:** Secure the gate, track all entry/exit, enforce ID cards, log vehicles

#### What Gets Built:

```
┌──────────────────────────────────────────────────────────────┐
│                    FRONT GATE CAMERA                         │
│                          │                                   │
│     ┌────────────────────┼────────────────────┐              │
│     │                    │                    │              │
│     ▼                    ▼                    ▼              │
│ ┌────────┐        ┌───────────┐        ┌───────────┐        │
│ │ SCRFD  │        │ YOLOv8    │        │ EasyOCR   │        │
│ │ Face   │        │ ID Card + │        │ License   │        │
│ │ Detect │        │ Vehicle   │        │ Plate     │        │
│ └───┬────┘        └─────┬─────┘        └─────┬─────┘        │
│     │                   │                    │              │
│     ▼                   ▼                    ▼              │
│ ┌────────┐        ┌───────────┐        ┌───────────┐        │
│ │ArcFace │        │ Alerts:   │        │ Vehicle   │        │
│ │ ID +   │        │ No ID     │        │ Entry Log │        │
│ │ByteTrack│       │ Wrong     │        │ + Owner   │        │
│ └───┬────┘        │ Vehicle   │        │ Matching  │        │
│     │             └───────────┘        └───────────┘        │
│     ▼                                                       │
│ ┌────────────────────────────────────────┐                   │
│ │ DASHBOARD                              │                   │
│ │ • Who entered/exited + time            │                   │
│ │ • Teacher arrival log                  │                   │
│ │ • Unauthorized person alerts           │                   │
│ │ • ID card violation alerts             │                   │
│ │ • Vehicle log + plate numbers          │                   │
│ │ • Live people count on campus          │                   │
│ │ • Vendor overstay alerts (>20 min)     │                   │
│ └────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
```

#### Models to Train in Phase 1:

**Model 1: ID Card Detection (YOLOv8)**
- Classes: `id_card_visible`, `no_id_card`
- Dataset: Extract frames from gate footage → label bounding boxes
- Frames needed: ~800 (400 wearing + 400 not wearing)
- Expected accuracy: ~92-95%

**Model 2: Vehicle Classification (YOLOv8)**
- Classes: `bike`, `car`, `auto`, `bus`, `truck`, `cycle`
- Dataset: Frames from gate footage with vehicles
- Frames needed: ~500
- Expected accuracy: ~95%+

**Model 3: License Plate OCR (EasyOCR fine-tune)**
- Fine-tune on Indian plate formats from your camera angle
- Plates needed: ~200 cropped plate images with text labels
- Expected accuracy: ~93-97%

#### Features that work without training:
- Face recognition (register faces via dashboard)
- Entry/exit logging with timestamps
- Teacher arrival time tracking
- Unknown person alerts
- Vendor overstay alerts (>20 min)
- Live campus headcount (hourly breakdown)

---

### Phase 2 — Campus Behavior Monitoring (Week 3-5)

> **Goal:** Detect fights, smoking, loitering, crowd issues, abandoned objects

#### Models to Train:

**Model 4: Smoking Detection (YOLOv8)**
- Classes: `cigarette`, `smoking_person`, `vape`
- Use public datasets + your campus footage
- Frames needed: ~500

**Model 5: Fight Detection (Action Classifier)**
- Classes: `normal`, `fighting`, `pushing`
- Label short 3-5 second video clips
- Clips needed: ~400 (200 normal + 200 fight)

**Model 6: Crowd Gathering (YOLOv8 + Rules)**
- Use person detection → if >N people in zone → alert
- Minimal training, mostly thresholding

**Model 7: Unattended Object (YOLOv8 + Static Tracker)**
- Classes: `bag`, `box`, `package`
- Alert if object detected without owner for >5 min
- Frames needed: ~300

**Model 8: Parking Violation (YOLOv8 + Zone Map)**
- Detect vehicle in no-parking zone
- Reuse vehicle model from Phase 1 + zone configuration

#### Config-based features (no training):
- Loitering/bunking alerts (dwell time > threshold)
- Restricted area intrusion (face + zone rules)
- Student movement tracking

---

### Phase 3 — Hostel & Advanced (Week 6-8)

> **Goal:** Hostel security, emergency detection, campus-wide scaling

#### Models to Train:

**Model 9: Emergency Detection**
- Classes: `fire`, `smoke`, `person_fallen`, `stampede`
- Mix of public datasets + campus-specific data
- Frames/clips needed: ~500

#### Config-based features:
- Hostel curfew violation (time rules)
- Boys/girls hostel zone monitoring (face ID + gender)
- Teacher movement tracking across all cameras
- Campus-wide person of interest tracking

---

## 🔧 Training Pipeline Steps

For each model, the process is:

### Step 1: Frame Extraction
```bash
# Extract 1 frame every 3 seconds from gate videos
python extract_frames.py --input training/videos/raw/ --output training/frames/ --interval 3
```

### Step 2: Labeling (using Roboflow or CVAT)
- Upload frames to Roboflow (free tier: 10k images)
- Draw bounding boxes around objects
- Export in YOLOv8 format

### Step 3: Train
```bash
# Train YOLOv8 on your labeled data
yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 device=0
```

### Step 4: Validate
```bash
# Test on validation set
yolo detect val model=runs/detect/train/weights/best.pt data=dataset.yaml
```

### Step 5: Deploy
- Copy `best.pt` to `backend/models/`
- Integrate into the CCTV pipeline

---

## 📅 Timeline Summary

| Phase | Duration | Features | Models to Train |
|-------|----------|----------|----------------|
| **Phase 1** | Week 1-2 | Gate security, ID cards, vehicles, teacher tracking | 3 models |
| **Phase 2** | Week 3-5 | Fights, smoking, loitering, crowds, abandoned objects | 5 models |
| **Phase 3** | Week 6-8 | Hostel, emergencies, campus-wide scaling | 1 model |

**Total custom models to train: 9**
**Total features delivered: 20**

---

## ⚡ Immediate Next Steps

1. Install Microsoft C++ Build Tools → install `insightface`
2. Build frame extraction script for your gate videos
3. Start labeling ID card + vehicle frames (Roboflow)
4. Train first 2 YOLO models
5. Integrate into the running backend

> [!IMPORTANT]
> **Start with Phase 1.** Get the gate working perfectly before expanding campus-wide.
> A working gate system alone solves 60% of your college's security needs.
