# SentinelAI — Smart Campus Surveillance System
## Project Blueprint & Knowledge Base

> **Purpose:** This file captures the full project understanding so context is never lost across sessions.
> **Last Updated:** 2026-05-23

---

## 1. Project Overview

**What:** AI-powered CCTV monitoring system for a college campus with 150+ cameras.  
**Why:** Manual monitoring of 150+ cameras is impossible with accuracy. Digitalize campus security.  
**Where:** Atria University, Bangalore, Karnataka  
**Who:** Team of 2 — **Chirag** and **Abhitha**  
**Deadline:** ~1 week (HOD demo)  
**Audience:** College management (HOD ma'am)  
**Demo format:** Live demo + dashboard analytics, using video analysis mode + 4-5 ngrok phone cameras  
**Potential:** Monetary compensation if done at industry standards. Gateway to more projects.

### Core Modules
1. **Face Recognition** — Identify teachers & known students ✅ WORKING
2. **Vehicle ANPR** — Detect plates of vehicles entering/leaving campus ⚠️ IN PROGRESS
3. **Vehicle-Face Linking** — If teacher's car leaves but teacher isn't in it → alert ❌ NOT BUILT
4. **Person ReID/Movement Tracking** — Cross-camera tracking ⚠️ CODE EXISTS, NOT FUNCTIONAL
5. **Fight/Ragging Detection** — Detect physical altercations ❌ NOT BUILT
6. **Classroom Monitoring** — Track teacher actions during classes ❌ NOT BUILT
7. **Crowd/Loitering Analytics** — Detect unusual gatherings ⚠️ CODE EXISTS, NOT FUNCTIONAL

---

## 2. Infrastructure (known)

- **Hardware:** RTX 5070 (12GB VRAM) — primary inference GPU
- **Cameras:** 150+ CCTV, well placed with ~0 blind spots. No direct access yet — testing with footage + ngrok
- **Camera Access:** TBD — team will tell college what to configure (RTSP likely)
- **Backend:** Python/FastAPI
- **Frontend:** Next.js dashboard (UI built, backend integration partial)
- **Database:** Exists (seen in dashboard)
- **Gate:** Main gate is manned. Guards only block yellow-plate (commercial) vehicles
- **Vehicle volume:** 5000-8000 per day. Mostly daytime, rarely night
- **Key use case:** Teacher vehicle theft prevention — link registered plates to owner faces

---

## 3. Current Pipeline Status

### Working ✅
- Vehicle detection (YOLOv8 custom `gate_vehicles_v2.pt`)
- Face recognition (InsightFace/ArcFace)
- Person detection & ReID (cross-camera tracking)
- Real-ESRGAN plate enhancement
- EasyOCR plate text reading (low accuracy ~50-60%)
- ByteTrack for object tracking
- Traffic monitoring dashboard

### Partially Working ⚠️
- ANPR plate reading (reads KA53 prefix but misreads middle letters MK→NK/HK)
- Qwen2-VL integrated but causes speed issues (~2-3s per inference)
- TrOCR integrated but reads garbage on CCTV angles

### Not Started / Unknown ❌
- Fight/ragging detection
- Teacher classroom monitoring
- Full 150-camera deployment pipeline
- Movement analytics dashboard
- Alerting system

---

## 4. ANPR Pipeline — Current Architecture

```
Frame → YOLO vehicle detect → crop vehicle → bottom 50% (plate area)
  → Real-ESRGAN 4x upscale
  → [Qwen2-VL] → [EasyOCR] → [TrOCR] → _fix_indian_plate → emit event
```

### Known Issues
- Qwen2-VL: High accuracy but 2-3s/frame, uses 4GB VRAM
- EasyOCR: Fast but systematic misreads (M→N, K→H at oblique angles)
- TrOCR: Reads only 6-char fragments, misses middle characters
- Prompt bias: Example plate in Qwen prompt caused hallucination
- No consensus voting across frames yet
- Plate cooldown too short → duplicate emissions

---

## 5. Answered Questions

### Campus & Infrastructure
- Camera access: Not available yet. Team will advise college on setup.
- Testing with: Video footage + ngrok mobile cameras (4-5 devices)
- System runs on: Single machine with RTX 5070

### ANPR
- Gate cameras only for ANPR
- 5000-8000 vehicles/day — need fast pipeline (<1s per vehicle)
- Day operation mostly, rare night
- Both real-time alerts AND log/lookup needed
- Yellow plates (commercial) blocked by guards manually
- Visitors/delivery vehicles: ignored

### Vehicle-Face Linking (unique feature)
- Register teacher/staff plate numbers linked to their face
- If registered car exits with unknown driver → snapshot alert to admin
- Can get staff vehicle plate list from college

### Demo
- ~1 week deadline
- HOD + college management audience
- Live demo + analytics dashboard
- Must not fail during demo (embarrassment risk)
- 4-5 ngrok phone cameras + video analysis mode

---

## 6. Design Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-23 | Replace vehicle model → YOLOv8m COCO | Custom model has class confusion from auto-labeled data. COCO pretrained is reliable for car/bike/bus/truck |
| 2026-05-23 | Replace plate detector → better pretrained | Current 6MB model misses >50% of plates. Need Roboflow/GitHub pretrained plate YOLO |
| 2026-05-23 | Implement consensus voting | Single-frame OCR ~60% accurate. Voting across 5-10 frames → ~90%+. How production ANPR works |
| 2026-05-23 | Move Qwen2-VL → dashboard only | 2-3s/inference too slow for 5000-8000 vehicles/day. Keep as manual "verify" button |
| 2026-05-23 | Keep TrOCR alongside EasyOCR | With proper plate crops (from new detector), TrOCR may improve. Two opinions > one |
| 2026-05-23 | Keep Real-ESRGAN 4x | Plate crops ~60x20px without it. Essential, non-negotiable |
| 2026-05-23 | Keep _fix_indian_plate() | Position-aware OCR correction works well |
| 2026-05-23 | Use InsightFace for face recognition | Already working, industry standard |
| 2026-05-23 | Root cause: vehicle model failure | 26,925 auto-labeled images. COCO doesn't know auto-rickshaws. Bicycle mixed with motorcycle. Wrong labels amplified errors |
| 2026-05-23 | Auto-rickshaws → classified as "car" | Acceptable for now. Fine-tune later if needed |

---

## 7. Test Results Archive

### Video 1: Red car — KA53MK0449 (overhead gate cam)
- EasyOCR best: `KA53AN0649` (7/10 match)
- Qwen2-VL best: `KA53MK0469` (8/10 match, with biased prompt)
- Challenge: Very oblique angle, plate at ~45° tilt

### Video 2: White Dzire — KA02MG2926 (overhead gate cam)
- EasyOCR best: `KA02NG2926` (9/10 match)
- Qwen2-VL best: `KA02MG2926` ✅ EXACT (42/52 frames, but with prompt bias)
- Challenge: Headlight glare partially obscures plate

---

## 8. Implementation Reference

**Full implementation plan:** See [demo_strategy.md](file:///C:/Users/lokes/.gemini/antigravity/brain/0c8cfefa-6298-44f7-a96e-4ad59735b848/artifacts/demo_strategy.md)

**Key files to modify:**
- `backend/app/vision/anpr.py` — Main ANPR pipeline
- `backend/app/vision/traffic.py` — Class ID constants
- `backend/models/` — Model weights directory

---

