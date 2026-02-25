# 🛡️ SentinelAI — AI-Powered CCTV Surveillance System

## Complete Project Documentation & In-Depth Explanation

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Technology Stack — Complete Breakdown](#3-technology-stack--complete-breakdown)
4. [Backend Deep Dive](#4-backend-deep-dive)
5. [Vision Pipeline — The Brain of the System](#5-vision-pipeline--the-brain-of-the-system)
6. [AI/ML Algorithms Explained](#6-aiml-algorithms-explained)
7. [Safety & Security Modules](#7-safety--security-modules)
8. [Traffic & ANPR System](#8-traffic--anpr-system)
9. [Real-Time Communication (WebSockets)](#9-real-time-communication-websockets)
10. [Database Architecture](#10-database-architecture)
11. [Authentication & Security](#11-authentication--security)
12. [Frontend Deep Dive](#12-frontend-deep-dive)
13. [DevOps & Deployment](#13-devops--deployment)
14. [Data Flow — End to End](#14-data-flow--end-to-end)
15. [Why Each Technology Was Chosen](#15-why-each-technology-was-chosen)
16. [Project Structure](#16-project-structure)

---

## 1. Project Overview

**SentinelAI** is a production-grade, AI-powered CCTV surveillance system that transforms ordinary camera feeds into an intelligent security platform. It processes live video streams in real-time using advanced computer vision and machine learning algorithms to:

- **Detect and recognize faces** in live video
- **Track people's entry and exit** with time-based logic
- **Identify unknown/unrecognized individuals** and queue them for admin review
- **Detect crowds and gatherings** that may indicate security incidents
- **Detect loiterers** — people remaining in one area for too long
- **Detect hazardous objects** (knives, scissors) using object detection
- **Read vehicle license plates** (ANPR — Automatic Number Plate Recognition)
- **Monitor vehicle-pedestrian proximity** for safety
- **Stream live annotated video** to the web dashboard in real-time
- **Provide analytics dashboards** with occupancy, peak times, and reports

The system operates in **three pipeline modes**:

| Mode | Description |
|------|-------------|
| **Face** | Face recognition, entry/exit tracking, crowd/loiter/hazard detection |
| **Traffic** | Vehicle detection, ANPR (license plates), vehicle-pedestrian safety |
| **Hybrid** | Both Face + Traffic modes running simultaneously |

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js 16)                     │
│  ┌──────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ ┌────────────┐ │
│  │Dashboard │ │Cameras │ │Events  │ │ Persons │ │  Alerts    │ │
│  │  Page    │ │Monitor │ │  Log   │ │  Mgmt   │ │  Page      │ │
│  └────┬─────┘ └───┬────┘ └───┬────┘ └────┬────┘ └─────┬──────┘ │
│       │           │          │            │            │         │
│       └───────────┴──────────┴────────────┴────────────┘         │
│                         │ REST API + WebSocket                   │
└─────────────────────────┼────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI + Python)                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              VISION PIPELINE (per camera)             │       │
│  │  ┌─────────┐  ┌───────────┐  ┌─────────┐  ┌───────┐│       │
│  │  │  Face   │→ │   Face    │→ │ Person  │→ │ Event ││       │
│  │  │Detector │  │Recognizer │  │ Tracker │  │Emitter││       │
│  │  └─────────┘  └───────────┘  └─────────┘  └───────┘│       │
│  │                                                      │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │       │
│  │  │ Crowd    │  │ Loiter   │  │ Hazard   │          │       │
│  │  │ Detector │  │ Detector │  │ Detector │          │       │
│  │  └──────────┘  └──────────┘  └──────────┘          │       │
│  │                                                      │       │
│  │  ┌──────────┐  ┌──────────┐                         │       │
│  │  │  ANPR    │  │ Traffic  │                         │       │
│  │  │(Plates)  │  │ Monitor  │                         │       │
│  │  └──────────┘  └──────────┘                         │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Routes  │  │ Services  │  │  Models  │  │  WebSocket    │  │
│  │  (API)   │  │ (Logic)   │  │(Pydantic)│  │  Manager      │  │
│  └──────────┘  └───────────┘  └──────────┘  └───────────────┘  │
│                                                                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DATABASE (Supabase / PostgreSQL)               │
│  persons │ face_encodings │ cameras │ detection_events           │
│  tracking_sessions │ unknown_faces │ admin_users                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack — Complete Breakdown

### 🐍 Backend Technologies

| Technology | Version | Purpose | Why Important |
|---|---|---|---|
| **Python** | 3.11+ | Core language | ML ecosystem, async support, massive library ecosystem |
| **FastAPI** | 0.109.2 | Web framework | Async-first, auto-generated docs, Pydantic validation, WebSocket support |
| **Uvicorn** | 0.27.1 | ASGI server | Production-grade async server, handles concurrent WebSocket connections |
| **OpenCV** | ≥4.9.0 | Video capture & processing | Industry-standard computer vision library for frame manipulation |
| **face_recognition** | ≥1.3.0 | Face detection + encoding | Built on dlib's state-of-the-art HOG/CNN face models |
| **dlib** | ≥20.0.0 | Face landmark detection | Powers face_recognition with 99.38% accuracy on LFW benchmark |
| **DeepFace** | ≥0.0.89 | Gender/attribute recognition | Pre-trained deep neural networks for facial attribute analysis |
| **Ultralytics (YOLOv8)** | ≥8.1.0 | Object detection | State-of-the-art real-time object detection (weapons, vehicles) |
| **EasyOCR** | ≥1.7.0 | Optical Character Recognition | Reads license plate text from images with multi-language support |
| **NumPy** | ≥1.26.0 | Numerical computing | Foundation for all vector/matrix operations (face encodings are 128-d vectors) |
| **SciPy** | ≥1.12.0 | Scientific computing | Distance calculations, statistical operations for analytics |
| **Pillow** | ≥10.2.0 | Image processing | Image format conversion, resizing, snapshot saving |
| **Supabase** | ≥2.3.0 | Database client (Python) | PostgreSQL BaaS with built-in auth, RLS, real-time subscriptions |
| **Pydantic** | ≥2.6.0 | Data validation | Type-safe request/response models, automatic serialization |
| **Pydantic Settings** | ≥2.1.0 | Configuration management | Environment-variable-based config with type validation |
| **python-jose** | ≥3.3.0 | JWT tokens | Secure JSON Web Token creation and verification |
| **Passlib + bcrypt** | ≥1.7.4 / ≥4.1.0 | Password hashing | Industry-standard bcrypt hashing for secure password storage |
| **aiofiles** | ≥23.2.1 | Async file I/O | Non-blocking file operations for snapshot saving |
| **httpx** | ≥0.27.0 | HTTP client | Async HTTP requests for health checks and external services |
| **websockets** | ≥13.0 | WebSocket protocol | Real-time bidirectional communication for live event streaming |

### ⚛️ Frontend Technologies

| Technology | Version | Purpose | Why Important |
|---|---|---|---|
| **Next.js** | 16.1.6 | React framework | Server-side rendering, file-based routing, optimized production builds |
| **React** | 19.2.3 | UI library | Component-based UI with hooks for state management |
| **React DOM** | 19.2.3 | DOM rendering | React's DOM reconciliation engine |
| **ESLint** | ^9 | Code quality | Enforces code style and catches common JavaScript errors |

### 🐳 Infrastructure

| Technology | Purpose | Why Important |
|---|---|---|
| **Docker** | Containerization | Reproducible builds, isolated environments |
| **Docker Compose** | Multi-container orchestration | Runs backend + frontend together with networking |
| **Supabase (PostgreSQL)** | Managed database | Scalable, managed PostgreSQL with real-time features and RLS |

---

## 4. Backend Deep Dive

### 4.1 Application Entry Point (`main.py`)

The `main.py` file is the **central nervous system** of the backend. It:

1. **Creates the FastAPI application** with lifespan management
2. **Sets up CORS middleware** for cross-origin frontend requests
3. **Registers all API route modules** (auth, persons, cameras, events, analytics)
4. **Manages the vision pipeline lifecycle** — starts when app boots, stops on shutdown
5. **Bridges sync↔async worlds** — the vision pipeline runs in background threads (synchronous) but events must be broadcast over async WebSockets. It uses `asyncio.run_coroutine_threadsafe()` to safely schedule async WebSocket broadcasts from sync callback threads
6. **Persists events to the database** — every entry/exit/detection event is saved to Supabase
7. **Manages unknown face enrollment** — unknown faces are saved with their face encoding, cropped snapshot, context snapshot, and full frame for admin review

### 4.2 Configuration System (`config.py`)

Uses **Pydantic Settings** — every configuration value is:
- **Type-validated** at startup (catches misconfiguration immediately)
- **Loadable from environment variables** or `.env` files
- **Documented with defaults** so the system works out-of-the-box

Key configuration groups:
- **Supabase** — Database URL and API keys
- **JWT Auth** — Secret key, algorithm (HS256), token expiry
- **Vision Pipeline** — Detection model (HOG/CNN), face match tolerance, frame skipping
- **Safety Modules** — Crowd proximity thresholds, loitering time windows, hazard detection intervals
- **Traffic/ANPR** — Vehicle detection intervals, proximity thresholds, OCR languages
- **Pipeline Mode** — Switch between `face`, `traffic`, or `hybrid` mode

### 4.3 Service Layer Architecture

The backend follows a **clean service-oriented architecture**:

```
Routes (API endpoints) → Services (Business logic) → Database (Supabase)
                                    ↕
                          Vision Pipeline (AI processing)
```

| Service | Responsibility |
|---|---|
| `AuthService` | User authentication, JWT token generation, password verification |
| `PersonService` | CRUD for known persons, face encoding management, history querying |
| `CameraService` | Camera registration, start/stop pipeline, status monitoring |
| `AnalyticsService` | Dashboard summary, peak times, occupancy, report generation |
| `MovementService` | Movement logs, detection event querying with filters |

---

## 5. Vision Pipeline — The Brain of the System

### 5.1 Overview

The `VisionPipeline` class (in `pipeline.py`, **1,526 lines**) is the **most critical component** of the entire system. It orchestrates the complete flow from raw video frames to meaningful security events.

### 5.2 Processing Flow (Face Mode)

```
Camera Frame
    │
    ▼
┌─────────────────────┐
│  Frame Capture      │  ← cv2.VideoCapture reads from webcam/RTSP/IP
│  (30 FPS target)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Frame Skip Check   │  ← Process every Nth frame (default: every 3rd)
│                     │     for performance optimization
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Hazard Detection   │  ← YOLOv8 (runs independently, every 30th frame)
│  (Background Thread)│     Detects: knives, scissors, weapons
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Face Detection     │  ← face_recognition library (HOG or CNN model)
│  (FaceDetector)     │     Finds face locations in the frame
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Face Encoding      │  ← Generates 128-dimensional face vectors
│  (FaceRecognizer)   │     for each detected face
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Face Recognition   │  ← Compares encoding against known faces database
│  (Vectorized Numpy) │     Using Euclidean distance with tolerance threshold
└─────────┬───────────┘
          │
     ┌────┴────┐
     │         │
     ▼         ▼
┌─────────┐ ┌──────────────┐
│ KNOWN   │ │  UNKNOWN     │
│ PERSON  │ │  PERSON      │
└────┬────┘ └──────┬───────┘
     │              │
     ▼              ▼
┌─────────┐ ┌──────────────┐
│ Tracker │ │ Unknown Face │
│(Entry/  │ │  Handler     │
│ Exit)   │ │ (Dedup+Save) │
└────┬────┘ └──────┬───────┘
     │              │
     ▼              ▼
┌─────────────────────┐
│  Safety Analytics   │  ← Crowd detection + Loitering detection
│  (CrowdDetector,    │     Using centroid positions from tracker
│   LoiteringDetector)│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Attribute Recog.   │  ← Gender (DeepFace) + Clothing color (HSV)
│  (Background Thread)│     Runs once per tracked person
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Event Emission     │  ← Broadcasts events via WebSocket
│  + DB Persistence   │     Saves to Supabase database
└─────────────────────┘
```

### 5.3 Camera Worker System

Each camera runs in a **dedicated background thread** via `CameraWorker`:

```python
CameraManager (Singleton)
    ├── CameraWorker (camera-1)
    │       └── VisionPipeline
    │               ├── FaceDetector
    │               ├── FaceRecognizer
    │               ├── PersonTracker
    │               ├── CrowdDetector
    │               ├── LoiteringDetector
    │               ├── HazardDetector
    │               ├── PlateRecognizer (traffic mode)
    │               └── TrafficMonitor (traffic mode)
    ├── CameraWorker (camera-2)
    │       └── VisionPipeline (independent instance)
    └── ... (up to 10 cameras)
```

**Key Design Decision:** Each camera gets its **own pipeline instance** running in its **own thread**. This provides:
- **Complete isolation** — a crash in one camera doesn't affect others
- **Independent frame rates** — each camera can process at its own speed
- **Thread-safe state** — each pipeline manages its own tracking state

### 5.4 JPEG Encoding Optimization

A critical performance optimization: **JPEG encoding happens in the worker thread, not the async event loop**.

```
Worker Thread (Background):
    frame = pipeline.process_frame()
    small = resize_frame(frame, 640x480)
    pipeline._last_jpeg = encode_to_jpeg(small, quality=65)

WebSocket Stream Endpoint (Async):
    jpeg = worker.pipeline._last_jpeg    ← Just reads cached bytes
    await websocket.send_bytes(jpeg)     ← Zero CPU work, just I/O
```

This prevents JPEG encoding (~5-15ms) from blocking the async event loop that serves all WebSocket clients.

---

## 6. AI/ML Algorithms Explained

### 6.1 Face Detection — HOG & CNN

**What it does:** Finds the locations of faces in each video frame.

**Two detection models available:**

| Model | Full Name | Speed | Accuracy | Best For |
|---|---|---|---|---|
| **HOG** | Histogram of Oriented Gradients | ⚡ Fast | Good | CPU-only systems |
| **CNN** | Convolutional Neural Network | 🐢 Slower | Excellent | GPU-accelerated systems |

**How HOG works:**
1. Converts image to grayscale
2. Computes gradient magnitude and direction for each pixel
3. Divides image into cells and creates histograms of gradient orientations
4. Slides a detection window across the image
5. Uses a trained SVM classifier to determine if each window contains a face

**How CNN works:**
1. Passes the image through a deep convolutional neural network
2. The CNN learns features hierarchically — edges → textures → parts → faces
3. Uses learned feature maps to predict face locations with higher accuracy

**Performance optimization:** Frames are **downscaled** (default: 50%) before detection, then locations are scaled back up. This gives a ~4x speedup with minimal accuracy loss.

### 6.2 Face Encoding — 128-Dimensional Embeddings

**What it does:** Converts each detected face into a compact 128-dimensional vector (a "face encoding").

**How it works:**
1. The detected face region is fed through a **pre-trained deep neural network** (based on dlib's ResNet)
2. The network was trained on a large face dataset using **triplet loss** — it learns that:
   - Same-person faces → close vectors (small distance)
   - Different-person faces → far-apart vectors (large distance)
3. The output is a **128-float vector** that uniquely represents the face's identity

**Why 128 dimensions?** This was determined empirically to be the optimal balance between accuracy and storage/computation cost. Each encoding is just **1,024 bytes** (128 × 8 bytes per float64).

### 6.3 Face Recognition — Euclidean Distance Matching

**What it does:** Compares a detected face against all known faces to find a match.

**Algorithm:**
```python
# Vectorized distance computation (fast with NumPy)
distances = face_recognition.face_distance(known_encodings_matrix, new_encoding)
best_match_index = np.argmin(distances)
best_distance = distances[best_match_index]

if best_distance <= tolerance (0.55):
    → MATCH FOUND (confidence = 1.0 - distance)
else:
    → UNKNOWN PERSON
```

**Why Euclidean distance?** In the learned embedding space, faces of the same person cluster together. Euclidean distance directly measures this "closeness." A tolerance of **0.55** means two faces must be very similar to be considered a match.

**Performance optimization:** Known face encodings are pre-loaded into a **NumPy matrix**. Comparison against all known faces happens in a single vectorized operation — `O(1)` with respect to code complexity, `O(n)` actual computation but using optimized BLAS routines.

### 6.4 YOLOv8 — Object Detection

**What it does:** Detects specific objects (weapons, vehicles, people) in video frames.

**How YOLO works:**
1. **You Only Look Once** — divides the image into a grid
2. Each cell predicts bounding boxes + class probabilities simultaneously
3. Non-maximum suppression removes duplicate detections
4. Runs in a single forward pass — much faster than sliding-window approaches

**YOLOv8 improvements over older versions:**
- Anchor-free detection (more accurate bounding boxes)
- Better feature pyramid network for multi-scale detection
- Improved loss function for faster convergence
- Native support for GPU acceleration via PyTorch

**Model used:** `yolov8n.pt` (nano model) — the smallest/fastest variant:
- **6.5 MB** model size
- ~2ms inference on GPU
- 80 COCO classes pre-trained
- Used for hazard detection, vehicle detection, and traffic monitoring

### 6.5 EasyOCR — License Plate Text Reading

**What it does:** Reads text from license plate images.

**How it works:**
1. **Text detection** — finds regions containing text using CRAFT (Character Region Awareness for Text) network
2. **Text recognition** — reads characters using a CRNN (Convolutional Recurrent Neural Network)
3. Returns detected text with confidence scores

**Why EasyOCR over Tesseract?** EasyOCR uses deep learning models that are significantly more accurate on real-world images (varying angles, lighting, blur) compared to Tesseract's traditional approach.

### 6.6 DeepFace — Gender/Attribute Recognition

**What it does:** Predicts gender (Male/Female) from face crops.

**How it works:**
1. Takes a cropped face image
2. Passes it through pre-trained CNNs (VGG-Face, Facenet, etc.)
3. The model outputs gender probabilities: `{"Man": 95.2, "Woman": 4.8}`
4. Uses `detector_backend="skip"` to avoid redundant face detection (we already have the crop)

**Lazy loading:** DeepFace models are only loaded on first use (they're large, ~500MB). This prevents unnecessary memory consumption on systems that don't use attribute recognition.

---

## 7. Safety & Security Modules

### 7.1 Crowd / Gathering Detection

**Purpose:** Detect when 3+ people cluster together for a sustained period, indicating potential incidents.

**Algorithm — Single-Linkage Clustering:**

```
1. Get centroid positions of all tracked persons
2. Compute pairwise Euclidean distances between all centroids
3. Build an adjacency graph: connect persons within proximity_px (150px)
4. Find connected components using depth-first search (DFS)
5. If any component has ≥ min_persons (3) people:
     → Track as a potential gathering
6. If the gathering persists for ≥ sustain_seconds (5s):
     → TRIGGER CROWD ALERT 🚨
```

**Why single-linkage?** It naturally captures groups of any shape (not just circular clusters). If person A is near B, and B is near C, all three form a gathering even if A is far from C.

**Smart merging:** If a new cluster overlaps >60% with an existing gathering (some people joined/left), it's treated as the same gathering to prevent alert spam.

### 7.2 Loitering / Idle Detection

**Purpose:** Identify people who remain stationary for extended periods, indicating potential loitering.

**Algorithm — Centroid Displacement Analysis:**

```
1. Maintain a centroid history deque for each tracked person
   (timestamp, x, y) — up to 600 entries
2. Over a configurable time window (default: 5 minutes):
   a. Compute max displacement from the initial position
   b. If max_displacement < movement_threshold (50px):
       → Person has barely moved
       → TRIGGER LOITERING ALERT 🚨
3. Cooldown of 10 minutes prevents repeated alerts for the same person
```

**Why max displacement instead of average?** A person pacing back and forth might have low average displacement but high max displacement. Max displacement catches both truly stationary people AND small-area pacers.

### 7.3 Hazard Detection (YOLOv8)

**Purpose:** Detect dangerous objects (knives, scissors, weapons) in the camera feed.

**Threat classes monitored:**
| COCO ID | Object | Confidence Threshold |
|---|---|---|
| 43 | Knife | 50% |
| 76 | Scissors | 55% |
| — | Gun (custom model) | 50% |
| — | Fire (custom model) | 40% |

**Design decisions:**
- **Background thread execution** — YOLOv8 inference runs in a `ThreadPoolExecutor` to avoid blocking the video pipeline
- **Frame throttling** — only processes every 30th frame (1 per second at 30fps) to minimize GPU load
- **Per-class confidence thresholds** — different objects have different false-positive rates
- **Alert cooldown** — 30 seconds between alerts for the same class on the same camera
- **Graceful degradation** — if `ultralytics` isn't installed, the module silently disables itself

### 7.4 Attribute Recognition

**Purpose:** Extract person attributes (gender, clothing color) for "Smart Search" filtering.

**Clothing Color Detection — HSV Histogram Analysis:**

```
1. Identify torso region (below face bounding box, ~2× face height)
2. Convert torso crop to HSV color space
3. For each predefined color range:
   - Create a binary mask using cv2.inRange()
   - Count matching pixels
4. The color with most matching pixels = dominant clothing color
```

**Supported colors:** Red, Orange, Yellow, Green, Blue, Purple, Pink, White, Gray, Black

**Performance optimization:** Attribute recognition runs exactly **once per tracked person** using a `ThreadPoolExecutor`. Results are cached, so a person detected 100 times only triggers one DeepFace analysis.

---

## 8. Traffic & ANPR System

### 8.1 Automatic Number Plate Recognition (ANPR)

**Purpose:** Detect vehicles and read their license plates automatically.

**Multi-stage pipeline:**

```
Stage 1: Vehicle Detection (YOLOv8)
    → Detects cars, motorcycles, buses, trucks
    → Extracts vehicle bounding box

Stage 2: Plate Region Extraction (Computer Vision)
    → Focuses on lower 60% of vehicle (where plates typically are)
    → Applies bilateral filter for noise reduction
    → Canny edge detection to find plate-like rectangles
    → Contour analysis: finds rectangles with aspect ratio 1.5:1 to 7:1

Stage 3: OCR (EasyOCR)
    → Reads text from the best plate candidate regions
    → Validates against known plate patterns (Indian format: KA01AB1234)

Stage 4: Normalization & Dedup
    → Strips non-alphanumeric characters
    → Validates: must contain both letters and digits, ≥4 characters
    → 60-second cooldown per plate to prevent duplicate events
```

**Indian plate pattern support:**
- Standard: `KA01AB1234` / `KA 01 AB 1234`
- Older: `MH 12 1234`
- Generic: `[A-Z0-9]{4,12}`

### 8.2 Traffic Monitor — Vehicle-Pedestrian Safety

**Purpose:** Prevent accidents by detecting when people are dangerously close to vehicles.

**Algorithm — Bounding Box Proximity Analysis:**

```
1. Detect all persons and vehicles in the frame (YOLOv8)
2. For each (person, vehicle) pair:
   a. Calculate IoU (Intersection over Union) — measures overlap
   b. Calculate minimum edge-to-edge distance
3. If IoU > 0:
     → CRITICAL ALERT (bounding boxes overlap!) 🚨
4. If distance < proximity_px (50px):
     → WARNING ALERT (too close!) ⚠️
```

**Visualization on video feed:**
- 🔵 Blue boxes: vehicles
- 🟢 Green boxes: safe persons
- 🔴 Red boxes: persons in danger zone
- Red line: connecting person ↔ vehicle centroids with distance label

---

## 9. Real-Time Communication (WebSockets)

### 9.1 WebSocket Architecture

The system uses **two types of WebSocket connections:**

| Channel | Type | Data Format | Purpose |
|---|---|---|---|
| `/api/events/live` | Text (JSON) | Event objects | Real-time detection events (entry, exit, alert) |
| `/api/cameras/{id}/stream` | Binary (JPEG) | Raw JPEG bytes | Live video feed streaming |

### 9.2 Connection Manager

`ConnectionManager` (singleton) manages all WebSocket connections:

```python
# Event connections — all clients receive all events
_event_connections: List[WebSocket]

# Stream connections — camera-specific subscribers
_stream_connections: Dict[camera_id, List[WebSocket]]
```

**Thread safety:** Uses `asyncio.Lock()` for safe concurrent access.

**Auto-cleanup:** Failed sends automatically disconnect and clean up dead connections.

### 9.3 Sync→Async Bridge

The vision pipeline runs in **synchronous background threads**, but WebSocket communication is **async**. The bridge:

```python
# In background thread (vision pipeline callback):
asyncio.run_coroutine_threadsafe(
    ws_manager.broadcast_event(event),
    main_async_loop
)
```

This safely schedules the async WebSocket broadcast from the sync thread.

---

## 10. Database Architecture

### 10.1 Supabase (PostgreSQL)

**Why Supabase?**
- Managed PostgreSQL — no server maintenance
- Row Level Security (RLS) — fine-grained access control
- Built-in auth system
- Real-time subscriptions
- Auto-generated REST API
- Free tier available for development

### 10.2 Schema Design

```sql
┌──────────────┐     ┌──────────────────┐
│   persons    │     │  face_encodings  │
│──────────────│     │──────────────────│
│ id (PK)      │╌╌╌╌>│ id (PK)          │
│ name         │     │ person_id (FK)   │
│ role         │     │ encoding (TEXT)   │  ← Base64 pickle of numpy array
│ department   │     │ quality (FLOAT)  │
│ phone        │     │ created_at       │
│ email        │     └──────────────────┘
│ avatar_url   │
│ is_active    │     ┌──────────────────┐
│ created_at   │     │ tracking_sessions│
│ updated_at   │     │──────────────────│
└──────┬───────┘╌╌╌╌>│ id (PK)          │
       │             │ person_id (FK)   │
       │             │ camera_id (FK)   │
       │             │ entry_time       │
       │             │ exit_time        │
       │             │ duration_sec     │
       │             │ status           │
       │             └──────────────────┘
       │
       │             ┌──────────────────┐
       ╰╌╌╌╌╌╌╌╌╌╌╌>│ detection_events │
                     │──────────────────│
                     │ id (PK)          │
                     │ person_id (FK)   │
                     │ camera_id (FK)   │
                     │ event_type       │  ← entry/exit/detection/unknown/security_alert
                     │ confidence       │
                     │ snapshot_url     │
                     │ metadata (JSONB) │  ← Flexible metadata for any event type
                     │ created_at       │
                     └──────────────────┘

┌──────────────────┐     ┌──────────────────┐
│    cameras       │     │  unknown_faces   │
│──────────────────│     │──────────────────│
│ id (PK)          │     │ id (PK)          │
│ name             │     │ camera_id (FK)   │
│ location         │     │ snapshot_url     │
│ stream_url       │     │ full_frame       │
│ camera_type      │     │ encoding (TEXT)  │
│ is_active        │     │ occurrence       │
│ config (JSONB)   │     │ first_seen       │
│ created_at       │     │ last_seen        │
└──────────────────┘     │ status           │  ← pending/enrolled/dismissed
                         │ assigned_to      │
┌──────────────────┐     └──────────────────┘
│  admin_users     │
│──────────────────│
│ id (PK)          │
│ email (UNIQUE)   │
│ password_hash    │  ← bcrypt hashed
│ name             │
│ role             │  ← superadmin/admin/operator
│ is_active        │
│ created_at       │
└──────────────────┘
```

### 10.3 Database Views

Pre-built SQL views for common queries:

| View | Purpose |
|---|---|
| `active_occupancy` | Currently present persons with camera info and duration |
| `daily_summary` | Event counts grouped by date and type |
| `movements` | Enriched detection events with person and camera details |
| `movement_heatmap` | Hourly event aggregation for analytics |
| `security_alerts` | Security alerts with camera and metadata details |

### 10.4 Mock Mode

When Supabase credentials aren't configured, the system **automatically falls back to mock mode**:

- `MockStore` provides an **in-memory data store** with the same API
- Stores persons, cameras, events, unknown faces, and encodings
- Thread-safe with `threading.Lock()`
- Memory-bounded (max 100 unknown faces, 500 events)
- Enables **full development without any database setup**

---

## 11. Authentication & Security

### 11.1 JWT Authentication

**Flow:**
```
1. POST /api/auth/login  →  { email, password }
2. Server verifies password against bcrypt hash
3. Server creates JWT token:
   - Payload: { sub: user_id, email, role, name }
   - Signed with HS256 algorithm
   - Expires in 60 minutes
4. Client stores token in localStorage
5. All API requests include: Authorization: Bearer <token>
```

### 11.2 Role-Based Access Control

| Role | Permissions |
|---|---|
| `superadmin` | Full system access, user management |
| `admin` | Camera management, person enrollment, alert handling |
| `operator` | Read-only access, view dashboards and events |

### 11.3 Password Security

- **bcrypt** hashing with automatic salt generation
- Passwords are never stored in plain text
- `passlib` handles hash verification with constant-time comparison (prevents timing attacks)

---

## 12. Frontend Deep Dive

### 12.1 Next.js 16 Architecture

The frontend uses **Next.js App Router** with file-based routing:

```
src/
├── app/
│   ├── page.js               → Dashboard (/)
│   ├── layout.js             → Root layout (shared across all pages)
│   ├── globals.css            → Global styles (42KB comprehensive design system)
│   ├── login/page.js          → Authentication (/login)
│   ├── cameras/page.js        → Camera management (/cameras)
│   ├── monitor/page.js        → Live video monitor (/monitor)
│   ├── events/page.js         → Event log (/events)
│   ├── persons/page.js        → Person management (/persons)
│   ├── analytics/page.js      → Analytics dashboard (/analytics)
│   ├── security-alerts/page.js → Security alerts (/security-alerts)
│   ├── traffic/page.js        → Traffic monitor (/traffic)
│   └── unknown-faces/page.js  → Unknown face review (/unknown-faces)
├── components/
│   ├── AppShell.js            → Main layout wrapper (sidebar + content)
│   ├── Sidebar.js             → Navigation sidebar
│   ├── StatCard.js            → Dashboard statistic card
│   ├── ActivityFeed.js        → Real-time event feed
│   ├── OccupancyRing.js       → SVG donut chart for occupancy
│   ├── HourlyChart.js         → Bar chart for peak times
│   └── Icons.js               → Custom SVG icon components
└── lib/
    └── api.js                 → API client (REST + WebSocket)
```

### 12.2 Key Frontend Pages

| Page | Features |
|---|---|
| **Dashboard** | Live stats, occupancy ring, activity feed, peak times chart, WebSocket events |
| **Camera Monitor** | Live video feed (WebSocket binary stream), camera start/stop controls |
| **Person Management** | CRUD for known persons, face photo upload, enrollment |
| **Events** | Sortable/filterable event log, event type badges |
| **Unknown Faces** | Review queue for unrecognized faces, enroll or dismiss actions |
| **Security Alerts** | Filtered view of security events (crowd, loitering, hazard) |
| **Traffic Monitor** | Vehicle events, ANPR plate detections, proximity alerts |
| **Analytics** | Reports, movement heatmaps, trend analysis |

### 12.3 API Client (`api.js`)

A centralized API client class that handles:
- **REST API calls** with automatic JSON serialization/deserialization
- **JWT token management** — stores in localStorage, auto-attaches to requests
- **401 handling** — redirects to login on token expiry
- **WebSocket connections** for live events and camera streams
- **Keep-alive pings** (every 30 seconds) to prevent WebSocket timeouts

### 12.4 Real-Time Video Streaming

```javascript
// Binary WebSocket: receives JPEG frames from backend
ws.binaryType = 'arraybuffer';
ws.onmessage = (event) => {
    const blob = new Blob([event.data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    // Update <img> src to display frame
};
```

---

## 13. DevOps & Deployment

### 13.1 Docker Architecture

**Backend Dockerfile** — Multi-stage build:
```
Stage 1 (base): Python 3.11-slim
    + System dependencies (cmake, libopencv, libboost)
    + Python dependencies (pip install)

Stage 2 (production):
    + Application code
    + Snapshot directory
    + Health check
    + Runs via: python run.py
```

**Docker Compose** orchestrates both services:
- `cctv-backend` on port 8000 (with webcam passthrough on Linux)
- `cctv-frontend` on port 3000 (depends on backend)
- Shared networking for internal communication
- Health checks with automatic restart

### 13.2 Environment Configuration

```env
# Backend (.env)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key
JWT_SECRET_KEY=your-secret-key
PIPELINE_MODE=face              # face | traffic | hybrid
DETECTION_MODEL=cnn             # hog | cnn
ENABLE_HAZARD_DETECTION=true
ENABLE_CROWD_DETECTION=true
ENABLE_LOITERING_DETECTION=true
ENABLE_ANPR=true

# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 14. Data Flow — End to End

### 14.1 Person Entry Detection Flow

```
1. Camera captures frame (30 FPS)
      ↓
2. FaceDetector finds face at (top, right, bottom, left)
      ↓
3. FaceRecognizer generates 128-d encoding
      ↓
4. Encoding compared against all known faces (vectorized NumPy)
      ↓
5. Match found! person_id = "abc123", confidence = 0.82
      ↓
6. PersonTracker.on_detection() called
      ↓
7. Tracker creates TrackState: first detection of "abc123"
      ↓
8. After 3 seconds of continuous detections:
      → Entry CONFIRMED
      ↓
9. Entry event created:
   {
     event_type: "entry",
     person_id: "abc123",
     person_name: "John Doe",
     camera_id: "cam-001",
     confidence: 0.82,
     timestamp: "2026-02-23T20:00:00Z"
   }
      ↓
10. Event saved to Supabase (detection_events table)
    + Tracking session created (tracking_sessions table)
      ↓
11. Event broadcast via WebSocket to all connected dashboards
      ↓
12. Frontend updates: stats, activity feed, occupancy ring
```

### 14.2 Unknown Face Detection Flow

```
1. Face detected but doesn't match any known encoding
      ↓
2. Check against recent unknowns (dedup cache):
   - If similar face seen recently → increment counter
   - If new unknown → proceed
      ↓
3. Save three snapshots:
   a. Face crop (tight, with padding)      → /snapshots/unknowns/
   b. Context crop (head + shoulders)       → /snapshots/unknowns/
   c. Full frame                            → /snapshots/frames/
      ↓
4. Save to unknown_faces table:
   - snapshot_url, encoding, camera_id
   - status = "pending"
      ↓
5. Broadcast "unknown" event via WebSocket
      ↓
6. Admin sees alert on dashboard → navigates to Unknown Faces page
      ↓
7. Admin can:
   a. ENROLL → Creates a new person + face encoding → future detections recognized
   b. DISMISS → Marks as dismissed → ignored
```

---

## 15. Why Each Technology Was Chosen

### Backend Choices

| Choice | Alternative Considered | Why This Was Chosen |
|---|---|---|
| **FastAPI** over Flask/Django | Flask is simpler, Django is heavier | FastAPI has native async/WebSocket support, auto-docs, and Pydantic integration |
| **face_recognition** over MediaPipe | MediaPipe is faster but less accurate | face_recognition uses dlib's 99.38% accuracy CNN model; recognition quality matters more than raw speed |
| **YOLOv8** over SSD/Faster-RCNN | SSD is older, Faster-RCNN is slower | YOLOv8 offers the best speed/accuracy tradeoff with native PyTorch support |
| **Supabase** over raw PostgreSQL | Raw Postgres needs more setup | Supabase provides RLS, auth, real-time, and admin dashboard out-of-the-box |
| **EasyOCR** over Tesseract | Tesseract is traditional | EasyOCR uses deep learning and handles real-world conditions (blur, angle) far better |
| **DeepFace** over custom models | Custom training takes time | DeepFace bundles multiple pre-trained models; works out-of-the-box |
| **ThreadPoolExecutor** over multiprocessing | Multiprocessing has IPC overhead | Thread pool is simpler for I/O-bound tasks; GIL isn't an issue since heavy work is in C extensions (OpenCV, numpy, PyTorch) |

### Frontend Choices

| Choice | Alternative Considered | Why This Was Chosen |
|---|---|---|
| **Next.js 16** over React CRA | CRA is deprecated | Next.js provides SSR, file-based routing, and production optimizations |
| **Vanilla CSS** over Tailwind | Tailwind has utility-class overhead | Complete control over design system, no build-time dependencies |
| **WebSocket** over polling | Polling wastes bandwidth | WebSocket provides true real-time updates with minimal latency |

### Infrastructure Choices

| Choice | Alternative | Why This Was Chosen |
|---|---|---|
| **Docker** over bare metal | Bare metal works but isn't reproducible | Docker ensures identical environments across dev/staging/production |
| **Background threads** over Celery | Celery adds Redis dependency | Threads are simpler for camera processing; no message broker needed |
| **Mock mode** fallback | Could require database always | Enables zero-dependency development; new developers can start immediately |

---

## 16. Project Structure

```
CCTV/
├── 📄 docker-compose.yml          # Multi-container orchestration
├── 📄 .gitignore                  # Git ignore rules
├── 📄 ARCHITECTURE.md             # Architecture documentation
├── 📄 PROJECT_EXPLANATION.md      # This file
│
├── 📁 backend/                    # Python FastAPI backend
│   ├── 📄 Dockerfile              # Container build instructions
│   ├── 📄 requirements.txt        # Python dependencies (38 packages)
│   ├── 📄 run.py                  # Application entry point (uvicorn runner)
│   ├── 📄 yolov8n.pt             # YOLOv8 nano model weights (6.5 MB)
│   ├── 📄 .env                    # Environment variables (secrets)
│   ├── 📄 .env.example            # Template for environment setup
│   │
│   ├── 📁 app/                    # Main application package
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main.py            # FastAPI app, lifespan, event handlers
│   │   ├── 📄 config.py          # Pydantic Settings configuration
│   │   ├── 📄 database.py        # Supabase client singleton
│   │   │
│   │   ├── 📁 core/              # Core infrastructure
│   │   │   ├── 📄 security.py    # JWT auth, password hashing, RBAC
│   │   │   ├── 📄 websocket.py   # WebSocket connection manager
│   │   │   ├── 📄 mock_store.py  # In-memory store for dev mode
│   │   │   └── 📄 exceptions.py  # Custom exception classes
│   │   │
│   │   ├── 📁 models/            # Pydantic data models
│   │   │   ├── 📄 auth.py        # Login/Token schemas
│   │   │   ├── 📄 camera.py      # Camera CRUD schemas
│   │   │   ├── 📄 event.py       # Event + tracking session schemas
│   │   │   ├── 📄 movement.py    # Movement log schemas
│   │   │   └── 📄 person.py      # Person + face encoding schemas
│   │   │
│   │   ├── 📁 routes/            # API endpoint definitions
│   │   │   ├── 📄 auth.py        # POST /api/auth/login, GET /me
│   │   │   ├── 📄 cameras.py     # CRUD + start/stop + WebSocket stream
│   │   │   ├── 📄 events.py      # GET /api/events (filterable)
│   │   │   ├── 📄 persons.py     # CRUD + face photo upload
│   │   │   ├── 📄 unknown_faces.py # Review queue + enroll/dismiss
│   │   │   ├── 📄 analytics.py   # Dashboard, peak times, reports
│   │   │   └── 📄 movements.py   # Movement logs
│   │   │
│   │   ├── 📁 services/          # Business logic layer
│   │   │   ├── 📄 auth_service.py
│   │   │   ├── 📄 camera_service.py
│   │   │   ├── 📄 person_service.py
│   │   │   ├── 📄 analytics_service.py
│   │   │   └── 📄 movement_service.py
│   │   │
│   │   ├── 📁 vision/            # 🧠 AI Vision Processing
│   │   │   ├── 📄 pipeline.py    # Main vision pipeline (1,526 lines)
│   │   │   ├── 📄 detector.py    # Face detection (HOG/CNN)
│   │   │   ├── 📄 recognizer.py  # Face recognition (128-d matching)
│   │   │   ├── 📄 tracker.py     # Entry/exit tracking state machine
│   │   │   ├── 📄 safety_analytics.py # Crowd + loitering + attributes
│   │   │   ├── 📄 hazard_detector.py  # YOLOv8 weapon detection
│   │   │   ├── 📄 anpr.py        # License plate recognition
│   │   │   ├── 📄 traffic.py     # Vehicle-pedestrian safety
│   │   │   └── 📄 camera_worker.py    # Background thread management
│   │   │
│   │   ├── 📁 utils/             # Utility functions
│   │   │   ├── 📄 image_utils.py # Frame encoding, resizing, snapshot saving
│   │   │   └── 📄 time_utils.py  # Timestamp formatting helpers
│   │   │
│   │   └── 📁 migrations/        # SQL schema files
│   │       ├── 📄 001_initial_schema.sql   # Core tables + RLS + views
│   │       └── 📄 002_safety_modules.sql   # Security alert schema updates
│   │
│   └── 📁 snapshots/             # Saved face crops and frames
│
├── 📁 frontend/                   # Next.js React frontend
│   ├── 📄 package.json            # npm dependencies
│   ├── 📄 next.config.mjs         # Next.js configuration
│   ├── 📄 .env.local              # Frontend environment variables
│   │
│   └── 📁 src/
│       ├── 📁 app/                # Next.js App Router pages
│       │   ├── 📄 page.js         # Dashboard
│       │   ├── 📄 layout.js       # Root layout + metadata
│       │   ├── 📄 globals.css     # Design system (42KB)
│       │   ├── 📁 login/          # Authentication page
│       │   ├── 📁 cameras/        # Camera management
│       │   ├── 📁 monitor/        # Live video feed
│       │   ├── 📁 events/         # Event log
│       │   ├── 📁 persons/        # Person management
│       │   ├── 📁 analytics/      # Analytics dashboard
│       │   ├── 📁 security-alerts/ # Security alerts
│       │   ├── 📁 traffic/        # Traffic monitor
│       │   └── 📁 unknown-faces/  # Unknown face review
│       │
│       ├── 📁 components/         # Reusable React components
│       │   ├── 📄 AppShell.js     # Layout wrapper
│       │   ├── 📄 Sidebar.js      # Navigation
│       │   ├── 📄 StatCard.js     # Metric cards
│       │   ├── 📄 ActivityFeed.js # Event feed
│       │   ├── 📄 OccupancyRing.js # SVG donut chart
│       │   ├── 📄 HourlyChart.js  # Bar chart
│       │   └── 📄 Icons.js        # SVG icons
│       │
│       └── 📁 lib/
│           └── 📄 api.js          # REST + WebSocket API client
│
└── 📁 .git/                       # Git version control
```

---

## 🎯 Summary — What Makes This Project Important

### 1. **Real-World Security Application**
This isn't a toy project — it solves real security challenges: identifying unauthorized entry, detecting weapons, preventing vehicle accidents, and managing physical access control.

### 2. **Multiple AI Models Working Together**
The system demonstrates the integration of:
- **dlib CNN/HOG** for face detection
- **ResNet-based embeddings** for face recognition
- **YOLOv8** for object detection
- **DeepFace** for attribute analysis
- **EasyOCR** for text recognition
- **Custom algorithms** for crowd detection, loitering, and proximity analysis

### 3. **Production-Grade Engineering**
- Thread-safe concurrent processing with per-camera isolation
- Mock mode for zero-dependency development
- Comprehensive error handling with graceful degradation
- Real-time WebSocket streaming with auto-reconnection
- JWT authentication with RBAC
- Database migrations and RLS policies
- Docker containerization for deployment

### 4. **Scalable Architecture**
- Supports up to 10 simultaneous cameras
- Each camera runs in an independent thread with its own pipeline
- Background thread pools for heavy AI inference
- Frame skipping and throttling for performance management
- Pre-encoded JPEG caching to prevent async event loop blocking

### 5. **Full-Stack Integration**
End-to-end system from camera hardware → AI processing → database → real-time dashboard, demonstrating mastery of:
- Computer Vision (OpenCV, face_recognition)
- Deep Learning (PyTorch/TF, YOLOv8, DeepFace)
- Backend Engineering (FastAPI, async Python, threading)
- Database Design (PostgreSQL, SQL views, RLS)
- Frontend Development (React, Next.js, WebSockets)
- DevOps (Docker, Docker Compose)

---

*Document generated on: February 23, 2026*
*Project: SentinelAI — AI-Powered CCTV Surveillance System v1.0.0*
