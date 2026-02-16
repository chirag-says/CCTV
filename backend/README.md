# 🔒 AI CCTV Surveillance System — Backend

Production-grade FastAPI backend for AI-powered video surveillance with face detection, recognition, entry/exit tracking, unknown face management, and analytics.

---

## 🏗️ Architecture

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry + event handlers
│   ├── config.py            # Pydantic settings (.env)
│   ├── database.py          # Supabase client singleton
│   │
│   ├── models/              # Pydantic request/response schemas
│   │   ├── auth.py          # Login, Token, AdminUser
│   │   ├── person.py        # Person, FaceEncoding
│   │   ├── camera.py        # Camera CRUD
│   │   ├── event.py         # DetectionEvent, TrackingSession, UnknownFace
│   │   └── movement.py      # Movement tracking models
│   │
│   ├── routes/              # API endpoint routers
│   │   ├── auth.py          # POST /api/auth/login, /register, /me
│   │   ├── persons.py       # CRUD /api/persons + face encoding upload
│   │   ├── cameras.py       # CRUD /api/cameras + start/stop + WS stream
│   │   ├── events.py        # GET /api/events + WS live events
│   │   ├── unknown_faces.py # GET/POST /api/unknown-faces + enroll/dismiss
│   │   ├── analytics.py     # Dashboard, peak times, occupancy, reports
│   │   └── movements.py     # Movement timeline, camera activity, summary
│   │
│   ├── services/            # Business logic layer
│   │   ├── auth_service.py
│   │   ├── person_service.py
│   │   ├── camera_service.py
│   │   ├── analytics_service.py
│   │   └── movement_service.py
│   │
│   ├── vision/              # AI pipeline
│   │   ├── detector.py      # Face detection (HOG/CNN)
│   │   ├── recognizer.py    # Face recognition + encoding cache
│   │   ├── tracker.py       # Entry/Exit tracking state machine
│   │   ├── pipeline.py      # Frame → Detect → Recognize → Track
│   │   └── camera_worker.py # Thread-based camera manager
│   │
│   ├── core/                # Shared infrastructure
│   │   ├── security.py      # JWT + bcrypt auth
│   │   ├── exceptions.py    # Custom HTTP exceptions
│   │   └── websocket.py     # WS connection manager
│   │
│   ├── utils/               # Utility helpers
│   │   ├── image_utils.py   # Snapshot saving, compression, streaming
│   │   └── time_utils.py    # UTC helpers, duration formatting
│   │
│   └── migrations/
│       └── 001_initial_schema.sql
│
├── requirements.txt
├── run.py                    # Application entry point
├── Dockerfile
├── .env.example
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd backend
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
copy .env.example .env
# Edit .env with your Supabase credentials
```

### 3. Setup Database
Run `app/migrations/001_initial_schema.sql` in your Supabase SQL Editor.

### 4. Run Server
```bash
python run.py
```

Server starts at `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 📡 API Overview

| Module          | Prefix                 | Description                  |
|----------------|------------------------|------------------------------|
| Auth           | `/api/auth`            | JWT login/register           |
| Persons        | `/api/persons`         | Known person CRUD + encoding |
| Cameras        | `/api/cameras`         | Camera CRUD + pipeline ctrl  |
| Events         | `/api/events`          | Detection event log          |
| Unknown Faces  | `/api/unknown-faces`   | Admin enrollment queue       |
| Analytics      | `/api/analytics`       | Dashboard & reports          |
| Movements      | `/api/movements`       | Movement timeline & activity |

### WebSocket Endpoints
- `ws://localhost:8000/api/events/live` — Real-time detection events
- `ws://localhost:8000/api/cameras/{id}/stream` — Live JPEG frame stream

---

## 🧠 Vision Pipeline

```
Frame → Downscale → Face Detection → Encoding → Recognition → Tracking → Event
                                                      │              │
                                                   MATCH         NO MATCH
                                                      │              │
                                                   Track         Save Unknown
                                                   Entry/Exit    Add to Queue
```

**Key Parameters** (configurable via `.env`):
- `FACE_MATCH_TOLERANCE`: 0.45 (strict matching)
- `FRAME_SKIP`: 3 (process every 3rd frame)
- `DETECTION_SCALE`: 0.5 (downscale for speed)
- `EXIT_THRESHOLD_SECONDS`: 300 (5 min absence → exit)
- `ENTRY_THRESHOLD_SECONDS`: 3 (3 sec presence → entry)

---

## 🐳 Docker

```bash
# From project root
docker compose up --build
```

---

## 🔐 Default Credentials

**Mock mode** (no Supabase):
- Email: `admin@cctv.local`
- Password: `admin123`
