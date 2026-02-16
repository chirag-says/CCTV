# 🔒 AI-Powered CCTV Surveillance System — Architecture Document

## 1. System Overview

A production-grade, AI-powered CCTV surveillance system that provides real-time face detection, recognition, entry/exit tracking, unknown person management, and analytics — designed as a startup-grade SaaS product.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js)                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │Dashboard │ │ Live     │ │ Admin    │ │ Analytics          │  │
│  │  Home    │ │ Monitor  │ │ Enroll   │ │ Reports            │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ REST API + WebSocket
┌────────────────────────────▼─────────────────────────────────────┐
│                      BACKEND (FastAPI)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Auth Module   │  │ Camera Mgmt  │  │ Person Management     │ │
│  ├──────────────┤  ├──────────────┤  ├────────────────────────┤ │
│  │ Analytics API │  │ Event Logger │  │ Admin Enrollment      │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    AI VISION PIPELINE                             │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ Frame    │  │ Face         │  │ Face Recognition        │   │
│  │ Capture  │──│ Detection    │──│ (Encoding Match)        │   │
│  └──────────┘  └──────────────┘  └─────────────────────────┘   │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ Tracker  │  │ Entry/Exit   │  │ Unknown Face Handler    │   │
│  │ State    │──│ Logic        │──│ (Snapshot + Queue)      │   │
│  └──────────┘  └──────────────┘  └─────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    SUPABASE (PostgreSQL)                          │
│  ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ persons  │ │ face_encod.. │ │ events   │ │ cameras      │   │
│  │ unknown_ │ │ sessions     │ │ analytics│ │ admin_users  │   │
│  │ faces    │ │              │ │          │ │              │   │
│  └──────────┘ └──────────────┘ └──────────┘ └──────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Supabase Storage (Snapshots)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Database Schema

### 3.1 `persons` — Known Individuals
| Column       | Type         | Description                    |
|-------------|-------------|-------------------------------|
| id          | UUID (PK)   | Unique identifier              |
| name        | VARCHAR(255)| Full name                      |
| role        | VARCHAR(50) | employee/visitor/vip/banned    |
| department  | VARCHAR(100)| Optional department            |
| phone       | VARCHAR(20) | Contact number                 |
| email       | VARCHAR(255)| Contact email                  |
| avatar_url  | TEXT        | Profile image URL              |
| is_active   | BOOLEAN     | Soft delete flag               |
| created_at  | TIMESTAMPTZ | Record creation time           |
| updated_at  | TIMESTAMPTZ | Last update time               |

### 3.2 `face_encodings` — Stored Face Encodings
| Column       | Type         | Description                    |
|-------------|-------------|-------------------------------|
| id          | UUID (PK)   | Unique identifier              |
| person_id   | UUID (FK)   | References persons.id          |
| encoding    | BYTEA       | 128-d face encoding vector     |
| source_image| TEXT        | URL of source image used       |
| quality     | FLOAT       | Encoding quality score (0-1)   |
| created_at  | TIMESTAMPTZ | When encoding was generated    |

### 3.3 `cameras` — Camera Registry
| Column        | Type         | Description                   |
|--------------|-------------|-------------------------------|
| id           | UUID (PK)   | Unique identifier              |
| name         | VARCHAR(100)| Camera display name            |
| location     | VARCHAR(255)| Physical location              |
| stream_url   | TEXT        | RTSP/HTTP stream URL           |
| camera_type  | VARCHAR(20) | webcam/rtsp/ip                 |
| is_active    | BOOLEAN     | Is camera currently active     |
| config       | JSONB       | Camera-specific settings       |
| created_at   | TIMESTAMPTZ | Registration time              |

### 3.4 `tracking_sessions` — Entry/Exit Sessions
| Column        | Type         | Description                   |
|--------------|-------------|-------------------------------|
| id           | UUID (PK)   | Unique identifier              |
| person_id    | UUID (FK)   | References persons.id          |
| camera_id    | UUID (FK)   | Camera that detected           |
| entry_time   | TIMESTAMPTZ | First detection timestamp      |
| exit_time    | TIMESTAMPTZ | Last seen + threshold          |
| duration_sec | INTEGER     | Computed duration              |
| status       | VARCHAR(20) | active/completed               |
| created_at   | TIMESTAMPTZ | Record creation                |

### 3.5 `detection_events` — All Detection Events
| Column        | Type         | Description                   |
|--------------|-------------|-------------------------------|
| id           | UUID (PK)   | Unique identifier              |
| person_id    | UUID (FK)   | References persons.id (nullable)|
| camera_id    | UUID (FK)   | Which camera                   |
| event_type   | VARCHAR(20) | entry/exit/detection/unknown   |
| confidence   | FLOAT       | Match confidence score         |
| snapshot_url | TEXT        | Frame snapshot URL             |
| metadata     | JSONB       | Additional event data          |
| created_at   | TIMESTAMPTZ | Event timestamp                |

### 3.6 `unknown_faces` — Unknown Face Queue
| Column        | Type         | Description                   |
|--------------|-------------|-------------------------------|
| id           | UUID (PK)   | Unique identifier              |
| camera_id    | UUID (FK)   | Where detected                 |
| snapshot_url | TEXT        | Cropped face image URL         |
| full_frame   | TEXT        | Full frame snapshot URL        |
| encoding     | BYTEA       | Temp face encoding             |
| occurrence   | INTEGER     | How many times seen            |
| first_seen   | TIMESTAMPTZ | First detection time           |
| last_seen    | TIMESTAMPTZ | Most recent detection          |
| status       | VARCHAR(20) | pending/enrolled/dismissed     |
| assigned_to  | UUID (FK)   | Admin reviewing (nullable)     |
| created_at   | TIMESTAMPTZ | Record creation                |

### 3.7 `admin_users` — System Administrators
| Column       | Type         | Description                    |
|-------------|-------------|-------------------------------|
| id          | UUID (PK)   | Unique identifier              |
| email       | VARCHAR(255)| Login email                    |
| password_hash| TEXT       | Hashed password                |
| name        | VARCHAR(255)| Display name                   |
| role        | VARCHAR(20) | superadmin/admin/operator       |
| is_active   | BOOLEAN     | Account status                 |
| created_at  | TIMESTAMPTZ | Account creation               |

---

## 4. API Endpoints

### 4.1 Authentication
| Method | Endpoint              | Description           |
|--------|----------------------|----------------------|
| POST   | /api/auth/login      | Admin login           |
| POST   | /api/auth/logout     | Admin logout          |
| GET    | /api/auth/me         | Current user info     |

### 4.2 Persons Management
| Method | Endpoint                    | Description                |
|--------|----------------------------|---------------------------|
| GET    | /api/persons               | List all known persons     |
| POST   | /api/persons               | Add new person             |
| GET    | /api/persons/{id}          | Get person details         |
| PUT    | /api/persons/{id}          | Update person info         |
| DELETE | /api/persons/{id}          | Soft delete person         |
| POST   | /api/persons/{id}/encodings| Upload face for encoding   |
| GET    | /api/persons/{id}/history  | Get person's tracking hist |

### 4.3 Camera Management
| Method | Endpoint                    | Description                |
|--------|----------------------------|---------------------------|
| GET    | /api/cameras               | List all cameras           |
| POST   | /api/cameras               | Register new camera        |
| GET    | /api/cameras/{id}          | Get camera details         |
| PUT    | /api/cameras/{id}          | Update camera config       |
| DELETE | /api/cameras/{id}          | Remove camera              |
| GET    | /api/cameras/{id}/stream   | WebSocket live stream      |
| POST   | /api/cameras/{id}/start    | Start processing           |
| POST   | /api/cameras/{id}/stop     | Stop processing            |

### 4.4 Unknown Faces (Admin Queue)
| Method | Endpoint                         | Description              |
|--------|----------------------------------|-------------------------|
| GET    | /api/unknown-faces              | List unknown faces queue |
| GET    | /api/unknown-faces/{id}         | Get unknown face detail  |
| POST   | /api/unknown-faces/{id}/enroll  | Enroll as known person   |
| POST   | /api/unknown-faces/{id}/dismiss | Dismiss unknown face     |

### 4.5 Events & Tracking
| Method | Endpoint                    | Description                |
|--------|----------------------------|---------------------------|
| GET    | /api/events                | List detection events      |
| GET    | /api/events/live           | WebSocket live events      |
| GET    | /api/sessions              | List tracking sessions     |
| GET    | /api/sessions/active       | Currently present people   |

### 4.6 Analytics
| Method | Endpoint                    | Description                |
|--------|----------------------------|---------------------------|
| GET    | /api/analytics/dashboard   | Dashboard summary data     |
| GET    | /api/analytics/peak-times  | Peak entry/exit times      |
| GET    | /api/analytics/occupancy   | Real-time occupancy        |
| GET    | /api/analytics/movement    | Movement logs & patterns   |
| GET    | /api/analytics/reports     | Generate reports           |

---

## 5. Recognition Pipeline Logic

```
Frame Captured
    │
    ▼
┌──────────────────┐
│ Pre-processing   │  Resize to 640px, convert BGR→RGB
│                  │  Skip frames (process every Nth frame)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Face Detection   │  face_recognition.face_locations()
│ (HOG or CNN)     │  Returns bounding boxes
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Face Encoding    │  face_recognition.face_encodings()
│                  │  128-dimensional vector per face
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Face Matching    │  Compare against known encodings
│                  │  Tolerance: 0.45 (strict)
│                  │  Use face_recognition.compare_faces()
│                  │  + face_recognition.face_distance()
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
  MATCH    NO MATCH
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│Track   │ │ Unknown Face │
│Person  │ │ Handler      │
│Entry/  │ │ Save snap    │
│Exit    │ │ Add to queue │
└────────┘ └──────────────┘
```

---

## 6. Entry/Exit Algorithm

```python
# Pseudo-code for Entry/Exit Tracking

ENTRY_THRESHOLD = 3          # seconds - minimum presence to confirm entry
EXIT_THRESHOLD = 300         # seconds (5 min) - absence before marking exit
COOLDOWN_PERIOD = 60         # seconds - prevent rapid re-entry

class PersonTracker:
    def __init__(self):
        self.active_tracks = {}  # person_id → TrackState

    def on_detection(self, person_id, camera_id, timestamp):
        if person_id in self.active_tracks:
            track = self.active_tracks[person_id]
            track.last_seen = timestamp
            track.detection_count += 1

            # Confirm entry after threshold
            if not track.entry_confirmed:
                if (timestamp - track.first_seen) >= ENTRY_THRESHOLD:
                    track.entry_confirmed = True
                    emit_event("ENTRY", person_id, camera_id)
        else:
            # New detection - start tracking
            self.active_tracks[person_id] = TrackState(
                first_seen=timestamp,
                last_seen=timestamp,
                camera_id=camera_id
            )

    def check_exits(self, current_time):
        for person_id, track in list(self.active_tracks.items()):
            if track.entry_confirmed:
                if (current_time - track.last_seen) >= EXIT_THRESHOLD:
                    emit_event("EXIT", person_id, track.camera_id)
                    del self.active_tracks[person_id]
```

---

## 7. Unknown Face Workflow

```
Unknown Face Detected
        │
        ▼
┌───────────────────┐
│ Similarity Check  │  Compare vs existing unknowns
│ Against Queue     │  (avoid duplicate entries)
└────────┬──────────┘
    ┌────┴────┐
    │         │
 SIMILAR    NEW
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│Update  │ │ Create new   │
│count + │ │ unknown_face │
│last_   │ │ Save snapshot│
│seen    │ │ Save encoding│
└────────┘ └──────────────┘
               │
               ▼
        ┌──────────────┐
        │ Admin Queue  │
        │ Notification │
        └──────┬───────┘
               │
          Admin Reviews
          ┌────┴────┐
          │         │
       ENROLL    DISMISS
          │         │
          ▼         ▼
     ┌────────┐ ┌────────┐
     │Create  │ │Mark as │
     │Person  │ │dismissed│
     │+Encode │ └────────┘
     └────────┘
```

---

## 8. Admin Enrollment Logic

1. Admin views unknown face queue (sorted by occurrence count)
2. Admin selects an unknown face entry
3. Admin fills in person details (name, role, department, etc.)
4. System creates a new `person` record
5. System moves the stored encoding to `face_encodings` table
6. System marks the `unknown_face` as `enrolled`
7. System reloads the in-memory encoding cache
8. Future detections will now match this person

---

## 9. Scalability Considerations

### 9.1 Camera Pipeline Scaling
- **Worker Pool**: Each camera runs in its own process/thread
- **Frame Skip**: Process every Nth frame (configurable per camera)
- **Resolution Scaling**: Downscale frames for detection, use original for snapshots
- **GPU Acceleration**: Optional CUDA support for CNN-based detection

### 9.2 Face Matching Optimization
- **In-Memory Cache**: Load all encodings into memory at startup
- **Batch Comparison**: Compare detected face against all known in one numpy operation
- **KD-Tree Index**: For large databases (>1000 faces), use spatial indexing
- **Encoding Refresh**: Background job to reload encodings every N minutes

### 9.3 Database Optimization
- **Connection Pooling**: Use async connection pool
- **Batch Inserts**: Buffer detection events, insert in batches
- **Partitioning**: Partition events table by date
- **Indexes**: On person_id, camera_id, created_at, event_type

### 9.4 Horizontal Scaling
- **Camera Workers**: Deploy as separate microservices
- **Message Queue**: Redis/RabbitMQ between detection and processing
- **Load Balancer**: Nginx for API distribution
- **CDN**: For snapshot image delivery

---

## 10. Performance Optimizations

| Optimization              | Impact     | Implementation                    |
|--------------------------|-----------|----------------------------------|
| Frame skip (every 3rd)   | 60% less  | Configurable per camera           |
| Resize to 480p           | 40% faster| Before face detection             |
| Batch encoding compare   | 5x faster | NumPy vectorized operations       |
| Async DB writes          | Non-block | Background task queue             |
| Encoding cache           | 10x faster| In-memory dict with periodic sync |
| JPEG compression         | 70% less  | For snapshot storage              |
| WebSocket vs polling     | 90% less  | For live updates                  |
| Process pool             | CPU util  | One process per camera            |

---

## 11. Project Structure

```
CCTV/
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app entry
│   │   ├── config.py           # Settings & env vars
│   │   ├── database.py         # Supabase connection
│   │   ├── models/             # Pydantic models
│   │   │   ├── __init__.py
│   │   │   ├── person.py
│   │   │   ├── camera.py
│   │   │   ├── event.py
│   │   │   └── auth.py
│   │   ├── routes/             # API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── persons.py
│   │   │   ├── cameras.py
│   │   │   ├── events.py
│   │   │   ├── unknown_faces.py
│   │   │   └── analytics.py
│   │   ├── services/           # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── auth_service.py
│   │   │   ├── person_service.py
│   │   │   ├── camera_service.py
│   │   │   └── analytics_service.py
│   │   ├── vision/             # AI Pipeline
│   │   │   ├── __init__.py
│   │   │   ├── detector.py     # Face detection
│   │   │   ├── recognizer.py   # Face recognition
│   │   │   ├── tracker.py      # Entry/Exit tracking
│   │   │   ├── pipeline.py     # Main processing pipeline
│   │   │   └── camera_worker.py# Camera stream handler
│   │   ├── core/               # Shared utilities
│   │   │   ├── __init__.py
│   │   │   ├── security.py     # JWT & password hashing
│   │   │   ├── exceptions.py   # Custom exceptions
│   │   │   └── websocket.py    # WebSocket manager
│   │   └── migrations/         # DB migration scripts
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/                   # Next.js Frontend
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── lib/
│   │   └── styles/
│   ├── package.json
│   └── next.config.js
├── docker-compose.yml
├── ARCHITECTURE.md
└── README.md
```
