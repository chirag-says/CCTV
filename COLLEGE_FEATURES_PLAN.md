# SentinelAI — College Management Features: Implementation Plan

**Scope:** Person Journey · Faculty Attendance · Daily Campus Summary · Teachers On Campus · Incident Evidence Pack
**Status:** Plan only — nothing implemented.
**Basis:** Full read of `backend/app/**`, `frontend/src/**`, `app/db/models.py`, `app/migrations/*.sql`, seed scripts.

Every claim below is tagged:
- **[CONFIRMED]** — read directly from code, with file:line.
- **[INFERENCE]** — reasoned from code, not directly stated.
- **[MISSING]** — does not exist; must be built.

---

# 1. Executive Summary

All five features can be built as **read-only aggregation layers over the `detection_events` table**, which is already populated, durable, indexed, and restart-safe. The vision pipeline needs **zero changes** for features 1–4 and only one optional change for feature 5.

The single most important finding:

> **`detection_events` already contains everything needed for the journey and attendance features.**
> Every time a known person becomes visible on a camera for ≥3s, an `entry` row is written with `person_id`, `camera_id`, `confidence` and `created_at`. When they leave that camera's view for ≥30s, an `exit` row is written with `duration_sec`. A person walking Main Gate → Academic Block → CSE Block produces exactly the entry sequence the "Rahul Sharma 08:42 / 08:47 / 09:15" mockup asks for — **it is already being recorded today.**

The second most important finding, and the one that dictates the whole design:

> **Entry/exit are PER-CAMERA and IN-MEMORY, not campus-level.**
> `PersonTracker` is instantiated per `VisionPipeline` (`pipeline.py:109`). "Entry" means *"became visible on camera X"*, not *"entered the campus"*. "Currently present" from the tracker means *"visible on some camera within the last 30 seconds"* — a teacher sitting in a classroom is **not** present by that definition, and all tracker state is lost on restart.

Therefore:
- **Journey (F1)** — near-free. The data is a direct read of existing rows. Reuses the existing `PersonTimeline` component.
- **Attendance (F2)** — derived nightly-free from event aggregates. **Do not use `tracking_sessions`** (see §10.2). **Do not treat one `entry` as attendance** (see §3.2).
- **On Campus (F4)** — must be a *derived* presence model (last-seen decay + gate rule), **not** the live tracker. The live tracker is layered on top as a "visible right now" badge.
- **Daily Summary (F3)** — pure aggregation of existing tables + one existing signal (`stream_offline`/`stream_recovered` events) for camera availability. Narrative generated from rule templates — **no LLM needed**.
- **Incidents (F5)** — `security_alert` rows already exist and are persisted. They are **missing snapshots**. Unknown-person evidence (face crop, context crop, full frame) **does** exist and is rich (20,222 unknown crops, 11,888 full frames on disk). **No video clips exist anywhere in the system** — do not design recording.

**Total new backend code: ~3 services + 3 routers (~900 lines). Total new DB objects: 1 column, 1 small table, 2 indexes. Total new frontend: 4 pages + 5 components.**

Realistic effort: **9–11 developer-days** for MUST HAVE scope.

---

# 2. Current System Capabilities

## 2.1 Data that already exists and is persisted

| Table | Purpose | Relevance |
|---|---|---|
| `persons` | id, **name, role, department**, phone, email, avatar_url, is_active | Faculty roster + departments **already stored** |
| `detection_events` | person_id, camera_id, **event_type**, subtype, confidence, snapshot_url, `metadata_json`, created_at | **The backbone for F1–F4** |
| `cameras` | id, name, **location**, stream_url, camera_type, is_active, **`config` JSON** | Names/locations for journey. `config` is a free JSON column — usable for zone/gate tagging with **no migration** |
| `tracking_sessions` | person_id, camera_id, entry_time, exit_time, duration_sec, status | Exists but **unreliable** — see §10.2 |
| `unknown_faces` | camera_id, **snapshot_url, full_frame**, encoding, occurrence, first_seen, last_seen, status, cluster_id | Stable cross-camera/cross-restart unknown identity + **evidence images** |
| `face_encodings` | Multi-view gallery per person, pose/quality metadata | Recognition, no change needed |
| `unknown_clusters`, `identity_review_candidates`, `gallery_updates` | AI identity review | Not needed for these features |

**[CONFIRMED]** `app/db/models.py:41-186`.

## 2.2 Event types actually written to `detection_events`

| `event_type` | `subtype` | Written by | person_id? | snapshot? |
|---|---|---|---|---|
| `entry` | — | `tracker.py:148` via `pipeline._handle_entry_event` | ✅ known persons only | ❌ |
| `exit` | — | `tracker.py:228` | ✅ known persons only | ❌ |
| `unknown` | — | `main.py:482` after unknown face persist | ❌ (NULL) | ✅ face crop |
| `security_alert` | `gathering` | `safety_analytics.py:390` | ❌ | ❌ |
| `security_alert` | `loitering` | `safety_analytics.py:547` | ❌ | ❌ |
| `security_alert` | `hazard` | `hazard_detector.py:320,426` | ❌ | ❌ |
| `security_alert` | `vehicle_proximity` | `traffic.py` | ❌ | ❌ |
| `security_alert` | `stream_offline` / `stream_recovered` | `stream_health.py:296,322,337` | ❌ | ❌ |
| `vehicle_entry` | — | `anpr.py:1179` | ❌ | ❌ (plate in metadata) |
| `vehicle_count` | — | `pipeline.py:1299` | **NOT PERSISTED** — broadcast only (`main.py:85`) | — |

**[CONFIRMED]** All of the above.

**[CONFIRMED]** There is **no `detection` event type ever emitted**. The `EVENT_ICONS.detection` entry in `PersonTimeline.js:15` is dead config.

## 2.3 FK guard behaviour for unknown persons — important

`main.py:158-170`: if `person_id` is not a known enrolled person, it is **moved out of the FK column into `metadata_json`**:
- `anon:<n>` handles → `metadata_json.anonymous = true`, `person_id = NULL`
- stable unknown ids → `metadata_json.unknown_person_id = <id>`, `person_id = NULL`

**[CONFIRMED]** This means unknown-person timelines must query `metadata_json->>'unknown_person_id'`, not `person_id`. The existing export already does this (`analytics_service.py:275`).

## 2.4 Existing APIs that can be reused directly

| Endpoint | File | Reuse for |
|---|---|---|
| `GET /api/persons` (`role`, `search`, `limit`) | `routes/persons.py:21` | Faculty roster |
| `GET /api/persons/{id}` | `routes/persons.py:51` | Person header |
| `GET /api/movements/person/{id}/timeline?days=&limit=` | `routes/movements.py:76` | **F1 journey raw feed** |
| `GET /api/events?event_type=&subtype=&camera_id=&start_date=` | `routes/events.py:46` | F5 incident list |
| `GET /api/analytics/dashboard` | `routes/analytics.py:14` | F3 base metrics |
| `GET /api/analytics/peak-times?days=` | `routes/analytics.py:22` | F3 peak arrival hour |
| `GET /api/analytics/export?format=csv\|xlsx\|json` | `routes/analytics.py:75` | **F2 export — copy this pattern verbatim** |
| `GET /api/cameras` | `routes/cameras.py:101` | Camera name/location lookup |
| `GET /api/cameras/active-persons` | `routes/cameras.py:75` | F4 "visible right now" |
| `GET /api/sessions/active` | `routes/events.py:158` | Same as above (duplicate of the one above — see §10.7) |
| `WS /api/events/live` | `routes/events.py:92` | **F4 real-time updates — already wired into the frontend** |
| `GET /api/unknown-faces` | `routes/unknown_faces.py:42` | F5 unknown-person evidence |

## 2.5 Existing frontend that can be reused

**[CONFIRMED]** Next.js 16 App Router, plain JS (no TS), **no chart/UI library** — everything is hand-rolled CSS in `globals.css` (2465 lines) using CSS custom properties with a light/dark theme.

Reusable primitives:
- `AppShell`, `Sidebar` (nav config is a plain array — `Sidebar.js:53`)
- `StatCard` (variants: primary/accent/success/warning/danger)
- `HourlyChart`, `OccupancyRing`, `ActivityHeatmap`, `ActivityFeed`, `CameraHealth`
- `PersonTimeline` (slide-out panel, already calls the timeline endpoint) — **F1 builds on this**
- `Lightbox` — **F5 evidence viewer, already exists**
- `SearchPalette`, `Skeleton`, `ErrorBoundary`, `ToastContext`
- CSS classes: `.card`, `.stats-grid`, `.data-table`, `.badge.{entry,exit,active,offline,pending}`, `.security-alert-card.severity-{high,medium,low}`, `.empty-state`, `.modal`, `.security-stats`
- `api.js` client with JWT + WS helpers (`lib/api.js`)

**[CONFIRMED]** `api.js` exposes `api.request(path)` publicly, so new endpoints can be called without editing the client — but adding named methods keeps it consistent.

## 2.6 What the seed data tells us

**[CONFIRMED]** `seed_teachers.py` creates 5 persons with `role="assistant_professor"`, `department="CSE"`. `seed_student.py` creates 2 with `role="student"`, `department="CSE"`.

So: **faculty vs student is already distinguishable via `role`**, but only by convention, and the Persons UI role dropdown (`persons/page.js:119-123`) offers only `employee/visitor/vip/banned` — it cannot even display or set the seeded values. Deployment scale is ~7 people, 1 department. Demo data will need widening to CSE/ECE/Mech/Civil to match the mockup.

---

# 3. Feature-by-Feature Analysis

## Feature 1 — "Where Is This Person?" / Person Journey

### What already exists
- **[CONFIRMED]** `MovementService.get_person_timeline(person_id, days, limit)` — `movement_service.py:185`. Returns all `detection_events` for a person, newest first.
- **[CONFIRMED]** `GET /api/movements/person/{id}/timeline` — `movements.py:76`.
- **[CONFIRMED]** `PersonTimeline.js` — a working slide-out timeline panel, grouped by date, with 24h/7d/30d/90d selector. Already wired into `/persons` (History button, `persons/page.js:232`) and `/persons/[id]`.
- **[CONFIRMED]** Entry events per camera are exactly the movement record needed. `tracker.py:139-141` even updates `track.camera_id` when a person moves — but that is within one pipeline, so it never fires cross-camera.

### What can be reused
- The timeline endpoint and service (extend, don't replace)
- `PersonTimeline` component (extend with a "Today / Journey" mode)
- `Icons.js` `EntryIcon`/`ExitIcon`/`CameraIcon`
- `/persons/[id]` page tab structure (`activeTab` state already exists, `persons/[id]/page.js:118`)

### What is missing
1. **[MISSING]** Camera **names** in the timeline response. `_event_to_dict()` (`movement_service.py:22`) returns `camera_id` only. `PersonTimeline.js:196` renders `event.camera_name || event.camera_id` → the UI currently shows a raw UUID. **This is a live bug that makes the feature look broken.**
2. **[MISSING]** A **date filter**. The endpoint takes `days` (relative lookback) only — cannot ask for "27 Aug".
3. **[MISSING]** Derived journey summary: first seen, last seen, camera-to-camera hops, dwell per location.
4. **[MISSING]** Journey for **unknown persons** — `person_id` is NULL for them (§2.3), so `/timeline` returns nothing for `Unknown #382`. Needed for F5.
5. **[MISSING]** Any visual camera-to-camera representation.

### Proposed implementation
A `JourneyService` that runs one query and folds the rows into "stops":

```
events for (person, local day) ordered ASC
  → collapse consecutive events on the same camera into one STOP
  → stop = { camera_id, camera_name, location, arrived_at, left_at, dwell_sec, event_count }
  → arrival = first stop.arrived_at, departure = last stop.left_at
  → transitions = [(stop[i].camera → stop[i+1].camera, gap_sec)]
```

Collapsing matters: a person lingering at one camera generates entry → exit → (cooldown) → entry → exit. Raw rows would show "Main Gate" five times. Collapsing yields the clean mockup output.

**Do not use ReID for this.** See §10.4.

### Backend changes
- **NEW** `app/services/journey_service.py` (~140 lines): `get_journey(person_id, date, tz)`, `get_unknown_journey(unknown_id, date)`.
- **CHANGE** `movement_service.py::_event_to_dict` → join `Camera`, add `camera_name` + `camera_location`. Fixes the existing bug and benefits every caller.
- **CHANGE** `movement_service.py::get_person_timeline` → accept optional `date` (local day) in addition to `days`.

### Database changes
**None.** Optional composite index — see §4.3.

### API changes
| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/persons/{id}/journey?date=YYYY-MM-DD` | **NEW** — derived stops/transitions/arrival/departure |
| `GET` | `/api/movements/person/{id}/timeline?date=&days=&limit=` | **EXTEND** — add `date`, return `camera_name` |
| `GET` | `/api/unknown-faces/{id}/journey?date=` | **NEW** — same shape, queries `metadata_json.unknown_person_id` |

### Frontend changes
- **MODIFY** `PersonTimeline.js`: add a `Today` tab alongside 24h/7d/30d; render `camera_name`; add a compact stop-strip at the top.
- **NEW** `components/JourneyStrip.js` (~120 lines): horizontal pill sequence `Main Gate → Academic Block → CSE Block`, dwell under each pill, connecting arrows with gap time. Pure CSS flexbox + existing tokens — no library.
- **MODIFY** `/persons/[id]/page.js`: add a `Journey` tab that renders `JourneyStrip` + day list + date picker.

### Edge cases
| Case | Handling |
|---|---|
| Same person visible on two overlapping cameras simultaneously | Two concurrent stops. Render as a merged stop with both camera names when intervals overlap >50%. |
| Person leaves and returns to the same camera later | Two stops separated by a gap ≥ `JOURNEY_STOP_GAP_SEC` (default 300s). Below that, merge. |
| Missing exit event (camera stopped mid-visit) | `left_at` falls back to the last event on that camera; UI labels it "last seen", never "departed". |
| Overnight | Journey is strictly bounded to the local campus day. A visit crossing midnight appears in both days, truncated. |
| Only one detection all day | Single stop, `dwell_sec = 0`, arrival == departure. |
| No cameras running that day | Empty state, not an error. |

### Complexity: **Low**
### Effort: **1.25 days** (0.5 BE, 0.75 FE)

---

## Feature 2 — Faculty Attendance

### What already exists
- **[CONFIRMED]** `persons.department` and `persons.role` are populated by the seed scripts.
- **[CONFIRMED]** `detection_events` with `(event_type='entry', person_id, confidence, created_at)` and index `ix_events_type_created` (`models.py:145`).
- **[CONFIRMED]** A complete CSV/XLSX/JSON export implementation with openpyxl styling in `routes/analytics.py:75-212` — **copy this, do not rewrite it**.
- **[CONFIRMED]** `AnalyticsService.generate_report(report_type, date)` — `analytics_service.py:310` — already does date-bounded aggregation.

### What can be reused
- The whole export pipeline (`AnalyticsService.get_entry_exit_export` shape + the route's CSV/XLSX/JSON writers)
- `PersonService.list_persons(role=...)` for the roster
- `StatCard`, `.data-table`, `.badge` CSS, `HourlyChart`

### What is missing
1. **[MISSING]** Any attendance concept at all — `grep -rin "attendance"` matches only two code comments.
2. **[MISSING]** A reliable faculty/student discriminator. `role` works by convention but is a free-text field (`models/person.py:15` documents a *different* set of values than what is seeded).
3. **[MISSING]** Campus timezone. Everything is UTC (`utils/time_utils.py:64` `today_start_utc()`). "Late before 09:15" is a **local-time** rule.
4. **[MISSING]** Historical durability if `/api/retention/cleanup` is ever run (deletes events >90d, `routes/retention.py:83`).

### The attendance rule — analysis and recommendation

**A single `entry` event must NOT equal attendance.** Reasons, all confirmed from code:

- `ENTRY_THRESHOLD_SECONDS = 3` (`config.py:186`) — three seconds of visibility creates an entry.
- A false accept by the recogniser produces a real `entry` row for the wrong person, with no downstream correction.
- `FACE_MATCH_TOLERANCE = 0.75` with `ENABLE_ADAPTIVE_THRESHOLDS = False` (`config.py:49`) — every face is scored at the same balanced threshold; there is no per-quality tightening.
- Conversely, a genuinely-present teacher who walks past four cameras produces **four** entries, so counting entries is also wrong.

**Recommended rule — two independent confirmations, computed from events only:**

```
For each faculty person P and campus-local day D:

  events = detection_events
           where (person_id = P)
             and event_type in ('entry','exit')
             and confidence >= ATTENDANCE_MIN_CONFIDENCE   (default 0.55)
             and created_at within [D 00:00 local, D 23:59:59 local]

  first_seen      = MIN(created_at)
  last_seen       = MAX(created_at)
  span_minutes    = (last_seen - first_seen) / 60
  entry_count     = COUNT(event_type='entry')
  camera_count    = COUNT(DISTINCT camera_id)

  PRESENT  if  entry_count >= ATTENDANCE_MIN_SIGHTINGS (default 2)
           OR  span_minutes >= ATTENDANCE_MIN_MINUTES  (default 10)
  LATE     if  PRESENT and first_seen_local > ATTENDANCE_LATE_AFTER (default 09:15)
  ABSENT   otherwise
```

Why this is the right trade-off here:

| Requirement | How it's satisfied |
|---|---|
| Minimum presence duration | `span_minutes >= 10` branch |
| Detection within an attendance window | Day-bounded; late cutoff is a config time |
| First arrival detection | `first_seen` = MIN over the day |
| Exit detection | **Deliberately not required.** Exits are unreliable (§10.2). `last_seen` is used as "last detected", labelled honestly. |
| Missing exit events | Irrelevant — the rule never reads exits for the present/absent decision |
| Multiple camera detections | `COUNT(DISTINCT camera_id)` is a *strengthening* signal, and multi-camera automatically satisfies `entry_count >= 2` |
| Restart safety | Reads only committed DB rows; no in-memory state |
| Duplicate detections | 60s per-camera cooldown (`tracker.py:69`) already throttles; `entry_count >= 2` tolerates it |

All four thresholds go in `config.py` so they can be tuned during the demo without a code change.

**Departure time:** report `last_seen` and label it **"Last detected"** in the UI, not "Departure". Claiming a departure time the system cannot actually observe is the fastest way to lose client trust.

### Proposed implementation

`AttendanceService` — pure read model, one grouped query per day:

```sql
SELECT person_id,
       MIN(created_at) AS first_seen,
       MAX(created_at) AS last_seen,
       COUNT(*) FILTER (WHERE event_type='entry') AS entry_count,
       COUNT(DISTINCT camera_id) AS camera_count
FROM detection_events
WHERE person_id = ANY(:faculty_ids)
  AND event_type IN ('entry','exit')
  AND confidence >= :min_conf
  AND created_at >= :day_start_utc AND created_at < :day_end_utc
GROUP BY person_id
```

One query for the whole campus-day. Roster comes from `persons`. Everything else is Python folding. At 68 faculty this is sub-millisecond.

### Backend changes
- **NEW** `app/services/attendance_service.py` (~260 lines):
  `get_day(date)` → per-person records; `get_summary(date)` → department rollup;
  `get_department(dept, date)`; `get_person_range(person_id, from, to)`; `export_rows(date)`.
- **NEW** `app/routes/attendance.py` (~120 lines) — routes below, plus export reusing the analytics writers.
- **CHANGE** `app/config.py` — add `CAMPUS_TIMEZONE`, `ATTENDANCE_MIN_SIGHTINGS`, `ATTENDANCE_MIN_MINUTES`, `ATTENDANCE_MIN_CONFIDENCE`, `ATTENDANCE_LATE_AFTER`, `ATTENDANCE_DAY_START`, `ATTENDANCE_DAY_END`, `FACULTY_ROLES`.
- **CHANGE** `app/utils/time_utils.py` — add `campus_day_bounds(date_str) -> (utc_start, utc_end)` and `to_campus_local(dt)`. Everything else stays UTC.
- **REFACTOR (small)** — lift the CSV/XLSX/JSON writer block out of `routes/analytics.py:145-212` into `app/utils/export_utils.py` so attendance and analytics share it. ~40 lines moved, no behaviour change.

### Database changes
- **NEW COLUMN** `persons.person_type VARCHAR(20)` — see §4.1. Required.
- **OPTIONAL** `attendance_days` snapshot table — see §4.2. Nice-to-have.

### API changes
| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/attendance/summary?date=` | Department rollup + totals + % |
| `GET` | `/api/attendance/day?date=&department=&status=` | Per-person rows |
| `GET` | `/api/attendance/person/{id}?from=&to=` | One person's history |
| `GET` | `/api/attendance/export?date=&format=csv\|xlsx\|json&department=` | Report download |

### Frontend changes
- **NEW** `/attendance/page.js` — date picker, department cards (click to drill down), totals bar, person table with ✓/⚠/✕ status, export buttons.
- **NEW** `components/AttendanceStatusBadge.js` — maps present/late/absent to existing `.badge` classes.
- **NEW** `components/DepartmentCard.js` — `18 / 22` + progress bar, reuses `.card` + `.stat-card` tokens.
- **MODIFY** `Sidebar.js` — add a `College` section.
- **MODIFY** `lib/api.js` — 4 methods.
- **MODIFY** `/persons/page.js` — role dropdown must include the real values (`assistant_professor`, `professor`, `hod`, `student`, `staff`) or switch to a `person_type` filter. **Currently the seeded teachers cannot be filtered at all.**

### Edge cases
| Case | Handling |
|---|---|
| Faculty with zero face encodings | Can never be recognised → always absent. Flag them in the UI as `No face enrolled` (data available: `encoding_count`). Prevents "the AI is broken" complaints. |
| Person deactivated mid-term | Roster respects `is_active`; historical days use the roster as-of-query (documented limitation). |
| Holiday / no cameras running | If zero entry events campus-wide for the day, return `no_data: true` and suppress the 0% headline. |
| Two people share a department string casing (`CSE` vs `cse`) | Normalise `UPPER(TRIM(department))` in the rollup. |
| Person with NULL department | Bucketed as `Unassigned`, shown last. |
| Query for a future date | 400. |
| Timezone | All bounds computed via `campus_day_bounds()`; storage stays UTC. |

### Complexity: **Medium** (rule design, not code)
### Effort: **2.25 days** (1.0 BE, 1.25 FE)

---

## Feature 3 — Daily Campus Summary

### What already exists
- **[CONFIRMED]** `AnalyticsService.get_dashboard_summary()` — today's entries/exits/unknowns, occupancy, camera status (`analytics_service.py:38`).
- **[CONFIRMED]** `get_peak_times(days)` — 24-bucket hourly entry histogram (`analytics_service.py:116`).
- **[CONFIRMED]** `generate_report('daily', date)` — date-bounded entries/exits/unique persons/avg duration (`analytics_service.py:310`).
- **[CONFIRMED]** Vehicles: `vehicle_entry` events **are** persisted with plate in `metadata_json` (`anpr.py:1179`).
- **[CONFIRMED]** Camera availability **is derivable** — `stream_offline` and `stream_recovered` are persisted as `security_alert` subtypes, and `stream_recovered` carries `metadata.downtime_sec` (`stream_health.py:322-326`).

### What can be reused
Essentially all of it. This feature is 80% composition of existing service methods.

### What is missing
1. **[MISSING]** Peak **occupancy** for the day. Occupancy is only a live in-memory count (`camera_manager.get_all_occupancy()`), never sampled or stored. **[INFERENCE]** It can be *reconstructed* from entry/exit events by replaying them as a +1/−1 running total over the day — accurate enough for a headline number, and free.
2. **[MISSING]** Peak **departure** period (only entries are histogrammed).
3. **[MISSING]** Faculty attendance % (comes from F2).
4. **[MISSING]** Camera availability %.
5. **[MISSING]** Any narrative generation.

### Proposed implementation
`CampusSummaryService.get_daily(date)` composes:

| Metric | Source | Confidence |
|---|---|---|
| Total people detected | `COUNT(DISTINCT person_id) + COUNT(DISTINCT metadata_json->>'unknown_person_id')` | High |
| Faculty present / % | `AttendanceService.get_summary(date)` | High |
| Students detected | Same query filtered by `person_type='student'` | High |
| People currently on campus | `PresenceService` (F4) | Medium |
| Peak occupancy | Entry/exit replay, +1/−1 running max | **Medium — label as "estimated"** |
| Peak arrival period | Hourly histogram of `entry`, argmax | High |
| Peak departure period | Hourly histogram of `exit`, argmax | Medium (exits unreliable) |
| Security incidents | `COUNT(security_alert)` excluding `stream_*` subtypes | High |
| Unknown persons | `COUNT(DISTINCT unknown_faces.id first_seen today)` | High |
| Vehicles | `COUNT(vehicle_entry)` | High |
| Camera availability % | `1 − (Σ downtime_sec from stream_recovered) / (active_cameras × day_seconds_elapsed)` | Medium |

**Narrative without an LLM — recommended.** A rule-ranked template generator:

```python
CANDIDATES = [
  (attendance_pct < avg7 - 10, "Faculty attendance is {d} points below the 7-day average."),
  (incidents > 0,              "{n} security incident{s} logged, {m} still open."),
  (unknown_today > avg7 * 1.5, "Unknown-person detections are {r}x the recent average."),
  (camera_avail < 98,          "{n} camera{s} experienced downtime totalling {t}."),
  (True,                       "Peak occupancy of {p} reached around {h}."),
]
# evaluate in order, take the first 3 that fire
```

~50 lines, deterministic, testable, zero dependencies, no latency, no cost. `vlm_service.py` + Ollama already exist if an LLM is ever wanted — but making the daily summary depend on a 7B VLM for three sentences would be an unforced infrastructure risk. **Recommendation: template generator.**

### Backend changes
- **NEW** `app/services/campus_summary_service.py` (~200 lines) including `_replay_occupancy()` and `_build_narrative()`.
- **CHANGE** `analytics_service.py` — add `get_hourly_distribution(event_type, date)` generalising `get_peak_times` (which is hardcoded to `entry`, `analytics_service.py:127`). Avoids duplicating that query twice.

### Database changes
**None.**

### API changes
| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/analytics/daily-summary?date=` | **NEW** — all metrics + narrative |

### Frontend changes
- **NEW** `/daily-summary/page.js` — hero metric grid + narrative card + arrival/departure dual `HourlyChart` + incident/unknown/vehicle strip + date picker.
- **MODIFY** `/page.js` (Dashboard) — add 3 college `StatCard`s (Faculty Attendance, On Campus, Today's Incidents) and a "View full daily summary →" link. Keeps the existing dashboard intact.
- **REUSE** `StatCard`, `HourlyChart`, `OccupancyRing`, `CameraHealth`.

### Edge cases
| Case | Handling |
|---|---|
| Camera availability when no cameras ran | Show `—`, not `0%` or `100%`. |
| Peak occupancy on a day with unbalanced entry/exit | Running total is clamped at ≥0; label "estimated". |
| Future/empty date | `no_data: true`, empty state. |
| Backend restarted mid-day | Event-derived metrics unaffected. Live "currently on campus" resets — see §10.6. |

### Complexity: **Low–Medium**
### Effort: **1.25 days** (0.75 BE, 0.5 FE)

---

## Feature 4 — Teachers Currently on Campus

### What already exists
- **[CONFIRMED]** `tracker.get_active_persons()` — live, per-camera, entry-confirmed persons with `first_seen`, `last_seen`, `duration_sec`, `camera_id` (`tracker.py:261`).
- **[CONFIRMED]** Two endpoints already expose this: `GET /api/cameras/active-persons` (`cameras.py:75`) and `GET /api/sessions/active` (`events.py:158`) — near-identical implementations.
- **[CONFIRMED]** `WS /api/events/live` broadcasts every entry/exit; `api.connectLiveEvents()` is already used by the Dashboard and Security Alerts pages.

### What can be reused
The live tracker (as a *secondary* signal), the WebSocket, `api.connectLiveEvents`, `.activity-item` / `.avatar` CSS.

### What is missing — and this is the crux
**[CONFIRMED]** The live tracker's definition of "present" is **"visible on a camera within the last `EXIT_THRESHOLD_SECONDS` (30s)"**. A teacher who arrived at 08:42 and is now teaching in a classroom is **not** in `get_active_persons()`. Building the "42 Present · On campus 3h 24m" screen on that data would show near-zero presence most of the day and would be visibly wrong to the client.

Also **[CONFIRMED]**: tracker state is in-memory per pipeline and is wiped on restart (`tracker.reset()`, and workers are recreated on `start_camera`).

### Proposed implementation — two-tier presence

```
PresenceService.get_on_campus():

  for each faculty person with an attendance record today:
      last_seen        = MAX(created_at) today
      minutes_since    = now - last_seen
      last_camera      = camera of the last event
      live             = person_id in camera_manager live tracker

      state =
        LIVE      if live                                  → "Visible now at {camera}"
        ON_CAMPUS if minutes_since < PRESENCE_TIMEOUT_MIN   → "Last seen {camera}, {n}m ago"
                     and not departed_via_gate
        DEPARTED  otherwise
```

`departed_via_gate` (optional, cheap, high-value):
`cameras.config` is an existing free JSON column (`models.py:50`). Tag gate cameras `{"is_gate": true}` from the camera edit form. Then: *if the last detection today was on a gate camera and was more than `GATE_DEPART_MIN` (default 10) minutes ago → DEPARTED*. This is the closest the system can honestly get to campus exit, and it needs **zero migration**.

Defaults: `PRESENCE_TIMEOUT_MIN = 90`, `GATE_DEPART_MIN = 10`.

Duration on campus = `now − first_seen_today` (or `last_seen − first_seen` once departed).

### Backend changes
- **NEW** `app/services/presence_service.py` (~150 lines). Depends on `AttendanceService` + `camera_manager` + `cameras.config`.
- **CHANGE** `camera_service.py` / camera update model — allow writing `config` (verify `CameraUpdate` in `models/camera.py` includes `config`; if not, add the field).

### Database changes
**None.** Gate flags live in the existing `cameras.config` JSON.

### API changes
| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/attendance/on-campus?department=&q=` | **NEW** — present faculty with state, camera, arrival, duration |

Do **not** add a third "active persons" endpoint. The two existing ones stay for the live monitor.

### Frontend changes
- **NEW** `/on-campus/page.js` — count header, search box, department filter, faculty cards.
- **NEW** `components/FacultyPresenceCard.js` — avatar, name, dept, state pill (`Visible now` green / `On campus` blue / `Departed` grey), current-or-last camera, arrival time, `On campus: 3h 24m`. Click → `/persons/{id}` Journey tab.
- **REAL-TIME** — poll `/on-campus` every 30s **and** subscribe to `WS /api/events/live`; on any `entry`/`exit` for a known `person_id`, patch that card optimistically and schedule a debounced refetch. Matches how `/page.js:43` and `/security-alerts/page.js:61` already work.

### Edge cases
| Case | Handling |
|---|---|
| Backend restarted at 14:00 | Event-derived presence survives intact; only the `LIVE` badge is briefly empty. **This is why the design is event-derived.** |
| Teacher in a room with no camera for 2h | Falls to `DEPARTED` after 90 min. Documented as a coverage limitation, tunable. |
| Person on two cameras at once | Dedup by `person_id`, keep the most recent event. |
| Late-evening query | Presence is bounded to `ATTENDANCE_DAY_END`; after that everyone is `DEPARTED`. |
| Unknown persons | Excluded — this view is faculty-only by design. |

### Complexity: **Medium**
### Effort: **1.25 days** (0.5 BE, 0.75 FE)

---

## Feature 5 — Incident Evidence Pack

### What already exists
- **[CONFIRMED]** `security_alert` events **are** persisted to `detection_events` with `subtype`, `camera_id`, `confidence`, `created_at`, and a rich `metadata_json` (person_ids, person_count, duration_sec, threat_class, bounding_box, model info).
- **[CONFIRMED]** `security_alert` is treated as CRITICAL priority in the event queue and never dropped (`main.py:94`).
- **[CONFIRMED]** A working `/security-alerts` page with severity styling, subtype filters, and live WS updates.
- **[CONFIRMED]** Unknown-person evidence is **rich and already on disk**: face crop, context crop, full frame — `snapshots/unknowns/` (20,222 files), `snapshots/frames/` (11,888 files), served at `/snapshots` (`main.py:629`).
- **[CONFIRMED]** `unknown_faces` rows have stable IDs across cameras and restarts (`UnknownRegistry`, `config.py:135-142`), `occurrence`, `first_seen`, `last_seen`.
- **[CONFIRMED]** `components/Lightbox.js` exists for full-size image viewing.
- **[CONFIRMED]** `SnapshotWorker` (`services/snapshot_worker.py`) with a `CRITICAL` priority tier that is never dropped — the correct mechanism if snapshots are added to alerts.

### What is missing
1. **[MISSING — the main gap]** `security_alert` events have **no `snapshot_url`**. `safety_analytics.py:389`, `hazard_detector.py:319` and `traffic.py` all build the alert dict without any image. So an incident page currently has no "what did it look like".
2. **[MISSING]** No incident **status** (Open/Investigating/Resolved), **notes**, **assignee**, or **severity** persisted. Severity is a hardcoded frontend map (`security-alerts/page.js:24-46`).
3. **[MISSING]** No incident detail page or route.
4. **[MISSING]** No human-readable incident number. Only UUIDs. The mockup wants `INCIDENT #1042`.
5. **[MISSING]** No "restricted area entry" detector — the mockup's example incident type does not exist. Existing subtypes are `gathering`, `loitering`, `hazard`, `vehicle_proximity`, `stream_offline`, `stream_recovered`.
6. **[CONFIRMED MISSING] There is no video recording anywhere in the system.** No clip writer, no ring buffer, no `VideoWriter` on live streams. `_last_jpeg` is a single in-memory frame. `video_analysis` is **upload-and-analyse**, not recording. **Do not design a recording subsystem.**

### Proposed implementation

**Evidence pack = what the system genuinely has:**

```
INCIDENT (detection_events row, event_type='security_alert')
  ├─ Header: incident number, subtype label, severity, camera name + location, local time
  ├─ Evidence
  │    ├─ Alert snapshot        ← NEW (small pipeline change, see below)
  │    ├─ Related unknown faces ← unknown_faces where camera_id = X
  │    │                          and first_seen within ±EVIDENCE_WINDOW (default 120s)
  │    │                          → face crop + full frame, already on disk
  │    └─ Related known persons ← entry/exit events, same camera, same window
  ├─ Timeline: every detection_event on that camera in [t−5min, t+5min]
  ├─ Other cameras: events on ALL other cameras in the same window (correlation)
  ├─ Person journey link: for each involved person/unknown → Feature 1
  └─ Review: status, severity override, assignee, notes[]   ← NEW table
```

**Adding the alert snapshot** — one narrow, well-bounded change:

`pipeline._handle_security_alert()` (`pipeline.py:1268`) already receives every alert. It has access to the current frame. Add there (not in each detector):

```python
def _handle_security_alert(self, alert):
    if self._current_frame is not None and settings.ALERT_SNAPSHOTS_ENABLED:
        get_snapshot_worker().submit(
            self._current_frame.copy(), f"alert_{uuid4().hex[:8]}",
            subdirectory="alerts", quality=88,
            priority=SnapshotPriority.CRITICAL,
            callback=lambda fp, url: alert.__setitem__("snapshot_url", url),
        )
    ...
```

⚠️ The callback is asynchronous — it may land *after* the event is enqueued. **Correct approach:** write the alert snapshot **synchronously** in the EventQueue consumer thread (`main.py::_process_pipeline_event`), the same pattern already used for the unknown face crop (`main.py:323`), by passing `frame` on the alert dict and stripping it before it reaches `metadata_json`. This keeps the vision thread free and guarantees the URL is present at insert time.

Cost: one `frame.copy()` (~0.5ms) per alert on the vision thread; alerts are heavily cooldown-throttled (15–600s per subtype), so this is negligible.

**Incident numbering:** a monotonic per-incident integer. Simplest correct option: a `seq` integer column on the new `incident_reviews` table with a Postgres sequence default — the row is created lazily on first view/status change. For the demo, `#1000 + row_number` computed on the fly is enough and needs nothing; but a stored `seq` is stable across queries and is worth the one column.

### Backend changes
- **NEW** `app/services/incident_service.py` (~220 lines): `list_incidents()`, `get_evidence_pack(event_id)`, `set_status()`, `add_note()`, `export_report()`.
- **NEW** `app/routes/incidents.py` (~110 lines).
- **CHANGE** `main.py::_process_pipeline_event` — save + attach `snapshot_url` for `security_alert` events (synchronous crop write, mirroring the unknown-face path).
- **CHANGE** `pipeline.py::_handle_security_alert` — attach `frame` to the alert dict.
- **CHANGE** `config.py` — `ALERT_SNAPSHOTS_ENABLED`, `INCIDENT_EVIDENCE_WINDOW_SEC`, `INCIDENT_SEVERITY_MAP`.
- **CHANGE** move the frontend's hardcoded severity map into backend config so the API returns `severity` — one source of truth.

### Database changes
- **NEW TABLE** `incident_reviews` — see §4.2. Additive-only, so `Base.metadata.create_all()` creates it automatically on next boot. **No manual SQL.**

### API changes
| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/incidents?status=&severity=&subtype=&date=&camera_id=` | List with review state joined |
| `GET` | `/api/incidents/{event_id}` | Full evidence pack |
| `PATCH` | `/api/incidents/{event_id}` | `{status, severity, assignee}` |
| `POST` | `/api/incidents/{event_id}/notes` | `{text}` → appends |
| `GET` | `/api/incidents/{event_id}/export?format=json\|csv` | Report download |

### Frontend changes
- **NEW** `/incidents/page.js` — table/card list with status + severity + date filters and counts by status. (Complements `/security-alerts`, which stays as the live monitoring wall.)
- **NEW** `/incidents/[id]/page.js` — the evidence pack layout above.
- **NEW** `components/EvidenceGrid.js` — thumbnail grid → existing `Lightbox`.
- **NEW** `components/IncidentStatusControl.js` — segmented Open/Investigating/Resolved.
- **NEW** `components/IncidentNotes.js` — append-only note list + textarea.
- **MODIFY** `/security-alerts/page.js` — make each card link to `/incidents/{id}`, and read `severity` from the API instead of the local map.
- **Export/share:** browser-side print-to-PDF of the detail page (`window.print()` + a `@media print` block in `globals.css`) is the cheapest professional-looking share. JSON/CSV export from the API covers machine use.

### Edge cases
| Case | Handling |
|---|---|
| Historical alerts (pre-change) have no snapshot | Evidence grid falls back to related unknown-face crops and full frames from the same camera/window. Show "No alert snapshot captured" rather than a broken image. |
| Alert on a camera that has since been deleted | `camera_id` FK is `ON DELETE CASCADE` (`models.py:131`) — **the alert row is destroyed with the camera.** Flag this in §10.5. |
| `stream_offline` / `stream_recovered` | Exclude from the incident list by default (they are health, not security). Offer a "System" filter tab. |
| Two operators change status concurrently | Last-write-wins on a single row; acceptable at this scale. |
| Unknown person involved | Link to `/api/unknown-faces/{id}/journey` (F1). |
| "Restricted area entry" | **Not implementable without a zone-definition feature.** See §11 — DO NOT BUILD NOW. Use `hazard`/`gathering`/`loitering` for the demo. |

### Complexity: **Medium**
### Effort: **2.25 days** (1.0 BE, 1.25 FE)

---

# 4. Database Changes

Only three items are genuinely necessary. **`init_db()` runs `Base.metadata.create_all()` (`db/session.py:57`), which creates missing TABLES but never adds COLUMNS or INDEXES to existing tables.** There is no Alembic. The repo convention is numbered SQL files in `app/migrations/` applied manually — follow it.

## 4.1 REQUIRED — `persons.person_type`

```sql
-- app/migrations/009_college_features.sql
ALTER TABLE persons ADD COLUMN IF NOT EXISTS person_type VARCHAR(20);

UPDATE persons SET person_type = 'faculty'
 WHERE person_type IS NULL
   AND role IN ('professor','associate_professor','assistant_professor','hod','lecturer','teacher');

UPDATE persons SET person_type = 'student'  WHERE person_type IS NULL AND role = 'student';
UPDATE persons SET person_type = 'staff'    WHERE person_type IS NULL AND role IN ('employee','staff');
UPDATE persons SET person_type = 'visitor'  WHERE person_type IS NULL;

CREATE INDEX IF NOT EXISTS ix_persons_type_dept ON persons (person_type, department);
```

**Why:** faculty vs student is currently encoded only in the free-text `role` field, whose documented values (`models/person.py:15`: employee/visitor/vip/banned) **do not match what the seed scripts actually write** (`assistant_professor`, `student`). Every attendance and presence query filters on this. A string-matching heuristic against `role` would silently break the moment someone types "Asst. Professor" in the UI. One indexed column removes a whole class of demo-day failure.

Also add `person_type` to `PersonBase`/`PersonUpdate` in `models/person.py` and to `_person_to_dict()` in `person_service.py`.

## 4.2 REQUIRED — `incident_reviews`

Additive table → `create_all()` handles it, **no manual SQL needed**. Add to `app/db/models.py`:

```python
class IncidentReview(Base):
    __tablename__ = "incident_reviews"
    id         = Column(String(36), primary_key=True, default=gen_uuid)
    event_id   = Column(String(36), ForeignKey("detection_events.id", ondelete="CASCADE"),
                        nullable=False, unique=True, index=True)
    seq        = Column(Integer, autoincrement=True, unique=True)   # human "#1042"
    status     = Column(String(20), default="open", index=True)     # open/investigating/resolved
    severity   = Column(String(20), nullable=True)                  # operator override
    assignee   = Column(String(36), nullable=True)                  # admin_users.id
    notes      = Column(JSON, default=list)  # [{author, text, at}]
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
```

**Why a table and not `detection_events.metadata_json`:** `metadata_json` is written by the vision event path on a hot table; mutating it from the API risks clobbering pipeline-written fields, and "list all open incidents" would become a JSON scan. A 9-column side table keeps the event log immutable and the query a plain indexed filter. Rows are created lazily on first status change or note.

## 4.3 RECOMMENDED — one composite index

```sql
CREATE INDEX IF NOT EXISTS ix_events_person_created
  ON detection_events (person_id, created_at)
  WHERE person_id IS NOT NULL;
```

**Why:** every attendance, journey and presence query is `WHERE person_id = ? AND created_at BETWEEN ? AND ?`. The existing indexes are `(event_type, created_at)` and `(camera_id, created_at)` (`models.py:144-147`) plus a bare `person_id` — none of which serve this shape. Given the snapshot counts on disk (32k images), `detection_events` is not small.

## 4.4 OPTIONAL — `attendance_days` snapshot table

```python
class AttendanceDay(Base):
    __tablename__ = "attendance_days"
    id          = Column(String(36), primary_key=True, default=gen_uuid)
    person_id   = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), index=True)
    day         = Column(String(10), index=True)      # 'YYYY-MM-DD' campus-local
    status      = Column(String(10))                  # present/late/absent
    first_seen  = Column(DateTime(timezone=True), nullable=True)
    last_seen   = Column(DateTime(timezone=True), nullable=True)
    entry_count = Column(Integer, default=0)
    camera_count= Column(Integer, default=0)
    department  = Column(String(100), nullable=True)  # snapshotted
    __table_args__ = (Index("ix_att_day_person", "day", "person_id", unique=True),)
```

**Why optional:** the live derivation is fast enough and always reflects current rules. This only matters if (a) `/api/retention/cleanup` deletes events older than 90 days, destroying attendance history, or (b) someone changes a threshold and past reports silently change. A nightly job writing yesterday's rows costs ~40 lines. **Recommend building it in Phase 7 if time allows, not before.**

## 4.5 NOT NEEDED
- ❌ No `departments` table — `persons.department` is sufficient at this scale.
- ❌ No `journeys` table — derived on read.
- ❌ No `camera_zones` table — use the existing `cameras.config` JSON.
- ❌ No `presence` table — derived on read.
- ❌ No changes to `tracking_sessions` — it is deliberately bypassed (§10.2).

---

# 5. API Changes

### NEW endpoints

| METHOD | ENDPOINT | PURPOSE | REQUEST | RESPONSE (abridged) |
|---|---|---|---|---|
| GET | `/api/attendance/summary` | Department rollup for a day | `?date=YYYY-MM-DD` | `{date, departments:[{name,present,late,absent,total,pct}], totals:{present,total,pct}, no_data}` |
| GET | `/api/attendance/day` | Per-person rows | `?date=&department=&status=` | `{date, records:[{person_id,name,department,role,status,first_seen,last_seen,minutes_on_campus,entry_count,camera_count,has_encodings}]}` |
| GET | `/api/attendance/person/{id}` | One person's history | `?from=&to=` | `{person, days:[{day,status,first_seen,last_seen}], summary:{present_days,late_days,absent_days,pct}}` |
| GET | `/api/attendance/export` | Download report | `?date=&format=csv\|xlsx\|json&department=` | file (`Content-Disposition`) |
| GET | `/api/attendance/on-campus` | Live faculty presence | `?department=&q=` | `{count, faculty:[{person_id,name,department,state,current_camera,camera_name,arrived_at,minutes_on_campus,last_seen}]}` |
| GET | `/api/persons/{id}/journey` | Person journey for a day | `?date=` | `{person, date, arrival, departure, stops:[{camera_id,camera_name,location,arrived_at,left_at,dwell_sec,event_count}], transitions:[...], total_cameras}` |
| GET | `/api/unknown-faces/{id}/journey` | Unknown-person journey | `?date=` | same shape |
| GET | `/api/analytics/daily-summary` | Campus daily summary | `?date=` | `{date, metrics:{...}, narrative:[str], peak_arrival_hour, peak_departure_hour, camera_availability_pct, no_data}` |
| GET | `/api/incidents` | Incident list | `?status=&severity=&subtype=&date=&camera_id=&limit=&offset=` | `{data:[{event_id,seq,subtype,severity,status,camera_name,created_at,summary}], total, counts_by_status}` |
| GET | `/api/incidents/{event_id}` | Evidence pack | — | `{incident, evidence:{alert_snapshot,unknown_faces[],full_frames[]}, timeline[], other_cameras[], persons_involved[], review}` |
| PATCH | `/api/incidents/{event_id}` | Update review state | `{status?, severity?, assignee?}` | updated review |
| POST | `/api/incidents/{event_id}/notes` | Add note | `{text}` | updated notes array |
| GET | `/api/incidents/{event_id}/export` | Report download | `?format=json\|csv` | file |

### EXTENDED endpoints (no duplicates created)

| METHOD | ENDPOINT | CHANGE |
|---|---|---|
| GET | `/api/movements/person/{id}/timeline` | Add `?date=` param; add `camera_name` + `camera_location` to every row (**fixes the current UUID-in-UI bug**) |
| GET | `/api/persons` | Add `person_type` and `department` query filters; return `person_type` |
| PUT | `/api/persons/{id}` | Accept `person_type` |
| PUT | `/api/cameras/{id}` | Ensure `config` is writable (for `{"is_gate":true,"zone":"Main Gate"}`) |

### Endpoints deliberately NOT created
- ❌ No new "active persons" endpoint — `/api/cameras/active-persons` already exists (and `/api/sessions/active` duplicates it).
- ❌ No new events-list endpoint — `/api/events` already supports every needed filter.
- ❌ No new export engine — the analytics CSV/XLSX/JSON writers are lifted into `utils/export_utils.py` and shared.

### Auth
Every new endpoint uses `Depends(get_current_user)`, matching every other route. `PATCH /api/incidents/*` should use `Depends(require_admin)`.

---

# 6. Frontend Changes

## New pages
| Route | Purpose |
|---|---|
| `/attendance` | Faculty attendance dashboard, department drill-down, date picker, export |
| `/on-campus` | Real-time faculty presence |
| `/daily-summary` | Daily campus summary + narrative |
| `/incidents` | Incident list with status/severity filters |
| `/incidents/[id]` | Incident evidence pack |

## Existing pages to modify
| File | Change |
|---|---|
| `components/Sidebar.js` | Add a `College` nav section: Attendance, On Campus, Daily Summary. Move Incidents under `Security`. |
| `app/page.js` (Dashboard) | Add 3 college `StatCard`s + link to `/daily-summary`. No restructuring. |
| `app/persons/[id]/page.js` | Add a `Journey` tab (the `activeTab` mechanism already exists) |
| `app/persons/page.js` | Fix the role dropdown to include real values; add a department filter column |
| `app/security-alerts/page.js` | Link cards to `/incidents/{id}`; read `severity` from the API |
| `components/PersonTimeline.js` | Render `camera_name`; add a `Today` mode; add `JourneyStrip` header |
| `lib/api.js` | ~13 new methods (thin wrappers over `request`) |
| `app/globals.css` | `@media print` block for incident reports; ~6 new classes for attendance/presence/journey |

## New components
| Component | Purpose | Reuses |
|---|---|---|
| `JourneyStrip.js` | Camera-to-camera pill sequence with arrows and dwell | CSS tokens, `CameraIcon` |
| `DepartmentCard.js` | `CSE 18/22` + progress bar | `.card`, `.stat-card` |
| `AttendanceStatusBadge.js` | ✓ / ⚠ / ✕ | `.badge.{active,pending,offline}` |
| `FacultyPresenceCard.js` | Presence row for `/on-campus` | `.activity-item`, `.avatar`, `.badge` |
| `EvidenceGrid.js` | Snapshot thumbnails → lightbox | `Lightbox` |
| `IncidentStatusControl.js` | Segmented status control | `.btn`, `.btn-primary` |
| `IncidentNotes.js` | Append-only notes | `.input`, `.btn` |
| `DatePicker.js` (tiny) | Shared date navigator (‹ 27 Aug 2026 ›) | `.btn`, native `<input type=date>` |

## Existing components reused unchanged
`AppShell`, `StatCard`, `HourlyChart`, `OccupancyRing`, `ActivityFeed`, `ActivityHeatmap`, `CameraHealth`, `Lightbox`, `Skeleton`, `ErrorBoundary`, `ToastContext`, `SearchPalette`, `Icons`.

## UX consistency rules
- Every page wrapped in `<AppShell>` with the standard `.page-header` block.
- Loading = existing spinner pattern; errors = existing retry card; empty = `.empty-state`.
- All times rendered with `toLocaleString` (already the convention — this is what converts UTC storage to the viewer's local time).
- Live views get the existing pulsing `LIVE` dot.
- No new dependencies. No chart library. Bars/rings are CSS, matching `HourlyChart`/`OccupancyRing`.

---

# 7. Event / Data Flow

**F1 — Journey**
```
Camera → VisionPipeline → FaceRecognizer → PersonTracker(entry/exit)
       → EventQueue → detection_events (person_id, camera_id, created_at)
       → JourneyService (collapse to stops) → /api/persons/{id}/journey
       → JourneyStrip + PersonTimeline
```

**F2 — Attendance**
```
detection_events (entry/exit, today, confidence ≥ min)
   ├─ GROUP BY person_id → first_seen, last_seen, entry_count, camera_count
   └─ JOIN persons (person_type='faculty')
       → AttendanceService rule (≥2 sightings OR ≥10 min span)
       → /api/attendance/{summary,day,export} → /attendance page
```

**F3 — Daily Summary**
```
detection_events ──┬─ entries/exits/unknown/vehicle counts
                   ├─ hourly histograms → peak arrival / departure
                   ├─ +1/−1 replay      → estimated peak occupancy
                   └─ stream_offline/recovered → camera availability %
persons + AttendanceService → faculty %
unknown_faces → unknown count
   → CampusSummaryService → template narrative (NO LLM)
   → /api/analytics/daily-summary → /daily-summary page
```

**F4 — On Campus**
```
AttendanceService(today) ─┐
camera_manager.tracker ───┼→ PresenceService → state = LIVE | ON_CAMPUS | DEPARTED
cameras.config.is_gate ───┘
   → /api/attendance/on-campus  ──→ /on-campus page (30s poll)
   → WS /api/events/live        ──→ optimistic card patch
```

**F5 — Incident**
```
CrowdDetector / LoiteringDetector / HazardDetector
   → pipeline._handle_security_alert(alert + frame)
   → EventQueue (CRITICAL)
   → main._process_pipeline_event: save alert snapshot (sync) → detection_events.snapshot_url  [NEW]
   → IncidentService.get_evidence_pack(event_id)
        ├─ alert snapshot
        ├─ unknown_faces  (same camera, ±120s) → crops + full frames
        ├─ detection_events (same camera, ±5min) → timeline
        ├─ detection_events (other cameras, ±5min) → correlation
        └─ incident_reviews → status / severity / notes
   → /incidents/[id]
```

---

# 8. Implementation Phases

Ordering is deliberately changed from the suggested one: **Attendance must come before Presence and before the Daily Summary**, because both consume it. The journey feature is moved earlier because it is the cheapest, most visually impressive, and lowest-risk win.

### Phase 0 — Data foundation *(0.75 day)*
- `009_college_features.sql`: `person_type` + backfill + indexes. Apply manually.
- Add `person_type` to Pydantic models, `person_service`, persons UI dropdown.
- Add `IncidentReview` to `models.py` (auto-created by `create_all`).
- `config.py`: campus timezone + all attendance/presence/incident thresholds.
- `time_utils.campus_day_bounds()` / `to_campus_local()`.
- **Fix** `camera_name` in the movements timeline (unblocks F1 + fixes a visible bug).
- Extend `seed_teachers.py` to cover CSE/ECE/Mechanical/Civil so the demo matches the mockup.
- ✅ **Checkpoint:** `/persons` shows correct roles/departments; the History panel shows camera names, not UUIDs.

### Phase 1 — Person Journey *(1.25 days)*
- `journey_service.py` + `/api/persons/{id}/journey` + unknown variant.
- `JourneyStrip` component; `Journey` tab on `/persons/[id]`; `PersonTimeline` Today mode.
- ✅ **Checkpoint:** open a teacher → see today's camera-by-camera journey.

### Phase 2 — Faculty Attendance *(2.25 days)*
- `attendance_service.py` + 4 endpoints; lift export writers into `utils/export_utils.py`.
- `/attendance` page + `DepartmentCard` + `AttendanceStatusBadge` + `DatePicker`.
- ✅ **Checkpoint:** the exact mockup — `CSE 18/22 … TOTAL 56/68 · 82.4%`, drill-down, CSV/XLSX export.

### Phase 3 — Teachers On Campus *(1.25 days)*
- `presence_service.py` + `/api/attendance/on-campus`; gate flags via `cameras.config`.
- `/on-campus` page + `FacultyPresenceCard` + WS live patching.
- ✅ **Checkpoint:** live roster with "Currently at CSE Block · Entered 08:42 · 3h 24m", clicking through to the journey.

### Phase 4 — Daily Campus Summary *(1.25 days)*
- `campus_summary_service.py` (occupancy replay, camera availability, template narrative).
- `/daily-summary` page + 3 dashboard cards.
- ✅ **Checkpoint:** the full "TODAY'S CAMPUS" metric wall with a generated 3-line narrative.

### Phase 5 — Incident Evidence Pack *(2.25 days)*
- Alert snapshot capture (pipeline + event-queue consumer).
- `incident_service.py` + 5 endpoints.
- `/incidents` + `/incidents/[id]` + `EvidenceGrid` + status control + notes + print stylesheet.
- ✅ **Checkpoint:** click a hazard alert → full evidence page with images, timeline, related cameras, status and notes.

### Phase 6 — Testing & polish *(1.0 day)*
- The §9 test matrix.
- Empty/error states, mobile widths, dark+light theme pass on all four new pages.
- Nav, badges, print stylesheet.
- **If time remains:** `attendance_days` nightly snapshot job (§4.4).

**Total: 10 developer-days.** Phases 1–3 alone (4.75 days) already deliver the three highest-impact demos.

---

# 9. Testing Plan

Existing tests live in `backend/tests/` (plain pytest, no fixtures framework, e.g. `test_identity_continuity.py`). Follow that style. **All attendance/presence logic is pure functions over event rows — seed synthetic `detection_events` and assert, no camera required.**

| # | Scenario | Setup | Expected |
|---|---|---|---|
| 1 | **Correct attendance** | 3 entries across 2 cameras, 08:42–17:21 | `present`, first=08:42, last=17:21 |
| 2 | **Late arrival** | first entry 09:47, cutoff 09:15 | `late` |
| 3 | **Absence** | no events | `absent`, `first_seen=None` |
| 4 | **Single fleeting detection** | 1 entry, span 0 min | `absent` (guards against false accepts) |
| 5 | **Single long presence** | 1 entry at 08:00, 1 exit at 08:40 | `present` via the span branch |
| 6 | **Missed exit** | entries only, no exits at all | `present`; `last_seen` = last entry; UI says "last detected" |
| 7 | **Multiple cameras** | entries on 4 cameras in 20 min | 1 attendance record, `camera_count=4`, journey has 4 stops |
| 8 | **Duplicate detections** | 6 entries on one camera within 5 min | Journey collapses to 1 stop; attendance unaffected |
| 9 | **Low confidence** | entries all at `confidence=0.3` | Excluded → `absent` |
| 10 | **Backend restart** | insert events, restart process, query | Identical result (no in-memory dependency) |
| 11 | **Stale active session** | `tracking_sessions.status='active'` from 3 days ago | Does **not** appear in on-campus (proves §10.2 is bypassed) |
| 12 | **On-campus decay** | last seen 100 min ago, timeout 90 | `DEPARTED` |
| 13 | **On-campus gate rule** | last seen at `is_gate` camera 15 min ago | `DEPARTED` |
| 14 | **On-campus live** | person in `tracker.get_active_persons()` | `LIVE` + camera name |
| 15 | **Unknown person journey** | events with `metadata_json.unknown_person_id` | Journey returns stops (proves the FK-guard path) |
| 16 | **Incident creation** | insert a `security_alert` | Appears in `/api/incidents`, `status='open'`, `seq` assigned on first review |
| 17 | **Incident evidence correlation** | alert at T; unknown face at T+30s, same camera | Unknown face appears in the pack |
| 18 | **Incident status + notes** | PATCH → resolved, POST 2 notes | Persisted, ordered, survives reload |
| 19 | **Historical date** | query `date=` 5 days ago | Correct historical rollup |
| 20 | **Department filter** | 4 departments seeded | Rollup sums to campus total; `NULL` dept → `Unassigned` |
| 21 | **Timezone boundary** | event at 23:50 local (=18:20 UTC) | Attributed to the correct campus-local day |
| 22 | **Real-time update** | WS `entry` for a faculty member | `/on-campus` card flips to `LIVE` without a full reload |
| 23 | **Camera availability** | `stream_offline` + `stream_recovered(downtime_sec=600)` | Availability % reflects 10 min down |
| 24 | **No-data day** | Sunday, zero events | `no_data:true`, no `0%` headline |
| 25 | **Export integrity** | Export a day as CSV and XLSX | Row count matches the API; headers stable; UTF-8 BOM present |

**Manual demo rehearsal (do this before the client demo):** walk a known face past 3 cameras, then verify — journey shows 3 stops with correct names; attendance flips to present; `/on-campus` shows LIVE then decays; daily summary counts update.

---

# 10. Risks / Existing Bugs That Must Be Fixed

Only issues that directly affect these five features.

### 10.1 🔴 `camera_name` is never returned by the timeline API — **must fix**
`movement_service.py:22` returns `camera_id` only, but `PersonTimeline.js:196` renders `event.camera_name || event.camera_id`. **The existing History panel shows raw UUIDs today.** The journey feature is unusable without this. *Fix: outer-join `Camera` in `_event_to_dict`.* One-line-class fix, high visibility.

### 10.2 🔴 `tracking_sessions` leaks `status='active'` forever — **must avoid**
`main.py:187-214`: a session is opened on `entry` and closed only when an `exit` event arrives carrying `session_id`. But:
- `pipeline.stop()` calls `tracker.check_exits()` (`pipeline.py:440`), which **only emits exits for tracks absent ≥ `EXIT_THRESHOLD_SECONDS` (30s)**. Anyone visible at shutdown is dropped silently — no exit, no session close.
- A crash or `kill` closes nothing at all.
- On restart the tracker is empty, so those sessions can never be closed.

**Consequence:** `SELECT * FROM tracking_sessions WHERE status='active'` returns an ever-growing set of ghosts. **Mitigation: do not use `tracking_sessions` for attendance or presence.** Both are derived from `detection_events`, which are immutable point-in-time facts. *(Optional cleanup: on startup, close sessions with `entry_time < today` — 6 lines, worth doing, but the features must not depend on it.)*

### 10.3 🔴 "Entry" means *camera-visible*, not *campus-entered* — **design constraint**
`PersonTracker` is per-pipeline (`pipeline.py:109`). One campus visit past 4 cameras = 4 entries + 4 exits. `ENTRY_THRESHOLD_SECONDS=3`. **This is why attendance requires two independent confirmations and why the journey collapses consecutive same-camera events.** Not a bug — but any design that reads "entry count" as "people who entered campus" will be wrong.

### 10.4 🟠 ReID is not trustworthy for cross-camera identity — **do not depend on it**
`reid.py:91` uses a **stock ImageNet-pretrained ResNet50**, not a ReID-trained model, with the classification head removed and raw 2048-d avgpool features. Cosine threshold 0.65. The gallery is **in-memory only**, capped at 500 with LRU eviction, and **wiped on every restart**. `ENABLE_REID=True` and it does run (`pipeline.py:2140-2174`), but it is an appearance heuristic on clothing colour, not identity.
**Recommendation: journey and presence use face-recognition `person_id` only.** The `/reid` page stays as-is. Never surface ReID trails as "the person's journey".

### 10.5 🟠 Deleting a camera destroys its entire event history — **flag before demo**
`models.py:131`: `detection_events.camera_id` is `ON DELETE CASCADE`. Deleting a camera silently deletes every event and every incident recorded on it. This is unrelated to these features but would wipe attendance history mid-project. *Minimum action: warn in the delete confirmation dialog. Proper fix (out of scope): `ON DELETE SET NULL` + soft-delete cameras.*

### 10.6 🟠 Everything is UTC; there is no campus timezone — **must fix for attendance**
`time_utils.today_start_utc()` (`:64`) returns UTC midnight. In IST (UTC+5:30) that is 05:30 local. Any "today" or "late after 09:15" rule computed in UTC is wrong. *Fix: `CAMPUS_TIMEZONE` config + `campus_day_bounds()`. Keep storing UTC; convert only at query bounds and display.* The frontend already displays correctly (`toLocaleString`).

### 10.7 🟡 Duplicate "active persons" endpoints
`GET /api/cameras/active-persons` (`cameras.py:75`) and `GET /api/sessions/active` (`events.py:158`) are functionally identical. Don't add a third. *(Consolidation is optional cleanup, not required.)*

### 10.8 🟡 `security_alert` events carry no snapshot — **blocks F5's headline value**
`safety_analytics.py:389`, `hazard_detector.py:319`, `traffic.py` all build alerts without an image. An evidence pack with no image is not evidence. *Fix in Phase 5 (§F5). Historical alerts fall back to same-window unknown crops and full frames.*

### 10.9 🟡 `/api/system-health` has no auth
`routes/system_health.py:33` — no `Depends(get_current_user)`, unlike every other route. Not required by these features, but if the daily summary consumes it, it inherits an open endpoint. *One-line fix; do it while touching the file.*

### 10.10 🟡 Persons role dropdown cannot represent the seeded data
`persons/page.js:119-123` offers `employee/visitor/vip/banned`; the seeds write `assistant_professor`/`student`. Editing any seeded teacher through the UI **silently overwrites their role** and breaks their faculty classification. *Fixed in Phase 0 by `person_type` + a corrected dropdown.*

### 10.11 ⚪ No video clips exist — confirmed, and that's fine
No `VideoWriter`, no ring buffer, no clip storage anywhere in `app/vision/**`. `_last_jpeg` is one in-memory frame. `video_analysis` is upload-and-analyse. **Do not build recording for this deliverable.** Snapshot-based evidence is what the system honestly has, and it is substantial (32k images already on disk).

---

# 11. Final Recommended Scope

## MUST HAVE
- `persons.person_type` column + backfill + corrected persons UI role/department controls
- `CAMPUS_TIMEZONE` + `campus_day_bounds()` — attendance is wrong without it
- `camera_name` fix in the movements timeline (fixes an existing visible bug)
- **Person Journey** — service, `/api/persons/{id}/journey`, `JourneyStrip`, Journey tab
- **Faculty Attendance** — service with the two-confirmation rule, 4 endpoints, `/attendance` page with department drill-down, CSV/XLSX export
- **Teachers On Campus** — `PresenceService` (event-derived, not tracker-derived), `/on-campus` page, WS live updates
- **Daily Campus Summary** — service, `/daily-summary` page, template narrative, 3 dashboard cards
- **Incident Evidence Pack** — `incident_reviews` table, alert snapshot capture, `/incidents` + `/incidents/[id]` with status + notes
- Sidebar `College` section; dark+light theme pass; empty/error states
- Widen seed data to 4 departments so the demo matches the mockup

## NICE TO HAVE
- `attendance_days` nightly snapshot (protects history from retention cleanup and rule changes)
- Gate-camera tagging via `cameras.config` (`{"is_gate":true}`) for a sharper "departed" signal
- Print stylesheet → one-click PDF incident report
- `ix_events_person_created` composite index (do it if the events table is already large)
- Startup cleanup for stale `tracking_sessions`
- Weekly / monthly attendance rollup view
- Attendance % trend sparkline on the dashboard
- `require_admin` on `/api/system-health`

## DO NOT BUILD NOW
- ❌ **Video recording / incident clips** — no recording infrastructure exists. This is a multi-week subsystem (storage, retention, seeking, playback) and it is not required for the client value here. Note it as a Phase 2 roadmap item.
- ❌ **"Restricted area entry" detection** — needs a zone-drawing UI, per-camera polygons, and point-in-polygon evaluation in the vision loop. Use the existing `hazard` / `gathering` / `loitering` subtypes for the demo.
- ❌ **LLM-generated summaries** — the template generator produces better, faster, deterministic output for three sentences. Ollama/VLM plumbing already exists if it is ever wanted.
- ❌ **Any new AI model, ReID retraining, or gait/appearance matching** — the existing ReID is an appearance heuristic; improving it is a research project, not a sprint task.
- ❌ **Rewriting `PersonTracker` into a campus-level tracker** — high-risk change to the hot path. The derived event-based model achieves the same outcome from data that already exists.
- ❌ **A `departments` table, `journeys` table, or `presence` table** — all derivable; each would be a migration and a sync problem for zero gain.
- ❌ **Student attendance** — the same service supports it via `person_type='student'`, but only 2 students are enrolled and student attendance carries different institutional rules. Ship faculty first.
- ❌ **Consolidating the duplicate active-persons endpoints** — cosmetic; not worth the regression risk before a deadline.

---

## Optimisation scorecard

| Goal | How this plan achieves it |
|---|---|
| **Max client impact** | Five visible, demo-able features. Journey and attendance land in the first 4 days. |
| **Min implementation time** | ~10 dev-days. Zero pipeline changes for 4 of 5 features. One column + one table of DB work. |
| **Max reuse** | Reuses `detection_events`, `persons`, `cameras`, `unknown_faces`, snapshot storage, the export engine, the WebSocket, `PersonTimeline`, `Lightbox`, `StatCard`, `HourlyChart`, the entire CSS design system. |
| **Reliability of attendance/presence** | Derived from immutable committed events, not in-memory state. Restart-safe. Two-confirmation rule guards against 3-second false accepts. Departure labelled honestly as "last detected". |
| **Clean professional UI** | Every page composed from existing components and CSS tokens; no new dependencies; theme- and mobile-consistent. |
| **Min architectural risk** | No new services, queues, models, or infrastructure. The one hot-path change (alert snapshots) is behind a config flag and reuses the existing `SnapshotWorker` CRITICAL tier. |
