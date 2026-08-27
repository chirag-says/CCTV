# SentinelAI — College Features: Reviewed Implementation Specification (v2)

**Supersedes** `COLLEGE_FEATURES_PLAN.md` (v1). v1 remains the record of the codebase investigation; **this document is the build contract.**

**Status:** Approved for implementation. Build phase-by-phase. Do not deviate. Do not introduce architecture not listed here.

**Changes from v1** — incorporating review, marked ✅ accepted / ⚠️ accepted with adjustment:

| # | Review correction | Disposition |
|---|---|---|
| 1 | Attendance `OR` rule too loose | ⚠️ Accepted with adjustment — see §2.1 |
| 2 | 90-min presence timeout too loose | ⚠️ Accepted with adjustment — see §3.1 |
| 3 | Journey should be a hero feature | ✅ Accepted in full — §4 |
| 4 | Trim Incident V1 scope | ✅ Accepted in full — §6 |
| 5 | Drop unknown-person journey | ✅ Accepted in full |
| 6 | Add "Faculty Search → Where are they?" | ✅ Accepted — and cheaper than estimated, §5 |
| 7 | Reorder phases | ✅ Accepted — §8 |
| — | Don't overclaim attendance | ✅ Accepted — formalised as §1.3 labelling contract |

---

# 0. Two adjustments the reviewer should sign off on

Everything else in the review is adopted verbatim. These two need a correction, and I'd rather raise them now than silently implement something different.

## 0.1 The attendance rule creates a *false absence* risk that the review's version makes worse

The proposed rule is correct and I'm adopting it:

```
PRESENT = 2+ high-confidence sightings
          AND (span >= ATTENDANCE_MIN_SPAN_MIN  OR  camera_count >= 2)
```

It genuinely fixes the `08:42 / 08:43 double-false-positive` hole. Adopted.

**But it introduces the mirror failure.** Consider a deployment where only the Main Gate camera is running — which is the likely demo configuration:

- Teacher arrives 08:42 (1 sighting), teaches all day in rooms with no camera, leaves 17:21 (1 sighting).
- `entry_count = 2` ✅, `camera_count = 1`, `span = 8h39m` ✅ → **PRESENT.** Fine.

Now the harder one:

- Teacher arrives 08:42, is seen once, and the gate camera is the only camera. They leave through a side gate.
- `entry_count = 1` → **ABSENT.** They were on campus all day.

A confidently-wrong `✕ Absent` next to a teacher who was demonstrably at work is the single worst failure mode this feature has — worse than a false present, because it is an accusation.

**Adjustment:** keep the strict rule as the default, but make the shortfall *visible instead of silent*.

- `ATTENDANCE_MIN_SIGHTINGS` and `ATTENDANCE_MIN_SPAN_MIN` are config, so a single-camera deployment can be tuned down without a code change.
- The attendance table always shows `Sightings` and `Cameras` columns. When a person is `absent` but has ≥1 sighting, the row renders `✕ Absent · 1 sighting` in muted text rather than a bare `✕`.
- No fourth status, no extra UI surface. One column, one conditional string.

This costs ~10 lines and converts an invisible false accusation into a visible, explainable data point.

## 0.2 The proposed presence model is *less* conservative than v1, not more — in the case that matters

The review's objection is right: `PRESENCE_TIMEOUT_MIN = 90` showing "On campus" for someone last seen 89 minutes ago is indefensible on a demo screen. Agreed, and dropped.

But the proposed replacement is:

```
ON CAMPUS = attendance confirmed today AND no reliable departure signal
```

Since a "reliable departure signal" requires a gate camera the person may never walk past again, this marks someone last seen at **08:42** as "On campus" at **23:00** — an 8-hour stale claim, strictly worse than the 90-minute one it replaces. The 15-minute `RECENTLY SEEN` tier doesn't prevent this, because `ON CAMPUS` is defined as the fallback below it.

**Adjustment — keep the review's three visible states, but fix what the headline number counts:**

| Internal state | Condition | UI |
|---|---|---|
| `LIVE` | in live tracker (< ~30s) | 🟢 **Visible now** · CSE Block |
| `ON_CAMPUS` | last seen < `PRESENCE_CONFIDENT_MIN` (default **20 min**) | 🔵 **On campus** · Last seen 4 min ago |
| `SEEN_EARLIER` | seen today, no gate exit, older than 20 min | 🔵 **On campus** · Last seen 2h 14m ago |
| `DEPARTED` | gate exit ≥ `GATE_DEPART_MIN`, or past `ATTENDANCE_DAY_END` | ⚪ **Departed** · 17:21 |

Three visible states exactly as requested. `SEEN_EARLIER` is not a fourth badge — it renders identically to `ON_CAMPUS`; **the last-seen age is what does the honesty work**, and it is always shown.

The headline is split so neither number can embarrass:

```
TEACHERS CURRENTLY ON CAMPUS

18 present now                    ← LIVE + ON_CAMPUS only
+ 6 seen earlier today            ← SEEN_EARLIER, muted subline
```

The list is sorted by recency, so stale entries sink to the bottom on their own. Nobody has to defend "89 minutes ago", and nobody has to defend "8 hours ago" either.

---

# 1. Product contract

## 1.1 Feature priority (final)

1. **Faculty Attendance** — the core institutional value
2. **Teachers Currently On Campus** — the live "wow"
3. **Person Journey** — hero feature, highest visual impact
4. **Daily Campus Summary** — the executive view
5. **Incident Evidence Pack** — trimmed V1

## 1.2 Naming rules — non-negotiable

- The feature is **"Person Journey"**. Never "tracking", "trail", "surveillance history", or "movement log" in any user-facing string.
- Faculty presence view is **"Currently on Campus"**.
- Never render the word "Tracked".

## 1.3 Labelling contract — what we may and may not claim

This table is binding on all UI copy. The system observes cameras; it does not observe campus boundaries.

| ❌ Never say | ✅ Always say | Why |
|---|---|---|
| "Entered campus at 08:42" | **"First detected 08:42"** | An `entry` event means *became visible on a camera*, not *crossed the campus boundary* |
| "Departed 17:21" | **"Last detected 17:21"** | Exit events are unreliable (v1 §10.2); absence of detection ≠ departure |
| "Left the building" | **"No longer detected"** | Same |
| "Present 8h 39m" | **"Time observed 8h 39m"** | It is the observation span, not attendance duration |
| "Tracked across 4 cameras" | **"Seen on 4 cameras"** | Tone |
| "0% attendance" (no-data day) | **"No data for this day"** | A day with no cameras running is not a day everyone was absent |

The only place a departure may be asserted as fact is when a gate camera produced the last detection and `GATE_DEPART_MIN` has elapsed — then, and only then, the UI may say **"Departed 17:21 · Main Gate"**.

---

# 2. Feature — Faculty Attendance

## 2.1 The rule (final)

```python
# app/services/attendance_service.py

# Per faculty person P, per campus-local day D:
events = detection_events
         WHERE person_id = P
           AND event_type IN ('entry', 'exit')
           AND confidence >= ATTENDANCE_MIN_CONFIDENCE      # default 0.55
           AND created_at IN campus_day_bounds(D)

first_seen   = MIN(created_at)
last_seen    = MAX(created_at)
sightings    = COUNT(*) WHERE event_type = 'entry'
camera_count = COUNT(DISTINCT camera_id)
span_min     = (last_seen - first_seen) / 60

confirmed = (
    sightings >= ATTENDANCE_MIN_SIGHTINGS               # default 2
    and (
        span_min >= ATTENDANCE_MIN_SPAN_MIN             # default 5
        or camera_count >= ATTENDANCE_MIN_CAMERAS       # default 2
    )
)

status = 'absent'                                    if not confirmed
       = 'late'    if to_campus_local(first_seen).time() > ATTENDANCE_LATE_AFTER   # default 09:15
       = 'present'
```

The `AND` gate is the review's correction. Two sightings alone are never enough — they must be *separated in time* or *corroborated by a second camera*. Two false recognitions one minute apart on one camera now correctly yield `absent`.

## 2.2 Config (all in `app/config.py`)

```python
CAMPUS_TIMEZONE: str = "Asia/Kolkata"
ATTENDANCE_DAY_START: str = "06:00"
ATTENDANCE_DAY_END: str = "21:00"
ATTENDANCE_LATE_AFTER: str = "09:15"
ATTENDANCE_MIN_CONFIDENCE: float = 0.0   # DISABLED — amended, see note below
ATTENDANCE_MIN_SIGHTINGS: int = 2
ATTENDANCE_MIN_SPAN_MIN: int = 5
ATTENDANCE_MIN_CAMERAS: int = 2
FACULTY_PERSON_TYPE: str = "faculty"
```

> **Amended during Phase 2 — `ATTENDANCE_MIN_CONFIDENCE` ships as `0.0`
> (disabled), not `0.55`.** Measured against the live database: `exit` events
> carry **no confidence at all** (`tracker.check_exits()` builds the event dict
> without the key — 6,238 of 6,238 are `0.0`), and `entry` events only carry one
> when the confirming frame came from the recognition path (6,361 of 6,400 are
> `0.0`). Where a confidence *is* recorded it clusters around 0.39–0.70, so a
> 0.55 gate discarded 5 of 7 genuine entries for a real person and would mark
> the entire faculty absent. An uncalibrated threshold that silently voids 99%
> of presence evidence is worse than none — the same reasoning the codebase
> already applies to `ENABLE_ADAPTIVE_THRESHOLDS`. Events with
> `confidence <= 0` are **never** filtered: unrecorded is not the same as low.
> Calibrate against enrolled faculty before raising it.
>
> **Also amended — physical-camera de-duplication.** The `cameras >= 2` branch
> is only sound if those are two different lenses. `MainGate` and `MainStream`
> are both `channel=1` of the same NVR, differing only by `subtype` (main vs
> sub stream), so one walk-past would have satisfied the rule from a single
> appearance. `normalise_stream_source()` collapses quality variants, Hikvision
> stream indices and differing credentials onto one key. Consequence for this
> deployment: **one physical camera**, so the `cameras >= 2` branch can never
> fire and the span branch carries every decision.

## 2.3 Query — one grouped query for the whole campus-day

```sql
SELECT person_id,
       MIN(created_at) AS first_seen,
       MAX(created_at) AS last_seen,
       COUNT(*) FILTER (WHERE event_type = 'entry') AS sightings,
       COUNT(DISTINCT camera_id) AS camera_count
FROM detection_events
WHERE person_id = ANY(:faculty_ids)
  AND event_type IN ('entry','exit')
  AND confidence >= :min_conf
  AND created_at >= :day_start_utc
  AND created_at <  :day_end_utc
GROUP BY person_id
```

Roster from `persons WHERE person_type='faculty' AND is_active`. Everything else is Python folding. Sub-millisecond at this scale.

## 2.4 API

| Method | Endpoint | Response |
|---|---|---|
| GET | `/api/attendance/summary?date=` | `{date, departments:[{name,present,late,absent,total,pct}], totals:{present,total,pct}, no_data}` |
| GET | `/api/attendance/day?date=&department=&status=` | `{date, records:[{person_id,name,department,role,status,first_seen,last_seen,minutes_observed,sightings,camera_count,has_encodings}]}` |
| GET | `/api/attendance/person/{id}?from=&to=` | `{person, days:[...], summary:{present_days,late_days,absent_days,pct}}` |
| GET | `/api/attendance/export?date=&format=csv\|xlsx\|json&department=` | File download |

## 2.5 UI — `/attendance`

```
FACULTY ATTENDANCE — TODAY              ‹ 27 Aug 2026 ›      [CSV] [Excel]

CSE            18 / 22        ████████████████░░░░  82%
ECE            16 / 19        ███████████████░░░░░  84%
Mechanical     12 / 15        ████████████████░░░░  80%
Civil          10 / 12        █████████████████░░░  83%

TOTAL          56 / 68                              82.4%
```

Clicking a department card drills into the person table:

| Status | Name | First detected | Last detected | Observed | Sightings | Cameras |
|---|---|---|---|---|---|---|
| ✓ Present | Rahul Sharma | 08:42 | 17:21 | 8h 39m | 14 | 4 |
| ✓ Present | Priya Mehta | 08:51 | 16:58 | 8h 07m | 9 | 3 |
| ⚠ Late | Raj Patel | 09:47 | 17:02 | 7h 15m | 11 | 3 |
| ✕ Absent | Neha Singh | — | — | — | 0 | 0 |
| ✕ Absent · 1 sighting | Anil Kumar | 08:39 | 08:39 | — | 1 | 1 |

The last row is §0.1 in action — the shortfall is visible, not silent.

Faculty with `encoding_count == 0` get a `No face enrolled` chip instead of `✕ Absent`. They cannot possibly be recognised, and blaming them for it makes the system look broken.

## 2.6 Components
- **NEW** `DepartmentCard.js` — count + progress bar (`.card` + `.stat-card` tokens)
- **NEW** `AttendanceStatusBadge.js` — maps to existing `.badge.{active,pending,offline}`
- **NEW** `DatePicker.js` — `‹ 27 Aug 2026 ›` + native `<input type=date>`
- **REUSE** `StatCard`, `.data-table`, export-download logic copied from `analytics/page.js:40-81`

## 2.7 Effort: **2.25 days** (1.0 BE, 1.25 FE)

---

# 3. Feature — Teachers Currently on Campus

## 3.1 Presence model (final — per §0.2)

```python
# app/services/presence_service.py

live_ids = {p['person_id'] for p in camera_manager live trackers}

for person in attendance_day(today) where status in ('present','late'):
    age_min      = (now - last_seen).minutes
    last_cam     = camera of last event
    gate_exit    = last_cam.config.get('is_gate') and age_min >= GATE_DEPART_MIN
    past_day_end = now_local > ATTENDANCE_DAY_END

    state = 'LIVE'          if person_id in live_ids
          = 'DEPARTED'      if gate_exit or past_day_end
          = 'ON_CAMPUS'     if age_min < PRESENCE_CONFIDENT_MIN     # 20
          = 'SEEN_EARLIER'
```

```python
PRESENCE_CONFIDENT_MIN: int = 20   # minutes — inside this, "On campus" is asserted confidently
GATE_DEPART_MIN: int = 10          # minutes at a gate camera before departure is asserted
```

Gate cameras are flagged via the **existing** `cameras.config` JSON column — `{"is_gate": true, "zone": "Main Gate"}`. **No migration.**

## 3.2 API

| Method | Endpoint | Response |
|---|---|---|
| GET | `/api/attendance/on-campus?department=&q=` | `{present_now, seen_earlier, faculty:[{person_id,name,department,role,state,camera_name,first_seen,last_seen,minutes_since_seen,minutes_observed,avatar_url}]}` |

Sorted: `LIVE` → `ON_CAMPUS` → `SEEN_EARLIER` → `DEPARTED`, each by recency descending.

## 3.3 UI — `/on-campus`

```
TEACHERS CURRENTLY ON CAMPUS

18 present now
+ 6 seen earlier today

🔍 Search faculty...                        [All ▾] [CSE] [ECE] [Mech] [Civil]

┌──────────────────────────────────────────────┐
│ RS   Rahul Sharma                🟢 Visible now │
│      CSE · Assistant Professor                │
│      CSE Block — Camera 04                    │
│      First detected 08:42 · Observed 3h 24m   │
│                          Today's Journey →    │
└──────────────────────────────────────────────┘
┌──────────────────────────────────────────────┐
│ PM   Priya Mehta      🔵 On campus · 4 min ago │
│      ECE · Assistant Professor                │
│      Academic Block — Camera 02               │
│      First detected 08:51 · Observed 3h 15m   │
└──────────────────────────────────────────────┘
```

**Real-time:** poll `/on-campus` every 30s **and** subscribe to `WS /api/events/live` (already wired — same pattern as `page.js:43` and `security-alerts/page.js:61`). On any `entry`/`exit` with a known `person_id`, patch that card optimistically and schedule a debounced refetch.

## 3.4 Components
- **NEW** `FacultyPresenceCard.js` — reuses `.activity-item`, `.avatar`, `.badge`
- **NEW** `PresenceStateDot.js` — 🟢/🔵/⚪ + relative-time string

## 3.5 Effort: **1.25 days** (0.5 BE, 0.75 FE)

---

# 4. Feature — Person Journey ⭐ HERO

Upgraded per review correction #3. This gets the most design attention of any screen.

## 4.1 Derivation

```
events for (person, campus-local day) ORDER BY created_at ASC
  → collapse consecutive events on the same camera into one STOP
  → split a stop when the gap exceeds JOURNEY_STOP_GAP_SEC (default 300)
  → stop = {camera_id, camera_name, location, zone_icon,
            arrived_at, left_at, dwell_sec, event_count}
  → transitions = [(stop[i] → stop[i+1], gap_sec)]
```

Collapsing is essential: lingering at one camera produces entry → exit → cooldown → entry → exit. Raw rows would render "Main Gate" five times.

```python
JOURNEY_STOP_GAP_SEC: int = 300
```

## 4.2 API

| Method | Endpoint | Response |
|---|---|---|
| GET | `/api/persons/{id}/journey?date=` | `{person, date, first_seen, last_seen, camera_count, observed_sec, stops:[...], transitions:[...], no_data}` |
| GET | `/api/movements/person/{id}/timeline?date=&days=&limit=` | **EXTENDED** — add `date`; add `camera_name` + `camera_location` to every row |

**Dropped per correction #5:** ~~`/api/unknown-faces/{id}/journey`~~. Unknown faces still appear as *evidence* inside incident packs (§6) — that is a plain query, not this endpoint.

## 4.3 UI — Journey tab on `/persons/[id]`

```
Rahul Sharma
CSE · Assistant Professor                        ‹ 27 Aug 2026 ›

FIRST DETECTED    LAST DETECTED    CAMERAS    TIME OBSERVED
08:42             17:21            4          8h 39m

TODAY

  08:42  ●  🚪  Main Gate
         │      First detected
         │      5 min
  08:47  ●  🏢  Academic Block
         │      25 min
  09:12  ●  💻  CSE Block
         │      3h 36m
  12:48  ●  🍴  Cafeteria
         │      18 min
  13:06  ●  💻  CSE Block
         │      4h 15m
  17:21  ●  🚪  Main Gate
                Last detected
```

**Design notes:**
- Vertical rail with connector line — pure CSS (`border-left` + absolutely-positioned dots). No library.
- Dwell time sits **on the connector**, between stops — it is the time *at* the preceding stop.
- Zone icon comes from `cameras.config.zone_icon`, defaulting to a generic camera glyph. Set alongside `is_gate` in the camera form.
- Metric strip uses `StatCard` with a compact variant.
- Copy follows §1.3 exactly: "First detected" / "Last detected", "Time observed".
- Empty state: "No detections for this day" — never "0 attendance".

## 4.4 Components
- **NEW** `JourneyTimeline.js` (~180 lines) — the vertical rail above
- **NEW** `JourneyMetrics.js` (~50 lines) — the four-metric strip
- **MODIFY** `PersonTimeline.js` — render `camera_name` (**fixes the live UUID bug**), add a `Today` mode
- **MODIFY** `persons/[id]/page.js` — add the `Journey` tab (`activeTab` mechanism already exists)

## 4.5 Effort: **1.75 days** (0.5 BE, 1.25 FE) — raised from 1.25 to fund the hero treatment

---

# 5. Feature — Faculty Search → "Where are they?" ⭐

Review correction #6. **Cheaper than estimated: the search UI already exists.**

## 5.1 What already exists
- **[CONFIRMED]** `components/SearchPalette.js` — Ctrl+K command palette, 300ms debounce, ↑/↓/Enter navigation, backed by `GET /api/search` which already searches persons by name.
- **[CONFIRMED BUG]** `SearchPalette.js:83` does `router.push(config.route)` — it **discards `result.id`**, so selecting a person navigates to `/persons` (the list) instead of that person. Fixing this one line delivers most of the feature.

## 5.2 Build
1. **FIX** `SearchPalette.js:83` → `router.push(\`${config.route}/${result.id}\`)` for `person` results.
2. **ENRICH** `GET /api/search` person results with `person_type`, `department`, and — when `person_type='faculty'` — a live `presence` block from `PresenceService`. One extra service call, only for faculty hits.
3. **NEW** `components/WhereIsPersonCard.js` — the answer card:

```
Rahul Sharma
CSE · Assistant Professor

🟢 Currently on campus

CURRENT LOCATION      CSE Block — Camera 04
FIRST DETECTED        08:42
LAST DETECTED         09:37

Today's Journey →
```

4. **MOUNT** a "Where is someone?" search box at the top of `/on-campus` and as a Dashboard card. Both reuse `SearchPalette` in inline (non-modal) mode via an `inline` prop.

This is the connective tissue: **Search → Person → Current location → Journey → Attendance**, all four features reachable in two clicks, with **zero new backend systems**.

## 5.3 Effort: **0.5 day** (0.15 BE, 0.35 FE)

---

# 6. Feature — Incident Evidence Pack (V1, trimmed)

Scope reduced per review correction #4.

## 6.1 V1 scope — build exactly this

```
INCIDENT #1042
HIGH — Loitering Detected

CAMERA        CSE Block — Camera 04
TIME          14:32:18
DURATION      4m 21s

EVIDENCE
[Alert snapshot]  [Related face]  [Full frame]

TIMELINE
14:28   Person detected
14:29   Loitering begins
14:32   Alert triggered
14:34   Person no longer detected

STATUS
[ OPEN ]  [ INVESTIGATING ]  [ RESOLVED ]
```

## 6.2 Deferred out of V1 (build later, not now)
- ❌ Investigation notes
- ❌ Assignee
- ❌ Severity override
- ❌ Export / share report
- ❌ "Other cameras" correlation panel
- ❌ Persons-involved journey links

The `incident_reviews` table **keeps** its `notes`, `assignee` and `severity` columns — they cost nothing unused and adding them later would need manual SQL (there is no Alembic). **Schema stays as designed; only the UI shrinks.**

## 6.3 The one pipeline change: alert snapshots

**[CONFIRMED]** `security_alert` events currently carry **no image** (`safety_analytics.py:389`, `hazard_detector.py:319`, `traffic.py`). An evidence pack with no image is not evidence — this is the single highest-value change in the feature.

**Implementation — mirror the existing unknown-face pattern exactly:**

1. `pipeline._handle_security_alert()` (`pipeline.py:1268`) attaches `frame.copy()` to the alert dict. Cost ~0.5ms; alerts are cooldown-throttled 15–600s per subtype, so this is negligible on the vision thread.
2. `main._process_pipeline_event()` writes the JPEG **synchronously** in the EventQueue consumer thread — the same place and manner as the unknown face crop (`main.py:323`) — then sets `snapshot_url` **before** the row is inserted, and strips `frame` from `metadata_json`.

> ⚠️ **Do not** use the async `SnapshotWorker` callback to set `snapshot_url`. The callback can land *after* the row is inserted, producing alerts with a permanently NULL snapshot. Synchronous write in the consumer thread is the correct pattern and is already proven in this codebase.

Gate the whole thing behind `ALERT_SNAPSHOTS_ENABLED: bool = True`.

> **Amended during Phase 6 — the frame is not copied or re-encoded.**
> The spec called for `frame.copy()` on the vision thread plus a fresh encode.
> `pipeline._last_jpeg` already holds the ANNOTATED frame the JPEG encoder
> thread produced for the live stream, so the alert handler takes a reference
> to those immutable bytes instead: no array copy, no re-encode, and the saved
> image carries the detection boxes the operator would have seen — better
> evidence than a clean frame. The consumer thread writes the bytes straight
> to disk via `save_jpeg_bytes()`. The key is underscore-prefixed
> (`_evidence_jpeg`) and popped unconditionally in `_process_pipeline_event`,
> because raw bytes raise `TypeError` on both JSON paths — `metadata_json` and
> the WebSocket broadcast. Verified end-to-end: bytes land byte-identical,
> `snapshot_url` is set before insert, nothing leaks into metadata.

## 6.4 Evidence assembly (V1)

| Slot | Source |
|---|---|
| Alert snapshot | `detection_events.snapshot_url` (new) |
| Related face | `unknown_faces` where `camera_id` matches and `first_seen` within ±`INCIDENT_EVIDENCE_WINDOW_SEC` (default 120) → `snapshot_url` |
| Full frame | same row → `full_frame` |
| Timeline | `detection_events` on that camera in `[t−5min, t+5min]`, rendered as sentences |

Historical alerts predating the change have no snapshot — fall back to related unknown crops and full frames, and render **"No alert snapshot captured"** rather than a broken image.

## 6.5 API (V1)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/incidents?status=&subtype=&date=&camera_id=&limit=&offset=` | List + `counts_by_status` |
| GET | `/api/incidents/{event_id}` | Evidence pack |
| PATCH | `/api/incidents/{event_id}` | `{status}` — `Depends(require_admin)` |

`stream_offline` / `stream_recovered` are **excluded by default** — they are health, not security.

## 6.6 Components
- **NEW** `/incidents/page.js`, `/incidents/[id]/page.js`
- **NEW** `EvidenceGrid.js` → feeds the **existing** `Lightbox`
- **NEW** `IncidentStatusControl.js` — segmented control
- **MODIFY** `security-alerts/page.js` — link cards to `/incidents/{id}`; read `severity` from the API instead of the hardcoded map at `:24-46`

## 6.7 Effort: **1.75 days** (0.75 BE, 1.0 FE) — down from 2.25

---

# 7. Database changes (unchanged from v1)

`init_db()` runs `Base.metadata.create_all()` — it creates missing **tables** but never adds **columns or indexes** to existing tables. No Alembic. Follow the `app/migrations/NNN_*.sql` convention for anything that isn't a new table.

## 7.1 REQUIRED — manual SQL: `app/migrations/009_college_features.sql`

```sql
ALTER TABLE persons ADD COLUMN IF NOT EXISTS person_type VARCHAR(20);

UPDATE persons SET person_type = 'faculty' WHERE person_type IS NULL
  AND role IN ('professor','associate_professor','assistant_professor','hod','lecturer','teacher');
UPDATE persons SET person_type = 'student' WHERE person_type IS NULL AND role = 'student';
UPDATE persons SET person_type = 'staff'   WHERE person_type IS NULL AND role IN ('employee','staff');
UPDATE persons SET person_type = 'visitor' WHERE person_type IS NULL;

CREATE INDEX IF NOT EXISTS ix_persons_type_dept ON persons (person_type, department);

CREATE INDEX IF NOT EXISTS ix_events_person_created
  ON detection_events (person_id, created_at) WHERE person_id IS NOT NULL;
```

Then add `person_type` to `models/person.py` (`PersonBase`, `PersonUpdate`), `person_service._person_to_dict()`, and the persons UI.

## 7.2 REQUIRED — additive table (auto-created, no SQL)

`IncidentReview` in `app/db/models.py` exactly as specified in v1 §4.2 — including the `notes`, `assignee` and `severity` columns that V1 will not yet use.

## 7.3 NOT NEEDED
No `departments`, `journeys`, `presence`, or `camera_zones` tables. Zone/gate config lives in the existing `cameras.config` JSON.

---

# 8. Implementation phases (reordered per correction #7)

Each phase ends with a green checkpoint and its slice of the §9 test matrix. **Report exactly what changed after each phase. Do not start the next phase until the checkpoint passes.**

### Phase 1 — Faculty foundation *(0.75 day)*
- `009_college_features.sql` — apply manually, verify with `\d persons`
- `person_type` through Pydantic → service → persons UI; fix the role dropdown (`persons/page.js:119-123` cannot represent the seeded values)
- `IncidentReview` model added (auto-created on boot)
- `config.py` — every constant from §2.2, §3.1, §4.1, §6.3
- `time_utils.campus_day_bounds()` / `to_campus_local()`
- **Fix** `camera_name` in `movement_service._event_to_dict()` — outer-join `Camera`
- Widen `seed_teachers.py` to CSE / ECE / Mechanical / Civil
- ✅ **Checkpoint:** `/persons` shows correct roles + departments and filters by them; the History panel shows camera **names**, not UUIDs.

### Phase 2 — Faculty Attendance ⭐ *(2.25 days)*
- `attendance_service.py` with the §2.1 rule; lift export writers from `routes/analytics.py:145-212` into `utils/export_utils.py`
- 4 endpoints; `/attendance` page; `DepartmentCard`, `AttendanceStatusBadge`, `DatePicker`
- ✅ **Checkpoint:** the mockup renders with real data — `CSE 18/22 … TOTAL 56/68 · 82.4%`, drill-down, CSV + XLSX download, and the `✕ Absent · 1 sighting` case renders correctly.

### Phase 3 — Currently On Campus ⭐ *(1.75 days, incl. search)*
- `presence_service.py` with the §3.1 four-state model; gate flags via `cameras.config`
- `/api/attendance/on-campus`; `/on-campus` page; `FacultyPresenceCard`; WS live patching
- **Faculty Search (§5)** — fix `SearchPalette.js:83`, enrich `/api/search`, `WhereIsPersonCard`
- ✅ **Checkpoint:** live roster with split headline (`18 present now / + 6 seen earlier`); a real face walking past a camera flips a card to 🟢 within ~5s; Ctrl+K → type a teacher's name → their live location card.
- ⚠️ The "Today's Journey →" link is rendered **disabled** until Phase 4.

### Phase 4 — Person Journey ⭐ *(1.75 days)*
- `journey_service.py`; `/api/persons/{id}/journey`; timeline endpoint `date` param
- `JourneyTimeline`, `JourneyMetrics`, Journey tab, `PersonTimeline` Today mode
- Enable the Phase 3 journey link
- ✅ **Checkpoint:** open a teacher → the §4.3 vertical journey with correct stops, dwell times and metric strip.

### Phase 5 — Daily Campus Summary *(1.25 days)*
- `campus_summary_service.py` — occupancy replay (+1/−1 over entry/exit), camera availability from `stream_offline`/`stream_recovered` `downtime_sec`, **template-based narrative (no LLM)**
- `/api/analytics/daily-summary`; `/daily-summary` page; 3 dashboard cards
- ✅ **Checkpoint:** full "TODAY'S CAMPUS" metric wall + 3-line generated narrative; a no-camera day shows "No data", never 0%.

### Phase 6 — Incident Evidence *(1.75 days)*
- Alert snapshot capture (§6.3 — synchronous, in the consumer thread)
- `incident_service.py`; 3 endpoints; `/incidents` + `/incidents/[id]`; `EvidenceGrid` → `Lightbox`; status control
- ✅ **Checkpoint:** trigger a loitering alert → the §6.1 page renders with a real image, timeline, and working status transitions.

### Phase 7 — Polish *(1.0 day)*
- Dashboard college cards; Sidebar `College` section
- Empty / error / loading states on all new pages
- Mobile widths; dark **and** light theme pass
- Demo dataset: 4 departments, realistic arrival spread, at least one late and one absent
- Full §9 matrix
- **If time remains:** `attendance_days` nightly snapshot (v1 §4.4)

**Total: 10.5 developer-days.** Phases 1–4 (6.5 days) deliver the four highest-impact screens.

---

# 9. Test matrix

Plain pytest in `backend/tests/`, matching `test_identity_continuity.py` style. All attendance/presence/journey logic is pure functions over event rows — **seed synthetic `detection_events`, no camera required.**

| # | Scenario | Setup | Expected |
|---|---|---|---|
| 1 | Correct attendance | 3 entries, 2 cameras, 08:42–17:21 | `present`, first 08:42, last 17:21 |
| 2 | Late arrival | first entry 09:47 | `late` |
| 3 | Absence | no events | `absent` |
| 4 | **Double false positive** | 2 entries, same camera, 60s apart | **`absent`** — the §2.1 `AND` gate |
| 5 | **Two cameras, short span** | 2 entries, 2 cameras, 90s apart | `present` via `camera_count` branch |
| 6 | **Two sightings, long span, one camera** | 08:42 + 17:21, gate only | `present` via `span` branch |
| 7 | **Single sighting** | 1 entry only | `absent`, `sightings=1` surfaced in the row |
| 8 | Low confidence | all entries at 0.3 | `absent` |
| 9 | Missed exit | entries only, no exits | `present`; `last_seen` = last entry, labelled "Last detected" |
| 10 | Duplicate detections | 6 entries, one camera, 5 min | journey collapses to 1 stop; attendance unaffected |
| 11 | Backend restart | insert, restart, query | identical result |
| 12 | Stale active session | 3-day-old `tracking_sessions.status='active'` | does **not** appear in on-campus |
| 13 | Presence LIVE | person in live tracker | 🟢, camera name shown |
| 14 | Presence ON_CAMPUS | last seen 4 min ago | 🔵 "4 min ago", counted in headline |
| 15 | **Presence SEEN_EARLIER** | last seen 2h ago, no gate exit | 🔵 "2h 14m ago", **excluded** from headline, in `+ N seen earlier` |
| 16 | Presence gate departure | last seen at `is_gate` camera 15 min ago | ⚪ Departed |
| 17 | Presence day end | now > `ATTENDANCE_DAY_END` | all ⚪ Departed |
| 18 | Journey stops | 4 cameras with revisit | correct stop order, dwell, revisit as separate stop |
| 19 | Journey gap split | same camera, 10-min gap | 2 stops, not 1 |
| 20 | Timezone boundary | event 23:50 IST (18:20 UTC) | attributed to the correct campus day |
| 21 | Historical date | `date=` 5 days ago | correct rollup |
| 22 | Department filter | 4 departments | rollup sums to total; NULL dept → `Unassigned` |
| 23 | No-data day | zero events campus-wide | `no_data:true`, no `0%` headline |
| 24 | Incident creation | insert `security_alert` | appears in `/api/incidents`, `status='open'` |
| 25 | Incident snapshot | trigger a live alert | `snapshot_url` non-NULL **at insert time** |
| 26 | Incident evidence correlation | alert at T, unknown face at T+30s same camera | face appears in the pack |
| 27 | Incident status | PATCH → resolved | persisted, survives reload |
| 28 | Search → person | Ctrl+K, select a faculty result | navigates to `/persons/{id}`, **not** `/persons` |
| 29 | Export integrity | CSV + XLSX for one day | row counts match API; stable headers; UTF-8 BOM |
| 30 | Real-time | WS `entry` for faculty | card flips to 🟢 without full reload |

**Demo rehearsal before the client:** walk a known face past 3 cameras → journey shows 3 stops with correct names; attendance flips to present; `/on-campus` shows 🟢 then decays to 🔵; daily summary counts update.

---

# 10. Standing constraints

Carried from v1 §10. These are binding.

1. **Never use `tracking_sessions`** for attendance or presence — it leaks `status='active'` forever (`pipeline.stop()` only exits tracks absent ≥30s; a crash exits nothing).
2. **Never surface ReID as a person's journey** — `reid.py:91` is stock ImageNet ResNet50, in-memory, LRU-capped, wiped on restart. Face-recognition `person_id` only.
3. **Never treat an `entry` event as "entered campus"** — it means *visible on camera X*. `ENTRY_THRESHOLD_SECONDS=3`.
4. **Never compute a day boundary in UTC** — always `campus_day_bounds()`. Store UTC, convert at query bounds and display.
5. **Never build video recording** — none exists (`_last_jpeg` is one in-memory frame; `video_analysis` is upload-and-analyse). Snapshot evidence only.
6. **Never violate the §1.3 labelling contract.**
7. **Warn before camera deletion** — `detection_events.camera_id` is `ON DELETE CASCADE` (`models.py:131`); deleting a camera destroys its entire event and incident history. Add a warning to the confirm dialog in Phase 7.

---

# 11. Do not build

Unchanged and reaffirmed:

- ❌ Video recording / incident clips
- ❌ Restricted-zone detection (needs zone-drawing UI + per-camera polygons + point-in-polygon in the vision loop)
- ❌ LLM-generated summaries (template generator is better, faster, deterministic, free)
- ❌ Any new AI model, ReID retraining, gait or appearance matching
- ❌ Rewriting `PersonTracker` into a campus-level tracker
- ❌ `departments` / `journeys` / `presence` / `camera_zones` tables
- ❌ Unknown-person journey endpoint *(dropped per correction #5 — unknown faces remain as incident evidence only)*
- ❌ Student attendance *(same service supports it via `person_type='student'`; ship faculty first)*
- ❌ Consolidating the duplicate active-persons endpoints
- ❌ Incident notes / assignee / severity override / export *(deferred out of V1; columns exist, UI does not)*
