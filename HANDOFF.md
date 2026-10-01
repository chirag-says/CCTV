# SentinelAI (CCTV-COLLEGE) — Engineering Handoff

Context for a fresh agent picking up this codebase. Written 2026-08-28.

---

## 0. What the system is

College campus CCTV analytics. FastAPI + PostgreSQL backend, Next.js frontend,
13 registered RTSP cameras. Per-camera `VisionPipeline` threads in one process
do face detection (SCRFD via insightface `buffalo_l`), recognition (ArcFace),
person tracking (ByteTrack), plus safety analytics, ANPR and a VLM-assisted
identity review flow.

Key paths:

| Path | Role |
|---|---|
| `backend/app/vision/pipeline.py` | ~3000 lines. The per-camera hot path. |
| `backend/app/vision/bytetrack.py` | `STrack` / `ByteTracker`. Geometry + identity state. |
| `backend/app/vision/unknown_registry.py` | Per-pipeline stable identities for unenrolled people. |
| `backend/app/vision/detector.py` | insightface loading + SCRFD wrapper. |
| `backend/app/main.py` | EventQueue consumers — where DB writes happen. |
| `backend/app/services/` | incident, journey, attendance, presence, clustering. |

Two architectural rules the codebase states and follows:
**tracking owns existence, recognition owns identity**; and
**the machine proposes, a human disposes** (no auto-merge into the face gallery).

---

## 1. CRITICAL — repo and environment state

Read this before touching anything.

### Nothing is committed

- Branch: `phase0-defects-and-auth`, HEAD = `58905f4 "before d"`.
- **HEAD already contains** coasting, person tracking, the unidentified-occupant
  lane, identity voting, `detection_events.unknown_person_id`, multi-view
  persistence, the incident presence query, and the SCRFD detector fixes.
- **Uncommitted working-tree changes:** `config.py`, `main.py`,
  `routes/unknown_faces.py`, `services/unknown_service.py`, `pipeline.py`.
- **Untracked new files:** `app/services/unknown_dedup.py`,
  `app/vision/load_governor.py`, `tests/test_load_governor.py`,
  `tests/test_unknown_dedup_index.py`.

Decide early whether to commit. Nothing has been verified in CI.

### `.env` overrides config defaults — always read it

`backend/.env` silently overrides `app/config.py`. This caused three wrong
conclusions during the last session. Currently set there:

```
FRAME_SKIP=4                      # was 2
INSIGHTFACE_DET_SIZE=640          # code default says 1024
INSIGHTFACE_DET_THRESH=0.60
UNKNOWN_SIMILARITY_THRESHOLD=0.62 # was 0.50
FACE_MATCH_TOLERANCE=0.4
```

### Migrations are hand-applied — there is no Alembic

`init_db()` calls `Base.metadata.create_all()`, which creates missing *tables*
but never adds a *column*. Every migration from `002` on must be run by hand.
If a model column is missing from the database, **every INSERT to that table
fails** — for `detection_events` that is silent, total loss of the event log.

```bash
cd backend
python apply_migration.py --check      # reports drift, exits non-zero
python apply_migration.py --list
python apply_migration.py app/migrations/013_whatever.sql
```

`psql` is NOT on PATH on this machine (it lives in
`C:\Program Files\PostgreSQL\15\bin`). `apply_migration.py` reads
`app.config.settings`, so it always targets the right database.

`verify_schema()` in `app/db/session.py` runs at boot and logs a CRITICAL
banner naming any missing column. It deliberately does not raise.

**Migrations 011 and 012 are already applied to the live database.**

### Other environment facts

- `DEBUG=True`, so uvicorn runs with `reload=True`. Editing a `.py` file
  restarts the server and re-reads `.env`. Camera workers take ~1–2 min to
  come back.
- `detection_events`'s JSON column is named `metadata_json` and is Postgres
  type **`json`, not `jsonb`** — the `?` key-existence operator does not work
  on it. Use `->>` with a NULL check.
- `detection_events.event_type` has a CHECK constraint allowing only
  `entry, exit, detection, unknown, security_alert`. Adding a value needs a
  migration. Prefer `event_type='detection'` plus a `subtype`.
- Only 2 of 13 cameras (`coe`, `security cabin outside`) actually produce
  events. The other 11 have live pipelines consuming detection budget.

---

## 2. What was changed, and why

In dependency order. All of it is in the working tree; most is in HEAD.

### Step 0 — Coasting (drawing)

**Problem:** the box vanished the instant someone turned their head.
`ByteTracker.update()` returns only `_tracked_stracks`; a track unmatched for
`BYTETRACK_TRACKED_GRACE` detection-frames moves to `_lost_stracks` — where it
stays a live re-association candidate with its identity intact, but stops being
drawn. At `FRAME_SKIP` 3 the old grace of 2 was ~0.2 real seconds.

**Fix:** `STrack.last_seen_at` / `last_seen_bbox` record the last *observed*
position (never the motion-predicted `bbox`, which `predict()` keeps drifting).
`ByteTracker.get_coasting_tracks()` exposes lost-but-recent tracks.
`VisionPipeline._coasting_boxes()` draws them dashed and dimmed.

**Deliberately a separate method, not added to `update()`'s return value** —
the pipeline runs recognition and `PersonTracker.on_detection` over everything
`update()` returns, so coasting tracks there would mark people present on
frames where nothing was detected. Coasting is a drawing concern.

Config: `BYTETRACK_COAST_ENABLED`, `BYTETRACK_COAST_SEC` (2.0),
`BYTETRACK_TRACKED_GRACE` 2 → 4.

### Step 1 — Person tracking (the architectural correction)

**Problem:** ByteTrack was fed SCRFD *face* boxes. A person who turned away
ceased to exist. There was no person in the system, only a face that sometimes
had a name.

**Fix:** `TRACKING_MODE="person"` (default) feeds YOLO **person** boxes to
ByteTrack. Two new helpers in `pipeline.py`:

- `_tracking_detections()` — picks the detection source. Falls back to face
  mode *permanently* if YOLO can't load. A frame where YOLO finds nobody is
  **not** a fallback (it returns empty and tracks coast) — switching source
  mid-stream would change what a track box means and churn every id.
- `_bind_face_to_track()` — binds a face to a track. Person mode uses
  **containment** (face centre inside the padded body box, largest face wins),
  because a face box and a body box barely overlap by area so IoU is the wrong
  test. Face mode keeps the original IoU path.

The per-track loop now separates `draw_location` (the body box — drawing,
centroid, presence) from `face_location` (the bound face — ArcFace, size gate,
pose, crops). **`face_location is None` is normal, not an error.**

Four interactions that had to be fixed alongside:

1. Presence: a known person facing away had a live track but nothing refreshed
   `PersonTracker`, so they aged toward `EXIT_THRESHOLD` while standing there.
   An established identity now refreshes presence off the body.
2. The `anon:` flood — see Step 1.5.
3. Crowd detection reads centroids from `PersonTracker`, which unidentified
   person tracks no longer feed. They are now passed to crowd detection (which
   counts *people*) but deliberately not to loitering (which reasons about a
   specific person over time).
4. The YOLO assist ran a *second* inference on the same frame, and its
   body-only fallback (`_handle_unknown_person_body`) builds a centroid-keyed
   pseudo-identity — exactly what person tracks now own. Disabled in person
   mode; the face-in-crop rescue is kept, capped at 3 bodies per round.

Config: `TRACKING_MODE`, `PERSON_DET_CONF`, `PERSON_FACE_BIND_PAD`.

**Live evidence it works:** 4.5 tracks/frame against 1.0 face/frame. Under
face-only tracking those two numbers are the same by construction.

**Main risk, still unmeasured:** ByteTrack's carry-over sets `person_id` on
IoU ≥ 0.3 *without* recognition verification, and body boxes overlap far more
than face boxes. Two students crossing in a corridor is the failure case.
Watch `IDENTITY SWAP on T<id>` at WARNING.

### Step 1.5 — Unidentified occupants

**Problem:** person tracking made "a tracked body we never identified" the
ordinary case. The old `anon:{camera}:{track_id}` handle was a string
pretending to be a person: it entered `PersonTracker._active_tracks` beside
real identities, and because it embedded the ByteTrack id it minted a new
"person" on every track recreation — which `_cooldowns` then held forever
(it had **no time-based expiry**, a real latent leak, now fixed).

**Fix:** a separate lane inside `PersonTracker`, keyed on the integer track id
and expired by time: `on_unidentified()`, `expire_unidentified()`,
`get_unidentified_centroids()` (namespaced `track:<id>` keys),
`unidentified_count`. Bounded by `UNIDENTIFIED_MAX_TRACKS` with stalest-first
eviction.

Occupancy events are written as `event_type='detection'`,
`subtype='unidentified_entry'|'unidentified_exit'` — **never `entry`/`exit`**,
because `analytics_service` counts `event_type == "entry"` with no filter and
would inflate footfall.

Also: `STrack.unknown_id` / `unknown_label` were added so a *stranger* who
turns away keeps their stable UnknownRegistry id instead of being re-counted
as an anonymous body (one human in two lanes in a single visit).

### Step 2 — Asymmetric identity voting

**Problem:** a single frame could flip or drop an identity.

**Fix:** `IdentityState` enum on `STrack`
(`PROVISIONAL / KNOWN / UNKNOWN_CANDIDATE / UNKNOWN_CONFIRMED`) plus a bounded
time-windowed vote list `(ts, person_id|None, confidence, confident)`.

| | Bar | Config |
|---|---|---|
| Promote to KNOWN | 2 confident matches, same person, spread ≥0.5 s | `IDENTITY_PROMOTE_*` |
| Demote from KNOWN | 3 **confident** contradictions in 10 s | `IDENTITY_DEMOTE_*` |
| Confirm UNKNOWN | 3 non-matches spanning ≥1 s | `UNKNOWN_CONFIRM_*` |

The span requirement matters: one well-lit pose produces several consecutive
agreeing frames, and that is *one* observation.

- **Absence of a face never records a vote**, which is what makes Step 1's
  guarantee hold.
- A **gray-zone** miss can no longer unseat a confirmed identity (it still
  drops a `PROVISIONAL` guess, and still counts toward confirming a stranger).
- A **definitive** miss no longer drops an identity in one frame.
- `UNKNOWN_MATURATION_SEC` (a 500 ms timer) is **superseded** and no longer
  read. Only a confirmed stranger creates an `unknown_faces` row — this is
  what prevents most stranded unknowns from existing at all.
- `_report_unknown_resolved()` emits `subtype='unknown_resolved'` carrying both
  ids when a track that was a confirmed stranger resolves to an enrolled
  person. **Deliberately does not auto-merge** — the existing review flow
  (`POST /unknown-faces/{id}/merge`) does that after human approval.
- The identity-swap counter now records only where a swap is **applied**, not
  where the vote threshold prevented one.

### Step 3 — `unknown_person_id` (migration 011, APPLIED)

Stranger ids were being written into `detection_events.metadata_json` — real
data, but unindexable and unqueryable. Migration 011 promotes it to a real
column with two indexes:

- `ix_events_unknown_created (unknown_person_id, created_at) WHERE NOT NULL`
- `ix_events_camera_created_all (camera_id, created_at)`

Not a foreign key: an `unknown_faces` row can be enrolled or dismissed, and
neither should cascade into the permanent event log.

**Backfilled 10,090 rows.** `main.py` writes the column *and* still writes the
JSON key, because `campus_summary_service._person_key` and the analytics export
still read the JSON. That duplicate write can be removed once those move.

### Step 3.5 — Multi-view persistence (migration 012, APPLIED)

**Measured failure:** cross-camera re-identification of unenrolled people was
**0.0%** — 415 strangers across 7 live cameras, none matched on two of them.
Cross-day was 0.3–0.5%.

**Root cause, and it was not the matching logic.** `UnknownRegistry` already
matched best-of-K over several views. But `unknown_faces` stores ONE encoding,
so every refresh, restart and second camera rebuilt each identity from its
single frozen first view — best-of-K silently became best-of-1. Exactly the
failure `unknown_registry.py`'s own docstring says multi-view was written to
fix. It was fixed in memory and never on disk.

Second cause: `get_pending_encodings(limit=500)` and `max_entries=500` against
**9,624** pending identities — each camera saw 5% of the pool, and the
shortfall was self-reinforcing.

**Fix:** `unknown_face_views` table (mirrors `face_encodings`), seeded 9,751
rows. Round trip closed both ways — the registry queues learned views, the
pipeline ships them, the consumer writes them; `get_pending_encodings` returns
all views and `maybe_refresh` rebuilds with them. Caps raised to 3000/3000/5.

**A trap for anyone raising the caps further:** naively raising them made
`resolve()` take **39.6 ms** (more than a frame budget). The matvec was never
the cost — `np.vstack` rebuilt the whole matrix after every mint, and
`_evict()` set `_dirty` unconditionally while trimming one identity per mint.
Fixed with an append-in-place buffer and batch eviction to 90%. **39.6 ms →
0.29 ms.** Search is not the constraint; memory is (the buffer is per-pipeline,
~61 MB × 13 cameras).

### Step 5 — Incident presence query

**Problem:** "who was in Classroom 204 at 14:00" was answered with
"`unknown_faces` rows whose `first_seen` falls in the window". `first_seen` is
the first moment that identity was *ever* observed, so it found strangers who
happened to be new that minute and **missed every repeat visitor** — the people
most likely to be involved.

**Fix:** `IncidentService._people_present()` queries `detection_events` using
the 011 column, returning known + unknown + an unidentified body count.

- **Presence is an interval, not a point.** Someone who entered at 13:50 and
  left at 14:20 was present at 14:00 with no event near it. Entry/exit pairs
  are reconstructed and tested for overlap, which is why the query reads
  `INCIDENT_PRESENCE_LOOKBACK_SEC` (30 min) further back than it reports on.
  Each person carries `basis`: `observed` or `spanned`.
- `_sibling_camera_ids()` merges duplicate registrations of one physical camera
  (`MainGate`/`MainStream`, `security cabin` ×2) via `normalise_stream_source`.
- `evidence.faces` / `full_frames` keep their exact shape (`EvidenceGrid.js`
  renders them); `evidence.people` is additive.

### B — Write-time dedup

**The actual bug was the threshold, and it was not the value in the code.**
`.env` set `UNKNOWN_SIMILARITY_THRESHOLD=0.50` — *stricter than*
`FACE_MATCH_STRICT` (0.52). The system would call two faces the same enrolled
person at 0.62, then refuse to call two sightings the same stranger unless
within 0.50, on faces that are by definition the harder ones.

Measured, 300 recent identities held out against a 6,000-identity corpus:

| threshold | merged into an existing row |
|---|---|
| 0.50 (was) | **0.0%** |
| 0.62 (now) | **55.3%** |

That 0.0% is the entire explanation for 81% of strangers having exactly one
visit.

**Calibrated against the labelled gallery** (22 people, 112 embeddings, 307
same-person and 5,909 different-person pairs):

| threshold | same-person merged | different-person merged |
|---|---|---|
| 0.50 | 23.1% | 0.00% |
| **0.62** | **33.9%** | **0.12%** |
| 0.72 | 53.4% | 0.46% |

Stopped at 0.62 deliberately: at 0.72 the false-merge rate quadruples, and
merging two strangers corrupts the very incident question the id exists to
answer. (Same-person figures are *pessimistic* — gallery embeddings are curated
for pose diversity, so they are deliberately far apart.)

**Also:** new `UnknownDedupIndex` (`app/services/unknown_dedup.py`) — cached,
vectorized, best-of-K across every stored view. The old path re-queried ~10k
rows and deserialized each in a Python loop, **280 ms per face** on the
EventQueue consumer thread. Now **0.11 ms**. New rows and learned views are
added live; enrolled/dismissed rows are dropped.

Multi-view only adds ~1 point today (55.3% → 56.3%) because views have barely
accumulated. Its value grows as they do.

### A — Load governor (the uncommitted delta)

`_apply_load_degradation` tiers on **one camera's** face count. With 13
pipelines each seeing ~1 face, all conclude "normal" while collectively
oversubscribing the GPU and GIL ~3×. No camera can see that from its own state.

`app/vision/load_governor.py` — process-wide singleton. Closed loop on measured
detection latency (not a capacity model, whose every term goes stale): holds
p95 near `LOAD_TARGET_DETECT_MS` (45 ms) by handing every pipeline the same
extra frame skip, decaying to zero when load clears.

Wide dead band (0.7×–1.2×) because detection latency is noisy and a governor
that chases fluctuations oscillates. Hard ceiling (`LOAD_MAX_EXTRA_SKIP=6`).

Skipping detection frames is only acceptable **because of Step 1** — the
tracker now coasts through gaps, so it costs recognition *freshness*, not track
continuity.

**Status: unit-tested, never exercised under real pressure.** It made zero
decisions live because p95 was already under target.

### SCRFD performance

Two fixes in `detector.py`:

1. **`onnxruntime`'s CUDA provider only worked by accident.** `onnxruntime-gpu`
   does not ship the CUDA runtime; the only `cublasLt64_12.dll` on the machine
   is inside `site-packages/torch/lib`. ORT doesn't look there, so it silently
   fell back to CPU — **3,648 ms vs 34 ms**, a ~100× regression. It usually
   worked only because some subsystem imported torch first, and every
   `import torch` in this codebase is inside a function, making it a startup
   race. `_ensure_cuda_dll_path()` now imports torch explicitly.
   **`os.add_dll_directory()` alone does NOT work** — ORT loads its provider
   DLL with flags that bypass user-added directories. Measured both ways.
2. **`allowed_modules=['detection','recognition']`.** `FaceAnalysis` was
   loading all five buffalo_l models and running `landmark_3d_68`,
   `landmark_2d_106` and `genderage` on every face for nothing. The pipeline
   reads only `face.embedding` and `face.kps` — and `kps` comes from SCRFD
   itself. Gender is DeepFace, in `safety_analytics.py`. 1.18× measured.

Result across the session: SCRFD p95 **108.8 ms → 32.1 ms**. Attribution is
spread across `FRAME_SKIP` 2→4, `allowed_modules`, and the dedup fix (at 0.50
*every* face minted a row, paying 280 ms of dedup plus inserts plus three
snapshot writes).

---

## 3. Measurement tools written for this work

```bash
cd backend
python apply_migration.py --check
python measure_unknown_reid.py --days 1
python measure_unknown_reid.py --since 2026-08-28T12:38
```

`measure_unknown_reid.py` reports, in the order they gate each other:

1. **views per identity** — must exceed 1.00, or the Step 3.5 write path isn't
   firing and nothing below can improve.
2. **cross-camera / cross-day re-ID** — the headline. Was exactly 0.0%.
3. **sightings per stranger** — a spike at exactly 2 means one entry/exit pair
   each, i.e. every stranger is a first-time visitor.

---

## 4. Baselines to compare against

All measured 2026-08-28 on the live database, **before** the dedup threshold
change and before any camera session on the new code.

```
total detection_events        31,419
events with unknown identity  10,090
distinct strangers             4,062
strangers with exactly 1 visit    81%   (3,281 of 4,069)
pending unknown_faces         11,327   (8,563 with occurrence=1)
cross-camera re-ID               0.0%   (415 strangers, 7 cameras)
cross-day re-ID                  0.4%
SCRFD p95                     108.8 ms  (min 9.6, median 33.5)
```

Live confirmations after the changes: 4.5 tracks/frame vs 1.0 face/frame
(person tracking); `unidentified bodies: 10–13` per incident (Step 1.5
occupancy events firing); incident rosters returning known + unknown + bodies.

---

## 5. What is NOT done

### Step 4 — unified journey page ("Where has this person been?")

The remaining planned feature. Generalize `journey_service` from `person_id` to
`(subject_type, subject_id)` so it serves unknown identities too; everything
downstream (stop collapsing, `normalise_stream_source`, campus-day bounds)
works unchanged. Then a UI listing known + unknown people, each linking to a
timeline.

**GATED.** Do not build this until `measure_unknown_reid.py` shows cross-camera
re-ID meaningfully above 0.0% and the "2 sightings" spike has fallen. Built on
today's data it would render 4,000 strangers who each visited once —
technically correct and useless.

### Deferred / open

- **Cross-camera body ReID.** `app/vision/reid.py` is a full ResNet50
  cross-camera engine that is effectively dead: only called from
  `_process_frame_hybrid`, and its body box is *guessed* from the face box
  (`bottom + face_h*3`), so it only runs when a face was already detected.
  Deferred until face-based re-ID is proven. A separate, nearer use: an
  appearance term in ByteTrack's association cost matrix (BoT-SORT style) if
  ID-switch rate climbs in crowds.
- **`UnknownRegistry` as a shared singleton.** Currently one per pipeline;
  cross-camera agreement comes via a 60 s DB refresh. Sharing it (like
  `get_reid_tracker()`) would make it instant and cut ~13× memory duplication.
- **Pre-existing bug, filed but unfixed:** `clean_frame = frame` is an alias, so
  `_attempt_yolo_assisted_detection` draws boxes onto the same array it later
  crops for snapshots — annotation gets burned into saved unknown-face images.
  Check `_handle_unknown_person_body` too.
- **The `anon:` guards** in `main.py:184` and `presence_service.py:103` are now
  unreachable from the pipeline but still correctly filter historical rows.
- 11 of 13 cameras produce no events while consuming detection budget.

---

## 6. Working notes for the next agent

- **Read `.env` before trusting `config.py`.** Three wrong conclusions came
  from not doing this.
- **`py_compile` is not an import check.** A bad import edit passed
  `py_compile` and went live via uvicorn auto-reload. Use
  `python -c "import app.vision.pipeline"`.
- **Measure before and after, on real data.** Nearly every conclusion in this
  session that came from reading code alone was wrong or incomplete; nearly
  every one that came from a measurement held. Specifically: the "det_size is
  too high" theory (it was already 640), the "GPU is saturated" theory (22%),
  the "add_dll_directory fixes CUDA" fix (it didn't), and the 0.70 threshold
  (it was 0.50).
- **The distribution of stored embeddings is censored by the dedup threshold**
  that produced it — you cannot read a natural valley from it. Calibrate
  against the labelled gallery (`face_encodings` grouped by `person_id`).
- Long-running benchmarks: model loading takes minutes. Run in background,
  write to a file, don't pipe through `grep` (it buffers).
- The test suite is 441 tests, all passing, no DB required except where
  explicitly noted.
