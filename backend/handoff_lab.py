"""
Handoff lab -- measure cross-camera handoff on real footage instead of guessing.

Every threshold in app/vision/handoff.py and topology.py was set by running
this against recordings of the COE cameras and LOOKING at who each number
ended up on. It is kept so the next change is made the same way.

    python handoff_lab.py record  runs/a --seconds 480 --zone COE
    python handoff_lab.py track   runs/a
    python handoff_lab.py replay  runs/a --events
    python handoff_lab.py sheet   runs/a

  record   synchronised clips from every camera in a zone, with a wall-clock
           timestamp per frame (what makes two cameras comparable).
  track    the production person detector + ByteTrack over those clips, one
           frame_metadata message per frame (exactly what the engine receives
           live), plus body crops and their OSNet embeddings.
  replay   feeds all cameras' messages, in time order, through the real
           HandoffEngine and reports what it learned and every number it
           carried. `--topology-in/--topology-out` chain recordings, so a
           second clip can be judged using only what the first one taught --
           the honest test. Replaying one clip twice is NOT: a recording
           agrees with itself perfectly.
  sheet    one row of body crops per number that crossed cameras. If a row
           shows two different people, that is a wrong join; nothing else
           measures that.

Recordings are footage of real people. They stay in the folder you name; keep
that folder out of version control and delete it when you are done.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _zone_cameras(zone: str):
    from app.db.models import Camera
    from app.db.session import SessionLocal
    from app.services.camera_service import camera_group

    with SessionLocal() as db:
        return [
            (str(c.id), c.name, c.stream_url) for c in db.query(Camera).all()
            if camera_group(c.name or "", c.location or "") == zone
        ]


# ── record ───────────────────────────────────────────────────────────────────

def record(args) -> None:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
    import cv2

    cameras = _zone_cameras(args.zone)
    if not cameras:
        sys.exit(f"no cameras in zone {args.zone!r}")
    os.makedirs(args.run, exist_ok=True)
    with open(os.path.join(args.run, "cameras.json"), "w") as f:
        json.dump({cid: name for cid, name, _url in cameras}, f, indent=1)
    until = time.time() + args.seconds

    def one(camera_id, name, url):
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        writer, last, frames = None, 0.0, 0
        with open(os.path.join(args.run, camera_id + ".ts"), "w") as stamps:
            while time.time() < until:
                ok, frame = capture.read()
                now = time.time()
                if not ok or frame is None:
                    capture.release()
                    time.sleep(0.5)
                    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    continue
                if now - last < 1.0 / args.fps:
                    continue
                last = now
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(args.run, camera_id + ".mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                writer.write(frame)
                stamps.write(f"{now:.3f}\n")
                frames += 1
        capture.release()
        if writer is not None:
            writer.release()
        print(f"{name}: {frames} frames")

    threads = [threading.Thread(target=one, args=c) for c in cameras]
    [t.start() for t in threads]
    [t.join() for t in threads]


# ── track ────────────────────────────────────────────────────────────────────

def track(args) -> None:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from app.config import settings
    from app.vision.appearance import AppearanceService, encode_body_crop, usable_body_box
    from app.vision.bytetrack import ByteTracker
    from app.vision.frame_metadata import build_frame_metadata

    names = json.load(open(os.path.join(args.run, "cameras.json")))
    model = YOLO(settings.PERSON_DET_MODEL or settings.HAZARD_MODEL_PATH)
    confidence = float(settings.PERSON_DET_CONF)
    min_px = int(settings.HANDOFF_APPEARANCE_MIN_PX)

    for camera_id, name in names.items():
        stamps = [float(x) for x in open(os.path.join(args.run, camera_id + ".ts")).read().split()]
        capture = cv2.VideoCapture(os.path.join(args.run, camera_id + ".mp4"))
        tracker = ByteTracker()
        crop_dir = os.path.join(args.run, "crops", camera_id)
        os.makedirs(crop_dir, exist_ok=True)
        looks, pending, last_crop = [], [], {}
        # The same service the server runs, so the vectors are the live ones.
        service = AppearanceService(on_embedding=lambda c, t, v, ts: looks.append((t, ts, v.tolist())))
        service.start()
        n = 0
        with open(os.path.join(args.run, camera_id + ".jsonl"), "w") as out:
            while n < len(stamps):
                ok, frame = capture.read()
                if not ok:
                    break
                t = stamps[n]
                h, w = frame.shape[:2]
                boxes, scores = [], []
                for result in model(frame, conf=0.25, classes=[0], verbose=False, device=0,
                                    imgsz=settings.PERSON_DET_IMGSZ):
                    for b in result.boxes or ():
                        if float(b.conf[0]) >= confidence:
                            boxes.append([float(v) for v in b.xyxy[0].tolist()])
                            scores.append(float(b.conf[0]))
                tracks = tracker.update(
                    np.array(boxes, np.float32).reshape(-1, 4), np.array(scores, np.float32))
                entries, xyxy = [], {}
                for tr in tracks:
                    top, right, bottom, left = tr.to_location()
                    xyxy[tr.track_id] = (left, top, right, bottom)
                    entries.append({"kind": "person", "box": (top, right, bottom, left),
                                    "track_id": tr.track_id, "label": f"Track #{tr.track_id}"})
                for track_id, box in xyxy.items():
                    if t - last_crop.get(track_id, 0.0) < 1.0:
                        continue
                    others = [b for other, b in xyxy.items() if other != track_id]
                    if not usable_body_box(box, w, h, others, min_px):
                        continue
                    jpeg = encode_body_crop(frame, box)
                    if jpeg is None:
                        continue
                    last_crop[track_id] = t
                    service.submit(camera_id, track_id, t, jpeg)
                    with open(os.path.join(crop_dir, f"T{track_id:05d}_{n:06d}.jpg"), "wb") as f:
                        f.write(jpeg)
                out.write(json.dumps(build_frame_metadata(
                    camera_id=camera_id, frame_width=w, frame_height=h,
                    entries=entries, timestamp=t, frame_id=n)) + "\n")
                n += 1
        capture.release()
        deadline = time.time() + 30
        while service.stats["embedded"] < service.stats["submitted"] and time.time() < deadline:
            time.sleep(0.1)
        service.stop()
        with open(os.path.join(args.run, camera_id + ".looks.json"), "w") as f:
            json.dump(looks, f)
        print(f"{name}: {n} frames, {len(last_crop)} tracks looked at, {len(looks)} looks")


# ── replay ───────────────────────────────────────────────────────────────────

def _replay(args):
    import numpy as np

    from app.vision.handoff import HandoffConfig, HandoffEngine
    from app.vision.topology import Topology

    names = json.load(open(os.path.join(args.run, "cameras.json")))
    stream = []
    for camera_id in names:
        with open(os.path.join(args.run, camera_id + ".jsonl")) as f:
            for line in f:
                message = json.loads(line)
                stream.append((message["timestamp"], 0, camera_id, message))
        if not args.no_looks:
            for track_id, t, vector in json.load(open(os.path.join(args.run, camera_id + ".looks.json"))):
                # A crop reaches the engine a moment after its frame did.
                stream.append((t + 0.25, 1, camera_id, (track_id, np.array(vector, np.float32))))
    stream.sort(key=lambda row: (row[0], row[1]))
    start = stream[0][0]

    log, alias = [], {}

    class Recorded(HandoffEngine):
        now = 0.0

        def _rebind(self, key, binding, gid, reason, detail):
            log.append((self.now, reason, f"{names[key[0]]} T{key[1]} -> #{gid} ({detail})"))
            super()._rebind(key, binding, gid, reason, detail)

        def _merge(self, gid_a, gid_b, reason, detail):
            a, b = self._people.get(gid_a), self._people.get(gid_b)
            merged = super()._merge(gid_a, gid_b, reason, detail)
            if merged:
                keep, drop = (a, b) if a.created_at <= b.created_at else (b, a)
                alias[drop.gid] = keep.gid
                log.append((self.now, reason, f"#{drop.gid} is #{keep.gid} ({detail})"))
            return merged

    topology = Topology()
    if args.topology_in:
        topology.load(args.topology_in)
    engine = Recorded(topology=topology)
    engine.set_config(HandoffConfig.from_rows([(cid, name, {}, "zone") for cid, name in names.items()]))

    shown, first_seen = {}, {}
    for t, kind, camera_id, payload in stream:
        engine.now = t - start
        if kind == 1:
            engine.add_appearance(camera_id, payload[0], payload[1], t)
            continue
        engine.process(payload)
        for obj in payload["objects"]:
            if "global_track_id" in obj:
                key = (camera_id, obj["track_id"])
                shown[key] = obj["global_track_id"]
                first_seen.setdefault(key, t - start)

    def final(gid):
        while gid in alias:
            gid = alias[gid]
        return gid

    people = {}
    for key, gid in shown.items():
        people.setdefault(final(gid), []).append(key)
    if args.topology_out:
        topology.save(args.topology_out)
    return engine, names, people, first_seen, log


def replay(args) -> None:
    engine, names, people, _first_seen, log = _replay(args)
    print("stats:", dict(engine.stats))
    learned = engine.snapshot()["learned"]
    for o in learned["overlaps"]:
        state = "IN USE" if o["ready"] else "not confirmed"
        print(f"  shared view  {o['a_name']} <-> {o['b_name']}: {state}, {o['walks']} walks, "
              f"{o['agreement']:.0%} agree")
    for r in learned["routes"]:
        print(f"  route        {r['from_name']} -> {r['to_name']}: {r['min_sec']}..{r['max_sec']} s, "
              f"{r['episodes']} occasions")
    crossed = sum(1 for keys in people.values() if len({k[0] for k in keys}) > 1)
    print(f"tracks {sum(len(k) for k in people.values())} -> numbers {len(people)}; "
          f"{crossed} numbers seen on more than one camera")
    if args.events:
        for t, reason, text in log:
            if reason != "stitch" or args.stitches:
                print(f"  +{t:6.1f}s {reason:7s} {text}")


# ── sheet ────────────────────────────────────────────────────────────────────

def sheet(args) -> None:
    import cv2
    import numpy as np

    _engine, names, people, first_seen, _log = _replay(args)
    out_dir = os.path.join(args.run, "sheets")
    os.makedirs(out_dir, exist_ok=True)

    def crops(camera_id, track_id, n=2):
        folder = os.path.join(args.run, "crops", camera_id)
        files = sorted(f for f in os.listdir(folder) if f.startswith(f"T{track_id:05d}_"))
        picks = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(n, len(files)))] if files else []
        return [cv2.imread(os.path.join(folder, f)) for f in picks]

    rows = []
    crossed = sorted(
        ((gid, sorted(keys, key=first_seen.get)) for gid, keys in people.items()
         if len({k[0] for k in keys}) > 1),
        key=lambda row: first_seen[row[1][0]],
    )
    for gid, keys in crossed:
        label = np.zeros((128, 80, 3), np.uint8)
        cv2.putText(label, f"#{gid}", (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cells = [label]
        for camera_id, track_id in keys[:10]:
            images = crops(camera_id, track_id)
            if not images:
                continue
            block = np.hstack([cv2.resize(i, (64, 128)) for i in images])
            cv2.putText(block, names[camera_id][-14:], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cells += [block, np.full((128, 6, 3), 60, np.uint8)]
        rows.append(np.hstack(cells))
    for page in range(0, len(rows), 14):
        chunk = rows[page: page + 14]
        width = max(r.shape[1] for r in chunk)
        canvas = np.vstack([np.hstack([r, np.zeros((128, width - r.shape[1], 3), np.uint8)]) for r in chunk])
        path = os.path.join(out_dir, f"crossed_{page // 14:02d}.jpg")
        cv2.imwrite(path, canvas)
        print(path)
    if not rows:
        print("no number crossed cameras in this recording")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("record")
    p.add_argument("run")
    p.add_argument("--zone", default="COE")
    p.add_argument("--seconds", type=float, default=480)
    p.add_argument("--fps", type=float, default=8)
    p = sub.add_parser("track")
    p.add_argument("run")
    for command in ("replay", "sheet"):
        p = sub.add_parser(command)
        p.add_argument("run")
        p.add_argument("--topology-in")
        p.add_argument("--topology-out")
        p.add_argument("--no-looks", action="store_true")
        p.add_argument("--events", action="store_true")
        p.add_argument("--stitches", action="store_true")
    args = parser.parse_args()
    {"record": record, "track": track, "replay": replay, "sheet": sheet}[args.command](args)


if __name__ == "__main__":
    main()
