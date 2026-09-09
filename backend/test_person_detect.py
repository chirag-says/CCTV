"""
Person detection and tracking regression check.

Phase 2 item 0a switches the person detector to fp16 with a pinned imgsz. That
is the only model Basic runs, so it is the hottest call in the product -- and
fp16 changes numbers, so "it got faster" is not on its own an acceptable
result. The brief asks for person-box counts and ByteTrack id churn compared
before and after, which is what this measures.

Deterministic by construction: fixed frames, in order, through a real
ByteTracker, with no wall-clock or threading involved. Two runs of the same
build produce byte-identical JSON.

    python test_person_detect.py --out before.json
    python test_person_detect.py --out after.json
    python test_person_detect.py --compare before.json after.json

WHAT COUNTS AS A REGRESSION
Not "any change". fp16 rounding will move a box by a pixel and a confidence in
the third decimal, and that is fine. What matters is whether the pipeline sees
the same PEOPLE and keeps the same TRACKS:

  * person count per frame -- a detection appearing or vanishing
  * total ByteTrack ids minted over the sequence -- id churn, which is what
    breaks identity continuity downstream
  * mean IoU of matched boxes -- whether boxes moved meaningfully
"""

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import cv2
import numpy as np

DEFAULT_FRAMES = list(range(0, 900, 15))     # 60 frames, evenly spread


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def collect(video, frame_indices, conf_floor=0.05):
    from app.config import settings
    from app.vision.bytetrack import ByteTracker
    from app.vision.pipeline import VisionPipeline

    # A pipeline object purely to reuse its detector loading and call
    # convention; nothing here starts a camera or a thread.
    pipe = VisionPipeline.__new__(VisionPipeline)
    pipe.camera_id = "persondet"
    pipe._yolo_person_detector = None
    pipe._yolo_person_loaded = False
    import threading
    pipe._yolo_person_lock = threading.Lock()
    pipe.hazard_detector = None
    if not pipe._load_yolo_person_detector():
        sys.exit("no YOLO person detector available")

    tracker = ByteTracker()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = {
        "video": video,
        "half": getattr(settings, "PERSON_DET_HALF", None),
        "imgsz": getattr(settings, "PERSON_DET_IMGSZ", None),
        # NOT settings.PERSON_DET_CONF. On the only clip in uploads/videos the
        # people are distant and score 0.08 to 0.44, so the product floor of
        # 0.35 admits almost none of them and the comparison would be between
        # two empty sets -- which passes trivially and proves nothing about
        # fp16. This measures the DETECTOR, so it takes everything the
        # detector emits and compares the full distribution.
        "conf_floor": conf_floor,
        "frames": [],
        "persons_total": 0,
        "ids_minted": 0,
    }

    seen_ids = set()

    for idx in frame_indices:
        if total and idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        dets = pipe._get_yolo_person_detections(frame)
        boxes, scores = [], []
        for d in dets:
            if d.get("confidence", 0.0) < conf_floor:
                continue
            b = d["bbox"]
            boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
            scores.append(round(float(d["confidence"]), 4))

        arr = (np.array(boxes, dtype=np.float32) if boxes
               else np.empty((0, 4), dtype=np.float32))
        sc = (np.array(scores, dtype=np.float32) if scores
              else np.empty(0, dtype=np.float32))
        tracks = tracker.update(arr, sc)
        ids = sorted(int(t.track_id) for t in tracks)
        seen_ids.update(ids)

        out["persons_total"] += len(boxes)
        out["frames"].append({
            "index": idx,
            "persons": len(boxes),
            "boxes": [[int(v) for v in b] for b in boxes],
            "scores": scores,
            "track_ids": ids,
        })

    out["ids_minted"] = len(seen_ids)
    cap.release()
    return out


def show(res):
    print(f"\nvideo            {res['video']}")
    print(f"half             {res['half']}")
    print(f"imgsz            {res['imgsz']}")
    print(f"frames sampled   {len(res['frames'])}")
    print(f"persons detected {res['persons_total']}")
    print(f"ByteTrack ids    {res['ids_minted']}")


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    print(f"before  half={a['half']} imgsz={a['imgsz']}  "
          f"{a['persons_total']} persons, {a['ids_minted']} ids")
    print(f"after   half={b['half']} imgsz={b['imgsz']}  "
          f"{b['persons_total']} persons, {b['ids_minted']} ids")

    fa = {f["index"]: f for f in a["frames"]}
    fb = {f["index"]: f for f in b["frames"]}
    common = sorted(set(fa) & set(fb))

    count_diff, ious = 0, []
    for i in common:
        pa, pb = fa[i]["persons"], fb[i]["persons"]
        if pa != pb:
            count_diff += 1
        for box_a in fa[i]["boxes"]:
            best = max((_iou(box_a, box_b) for box_b in fb[i]["boxes"]), default=0.0)
            ious.append(best)

    mean_iou = sum(ious) / len(ious) if ious else 1.0
    dp = b["persons_total"] - a["persons_total"]
    di = b["ids_minted"] - a["ids_minted"]
    pct = (100.0 * dp / a["persons_total"]) if a["persons_total"] else 0.0

    print(f"\nframes compared            {len(common)}")
    print(f"frames with a count change {count_diff}")
    print(f"person count delta         {dp:+d}  ({pct:+.2f}%)")
    print(f"ByteTrack id delta         {di:+d}")
    print(f"mean IoU of matched boxes  {mean_iou:.4f}")

    # Thresholds are judgement, and stated rather than hidden: a couple of
    # percent of boxes and a couple of ids is fp16 rounding; ten percent is a
    # different detector.
    bad = []
    if abs(pct) > 2.0:
        bad.append(f"person count moved {pct:+.2f}%")
    if abs(di) > max(2, 0.1 * a["ids_minted"]):
        bad.append(f"id churn moved {di:+d}")
    if mean_iou < 0.95:
        bad.append(f"mean IoU {mean_iou:.4f} below 0.95")
    if bad:
        print("\nREVIEW: " + "; ".join(bad))
        return 1
    print("\nPASS: detections and tracking are equivalent")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.05,
                    help="detector confidence floor for this "
                         "comparison (default 0.05, deliberately "
                         "below the product floor)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))

    video = args.video
    if video is None:
        found = sorted(glob.glob("uploads/videos/*.mp4"))
        if not found:
            sys.exit("no video: pass --video")
        video = found[0]

    frames = ([int(x) for x in args.frames.split(",")] if args.frames
              else DEFAULT_FRAMES)
    res = collect(video, frames, conf_floor=args.conf)
    show(res)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
