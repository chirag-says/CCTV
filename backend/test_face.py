#!python3.11
"""
Face accuracy check: SCRFD detection + ArcFace match distances.

The Phase 1 rules say accuracy must not regress and that match distances are
to be compared before and after. The previous version of this file could not
do either. It read
    training/videos/raw/Security _Gate_Inside_*.mp4
which does not exist in this repository, and it printed only bounding boxes
and det_score -- never a match distance, so there was nothing to compare.

This version:
  * takes footage from uploads/videos/*.mp4 by default, overridable
  * samples FIXED frame indices, so two runs see identical pixels
  * runs the pipeline's own FaceDetector and FaceRecognizer rather than a
    separate FaceAnalysis, so it measures what the cameras actually do
  * prints, and writes as JSON, the cosine distance of every detected face
    against the enrolled gallery

Usage:
    python test_face.py                             # print a table
    python test_face.py --out before.json
    python test_face.py --out after.json
    python test_face.py --compare before.json after.json

Distances are the figure to compare. They are a property of the models and
the pixels, so ANY change in them means an accuracy change -- which is
exactly what the fp16 and TensorRT work in Phase 2 has to be held to.
"""

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import cv2
import numpy as np

# Sampled deterministically rather than "a frame with people in it", so the
# comparison is of models and not of which frame happened to decode.
DEFAULT_FRAMES = [0, 60, 125, 200, 300, 420, 560, 700, 820, 900]


def _engines(load_gallery=True):
    from app.vision.detector import FaceDetector
    from app.vision.recognizer import FaceRecognizer

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    gallery = 0
    if load_gallery:
        try:
            from app.services.person_service import PersonService
            encodings = PersonService.get_all_encodings()
            recognizer.load_encodings(encodings)
            gallery = recognizer.known_count
        except Exception as e:
            print(f"  (no gallery: {e})")
    return detector, recognizer, gallery


def _measure(detector, recognizer, gallery, frame):
    """Detect, embed and match every face in one image."""
    locations, embeddings, _landmarks, _ = detector.detect_faces_with_embeddings(frame)
    faces = []
    for i, loc in enumerate(locations):
        top, right, bottom, left = [int(v) for v in loc]
        rec = {
            "bbox": [left, top, right, bottom],
            "w": right - left,
            "h": bottom - top,
            "match": None,
        }
        emb = embeddings[i] if i < len(embeddings) else None
        if emb is not None:
            rec["embedding_norm"] = round(float(np.linalg.norm(emb)), 6)
            if gallery:
                m = recognizer.recognize_face(emb)
                rec["match"] = {
                    "matched": bool(m.matched),
                    "person_name": m.person_name,
                    # The number that must not move.
                    "distance": round(float(m.distance), 6),
                    "confidence": round(float(m.confidence), 6),
                    "margin": round(float(m.margin), 6),
                    "quality_tier": m.quality_tier,
                    "rejection_reason": m.rejection_reason,
                }
        faces.append(rec)
    return faces


def collect_images(pattern, load_gallery=True, pad_frac=0.5):
    """Measure a directory of stills.

    Preferred over video on this deployment: uploads/videos holds a gate clip
    in which SCRFD finds no face at all, so it yields no distances to compare.
    snapshots/faces holds the enrolled people, which is what the gallery is
    made of and so what a match-distance regression should be run against.

    WHY THE PADDING
    These are tight crops -- roughly 1000x700 of almost nothing but face --
    and SCRFD finds zero faces in them as-is. Verified directly: app.get() on
    the raw crop returns 0, and the same crop with a border returns 1. SCRFD
    matches anchors at a scale relative to the input, so a face that fills the
    frame falls outside every anchor it has. The real pipeline never hits this
    because it detects on a whole 720p frame where a face is a small part of
    the image; padding restores that proportion.

    The padding is fixed and deterministic, so it cancels out of any
    before/after comparison.
    """
    detector, recognizer, gallery = _engines(load_gallery)
    paths = sorted(glob.glob(pattern))       # sorted, so runs are comparable
    out = {
        "source": pattern,
        "image_count": len(paths),
        "gallery_encodings": gallery,
        "pad_frac": pad_frac,
        "frames": [],
        "faces_total": 0,
    }
    _stamp_settings(out)

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        if pad_frac:
            h, w = frame.shape[:2]
            py, px = int(h * pad_frac), int(w * pad_frac)
            frame = cv2.copyMakeBorder(
                frame, py, py, px, px, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        faces = _measure(detector, recognizer, gallery, frame)
        out["faces_total"] += len(faces)
        out["frames"].append({"index": os.path.basename(path), "faces": faces})
    return out


def _stamp_settings(out):
    try:
        from app.config import settings
        out["det_size"] = settings.INSIGHTFACE_DET_SIZE
        out["det_thresh"] = settings.INSIGHTFACE_DET_THRESH
        out["fp16"] = getattr(settings, "ENABLE_FP16", None)
    except Exception:
        pass


def collect(video, frame_indices, load_gallery=True):
    detector, recognizer, gallery = _engines(load_gallery)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = {
        "video": video,
        "frame_count": total,
        "gallery_encodings": gallery,
        "frames": [],
        "det_size": None,
        "faces_total": 0,
    }
    _stamp_settings(out)

    for idx in frame_indices:
        if total and idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        faces = _measure(detector, recognizer, gallery, frame)
        out["faces_total"] += len(faces)
        out["frames"].append({"index": idx, "faces": faces})

    cap.release()
    return out


def show(res):
    print(f"\nsource           {res.get('video') or res.get('source')}")
    print(f"images / frames  {res.get('frame_count') or res.get('image_count')}")
    print(f"det_size         {res.get('det_size')}  thresh {res.get('det_thresh')}")
    print(f"gallery          {res['gallery_encodings']} encodings")
    print(f"faces detected   {res['faces_total']}")
    header = ("frame", "face", "w", "h", "distance", "conf", "margin")
    print("\n{:>22s} {:>4s} {:>4s} {:>4s} {:>9s} {:>6s} {:>7s}  name".format(*header))

    def num(value, width, places):
        return ("-" if value is None else f"{value:.{places}f}").rjust(width)

    for fr in res["frames"]:
        for i, f in enumerate(fr["faces"]):
            m = f["match"] or {}
            distance = m.get("distance")
            if distance is not None:
                name = m.get("person_name") or "UNKNOWN"
            else:
                name = "no embedding"
            print(
                f"{str(fr['index'])[-22:]:>22s} {i:4d} {f['w']:4d} {f['h']:4d} "
                f"{num(distance, 9, 6)} "
                f"{num(m.get('confidence'), 6, 3)} "
                f"{num(m.get('margin'), 7, 3)}"
                f"  {name}"
            )


def compare(a_path, b_path):
    """Did any match distance move? That is the whole question."""
    a = json.load(open(a_path))
    b = json.load(open(b_path))

    print(f"before  {a_path}  {a['faces_total']} faces")
    print(f"after   {b_path}  {b['faces_total']} faces")

    if a["faces_total"] != b["faces_total"]:
        print(f"\nFAIL: face count changed, {a['faces_total']} -> {b['faces_total']}")
        return 1

    worst, n, moved = 0.0, 0, []
    for fa, fb in zip(a["frames"], b["frames"]):
        for i, (x, y) in enumerate(zip(fa["faces"], fb["faces"])):
            ma, mb = x.get("match"), y.get("match")
            if not ma or not mb:
                continue
            n += 1
            d = abs(ma["distance"] - mb["distance"])
            worst = max(worst, d)
            if d > 1e-6 or ma["person_name"] != mb["person_name"]:
                moved.append((fa["index"], i, ma, mb))

    print(f"\ncompared {n} matched faces")
    print(f"largest distance change  {worst:.8f}")
    if not moved:
        print("PASS: every match distance and identity is unchanged")
        return 0
    print(f"\n{len(moved)} face(s) changed:")
    for idx, i, ma, mb in moved[:20]:
        print(f"  frame {idx} face {i}: {ma['person_name']} d={ma['distance']:.6f}"
              f"  ->  {mb['person_name']} d={mb['distance']:.6f}")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=None)
    ap.add_argument("--images", default=None,
                    help="glob of stills to measure instead of a video, e.g. "
                         "'snapshots/faces/*.jpg'. Preferred on this "
                         "deployment: the only clip in uploads/videos "
                         "contains no face SCRFD can find, so it yields "
                         "nothing to compare.")
    ap.add_argument("--frames", default=None,
                    help="comma-separated frame indices (default: a fixed spread)")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    ap.add_argument("--no-gallery", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))

    if args.images:
        res = collect_images(args.images, load_gallery=not args.no_gallery)
        show(res)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"\nwrote {args.out}")
        return

    video = args.video
    if video is None:
        found = sorted(glob.glob("uploads/videos/*.mp4"))
        if not found:
            sys.exit("no video: pass --video, or put one in uploads/videos/")
        video = found[0]

    frames = ([int(x) for x in args.frames.split(",")] if args.frames
              else DEFAULT_FRAMES)

    res = collect(video, frames, load_gallery=not args.no_gallery)
    show(res)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
