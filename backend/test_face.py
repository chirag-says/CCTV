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


def collect(video, frame_indices, load_gallery=True):
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
    try:
        from app.config import settings
        out["det_size"] = settings.INSIGHTFACE_DET_SIZE
        out["det_thresh"] = settings.INSIGHTFACE_DET_THRESH
        out["fp16"] = getattr(settings, "ENABLE_FP16", None)
    except Exception:
        pass

    for idx in frame_indices:
        if total and idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        locations, embeddings, landmarks, _ = detector.detect_faces_with_embeddings(frame)
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

        out["faces_total"] += len(faces)
        out["frames"].append({"index": idx, "faces": faces})

    cap.release()
    return out


def show(res):
    print(f"\nvideo            {res['video']}")
    print(f"frames in file   {res['frame_count']}")
    print(f"det_size         {res.get('det_size')}  thresh {res.get('det_thresh')}")
    print(f"gallery          {res['gallery_encodings']} encodings")
    print(f"faces detected   {res['faces_total']}")
    header = ("frame", "face", "w", "h", "distance", "conf", "margin")
    print("\n{:>6s} {:>4s} {:>4s} {:>4s} {:>9s} {:>6s} {:>7s}  name".format(*header))

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
                f"{fr['index']:6d} {i:4d} {f['w']:4d} {f['h']:4d} "
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
    ap.add_argument("--frames", default=None,
                    help="comma-separated frame indices (default: a fixed spread)")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    ap.add_argument("--no-gallery", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))

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
