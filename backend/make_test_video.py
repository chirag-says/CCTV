"""
Build a representative 720p test clip for load_test.py.

WHY THIS EXISTS
The only clip in uploads/videos shows a gate from far away. Its people score
0.08 to 0.44 on YOLOv8n, so almost none clear the 0.35 product floor, and
SCRFD finds no face in it at all. Load tests on that footage measure the
detector call and the frame plumbing while ByteTrack, the identity ledger,
the recognizer and the drawing path all sit idle -- which is most of what
Phase 2 items 3 to 6 change. The Plus acceptance ("15 Plus cameras at 10 fps
tracking with faces at 5 fps") cannot be measured on footage with no faces.

snapshots/frames holds 14,524 real full frames captured from these cameras,
with 5 to 11 people and 1 to 4 faces each. They are SAMPLED, though, not
recorded: the longest run of consecutive frames less than 250 ms apart is 11,
so they cannot simply be concatenated into video. Played back to back they
would be a slideshow, and ByteTrack would mint a fresh id on every frame --
which would overstate the tracking cost as badly as the current clip
understates it.

WHAT THIS BUILDS
A crop window panned slowly across each real full frame, held for `--hold`
seconds before cutting to the next frame. Within a hold the content is real
and the motion is smooth and sub-pixel, so detection, tracking and
recognition all behave as they would on video. Between holds there is a cut,
which is a scene change the tracker is entitled to churn on -- real cameras
have those too.

This is SYNTHETIC MOTION over REAL CONTENT, and every number measured on it
should say so.

    python make_test_video.py --out uploads/videos/synth_people_720p.mp4
"""

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np


def pick_frames(pattern, want, min_persons, seed):
    """Choose full frames that actually contain people.

    Screened with the same detector the pipeline uses, because "has people in
    it" is exactly the property the current clip fails on, and eyeballing a
    directory of 14,524 files is not a screen.
    """
    from ultralytics import YOLO

    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"no frames matched {pattern}")
    random.Random(seed).shuffle(paths)

    model = YOLO("yolov8n.pt")
    chosen = []
    for path in paths:
        if len(chosen) >= want:
            break
        img = cv2.imread(path)
        if img is None or img.shape[0] < 720:
            continue
        res = model.predict(img, conf=0.25, classes=[0], verbose=False, device=0)[0]
        n = 0 if res.boxes is None else len(res.boxes)
        if n >= min_persons:
            chosen.append((path, n))
    return chosen


def pan(img, out_w, out_h, n_frames, rng):
    """Yield n_frames crops panning across img, scaled to out_w x out_h.

    The crop is the largest window of the target aspect ratio that fits, shrunk
    by a margin so there is room to travel. Motion is a few pixels per frame:
    enough for ByteTrack to have something to associate, small enough that a
    person does not teleport.
    """
    h, w = img.shape[:2]
    aspect = out_w / out_h
    # Largest window of the right shape, minus a margin to pan within.
    cw = int(min(w, h * aspect) * 0.82)
    ch = int(cw / aspect)
    cw, ch = min(cw, w), min(ch, h)

    max_dx, max_dy = w - cw, h - ch
    # A direction and speed per clip, so different frames pan differently.
    ang = rng.uniform(0, 2 * np.pi)
    dx_total = np.cos(ang) * max_dx
    dy_total = np.sin(ang) * max_dy
    x0 = (max_dx - dx_total) / 2.0
    y0 = (max_dy - dy_total) / 2.0

    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        x = int(round(np.clip(x0 + dx_total * t, 0, max_dx)))
        y = int(round(np.clip(y0 + dy_total * t, 0, max_dy)))
        crop = img[y:y + ch, x:x + cw]
        yield cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-glob", default="snapshots/frames/fullframe_*.jpg")
    ap.add_argument("--out", default="uploads/videos/synth_people_720p.mp4")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--hold", type=float, default=1.6,
                    help="seconds of pan per source frame before cutting")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-persons", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    per_clip = max(2, int(round(args.hold * args.fps)))
    want = int(np.ceil(args.seconds * args.fps / per_clip))
    print(f"target {args.seconds:.0f}s at {args.fps:g} fps = "
          f"{int(args.seconds * args.fps)} frames, {per_clip} per source, "
          f"{want} source frames needed")

    chosen = pick_frames(args.frames_glob, want, args.min_persons, args.seed)
    if not chosen:
        sys.exit("no frames with enough people found")
    print(f"selected {len(chosen)} frames "
          f"({sum(n for _, n in chosen)} person detections total)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
    if not writer.isOpened():
        sys.exit(f"cannot open writer for {args.out}")

    rng = random.Random(args.seed)
    written = 0
    for path, _n in chosen:
        img = cv2.imread(path)
        if img is None:
            continue
        for frame in pan(img, args.width, args.height, per_clip, rng):
            writer.write(frame)
            written += 1
    writer.release()

    size_mb = os.path.getsize(args.out) / 2**20
    print(f"wrote {args.out}: {written} frames, "
          f"{written / args.fps:.1f}s, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
