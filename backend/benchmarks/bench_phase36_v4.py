"""
PHASE 3.6 v4: FULL-FRAME FACE->BODY LINKED VALIDATION
======================================================
Strategy: Run SCRFD on FULL FRAMES (not person crops) to detect faces,
match them against the gallery, then match face bbox to YOLO person bbox
to link identity → body crop.

Key insight: SCRFD works MUCH better on full frames (larger context,
original resolution) than on person crops. The pipeline runs SCRFD on
full frames in production too.

The unknowns directory ONLY has non-registered people (by definition).
We need to find the registered people in full frames.

Flow:
  1. Load ArcFace gallery (8 registered identities)
  2. For each frame: YOLO → person boxes, SCRFD → face boxes + embeddings
  3. Match face boxes to person boxes (IoU/containment)
  4. Match face embeddings to gallery → verified identity
  5. Identity-linked body crop → extract OSNet + ResNet features
  6. Build cross-frame pairs and evaluate

Usage:
    cd backend
    python -m benchmarks.bench_phase36_v4
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

BACKEND_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FRAMES_DIR = BACKEND_DIR / "snapshots" / "frames"
FACES_DIR = BACKEND_DIR / "snapshots" / "faces"
MODELS_DIR = BACKEND_DIR / "models"


def build_osnet():
    import torchreid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1, pretrained=False)
    wp = MODELS_DIR / 'osnet_x1_0_msmt17.pth'
    if wp.exists():
        state = torch.load(str(wp), map_location='cpu', weights_only=True)
        if 'state_dict' in state: state = state['state_dict']
        md = model.state_dict()
        filt = {k: v for k, v in state.items() if k in md and v.shape == md[k].shape}
        md.update(filt)
        model.load_state_dict(md, strict=False)
        print(f"  OSNet MSMT17: {len(filt)}/{len(md)} keys")
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, tf, device


def build_resnet50():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(nn.Sequential(*list(backbone.children())[:-1]), nn.Flatten()).to(device).eval()
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, tf, device


def extract_feat(crop_bgr, model, tf, device):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = tf(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model(t)
        if isinstance(f, (tuple, list)): f = f[0]
    f = f.cpu().numpy().flatten()
    n = np.linalg.norm(f)
    return f / n if n > 0 else f


def build_face_gallery():
    from insightface.app import FaceAnalysis
    print("  Initializing InsightFace...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    gallery = {}
    for fp in sorted(FACES_DIR.glob("*.jpg")):
        parts = fp.stem.split("_")
        name = parts[1].lower() if len(parts) >= 2 else parts[0].lower()
        img = cv2.imread(str(fp))
        if img is None: continue
        h, w = img.shape[:2]
        for pm in [0.5, 1.0, 1.5]:
            ph, pw = int(h*pm), int(w*pm)
            padded = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_CONSTANT, value=(128,128,128))
            faces = app.get(padded)
            if faces: break
        if faces:
            emb = faces[0].embedding
            norm = np.linalg.norm(emb)
            if norm > 0: emb = emb / norm
            gallery.setdefault(name, []).append(emb)

    for name, embs in gallery.items():
        print(f"    {name}: {len(embs)} embeddings")
    return app, gallery


def recognize_face(emb, gallery, threshold=0.40):
    """Lower threshold (0.40) for CCTV conditions — side/partial faces."""
    best_name, best_sim = None, 0.0
    for name, refs in gallery.items():
        for ref in refs:
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_sim = sim
                best_name = name
    return (best_name, best_sim) if best_sim >= threshold else (None, best_sim)


def face_in_person(face_box, person_box):
    """Check if a face bbox is inside/overlaps with a person bbox.
    face_box/person_box: (x1,y1,x2,y2)"""
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    # Face center must be inside person box
    fc_x = (fx1 + fx2) / 2
    fc_y = (fy1 + fy2) / 2
    return px1 <= fc_x <= px2 and py1 <= fc_y <= py2


def main():
    print("=" * 60)
    print("PHASE 3.6 v4: FULL-FRAME FACE->BODY LINKED VALIDATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    face_app, face_gallery = build_face_gallery()

    print("\n  Loading OSNet MSMT17...")
    osnet_model, osnet_tf, device = build_osnet()
    print("  Loading ResNet50 ImageNet...")
    resnet_model, resnet_tf, _ = build_resnet50()

    print("  Loading YOLO11m...")
    from ultralytics import YOLO
    yolo = YOLO("yolo11m.pt")

    # Process ALL available frames
    frame_files = sorted(list(FRAMES_DIR.glob("fullframe_*.jpg")))
    print(f"\n  Total frames available: {len(frame_files)}")
    # Process all of them — we need maximum coverage
    # But cap at 10000 for practical runtime
    rng = np.random.RandomState(42)
    if len(frame_files) > 10000:
        indices = rng.choice(len(frame_files), 10000, replace=False)
        frame_files = [frame_files[i] for i in sorted(indices)]
    print(f"  Processing: {len(frame_files)} frames")

    verified = []  # {identity, frame_name, crop, osnet_feat, resnet_feat, height, face_sim}
    unverified = []
    face_detected_count = 0
    face_recognized_count = 0

    for fi, fpath in enumerate(frame_files):
        if fi % 500 == 0:
            print(f"    Frame {fi}/{len(frame_files)} | verified={len(verified)} | "
                  f"faces_detected={face_detected_count} | faces_recognized={face_recognized_count}")

        try:
            frame = cv2.imread(str(fpath))
        except Exception:
            continue
        if frame is None:
            continue

        # Step 1: YOLO person detection
        det = yolo.predict(frame, imgsz=640, classes=[0], conf=0.4, verbose=False)
        person_boxes = []
        for b in det[0].boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            h, w = y2 - y1, x2 - x1
            if h >= 60 and w >= 25:
                person_boxes.append((x1, y1, x2, y2))

        if not person_boxes:
            continue

        # Step 2: SCRFD face detection on FULL FRAME (not individual crops!)
        faces = face_app.get(frame)
        face_detected_count += len(faces) if faces else 0

        # Step 3: Match faces to person boxes, recognize
        face_identities = {}  # person_box_index -> (identity, face_sim)
        if faces:
            for face in faces:
                if face.embedding is None:
                    continue
                emb = face.embedding
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                identity, face_sim = recognize_face(emb, face_gallery)
                if not identity:
                    continue

                face_recognized_count += 1

                # Which person box contains this face?
                face_bbox = face.bbox.astype(int)  # x1, y1, x2, y2
                for pi, pbox in enumerate(person_boxes):
                    if face_in_person(face_bbox, pbox):
                        # If multiple faces match same person box, keep highest sim
                        if pi not in face_identities or face_sim > face_identities[pi][1]:
                            face_identities[pi] = (identity, face_sim)
                        break

        # Step 4: Extract body features for all person boxes
        for pi, pbox in enumerate(person_boxes):
            x1, y1, x2, y2 = pbox
            crop = frame[y1:y2, x1:x2].copy()
            h, w = crop.shape[:2]

            osnet_feat = extract_feat(crop, osnet_model, osnet_tf, device)
            resnet_feat = extract_feat(crop, resnet_model, resnet_tf, device)

            entry = {
                "identity": face_identities.get(pi, (None, 0))[0],
                "face_sim": face_identities.get(pi, (None, 0))[1],
                "frame_name": fpath.name,
                "frame_idx": fi,
                "height_px": h, "width_px": w,
                "crop": crop,
                "osnet_feat": osnet_feat,
                "resnet_feat": resnet_feat,
            }

            if entry["identity"]:
                verified.append(entry)
            else:
                if len(unverified) < 2000:
                    unverified.append(entry)

    del yolo
    torch.cuda.empty_cache()

    print(f"\n  RESULTS:")
    print(f"  Total faces detected: {face_detected_count}")
    print(f"  Faces recognized: {face_recognized_count}")
    print(f"  Face-verified body crops: {len(verified)}")
    print(f"  Unverified body crops: {len(unverified)}")

    id_counts = defaultdict(int)
    for v in verified:
        id_counts[v["identity"]] += 1
    for name, cnt in sorted(id_counts.items(), key=lambda x: -x[1]):
        print(f"    {name}: {cnt} crops across frames")

    if len(verified) < 4:
        print("\n  ERROR: Too few face-verified crops.")
        results = {"error": "too_few_verified", "verified": len(verified),
                   "faces_detected": face_detected_count, "faces_recognized": face_recognized_count,
                   "identities": dict(id_counts)}
        with open(RESULTS_DIR / "phase36_v4.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # ── Build pairs ──
    print("\n" + "=" * 60)
    print("BUILDING FACE-VERIFIED PAIRS")
    print("=" * 60)

    identity_groups = defaultdict(list)
    for i, v in enumerate(verified):
        identity_groups[v["identity"]].append(i)

    # Positive: same identity, different frames (cap pairs per identity)
    positive_pairs = []
    for identity, indices in identity_groups.items():
        # Only cross-frame pairs
        frame_groups = defaultdict(list)
        for idx in indices:
            frame_groups[verified[idx]["frame_name"]].append(idx)
        frames_list = list(frame_groups.keys())
        for fi in range(len(frames_list)):
            for fj in range(fi+1, len(frames_list)):
                for ia in frame_groups[frames_list[fi]][:3]:  # limit per frame
                    for ib in frame_groups[frames_list[fj]][:3]:
                        osnet_sim = float(np.dot(verified[ia]["osnet_feat"], verified[ib]["osnet_feat"]))
                        resnet_sim = float(np.dot(verified[ia]["resnet_feat"], verified[ib]["resnet_feat"]))
                        positive_pairs.append({
                            "idx_a": ia, "idx_b": ib, "identity": identity,
                            "osnet_sim": round(osnet_sim, 4), "resnet_sim": round(resnet_sim, 4),
                            "height_a": verified[ia]["height_px"], "height_b": verified[ib]["height_px"],
                            "face_sim_a": verified[ia]["face_sim"], "face_sim_b": verified[ib]["face_sim"],
                            "frame_a": verified[ia]["frame_name"], "frame_b": verified[ib]["frame_name"],
                        })
        if len(positive_pairs) > 5000:
            break  # cap total pairs

    # Negative: different identities
    negative_pairs = []
    identities = list(identity_groups.keys())
    rng2 = np.random.RandomState(42)
    for i in range(len(identities)):
        for j in range(i+1, len(identities)):
            ia_list = identity_groups[identities[i]]
            ib_list = identity_groups[identities[j]]
            n_sample = min(30, len(ia_list) * len(ib_list))
            for _ in range(n_sample):
                ia = rng2.choice(ia_list)
                ib = rng2.choice(ib_list)
                osnet_sim = float(np.dot(verified[ia]["osnet_feat"], verified[ib]["osnet_feat"]))
                resnet_sim = float(np.dot(verified[ia]["resnet_feat"], verified[ib]["resnet_feat"]))
                negative_pairs.append({
                    "idx_a": ia, "idx_b": ib,
                    "identity_a": identities[i], "identity_b": identities[j],
                    "osnet_sim": round(osnet_sim, 4), "resnet_sim": round(resnet_sim, 4),
                    "height_a": verified[ia]["height_px"], "height_b": verified[ib]["height_px"],
                })

    print(f"  Positive (face-verified, cross-frame): {len(positive_pairs)}")
    print(f"  Negative (face-verified, cross-identity): {len(negative_pairs)}")

    if not positive_pairs:
        print("  No cross-frame positive pairs found!")
        results = {"error": "no_crossframe_positives", "verified": len(verified),
                   "identities": dict(id_counts)}
        with open(RESULTS_DIR / "phase36_v4.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    # ── Threshold sweep ──
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP (FACE-VERIFIED)")
    print("=" * 60)

    pos_o = np.array([p["osnet_sim"] for p in positive_pairs])
    neg_o = np.array([p["osnet_sim"] for p in negative_pairs])
    pos_r = np.array([p["resnet_sim"] for p in positive_pairs])
    neg_r = np.array([p["resnet_sim"] for p in negative_pairs])

    thresholds = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.65, 0.68, 0.70]

    def sweep(pos, neg, name):
        rows = []
        print(f"\n  {name} | pos={len(pos)} neg={len(neg)}")
        print(f"  {'Thresh':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FAR':>8}")
        for t in thresholds:
            tp=int(np.sum(pos>=t)); fn=int(np.sum(pos<t))
            fp=int(np.sum(neg>=t)); tn=int(np.sum(neg<t))
            pr=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
            f1=2*pr*rec/(pr+rec) if pr+rec else 0; far=fp/(fp+tn) if fp+tn else 0
            rows.append({"threshold":t,"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":round(pr,4),
                         "recall":round(rec,4),"f1":round(f1,4),"far":round(far,4)})
            print(f"  {t:8.2f} {tp:6d} {fp:6d} {fn:6d} {tn:6d} {pr:8.4f} {rec:8.4f} {f1:8.4f} {far:8.4f}")
        return rows

    osnet_sweep = sweep(pos_o, neg_o, "OSNet MSMT17")
    resnet_sweep = sweep(pos_r, neg_r, "ResNet50 ImageNet")

    # ── Head-to-head ──
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD (FACE-VERIFIED)")
    print("=" * 60)

    comp = {}
    for name, pos, neg in [("osnet", pos_o, neg_o), ("resnet", pos_r, neg_r)]:
        comp[name] = {
            "same_mean": round(float(np.mean(pos)), 4),
            "same_std": round(float(np.std(pos)), 4),
            "same_min": round(float(np.min(pos)), 4),
            "same_p5": round(float(np.percentile(pos, 5)), 4),
            "same_median": round(float(np.median(pos)), 4),
            "diff_mean": round(float(np.mean(neg)), 4) if len(neg) else None,
            "diff_p95": round(float(np.percentile(neg, 95)), 4) if len(neg) else None,
            "diff_max": round(float(np.max(neg)), 4) if len(neg) else None,
            "separation_gap": round(float(np.mean(pos)) - float(np.percentile(neg, 95)), 4) if len(neg) else None,
        }

    print(f"\n  {'Metric':<25} {'ResNet50':>12} {'OSNet':>12}")
    print(f"  {'-'*50}")
    for k in ["same_mean","same_min","same_p5","diff_mean","diff_p95","diff_max","separation_gap"]:
        print(f"  {k:<25} {comp['resnet'].get(k,'N/A'):>12} {comp['osnet'].get(k,'N/A'):>12}")

    # ── Query-level test ──
    print("\n" + "=" * 60)
    print("QUERY-LEVEL RANK-1 TEST")
    print("=" * 60)

    qr = {}
    for gs in [10, 25, 50, 100]:
        gidx, gids = [], []
        for ident, indices in identity_groups.items():
            gidx.append(("v", indices[0])); gids.append(ident)
        nfill = gs - len(gidx)
        for fi in range(min(nfill, len(unverified))):
            gidx.append(("u", fi)); gids.append(f"unk_{fi}")

        for mn, fk in [("osnet","osnet_feat"), ("resnet","resnet_feat")]:
            gf = np.array([verified[i]["osnet_feat" if mn=="osnet" else "resnet_feat"] if s=="v"
                           else unverified[i]["osnet_feat" if mn=="osnet" else "resnet_feat"]
                           for s,i in gidx])
            correct=incorrect=nomatch=total=0
            for ident, indices in identity_groups.items():
                for qi in indices[1:]:
                    qf = verified[qi][fk]
                    sims = np.dot(gf, qf)
                    si = np.argsort(-sims)
                    t1s = float(sims[si[0]])
                    t1id = gids[si[0]]
                    if t1s < 0.60: nomatch += 1
                    elif t1id == ident: correct += 1
                    else: incorrect += 1
                    total += 1
            key = f"{mn}_gallery_{len(gidx)}"
            qr[key] = {"total": total, "correct": correct, "incorrect": incorrect, "nomatch": nomatch,
                        "rank1": round(correct/total,4) if total else 0,
                        "false_id": round(incorrect/total,4) if total else 0,
                        "false_rej": round(nomatch/total,4) if total else 0}

    print(f"\n  {'Config':<25} {'Rank-1':>8} {'FalseID':>8} {'FalseRej':>8} {'N':>6}")
    for k,v in sorted(qr.items()):
        print(f"  {k:<25} {v['rank1']:8.4f} {v['false_id']:8.4f} {v['false_rej']:8.4f} {v['total']:6d}")

    # ── Difficult cases ──
    difficult = []
    for p in negative_pairs:
        if p["osnet_sim"] >= 0.50:
            difficult.append({"type": "FALSE_MATCH", "osnet_sim": p["osnet_sim"], "resnet_sim": p["resnet_sim"],
                              "id_a": p["identity_a"], "id_b": p["identity_b"],
                              "h_a": p["height_a"], "h_b": p["height_b"]})
    for p in positive_pairs:
        if p["osnet_sim"] < 0.45:
            difficult.append({"type": "FALSE_REJECT", "osnet_sim": p["osnet_sim"], "resnet_sim": p["resnet_sim"],
                              "identity": p["identity"], "h_a": p["height_a"], "h_b": p["height_b"]})

    print(f"\n  Difficult: {len([d for d in difficult if d['type']=='FALSE_MATCH'])} false match risks, "
          f"{len([d for d in difficult if d['type']=='FALSE_REJECT'])} false reject risks")

    # ── Save visual samples ──
    gt_dir = RESULTS_DIR / "phase36_ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(positive_pairs[:30]):
        a, b = verified[p["idx_a"]]["crop"], verified[p["idx_b"]]["crop"]
        th = 256
        ar = cv2.resize(a, (max(1, int(a.shape[1]*th/a.shape[0])), th))
        br = cv2.resize(b, (max(1, int(b.shape[1]*th/b.shape[0])), th))
        combined = np.hstack([ar, np.ones((th,5,3),dtype=np.uint8)*255, br])
        cv2.imwrite(str(gt_dir / f"pos_{i:03d}_{p['identity']}_o{p['osnet_sim']:.3f}.jpg"), combined)
    for i, p in enumerate(negative_pairs[:30]):
        a, b = verified[p["idx_a"]]["crop"], verified[p["idx_b"]]["crop"]
        th = 256
        ar = cv2.resize(a, (max(1, int(a.shape[1]*th/a.shape[0])), th))
        br = cv2.resize(b, (max(1, int(b.shape[1]*th/b.shape[0])), th))
        combined = np.hstack([ar, np.ones((th,5,3),dtype=np.uint8)*255, br])
        cv2.imwrite(str(gt_dir / f"neg_{i:03d}_{p['identity_a']}_vs_{p['identity_b']}_o{p['osnet_sim']:.3f}.jpg"), combined)

    # ── Save results ──
    results = {
        "metadata": {"timestamp": datetime.now().isoformat(), "version": "v4_fullframe"},
        "ground_truth": {
            "method": "SCRFD on full frames → face-person bbox matching → ArcFace recognition → body features",
            "frames_processed": len(frame_files),
            "faces_detected": face_detected_count,
            "faces_recognized": face_recognized_count,
            "verified_crops": len(verified),
            "unverified_crops": len(unverified),
            "identities": dict(id_counts),
            "positive_pairs": len(positive_pairs),
            "negative_pairs": len(negative_pairs),
        },
        "comparison": comp,
        "threshold_sweep_osnet": osnet_sweep,
        "threshold_sweep_resnet": resnet_sweep,
        "query_level": qr,
        "difficult_cases": difficult[:50],
        "distributions": {
            "osnet_pos": {"mean": round(float(np.mean(pos_o)),4), "std": round(float(np.std(pos_o)),4),
                          "min": round(float(np.min(pos_o)),4), "p5": round(float(np.percentile(pos_o,5)),4),
                          "median": round(float(np.median(pos_o)),4)},
            "osnet_neg": {"mean": round(float(np.mean(neg_o)),4), "p95": round(float(np.percentile(neg_o,95)),4),
                          "max": round(float(np.max(neg_o)),4)} if len(neg_o) else {},
        },
    }
    out = RESULTS_DIR / "phase36_v4.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out}")

    print("\n" + "=" * 60)
    print("PHASE 3.6 v4 COMPLETE")
    print(f"Ended: {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
