"""
PHASE 3.6 v3: FACE-ANCHORED BODY RE-ID VALIDATION
===================================================
Correct approach: Use unknown_UUID face crops → ArcFace recognition → 
link to context_UUID body crops via UUID.

This avoids running SCRFD on body crops (which mostly don't have visible faces)
and instead uses the already-extracted face crops from the live system.

Strategy:
  1. Load ArcFace gallery from registered face images
  2. Load unknown_UUID.jpg face crops → pad → SCRFD+ArcFace → recognize
  3. If recognized → link to context_UUID.jpg body crop via same UUID
  4. Extract OSNet + ResNet50 body features from context crops
  5. Build face-verified positive/negative body pairs

Usage:
    cd backend
    python -m benchmarks.bench_phase36_v3
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
UNKNOWNS_DIR = BACKEND_DIR / "snapshots" / "unknowns"
FACES_DIR = BACKEND_DIR / "snapshots" / "faces"
MODELS_DIR = BACKEND_DIR / "models"


# =================================================================
# MODEL BUILDERS
# =================================================================

def build_osnet():
    import torchreid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1, pretrained=False)
    wp = MODELS_DIR / 'osnet_x1_0_msmt17.pth'
    if wp.exists():
        state = torch.load(str(wp), map_location='cpu', weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
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
        if isinstance(f, (tuple, list)):
            f = f[0]
    f = f.cpu().numpy().flatten()
    n = np.linalg.norm(f)
    return f / n if n > 0 else f


# =================================================================
# FACE GALLERY
# =================================================================

def build_face_gallery():
    from insightface.app import FaceAnalysis
    print("  Initializing InsightFace (SCRFD + ArcFace)...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    gallery = {}
    face_files = sorted(FACES_DIR.glob("*.jpg"))
    print(f"  Loading {len(face_files)} registered face images...")

    for fp in face_files:
        parts = fp.stem.split("_")
        name = parts[1].lower() if len(parts) >= 2 else parts[0].lower()
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # Pad tight crops for SCRFD detection
        h, w = img.shape[:2]
        for pad_mult in [0.5, 1.0, 1.5]:
            pad_h, pad_w = int(h * pad_mult), int(w * pad_mult)
            padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                         cv2.BORDER_CONSTANT, value=(128, 128, 128))
            faces = app.get(padded)
            if faces:
                break

        if faces:
            emb = faces[0].embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            gallery.setdefault(name, []).append(emb)
        else:
            print(f"    WARNING: No face in {fp.name}")

    for name, embs in gallery.items():
        print(f"    {name}: {len(embs)} embeddings")
    return app, gallery


def recognize_face(emb, gallery, threshold=0.45):
    best_name, best_sim = None, 0.0
    for name, refs in gallery.items():
        for ref in refs:
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_sim = sim
                best_name = name
    return (best_name, best_sim) if best_sim >= threshold else (None, best_sim)


# =================================================================
# MAIN: UUID-LINKED FACE→BODY GROUND TRUTH
# =================================================================

def main():
    print("=" * 60)
    print("PHASE 3.6 v3: FACE-ANCHORED BODY RE-ID VALIDATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    face_app, face_gallery = build_face_gallery()

    print("\n  Loading OSNet MSMT17...")
    osnet_model, osnet_tf, device = build_osnet()
    print("  Loading ResNet50 ImageNet...")
    resnet_model, resnet_tf, _ = build_resnet50()

    # ── Step 1: Find all UUID-linked face/context pairs ──
    print("\n  Scanning UUID-linked face/context pairs...")
    face_files = sorted(UNKNOWNS_DIR.glob("unknown_*.jpg"))
    
    # Build UUID → file mapping
    uuid_pairs = {}  # uuid -> {face_path, context_path}
    for fp in face_files:
        # unknown_UUID.jpg → extract UUID
        uuid = fp.stem.replace("unknown_", "")
        context_path = UNKNOWNS_DIR / f"context_{uuid}.jpg"
        if context_path.exists():
            uuid_pairs[uuid] = {"face": fp, "context": context_path}

    print(f"  Found {len(uuid_pairs)} UUID-linked face/context pairs")

    # Sample up to 8000 for processing speed
    rng = np.random.RandomState(42)
    uuids = sorted(uuid_pairs.keys())
    if len(uuids) > 8000:
        indices = rng.choice(len(uuids), 8000, replace=False)
        uuids = [uuids[i] for i in sorted(indices)]

    # ── Step 2: Recognize faces and build verified body crops ──
    print(f"\n  Processing {len(uuids)} face/context pairs...")
    verified = []  # {identity, uuid, face_sim, context_crop, osnet_feat, resnet_feat, height}
    unverified = []  # body crops without identity match

    for i, uuid in enumerate(uuids):
        if i % 1000 == 0:
            print(f"    {i}/{len(uuids)} | verified={len(verified)} | unverified={len(unverified)}")

        pair = uuid_pairs[uuid]

        # Read face crop and recognize
        face_img = cv2.imread(str(pair["face"]))
        if face_img is None:
            continue

        # Pad face crop for SCRFD detection (same as gallery)
        h, w = face_img.shape[:2]
        faces = None
        for pad_mult in [0.5, 1.0, 1.5]:
            pad_h, pad_w = int(h * pad_mult), int(w * pad_mult)
            padded = cv2.copyMakeBorder(face_img, pad_h, pad_h, pad_w, pad_w,
                                         cv2.BORDER_CONSTANT, value=(128, 128, 128))
            faces = face_app.get(padded)
            if faces:
                break

        identity = None
        face_sim = 0.0
        if faces:
            best_face = max(faces, key=lambda f: f.det_score)
            if best_face.embedding is not None:
                emb = best_face.embedding
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                identity, face_sim = recognize_face(emb, face_gallery, threshold=0.45)

        # Read context (body) crop and extract features
        context_img = cv2.imread(str(pair["context"]))
        if context_img is None:
            continue

        ch, cw = context_img.shape[:2]
        if ch < 60 or cw < 25:
            continue

        osnet_feat = extract_feat(context_img, osnet_model, osnet_tf, device)
        resnet_feat = extract_feat(context_img, resnet_model, resnet_tf, device)

        entry = {
            "identity": identity,
            "uuid": uuid,
            "face_sim": round(face_sim, 4),
            "height_px": ch,
            "width_px": cw,
            "crop": context_img,
            "osnet_feat": osnet_feat,
            "resnet_feat": resnet_feat,
        }

        if identity:
            verified.append(entry)
        else:
            if len(unverified) < 2000:
                unverified.append(entry)

    print(f"\n  RESULTS:")
    print(f"  Face-verified body crops: {len(verified)}")
    print(f"  Unverified body crops: {len(unverified)}")

    id_counts = defaultdict(int)
    for v in verified:
        id_counts[v["identity"]] += 1
    for name, cnt in sorted(id_counts.items(), key=lambda x: -x[1]):
        print(f"    {name}: {cnt} crops")

    if len(verified) < 4:
        print("\n  ERROR: Too few face-verified crops for meaningful analysis.")
        print("  This footage doesn't contain enough recognizable registered people.")
        # Save minimal results
        res = {"error": "too_few_verified", "verified": len(verified), "unverified": len(unverified),
               "identities": dict(id_counts)}
        with open(RESULTS_DIR / "phase36_v3.json", "w") as f:
            json.dump(res, f, indent=2)
        return

    # ── Step 3: Build verified pairs ──
    print("\n" + "=" * 60)
    print("BUILDING FACE-VERIFIED PAIRS")
    print("=" * 60)

    identity_groups = defaultdict(list)
    for i, v in enumerate(verified):
        identity_groups[v["identity"]].append(i)

    # Positive pairs: same identity, different UUIDs
    positive_pairs = []
    for identity, indices in identity_groups.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                ia, ib = indices[i], indices[j]
                osnet_sim = float(np.dot(verified[ia]["osnet_feat"], verified[ib]["osnet_feat"]))
                resnet_sim = float(np.dot(verified[ia]["resnet_feat"], verified[ib]["resnet_feat"]))
                positive_pairs.append({
                    "idx_a": ia, "idx_b": ib, "identity": identity,
                    "osnet_sim": round(osnet_sim, 4), "resnet_sim": round(resnet_sim, 4),
                    "height_a": verified[ia]["height_px"], "height_b": verified[ib]["height_px"],
                    "face_sim_a": verified[ia]["face_sim"], "face_sim_b": verified[ib]["face_sim"],
                })

    # Negative pairs: different identities
    negative_pairs = []
    identities = list(identity_groups.keys())
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            id_a_idx = identity_groups[identities[i]]
            id_b_idx = identity_groups[identities[j]]
            n_sample = min(20, len(id_a_idx) * len(id_b_idx))
            rng2 = np.random.RandomState(42)
            for _ in range(n_sample):
                ia = rng2.choice(id_a_idx)
                ib = rng2.choice(id_b_idx)
                osnet_sim = float(np.dot(verified[ia]["osnet_feat"], verified[ib]["osnet_feat"]))
                resnet_sim = float(np.dot(verified[ia]["resnet_feat"], verified[ib]["resnet_feat"]))
                negative_pairs.append({
                    "idx_a": ia, "idx_b": ib,
                    "identity_a": identities[i], "identity_b": identities[j],
                    "osnet_sim": round(osnet_sim, 4), "resnet_sim": round(resnet_sim, 4),
                    "height_a": verified[ia]["height_px"], "height_b": verified[ib]["height_px"],
                })

    print(f"  Positive pairs (same person, face-verified): {len(positive_pairs)}")
    print(f"  Negative pairs (diff person, face-verified): {len(negative_pairs)}")

    if len(positive_pairs) == 0:
        print("  WARNING: No positive pairs — only 1 identity has multiple crops.")
        res = {"error": "no_positive_pairs", "verified": len(verified),
               "identities": dict(id_counts)}
        with open(RESULTS_DIR / "phase36_v3.json", "w") as f:
            json.dump(res, f, indent=2)
        return

    # ── Step 4: Threshold sweep ──
    print("\n" + "=" * 60)
    print("PART 4: THRESHOLD SWEEP (FACE-VERIFIED PAIRS)")
    print("=" * 60)

    pos_o = np.array([p["osnet_sim"] for p in positive_pairs])
    neg_o = np.array([p["osnet_sim"] for p in negative_pairs])
    pos_r = np.array([p["resnet_sim"] for p in positive_pairs])
    neg_r = np.array([p["resnet_sim"] for p in negative_pairs])

    thresholds = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.65, 0.68, 0.70]

    def sweep(pos, neg, name):
        rows = []
        print(f"\n  {name}:")
        print(f"  Positive: {len(pos)} | Negative: {len(neg)}")
        print(f"  {'Thresh':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FAR':>8} {'FRR':>8}")
        print(f"  {'-'*80}")
        for t in thresholds:
            tp = int(np.sum(pos >= t)); fn = int(np.sum(pos < t))
            fp = int(np.sum(neg >= t)); tn = int(np.sum(neg < t))
            pr = tp/(tp+fp) if (tp+fp) else 0; rec = tp/(tp+fn) if (tp+fn) else 0
            f1 = 2*pr*rec/(pr+rec) if (pr+rec) else 0
            far = fp/(fp+tn) if (fp+tn) else 0; frr = fn/(tp+fn) if (tp+fn) else 0
            rows.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "precision": round(pr,4), "recall": round(rec,4), "f1": round(f1,4),
                         "far": round(far,4), "frr": round(frr,4)})
            print(f"  {t:8.2f} {tp:6d} {fp:6d} {fn:6d} {tn:6d} {pr:8.4f} {rec:8.4f} {f1:8.4f} {far:8.4f} {frr:8.4f}")
        return rows

    osnet_sweep = sweep(pos_o, neg_o, "OSNet MSMT17")
    resnet_sweep = sweep(pos_r, neg_r, "ResNet50 ImageNet")

    # ── Step 5: Head-to-head comparison ──
    print("\n" + "=" * 60)
    print("PART 5: HEAD-TO-HEAD COMPARISON (FACE-VERIFIED)")
    print("=" * 60)

    comp = {}
    for name, pos, neg in [("osnet_msmt17", pos_o, neg_o), ("resnet50_imagenet", pos_r, neg_r)]:
        comp[name] = {
            "same_person_mean": round(float(np.mean(pos)), 4),
            "same_person_std": round(float(np.std(pos)), 4),
            "same_person_min": round(float(np.min(pos)), 4),
            "same_person_p5": round(float(np.percentile(pos, 5)), 4),
            "same_person_p25": round(float(np.percentile(pos, 25)), 4),
            "same_person_median": round(float(np.median(pos)), 4),
            "diff_person_mean": round(float(np.mean(neg)), 4) if len(neg) else None,
            "diff_person_p95": round(float(np.percentile(neg, 95)), 4) if len(neg) else None,
            "diff_person_max": round(float(np.max(neg)), 4) if len(neg) else None,
            "separation_gap": round(float(np.mean(pos)) - float(np.percentile(neg, 95)), 4) if len(neg) else None,
        }

    print(f"\n  {'Metric':<30} {'ResNet50':>15} {'OSNet MSMT17':>15}")
    print(f"  {'-'*60}")
    for key in ["same_person_mean", "same_person_min", "same_person_p5",
                 "diff_person_mean", "diff_person_p95", "diff_person_max", "separation_gap"]:
        r = comp.get("resnet50_imagenet", {}).get(key, "N/A")
        o = comp.get("osnet_msmt17", {}).get(key, "N/A")
        print(f"  {key:<30} {str(r):>15} {str(o):>15}")

    # ── Step 6: Query-level Rank-1 test ──
    print("\n" + "=" * 60)
    print("PART 3: QUERY-LEVEL RANK-1 TEST")
    print("=" * 60)

    gallery_sizes = [10, 25, 50, 100]
    query_results = {}

    for gs in gallery_sizes:
        # Gallery: one crop per known identity + unverified fillers
        gallery_idx = []
        gallery_ids = []
        for identity, indices in identity_groups.items():
            gallery_idx.append(("verified", indices[0]))
            gallery_ids.append(identity)

        n_fill = gs - len(gallery_idx)
        for fi in range(min(n_fill, len(unverified))):
            gallery_idx.append(("unverified", fi))
            gallery_ids.append(f"unknown_{fi}")

        # Build gallery feature matrices
        g_osnet = []
        g_resnet = []
        for src, idx in gallery_idx:
            if src == "verified":
                g_osnet.append(verified[idx]["osnet_feat"])
                g_resnet.append(verified[idx]["resnet_feat"])
            else:
                g_osnet.append(unverified[idx]["osnet_feat"])
                g_resnet.append(unverified[idx]["resnet_feat"])
        g_osnet = np.array(g_osnet)
        g_resnet = np.array(g_resnet)
        actual_gs = len(gallery_idx)

        for model_name, g_feats, feat_key in [("osnet", g_osnet, "osnet_feat"),
                                                ("resnet", g_resnet, "resnet_feat")]:
            correct = incorrect = no_match = total = 0
            margins = []

            for identity, indices in identity_groups.items():
                for qi in indices[1:]:  # skip gallery entry
                    q_feat = verified[qi][feat_key]
                    sims = np.dot(g_feats, q_feat)
                    sorted_i = np.argsort(-sims)
                    top1_sim = float(sims[sorted_i[0]])
                    top1_id = gallery_ids[sorted_i[0]]
                    top2_sim = float(sims[sorted_i[1]]) if len(sorted_i) > 1 else 0
                    margin = top1_sim - top2_sim

                    t = 0.60
                    if top1_sim < t:
                        no_match += 1
                    elif top1_id == identity:
                        correct += 1
                    else:
                        incorrect += 1
                    margins.append(margin)
                    total += 1

            key = f"{model_name}_gallery_{actual_gs}"
            query_results[key] = {
                "model": model_name, "gallery_size": actual_gs, "total_queries": total,
                "correct_rank1": correct, "incorrect_rank1": incorrect, "no_match": no_match,
                "rank1_accuracy": round(correct/total, 4) if total else 0,
                "false_identity_rate": round(incorrect/total, 4) if total else 0,
                "false_rejection_rate": round(no_match/total, 4) if total else 0,
                "margin_mean": round(float(np.mean(margins)), 4) if margins else 0,
            }

    print(f"\n  {'Config':<30} {'Rank-1':>8} {'FalseID':>8} {'FalseRej':>8} {'Queries':>8}")
    print(f"  {'-'*62}")
    for key, qr in sorted(query_results.items()):
        print(f"  {key:<30} {qr['rank1_accuracy']:8.4f} {qr['false_identity_rate']:8.4f} "
              f"{qr['false_rejection_rate']:8.4f} {qr['total_queries']:8d}")

    # ── Step 7: Difficult cases ──
    print("\n" + "=" * 60)
    print("PART 6: DIFFICULT CASES")
    print("=" * 60)

    difficult = []
    for p in negative_pairs:
        if p["osnet_sim"] >= 0.50:
            difficult.append({"type": "FALSE_MATCH_RISK", **{k: v for k, v in p.items() if k != "crop"}})
    for p in positive_pairs:
        if p["osnet_sim"] < 0.50:
            difficult.append({"type": "FALSE_REJECT_RISK", **{k: v for k, v in p.items() if k != "crop"}})

    print(f"  False match risks (neg osnet >= 0.50): {sum(1 for d in difficult if d['type'] == 'FALSE_MATCH_RISK')}")
    print(f"  False reject risks (pos osnet < 0.50): {sum(1 for d in difficult if d['type'] == 'FALSE_REJECT_RISK')}")
    for dc in sorted(difficult, key=lambda x: -x.get("osnet_sim", 0))[:15]:
        print(f"    {dc['type']}: osnet={dc.get('osnet_sim', 0):.4f} resnet={dc.get('resnet_sim', 0):.4f} "
              f"heights={dc.get('height_a', '?')}/{dc.get('height_b', '?')}")

    # ── Save visual samples ──
    gt_dir = RESULTS_DIR / "phase36_ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(positive_pairs[:25]):
        a = verified[p["idx_a"]]["crop"]
        b = verified[p["idx_b"]]["crop"]
        th = 256
        ar = cv2.resize(a, (int(a.shape[1]*th/a.shape[0]), th))
        br = cv2.resize(b, (int(b.shape[1]*th/b.shape[0]), th))
        combined = np.hstack([ar, np.ones((th,5,3), dtype=np.uint8)*255, br])
        cv2.imwrite(str(gt_dir / f"pos_{i:03d}_{p['identity']}_osnet{p['osnet_sim']:.3f}.jpg"), combined)

    for i, p in enumerate(negative_pairs[:25]):
        a = verified[p["idx_a"]]["crop"]
        b = verified[p["idx_b"]]["crop"]
        th = 256
        ar = cv2.resize(a, (int(a.shape[1]*th/a.shape[0]), th))
        br = cv2.resize(b, (int(b.shape[1]*th/b.shape[0]), th))
        combined = np.hstack([ar, np.ones((th,5,3), dtype=np.uint8)*255, br])
        cv2.imwrite(str(gt_dir / f"neg_{i:03d}_{p.get('identity_a','?')}_vs_{p.get('identity_b','?')}_osnet{p['osnet_sim']:.3f}.jpg"), combined)

    # ── Assemble results ──
    results = {
        "metadata": {"timestamp": datetime.now().isoformat(), "version": "v3_uuid_linked"},
        "ground_truth": {
            "method": "UUID-linked face→body. Face recognized via ArcFace gallery. Body features from context crops.",
            "verified_crops": len(verified),
            "unverified_crops": len(unverified),
            "identities": dict(id_counts),
            "positive_pairs": len(positive_pairs),
            "negative_pairs": len(negative_pairs),
        },
        "comparison": comp,
        "threshold_sweep_osnet": osnet_sweep,
        "threshold_sweep_resnet": resnet_sweep,
        "query_level": query_results,
        "difficult_cases": difficult[:50],
        "distributions": {
            "osnet_positive": {"mean": round(float(np.mean(pos_o)),4), "std": round(float(np.std(pos_o)),4),
                               "min": round(float(np.min(pos_o)),4), "max": round(float(np.max(pos_o)),4),
                               "p5": round(float(np.percentile(pos_o,5)),4), "p25": round(float(np.percentile(pos_o,25)),4),
                               "median": round(float(np.median(pos_o)),4)},
            "osnet_negative": {"mean": round(float(np.mean(neg_o)),4), "std": round(float(np.std(neg_o)),4),
                               "max": round(float(np.max(neg_o)),4), "p95": round(float(np.percentile(neg_o,95)),4),
                               "p99": round(float(np.percentile(neg_o,99)),4)} if len(neg_o) else {},
            "resnet_positive": {"mean": round(float(np.mean(pos_r)),4), "std": round(float(np.std(pos_r)),4),
                                "min": round(float(np.min(pos_r)),4)},
            "resnet_negative": {"mean": round(float(np.mean(neg_r)),4),
                                "p95": round(float(np.percentile(neg_r,95)),4)} if len(neg_r) else {},
        },
    }

    out_path = RESULTS_DIR / "phase36_v3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("PHASE 3.6 v3 VALIDATION COMPLETE")
    print(f"Ended: {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
