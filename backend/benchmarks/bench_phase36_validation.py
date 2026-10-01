"""
PHASE 3.6: FINAL CROSS-CAMERA RE-ID VALIDATION
================================================
Face-anchored ground truth for OSNet MSMT17 validation.

Strategy:
  1. Build ArcFace gallery from registered face images
  2. Process CCTV frames: YOLO → SCRFD → ArcFace
  3. When face matches a registered person → verified body crop + identity
  4. Cross-frame same-identity = VERIFIED positive pair
  5. Cross-identity = VERIFIED negative pair

This gives us GENUINE face-verified body ReID ground truth.

Usage:
    cd backend
    python -m benchmarks.bench_phase36_validation
"""

import sys
import os
import time
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, List, Tuple

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


# =================================================================
# MODEL BUILDERS (no singleton, no settings dependency)
# =================================================================

def build_osnet():
    """Build OSNet x1.0 MSMT17 directly."""
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
        print(f"  OSNet MSMT17: {len(filt)}/{len(md)} keys loaded")
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, tf, device


def build_resnet50():
    """Build ResNet50 ImageNet directly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(
        nn.Sequential(*list(backbone.children())[:-1]), nn.Flatten(),
    ).to(device).eval()
    tf = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, tf, device


def extract_feat(crop_bgr, model, tf, device):
    """Extract L2-normalized feature."""
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
# FACE RECOGNITION SETUP
# =================================================================

def build_face_gallery():
    """Build ArcFace gallery from registered face images.
    
    The registered face images are tight crops — SCRFD needs context to detect.
    Strategy: pad with gray border (50% extra each side) to give the detector room.
    Also lower det_thresh for these reference images.
    """
    from insightface.app import FaceAnalysis

    print("  Initializing InsightFace (SCRFD + ArcFace)...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load registered faces
    gallery = {}  # name -> list of 512-d embeddings
    face_files = sorted(FACES_DIR.glob("*.jpg"))
    print(f"  Loading {len(face_files)} registered face images...")

    loaded = 0
    failed = 0
    for fp in face_files:
        # Parse name: student_chirag_1.jpg -> chirag
        parts = fp.stem.split("_")
        if len(parts) >= 2:
            name = parts[1].lower()  # e.g. "chirag", "abhitha"
        else:
            name = parts[0].lower()

        img = cv2.imread(str(fp))
        if img is None:
            continue

        # Tight crops need padding for SCRFD to detect the face
        # Add 50% gray border on all sides
        h, w = img.shape[:2]
        pad_h, pad_w = h // 2, w // 2
        padded = cv2.copyMakeBorder(img, pad_h, pad_h, pad_w, pad_w,
                                     cv2.BORDER_CONSTANT, value=(128, 128, 128))

        # Try with padded image first, then original, then more padding
        faces = None
        for attempt_img, attempt_name in [(padded, "padded"), (img, "original")]:
            faces = app.get(attempt_img)
            if faces:
                break

        # Last resort: 100% padding
        if not faces:
            pad_h2, pad_w2 = h, w
            padded2 = cv2.copyMakeBorder(img, pad_h2, pad_h2, pad_w2, pad_w2,
                                          cv2.BORDER_CONSTANT, value=(128, 128, 128))
            faces = app.get(padded2)

        if faces:
            emb = faces[0].embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            if name not in gallery:
                gallery[name] = []
            gallery[name].append(emb)
            loaded += 1
        else:
            print(f"    WARNING: No face detected in {fp.name} (tried padding)")
            failed += 1

    print(f"  Gallery: {loaded} embeddings loaded, {failed} failed")
    for name, embs in gallery.items():
        print(f"    {name}: {len(embs)} face embeddings")

    return app, gallery


def recognize_face(face_embedding, gallery, threshold=0.5):
    """Match a face embedding against the gallery. Returns (name, similarity) or (None, 0)."""
    best_name = None
    best_sim = 0.0
    for name, embs in gallery.items():
        for ref_emb in embs:
            sim = float(np.dot(face_embedding, ref_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name
    if best_sim >= threshold:
        return best_name, best_sim
    return None, best_sim


# =================================================================
# PART 1+2: BUILD FACE-VERIFIED GROUND TRUTH
# =================================================================

def build_face_verified_ground_truth():
    """Build genuine face-verified body ReID pairs."""
    print("\n" + "=" * 60)
    print("PHASE 3.6: FACE-VERIFIED GROUND TRUTH CONSTRUCTION")
    print("=" * 60)

    from ultralytics import YOLO

    # Load face recognition
    face_app, face_gallery = build_face_gallery()

    # Load both ReID models
    print("  Loading OSNet MSMT17...")
    osnet_model, osnet_tf, device = build_osnet()
    print("  Loading ResNet50 ImageNet...")
    resnet_model, resnet_tf, _ = build_resnet50()

    # ── DATA SOURCE 1: Unknown context images (person-focused crops) ──
    # These are ALREADY person crops from live detections with faces visible
    UNKNOWNS_DIR = BACKEND_DIR / "snapshots" / "unknowns"
    context_files = sorted(list(UNKNOWNS_DIR.glob("context_*.jpg")))
    # Sample up to 5000 context images
    rng = np.random.RandomState(42)
    if len(context_files) > 5000:
        indices = rng.choice(len(context_files), 5000, replace=False)
        context_files = [context_files[i] for i in sorted(indices)]
    print(f"\n  Data source 1: {len(context_files)} unknowns context images")

    # ── DATA SOURCE 2: Full frames with YOLO detection ──
    frame_files = sorted(list(FRAMES_DIR.glob("fullframe_*.jpg")))
    n_frames = min(3000, len(frame_files))
    if len(frame_files) > n_frames:
        indices = rng.choice(len(frame_files), n_frames, replace=False)
        frame_files = [frame_files[i] for i in sorted(indices)]
    print(f"  Data source 2: {len(frame_files)} full frames (YOLO detection)")
    print(f"  Total processing: {len(context_files) + len(frame_files)} images")

    # Collect face-verified body crops
    verified_crops = []  # {identity, frame_name, bbox, crop, osnet_feat, resnet_feat, face_sim, height}
    unverified_crops = []  # body crops without face match (for negatives)

    face_match_count = 0
    no_face_count = 0

    def _process_crop(crop, source_name, source_idx):
        """Process a single person crop: face detect → recognize → extract body features."""
        nonlocal face_match_count, no_face_count
        h, w = crop.shape[:2]
        if h < 60 or w < 25:
            return

        # Face detection + recognition
        faces = face_app.get(crop)
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
                if identity:
                    face_match_count += 1
                else:
                    no_face_count += 1
        else:
            no_face_count += 1

        # Extract body features from both models
        osnet_feat = extract_feat(crop, osnet_model, osnet_tf, device)
        resnet_feat = extract_feat(crop, resnet_model, resnet_tf, device)

        entry = {
            "identity": identity,
            "frame_name": source_name,
            "frame_idx": source_idx,
            "bbox": [0, 0, w, h],
            "height_px": h,
            "width_px": w,
            "crop": crop,
            "osnet_feat": osnet_feat,
            "resnet_feat": resnet_feat,
            "face_sim": round(face_sim, 4),
            "has_face": bool(faces),
        }

        if identity:
            verified_crops.append(entry)
        else:
            if len(unverified_crops) < 2000:  # cap unverified to save memory
                unverified_crops.append(entry)

    # ── PASS 1: Process unknowns context images (already person crops) ──
    print("\n  PASS 1: Processing unknowns context images...")
    for ci, cpath in enumerate(context_files):
        if ci % 500 == 0:
            print(f"    Context {ci}/{len(context_files)} | verified={len(verified_crops)} | unverified={len(unverified_crops)}")

        crop = cv2.imread(str(cpath))
        if crop is None:
            continue
        _process_crop(crop, cpath.name, ci)

    print(f"    Context DONE: verified={len(verified_crops)} | unverified={len(unverified_crops)}")

    # ── PASS 2: Process full frames with YOLO ──
    print("\n  PASS 2: Processing full frames with YOLO...")
    from ultralytics import YOLO
    yolo = YOLO("yolo11m.pt")

    for fi, fpath in enumerate(frame_files):
        if fi % 500 == 0:
            print(f"    Frame {fi}/{len(frame_files)} | verified={len(verified_crops)} | unverified={len(unverified_crops)}")

        frame = cv2.imread(str(fpath))
        if frame is None:
            continue

        det = yolo.predict(frame, imgsz=640, classes=[0], conf=0.4, verbose=False)
        boxes = det[0].boxes

        for j, b in enumerate(boxes):
            if j >= 8:
                break
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2].copy()
            _process_crop(crop, fpath.name, fi)

    del yolo
    torch.cuda.empty_cache()

    # Summary
    identity_counts = defaultdict(int)
    for vc in verified_crops:
        identity_counts[vc["identity"]] += 1

    print(f"\n  Face-verified body crops: {len(verified_crops)}")
    print(f"  Unverified body crops: {len(unverified_crops)}")
    print(f"  Face match rate: {face_match_count}/{face_match_count + no_face_count}")
    print(f"  Identities found:")
    for name, count in sorted(identity_counts.items(), key=lambda x: -x[1]):
        print(f"    {name}: {count} crops across frames")

    return verified_crops, unverified_crops, face_app, face_gallery


# =================================================================
# PART 3-7: VALIDATION ANALYSIS
# =================================================================

def run_validation(verified_crops, unverified_crops):
    """Full validation: pairs, threshold sweep, gallery test, difficult cases."""
    print("\n" + "=" * 60)
    print("BUILDING VERIFIED PAIRS")
    print("=" * 60)

    results = {"metadata": {"timestamp": datetime.now().isoformat()}}

    identity_groups = defaultdict(list)
    for i, vc in enumerate(verified_crops):
        identity_groups[vc["identity"]].append(i)

    # POSITIVE PAIRS: same identity, different frames
    positive_pairs = []
    for identity, indices in identity_groups.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_a, idx_b = indices[i], indices[j]
                # Must be from different frames
                if verified_crops[idx_a]["frame_name"] == verified_crops[idx_b]["frame_name"]:
                    continue
                osnet_sim = float(np.dot(verified_crops[idx_a]["osnet_feat"],
                                         verified_crops[idx_b]["osnet_feat"]))
                resnet_sim = float(np.dot(verified_crops[idx_a]["resnet_feat"],
                                          verified_crops[idx_b]["resnet_feat"]))
                positive_pairs.append({
                    "idx_a": idx_a, "idx_b": idx_b,
                    "identity": identity,
                    "label": "SAME_PERSON",
                    "verification": "FACE_VERIFIED",
                    "osnet_sim": round(osnet_sim, 4),
                    "resnet_sim": round(resnet_sim, 4),
                    "frame_a": verified_crops[idx_a]["frame_name"],
                    "frame_b": verified_crops[idx_b]["frame_name"],
                    "height_a": verified_crops[idx_a]["height_px"],
                    "height_b": verified_crops[idx_b]["height_px"],
                    "face_sim_a": verified_crops[idx_a]["face_sim"],
                    "face_sim_b": verified_crops[idx_b]["face_sim"],
                })

    # NEGATIVE PAIRS: different identities
    negative_pairs = []
    identities = list(identity_groups.keys())
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            id_a_indices = identity_groups[identities[i]]
            id_b_indices = identity_groups[identities[j]]
            # Sample up to 10 pairs per identity combination
            rng = np.random.RandomState(42)
            n_pairs = min(10, len(id_a_indices) * len(id_b_indices))
            for _ in range(n_pairs):
                idx_a = rng.choice(id_a_indices)
                idx_b = rng.choice(id_b_indices)
                osnet_sim = float(np.dot(verified_crops[idx_a]["osnet_feat"],
                                         verified_crops[idx_b]["osnet_feat"]))
                resnet_sim = float(np.dot(verified_crops[idx_a]["resnet_feat"],
                                          verified_crops[idx_b]["resnet_feat"]))
                negative_pairs.append({
                    "idx_a": idx_a, "idx_b": idx_b,
                    "identity_a": identities[i],
                    "identity_b": identities[j],
                    "label": "DIFFERENT_PERSON",
                    "verification": "FACE_VERIFIED",
                    "osnet_sim": round(osnet_sim, 4),
                    "resnet_sim": round(resnet_sim, 4),
                    "frame_a": verified_crops[idx_a]["frame_name"],
                    "frame_b": verified_crops[idx_b]["frame_name"],
                    "height_a": verified_crops[idx_a]["height_px"],
                    "height_b": verified_crops[idx_b]["height_px"],
                })

    # Also add same-frame different-identity negatives
    for frame_name in set(vc["frame_name"] for vc in verified_crops):
        frame_crops = [(i, vc) for i, vc in enumerate(verified_crops) if vc["frame_name"] == frame_name]
        for i in range(len(frame_crops)):
            for j in range(i + 1, len(frame_crops)):
                idx_a, vc_a = frame_crops[i]
                idx_b, vc_b = frame_crops[j]
                if vc_a["identity"] != vc_b["identity"]:
                    osnet_sim = float(np.dot(vc_a["osnet_feat"], vc_b["osnet_feat"]))
                    resnet_sim = float(np.dot(vc_a["resnet_feat"], vc_b["resnet_feat"]))
                    negative_pairs.append({
                        "idx_a": idx_a, "idx_b": idx_b,
                        "identity_a": vc_a["identity"],
                        "identity_b": vc_b["identity"],
                        "label": "DIFFERENT_PERSON",
                        "verification": "FACE_VERIFIED_SAME_FRAME",
                        "osnet_sim": round(osnet_sim, 4),
                        "resnet_sim": round(resnet_sim, 4),
                        "frame_a": vc_a["frame_name"],
                        "frame_b": vc_b["frame_name"],
                        "height_a": vc_a["height_px"],
                        "height_b": vc_b["height_px"],
                    })

    print(f"  FACE-VERIFIED positive pairs: {len(positive_pairs)}")
    print(f"  FACE-VERIFIED negative pairs: {len(negative_pairs)}")

    if len(positive_pairs) == 0:
        print("\n  WARNING: No verified positive pairs found!")
        print("  This means no registered person was seen across multiple frames.")
        print("  POSSIBLE CAUSES:")
        print("    - Not enough frames with visible registered faces")
        print("    - Face recognition threshold too high")
        print("    - Registered people not present in footage")
        results["ground_truth"] = {
            "positive_pairs": 0, "negative_pairs": len(negative_pairs),
            "error": "No face-verified positive pairs found"
        }
        results["all_pairs"] = {"positive": positive_pairs, "negative": negative_pairs}
        return results

    # ============================================================
    # PART 4: THRESHOLD SWEEP
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 4: THRESHOLD SWEEP (FACE-VERIFIED PAIRS)")
    print("=" * 60)

    pos_osnet = np.array([p["osnet_sim"] for p in positive_pairs])
    neg_osnet = np.array([p["osnet_sim"] for p in negative_pairs])
    pos_resnet = np.array([p["resnet_sim"] for p in positive_pairs])
    neg_resnet = np.array([p["resnet_sim"] for p in negative_pairs])

    thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.65, 0.68]

    print(f"\n  OSNet MSMT17 (face-verified pairs):")
    print(f"  Positive pairs: {len(pos_osnet)} | Negative pairs: {len(neg_osnet)}")
    print(f"  {'Thresh':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FAR':>8} {'FRR':>8}")
    print(f"  {'-'*80}")

    osnet_sweep = []
    for t in thresholds:
        tp = int(np.sum(pos_osnet >= t))
        fn = int(np.sum(pos_osnet < t))
        fp = int(np.sum(neg_osnet >= t))
        tn = int(np.sum(neg_osnet < t))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        row = {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "precision": round(prec, 4), "recall": round(rec, 4),
               "f1": round(f1, 4), "far": round(far, 4), "frr": round(frr, 4)}
        osnet_sweep.append(row)
        print(f"  {t:8.2f} {tp:6d} {fp:6d} {fn:6d} {tn:6d} {prec:8.4f} {rec:8.4f} {f1:8.4f} {far:8.4f} {frr:8.4f}")

    print(f"\n  ResNet50 ImageNet (same face-verified pairs):")
    print(f"  {'Thresh':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FAR':>8} {'FRR':>8}")
    print(f"  {'-'*80}")

    resnet_sweep = []
    for t in thresholds:
        tp = int(np.sum(pos_resnet >= t))
        fn = int(np.sum(pos_resnet < t))
        fp = int(np.sum(neg_resnet >= t))
        tn = int(np.sum(neg_resnet < t))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        row = {"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
               "precision": round(prec, 4), "recall": round(rec, 4),
               "f1": round(f1, 4), "far": round(far, 4), "frr": round(frr, 4)}
        resnet_sweep.append(row)
        print(f"  {t:8.2f} {tp:6d} {fp:6d} {fn:6d} {tn:6d} {prec:8.4f} {rec:8.4f} {f1:8.4f} {far:8.4f} {frr:8.4f}")

    # ============================================================
    # PART 5: HEAD-TO-HEAD COMPARISON
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 5: RESNET50 VS OSNET (FACE-VERIFIED)")
    print("=" * 60)

    comparison = {}
    for name, pos, neg in [("osnet_msmt17", pos_osnet, neg_osnet),
                            ("resnet50_imagenet", pos_resnet, neg_resnet)]:
        comparison[name] = {
            "same_person_mean": round(float(np.mean(pos)), 4) if len(pos) > 0 else None,
            "same_person_std": round(float(np.std(pos)), 4) if len(pos) > 0 else None,
            "same_person_min": round(float(np.min(pos)), 4) if len(pos) > 0 else None,
            "same_person_max": round(float(np.max(pos)), 4) if len(pos) > 0 else None,
            "diff_person_mean": round(float(np.mean(neg)), 4) if len(neg) > 0 else None,
            "diff_person_std": round(float(np.std(neg)), 4) if len(neg) > 0 else None,
            "diff_person_p95": round(float(np.percentile(neg, 95)), 4) if len(neg) > 0 else None,
            "diff_person_max": round(float(np.max(neg)), 4) if len(neg) > 0 else None,
            "separation_gap": round(float(np.mean(pos)) - float(np.percentile(neg, 95)), 4) if len(pos) > 0 and len(neg) > 0 else None,
        }

    print(f"\n  {'Metric':<30} {'ResNet50':>15} {'OSNet MSMT17':>15}")
    print(f"  {'-'*60}")
    for key in ["same_person_mean", "same_person_min", "diff_person_mean",
                 "diff_person_p95", "diff_person_max", "separation_gap"]:
        r = comparison.get("resnet50_imagenet", {}).get(key, "N/A")
        o = comparison.get("osnet_msmt17", {}).get(key, "N/A")
        print(f"  {key:<30} {str(r):>15} {str(o):>15}")

    # ============================================================
    # PART 3: QUERY-LEVEL (RANK-1) TEST
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 3: QUERY-LEVEL RANK-1 TEST")
    print("=" * 60)

    identity_groups = defaultdict(list)
    for i, vc in enumerate(verified_crops):
        identity_groups[vc["identity"]].append(i)

    gallery_sizes = [10, 25, 50, 100]
    query_results = {}

    for gs in gallery_sizes:
        # Build gallery: take first crop from each identity, then fill with unverified
        gallery_indices = []
        gallery_identities = []
        for identity, indices in identity_groups.items():
            gallery_indices.append(indices[0])
            gallery_identities.append(identity)

        # Fill remaining gallery with unverified crops (unknown people)
        n_fillers = gs - len(gallery_indices)
        if n_fillers > 0 and len(unverified_crops) > 0:
            filler_indices = list(range(min(n_fillers, len(unverified_crops))))
            for fi in filler_indices:
                gallery_indices.append(-(fi + 1))  # negative = unverified
                gallery_identities.append(f"unknown_{fi}")

        # Build gallery features
        gallery_feats_osnet = []
        gallery_feats_resnet = []
        for gi in gallery_indices:
            if gi >= 0:
                gallery_feats_osnet.append(verified_crops[gi]["osnet_feat"])
                gallery_feats_resnet.append(verified_crops[gi]["resnet_feat"])
            else:
                ui = -(gi + 1)
                gallery_feats_osnet.append(unverified_crops[ui]["osnet_feat"])
                gallery_feats_resnet.append(unverified_crops[ui]["resnet_feat"])

        gallery_feats_osnet = np.array(gallery_feats_osnet)
        gallery_feats_resnet = np.array(gallery_feats_resnet)

        actual_gs = len(gallery_indices)

        # Query: use remaining verified crops (not in gallery)
        for model_name, gallery_feats in [("osnet", gallery_feats_osnet), ("resnet", gallery_feats_resnet)]:
            correct_rank1 = 0
            incorrect_rank1 = 0
            no_match = 0
            total_queries = 0
            margins = []

            for identity, indices in identity_groups.items():
                query_indices = indices[1:]  # skip the gallery entry
                for qi in query_indices:
                    vc = verified_crops[qi]
                    feat_key = "osnet_feat" if model_name == "osnet" else "resnet_feat"
                    query_feat = vc[feat_key]

                    # Search gallery
                    sims = np.dot(gallery_feats, query_feat)
                    sorted_idx = np.argsort(-sims)
                    top1_idx = sorted_idx[0]
                    top1_sim = float(sims[top1_idx])
                    top1_identity = gallery_identities[top1_idx]

                    top2_sim = float(sims[sorted_idx[1]]) if len(sorted_idx) > 1 else 0
                    margin = top1_sim - top2_sim

                    # Apply threshold
                    t = 0.60
                    if top1_sim < t:
                        no_match += 1
                    elif top1_identity == identity:
                        correct_rank1 += 1
                    else:
                        incorrect_rank1 += 1

                    margins.append(margin)
                    total_queries += 1

            if total_queries > 0:
                rank1_acc = correct_rank1 / total_queries
                false_id_rate = incorrect_rank1 / total_queries
                false_reject = no_match / total_queries
            else:
                rank1_acc = false_id_rate = false_reject = 0

            key = f"{model_name}_gallery_{actual_gs}"
            query_results[key] = {
                "model": model_name, "gallery_size": actual_gs,
                "total_queries": total_queries,
                "correct_rank1": correct_rank1,
                "incorrect_rank1": incorrect_rank1,
                "no_match": no_match,
                "rank1_accuracy": round(rank1_acc, 4),
                "false_identity_rate": round(false_id_rate, 4),
                "false_rejection_rate": round(false_reject, 4),
                "margin_mean": round(float(np.mean(margins)), 4) if margins else 0,
                "margin_min": round(float(np.min(margins)), 4) if margins else 0,
            }

    # Print query results
    print(f"\n  {'Config':<30} {'Rank-1':>8} {'FalseID':>8} {'FalseRej':>8} {'Queries':>8}")
    print(f"  {'-'*62}")
    for key, qr in sorted(query_results.items()):
        print(f"  {key:<30} {qr['rank1_accuracy']:8.4f} {qr['false_identity_rate']:8.4f} "
              f"{qr['false_rejection_rate']:8.4f} {qr['total_queries']:8d}")

    # ============================================================
    # PART 6: DIFFICULT CASES
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 6: DIFFICULT CASES")
    print("=" * 60)

    # Find worst false matches and misses
    difficult = []
    for p in negative_pairs:
        if p["osnet_sim"] >= 0.55:
            difficult.append({
                "type": "FALSE_MATCH_RISK",
                "identity_a": p.get("identity_a", "?"),
                "identity_b": p.get("identity_b", "?"),
                "osnet_sim": p["osnet_sim"],
                "resnet_sim": p["resnet_sim"],
                "height_a": p["height_a"],
                "height_b": p["height_b"],
            })
    for p in positive_pairs:
        if p["osnet_sim"] < 0.55:
            difficult.append({
                "type": "FALSE_REJECT_RISK",
                "identity": p["identity"],
                "osnet_sim": p["osnet_sim"],
                "resnet_sim": p["resnet_sim"],
                "height_a": p["height_a"],
                "height_b": p["height_b"],
            })

    print(f"  False match risks (neg sim >= 0.55): {sum(1 for d in difficult if d['type'] == 'FALSE_MATCH_RISK')}")
    print(f"  False reject risks (pos sim < 0.55): {sum(1 for d in difficult if d['type'] == 'FALSE_REJECT_RISK')}")
    for dc in sorted(difficult, key=lambda x: -x.get("osnet_sim", 0))[:10]:
        print(f"    {dc['type']}: osnet={dc['osnet_sim']:.4f} resnet={dc['resnet_sim']:.4f} "
              f"heights={dc['height_a']}/{dc['height_b']}")

    # ============================================================
    # ASSEMBLE RESULTS
    # ============================================================
    # Remove non-serializable crop arrays
    results["ground_truth"] = {
        "verified_identities": list(identity_groups.keys()),
        "verified_crops_per_identity": {k: len(v) for k, v in identity_groups.items()},
        "total_verified_crops": len(verified_crops),
        "total_unverified_crops": len(unverified_crops),
        "positive_pairs": len(positive_pairs),
        "negative_pairs": len(negative_pairs),
        "verification_method": "ArcFace face recognition against registered gallery",
    }
    results["threshold_sweep_osnet"] = osnet_sweep
    results["threshold_sweep_resnet"] = resnet_sweep
    results["comparison"] = comparison
    results["query_level"] = query_results
    results["difficult_cases"] = {
        "false_match_risks": [d for d in difficult if d["type"] == "FALSE_MATCH_RISK"],
        "false_reject_risks": [d for d in difficult if d["type"] == "FALSE_REJECT_RISK"],
    }

    # Distribution stats
    results["distributions"] = {
        "osnet_positive": {
            "mean": round(float(np.mean(pos_osnet)), 4),
            "std": round(float(np.std(pos_osnet)), 4),
            "min": round(float(np.min(pos_osnet)), 4),
            "max": round(float(np.max(pos_osnet)), 4),
            "p5": round(float(np.percentile(pos_osnet, 5)), 4),
            "p25": round(float(np.percentile(pos_osnet, 25)), 4),
            "p50": round(float(np.percentile(pos_osnet, 50)), 4),
        } if len(pos_osnet) > 0 else {},
        "osnet_negative": {
            "mean": round(float(np.mean(neg_osnet)), 4),
            "std": round(float(np.std(neg_osnet)), 4),
            "min": round(float(np.min(neg_osnet)), 4),
            "max": round(float(np.max(neg_osnet)), 4),
            "p95": round(float(np.percentile(neg_osnet, 95)), 4),
            "p99": round(float(np.percentile(neg_osnet, 99)), 4),
        } if len(neg_osnet) > 0 else {},
        "resnet_positive": {
            "mean": round(float(np.mean(pos_resnet)), 4),
            "std": round(float(np.std(pos_resnet)), 4),
            "min": round(float(np.min(pos_resnet)), 4),
            "max": round(float(np.max(pos_resnet)), 4),
        } if len(pos_resnet) > 0 else {},
        "resnet_negative": {
            "mean": round(float(np.mean(neg_resnet)), 4),
            "std": round(float(np.std(neg_resnet)), 4),
            "p95": round(float(np.percentile(neg_resnet, 95)), 4),
            "p99": round(float(np.percentile(neg_resnet, 99)), 4),
        } if len(neg_resnet) > 0 else {},
    }

    # Save positive/negative pair details (without crop arrays)
    results["positive_pair_details"] = [{k: v for k, v in p.items()} for p in positive_pairs[:200]]
    results["negative_pair_details"] = [{k: v for k, v in p.items()} for p in negative_pairs[:200]]

    # Save visual samples
    gt_dir = RESULTS_DIR / "phase36_ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(positive_pairs[:20]):
        crop_a = verified_crops[p["idx_a"]]["crop"]
        crop_b = verified_crops[p["idx_b"]]["crop"]
        th = 256
        a_r = cv2.resize(crop_a, (int(crop_a.shape[1] * th / crop_a.shape[0]), th))
        b_r = cv2.resize(crop_b, (int(crop_b.shape[1] * th / crop_b.shape[0]), th))
        combined = np.hstack([a_r, np.ones((th, 5, 3), dtype=np.uint8) * 255, b_r])
        cv2.imwrite(str(gt_dir / f"pos_{i:03d}_{p['identity']}_osnet{p['osnet_sim']:.3f}.jpg"), combined)

    for i, p in enumerate(negative_pairs[:20]):
        crop_a = verified_crops[p["idx_a"]]["crop"]
        crop_b = verified_crops[p["idx_b"]]["crop"]
        th = 256
        a_r = cv2.resize(crop_a, (int(crop_a.shape[1] * th / crop_a.shape[0]), th))
        b_r = cv2.resize(crop_b, (int(crop_b.shape[1] * th / crop_b.shape[0]), th))
        combined = np.hstack([a_r, np.ones((th, 5, 3), dtype=np.uint8) * 255, b_r])
        id_a = p.get("identity_a", "?")
        id_b = p.get("identity_b", "?")
        cv2.imwrite(str(gt_dir / f"neg_{i:03d}_{id_a}_vs_{id_b}_osnet{p['osnet_sim']:.3f}.jpg"), combined)

    return results


# =================================================================
# MAIN
# =================================================================

def main():
    print("=" * 60)
    print("PHASE 3.6: FINAL CROSS-CAMERA RE-ID VALIDATION")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    verified_crops, unverified_crops, face_app, face_gallery = build_face_verified_ground_truth()

    results = run_validation(verified_crops, unverified_crops)

    # Save results
    out_path = RESULTS_DIR / "phase36_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("PHASE 3.6 VALIDATION COMPLETE")
    print(f"Ended: {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
