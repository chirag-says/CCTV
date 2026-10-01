"""
Part 5 fix: Direct model comparison bypassing singleton.
Creates both models independently to get true head-to-head numbers.
"""
import sys, os, time, json
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


def build_resnet50():
    """Build ResNet50 ImageNet feature extractor directly (no singleton)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(
        nn.Sequential(*list(backbone.children())[:-1]),
        nn.Flatten(),
    ).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform, device, 2048


def build_osnet_msmt17():
    """Build OSNet x1.0 MSMT17 feature extractor directly (no singleton)."""
    import torchreid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchreid.models.build_model(
        name='osnet_x1_0', num_classes=1, pretrained=False
    )

    weights_path = BACKEND_DIR / 'models' / 'osnet_x1_0_msmt17.pth'
    if weights_path.exists():
        state = torch.load(str(weights_path), map_location='cpu', weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict, strict=False)
        print(f"  OSNet MSMT17: loaded {len(filtered)}/{len(model_dict)} keys")

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform, device, 512


def extract_feat(crop_bgr, model, transform, device):
    """Extract L2-normalized feature from a BGR crop."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
    feat = feat.cpu().numpy().flatten()
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def main():
    print("=" * 60)
    print("PART 5 FIX: RESNET50 VS OSNET MSMT17 HEAD-TO-HEAD")
    print("=" * 60)

    # Load both models independently
    print("\n  Loading ResNet50 ImageNet...")
    resnet_model, resnet_tf, device, resnet_dim = build_resnet50()
    print(f"    ResNet50: {resnet_dim}-dim on {device}")

    print("  Loading OSNet x1.0 MSMT17...")
    osnet_model, osnet_tf, _, osnet_dim = build_osnet_msmt17()
    print(f"    OSNet: {osnet_dim}-dim on {device}")

    # Load YOLO for person detection
    from ultralytics import YOLO
    yolo = YOLO("yolo11m.pt")

    # Sample 300 frames
    frame_files = sorted(list(FRAMES_DIR.glob("fullframe_*.jpg")))
    rng = np.random.RandomState(42)
    if len(frame_files) > 300:
        indices = rng.choice(len(frame_files), 300, replace=False)
        frame_files = [frame_files[i] for i in sorted(indices)]

    print(f"\n  Processing {len(frame_files)} frames...")

    all_crops = []
    for i, fpath in enumerate(frame_files):
        if i % 100 == 0:
            print(f"    Frame {i}/{len(frame_files)}...")
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue
        det = yolo.predict(frame, imgsz=640, classes=[0], conf=0.4, verbose=False)
        boxes = det[0].boxes
        for j, b in enumerate(boxes):
            if j >= 5:
                break
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            h, w = y2 - y1, x2 - x1
            if h < 60 or w < 25:
                continue
            crop = frame[y1:y2, x1:x2].copy()

            # Extract from BOTH models independently
            osnet_feat = extract_feat(crop, osnet_model, osnet_tf, device)
            resnet_feat = extract_feat(crop, resnet_model, resnet_tf, device)

            all_crops.append({
                "frame_name": fpath.name,
                "height_px": h,
                "osnet_feat": osnet_feat,
                "resnet_feat": resnet_feat,
            })

    del yolo
    torch.cuda.empty_cache()

    n = len(all_crops)
    print(f"  Extracted {n} crops with dual features")
    print(f"  OSNet feat sample: dim={all_crops[0]['osnet_feat'].shape}, "
          f"norm={np.linalg.norm(all_crops[0]['osnet_feat']):.4f}")
    print(f"  ResNet feat sample: dim={all_crops[0]['resnet_feat'].shape}, "
          f"norm={np.linalg.norm(all_crops[0]['resnet_feat']):.4f}")

    # Verify features are actually different
    cos_between = float(np.dot(all_crops[0]['osnet_feat'][:512],
                               all_crops[0]['resnet_feat'][:512]))
    print(f"  Cross-model cos(crop[0]): {cos_between:.4f} (should NOT be ~1.0)")

    # Build similarity matrices
    osnet_feats = np.array([c["osnet_feat"] for c in all_crops])
    resnet_feats = np.array([c["resnet_feat"] for c in all_crops])
    osnet_sim = np.dot(osnet_feats, osnet_feats.T)
    resnet_sim = np.dot(resnet_feats, resnet_feats.T)

    # Frame groups
    frame_groups = defaultdict(list)
    for idx, c in enumerate(all_crops):
        frame_groups[c["frame_name"]].append(idx)

    # Negative pairs (same-frame = different person)
    neg_osnet, neg_resnet = [], []
    for indices in frame_groups.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                neg_osnet.append(float(osnet_sim[indices[i], indices[j]]))
                neg_resnet.append(float(resnet_sim[indices[i], indices[j]]))
    neg_osnet = np.array(neg_osnet)
    neg_resnet = np.array(neg_resnet)

    # Positive pairs: use EACH model's own top-3% cross-frame pairs
    same_frame_mask = np.zeros((n, n), dtype=bool)
    for indices in frame_groups.values():
        for i in indices:
            for j in indices:
                same_frame_mask[i, j] = True

    # For fair comparison, use a COMMON set of positive pair indices
    # selected by high similarity in EITHER model
    osnet_cross = osnet_sim.copy()
    osnet_cross[same_frame_mask] = 0
    np.fill_diagonal(osnet_cross, 0)

    resnet_cross = resnet_sim.copy()
    resnet_cross[same_frame_mask] = 0
    np.fill_diagonal(resnet_cross, 0)

    # Use union of top-3% from each model
    osnet_valid = osnet_cross[np.triu_indices(n, k=1)]
    resnet_valid = resnet_cross[np.triu_indices(n, k=1)]

    osnet_t97 = np.percentile(osnet_valid[osnet_valid > 0], 97)
    resnet_t97 = np.percentile(resnet_valid[resnet_valid > 0], 97)

    # Get pair indices where EITHER model has high similarity
    high_osnet = set(map(tuple, np.argwhere((osnet_cross >= osnet_t97) & (np.triu(np.ones((n,n)), k=1).astype(bool)))))
    high_resnet = set(map(tuple, np.argwhere((resnet_cross >= resnet_t97) & (np.triu(np.ones((n,n)), k=1).astype(bool)))))
    common_pairs = high_osnet | high_resnet

    pos_osnet, pos_resnet = [], []
    for i, j in common_pairs:
        pos_osnet.append(float(osnet_sim[i, j]))
        pos_resnet.append(float(resnet_sim[i, j]))
    pos_osnet = np.array(pos_osnet)
    pos_resnet = np.array(pos_resnet)

    print(f"\n  Negative pairs: {len(neg_osnet)}")
    print(f"  Positive proxy pairs (union): {len(pos_osnet)}")

    # Compute metrics for each model at its own optimal threshold
    comparison = {"n_positive": len(pos_osnet), "n_negative": len(neg_osnet)}

    for name, pos, neg in [("osnet_msmt17", pos_osnet, neg_osnet),
                            ("resnet50_imagenet", pos_resnet, neg_resnet)]:
        best_f1, best_t = 0, 0
        for t in np.arange(0.30, 0.85, 0.01):
            tp = np.sum(pos >= t)
            fn = np.sum(pos < t)
            fp = np.sum(neg >= t)
            tn = np.sum(neg < t)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        tp = int(np.sum(pos >= best_t))
        fn = int(np.sum(pos < best_t))
        fp = int(np.sum(neg >= best_t))
        tn = int(np.sum(neg < best_t))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0

        comparison[name] = {
            "embedding_dim": 512 if "osnet" in name else 2048,
            "same_person_mean": round(float(np.mean(pos)), 4),
            "same_person_std": round(float(np.std(pos)), 4),
            "diff_person_mean": round(float(np.mean(neg)), 4),
            "diff_person_std": round(float(np.std(neg)), 4),
            "diff_person_p95": round(float(np.percentile(neg, 95)), 4),
            "separation_gap": round(float(np.mean(pos)) - float(np.percentile(neg, 95)), 4),
            "best_threshold": round(best_t, 2),
            "best_f1": round(best_f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "far": round(far, 4),
            "frr": round(frr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    # Print comparison
    print(f"\n  {'Metric':<25} {'ResNet50':>15} {'OSNet MSMT17':>15}")
    print(f"  {'-'*55}")
    for key in ["embedding_dim", "same_person_mean", "same_person_std",
                 "diff_person_mean", "diff_person_std", "diff_person_p95",
                 "separation_gap", "best_threshold", "best_f1",
                 "precision", "recall", "far", "frr"]:
        r = comparison.get("resnet50_imagenet", {}).get(key, "N/A")
        o = comparison.get("osnet_msmt17", {}).get(key, "N/A")
        print(f"  {key:<25} {str(r):>15} {str(o):>15}")

    # Also do per-model threshold sweeps
    thresholds = [0.45, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]
    print(f"\n  OSNet MSMT17 threshold sweep:")
    print(f"  {'Thresh':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FAR':>8}")
    for t in thresholds:
        tp = int(np.sum(pos_osnet >= t))
        fn = int(np.sum(pos_osnet < t))
        fp = int(np.sum(neg_osnet >= t))
        tn = int(np.sum(neg_osnet < t))
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        far = fp/(fp+tn) if (fp+tn)>0 else 0
        print(f"  {t:8.2f} {tp:6d} {fp:6d} {fn:6d} {tn:6d} {prec:8.4f} {rec:8.4f} {f1:8.4f} {far:8.4f}")

    print(f"\n  ResNet50 ImageNet threshold sweep:")
    print(f"  {'Thresh':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8} {'FAR':>8}")
    for t in thresholds:
        tp = int(np.sum(pos_resnet >= t))
        fn = int(np.sum(pos_resnet < t))
        fp = int(np.sum(neg_resnet >= t))
        tn = int(np.sum(neg_resnet < t))
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        far = fp/(fp+tn) if (fp+tn)>0 else 0
        print(f"  {t:8.2f} {tp:6d} {fp:6d} {fn:6d} {tn:6d} {prec:8.4f} {rec:8.4f} {f1:8.4f} {far:8.4f}")

    # Save
    out_path = RESULTS_DIR / "phase35_part5_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("PART 5 COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
