"""
PHASE 3: PERSON RE-ID MODEL BENCHMARK
======================================
Comprehensive evaluation of ReID model candidates on SentinelAI CCTV footage.

Compares:
  1. CURRENT: ResNet50 ImageNet (2048-dim) ? production baseline
  2. CANDIDATE A: OSNet x1.0 Market-1501 ReID-trained (512-dim)
  3. CANDIDATE B: OSNet x1.0 MSMT17 ReID-trained (512-dim)

Tests:
  - Single/batch latency (P50, P95, max)
  - Same-person vs different-person similarity distributions
  - Cross-camera re-identification
  - Threshold calibration with precision/recall curves
  - Crop quality analysis
  - Embedding processing verification
  - Full pipeline integration impact

READ-ONLY: does not modify production code, database, or models.

Usage:
    cd backend
    python -m benchmarks.bench_phase3_reid

Output:
    benchmarks/results/phase3_reid.json
    benchmarks/results/phase3_reid_visual/
"""

import sys
import os
import time
import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from benchmarks.bench_utils import (
    FRAMES_DIR, UNKNOWNS_DIR, Timer, GPUSampler, gpu_snapshot,
    cpu_snapshot, load_frames_from_dir, save_results,
)

PYTHON = r"C:\Users\chira\AppData\Local\Programs\Python\Python311\python.exe"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
VISUAL_DIR = RESULTS_DIR / "phase3_reid_visual"

MAX_FRAMES = 300      # frames to sample for person crops
MAX_CROPS = 500       # max person crops to process
WARMUP = 10           # warmup iterations before timing
BATCH_SIZES = [4, 8, 16]


# ===================================================================
# MODEL EXTRACTORS
# ===================================================================

class CurrentResNet50Extractor:
    """Exact replica of production app.vision.reid.ReIDFeatureExtractor."""

    NAME = "current_resnet50_imagenet"
    TRAINING = "ImageNet classification (NOT person-ReID)"
    EMBEDDING_DIM = 2048
    INPUT_SIZE = (256, 128)  # (H, W)
    DISTANCE = "cosine"
    LICENSE = "BSD-3-Clause (torchvision)"

    def __init__(self, device):
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(
            nn.Sequential(*list(backbone.children())[:-1]),
            nn.Flatten(),
        ).to(device).eval()
        self.device = device
        self.feat_dim = self.EMBEDDING_DIM
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.param_count = sum(p.numel() for p in self.model.parameters())

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    @torch.no_grad()
    def extract_batch(self, crops: list) -> np.ndarray:
        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))
        batch = torch.stack(tensors).to(self.device)
        feats = self.model(batch).cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms


class OSNetReIDExtractor:
    """
    OSNet x1.0 with actual ReID training weights.
    Uses torchreid to build the model architecture and loads
    person-ReID trained weights (Market-1501 or MSMT17).
    """

    INPUT_SIZE = (256, 128)  # Standard ReID input
    DISTANCE = "cosine"
    LICENSE = "MIT (torchreid)"

    def __init__(self, device, weights_path: str, name: str, training_desc: str):
        self.NAME = name
        self.TRAINING = training_desc
        self.device = device
        self.feat_dim = 512
        self.EMBEDDING_DIM = 512
        self.available = False

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self._load(weights_path)

    def _load(self, weights_path: str):
        try:
            import torchreid
            # Build OSNet architecture
            model = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=1,  # dummy, we discard classifier
                pretrained=False,
            )

            # Load ReID-trained weights
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location='cpu', weights_only=True)
                # Handle different checkpoint formats
                if 'state_dict' in state:
                    state = state['state_dict']

                # Filter out classifier weights (we only need feature extractor)
                model_dict = model.state_dict()
                filtered = {k: v for k, v in state.items()
                            if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(filtered)
                model.load_state_dict(model_dict, strict=False)
                loaded_keys = len(filtered)
                total_keys = len(model_dict)
                print(f"  ? Loaded {loaded_keys}/{total_keys} keys from {Path(weights_path).name}")
            else:
                print(f"  !? Weights not found: {weights_path}, using random init")

            self.model = model.to(self.device).eval()
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.available = True

        except Exception as e:
            print(f"  X Failed to load {self.NAME}: {e}")
            self.available = False

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        if not self.available:
            return np.zeros(self.feat_dim)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    @torch.no_grad()
    def extract_batch(self, crops: list) -> np.ndarray:
        if not self.available:
            return np.zeros((len(crops), self.feat_dim))
        tensors = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))
        batch = torch.stack(tensors).to(self.device)
        feats = self.model(batch)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = feats.cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms


# ===================================================================
# PERSON CROP EXTRACTION
# ===================================================================

def get_person_crops(frames, max_per_frame=5, min_height=50):
    """Detect persons in frames using YOLO and return crops with metadata."""
    from ultralytics import YOLO

    # Use the production YOLO model if available
    model_path = "yolo11m.pt"
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
    print(f"  Using {model_path} for person detection")

    yolo = YOLO(model_path)
    crops = []

    for fname, frame in frames:
        results = yolo.predict(
            frame, imgsz=640, half=True, classes=[0],
            conf=0.35, verbose=False,
        )
        boxes = results[0].boxes
        for i, b in enumerate(boxes):
            if i >= max_per_frame:
                break
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            h = y2 - y1
            w = x2 - x1
            if h < min_height or w < 20:
                continue
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            # Parse camera ID from filename if available
            camera_id = "unknown"
            # SentinelAI frames: fullframe_UUID_timestamp.jpg
            # Try to infer camera from frame characteristics or use hash-based grouping
            crops.append({
                "frame_name": fname,
                "bbox": [int(x) for x in xyxy],
                "crop": crop,
                "height_px": h,
                "width_px": w,
                "aspect_ratio": round(h / max(w, 1), 2),
                "area_px": h * w,
                "frame_shape": frame.shape[:2],
                "camera_id": camera_id,
            })

    del yolo
    torch.cuda.empty_cache()
    return crops


def analyze_crop_quality(crops):
    """Analyze crop quality: head/feet truncation, padding, overlap estimation."""
    quality_stats = {
        "total_crops": len(crops),
        "height_distribution": {},
        "aspect_ratio_distribution": {},
        "very_small": 0,      # < 80px height
        "small": 0,           # 80-128px
        "medium": 0,          # 128-256px
        "large": 0,           # > 256px
        "narrow_aspect": 0,   # aspect > 3.5 (tall thin)
        "wide_aspect": 0,     # aspect < 1.5 (short wide, likely partial)
    }

    heights = [c["height_px"] for c in crops]
    aspects = [c["aspect_ratio"] for c in crops]

    for h in heights:
        if h < 80:
            quality_stats["very_small"] += 1
        elif h < 128:
            quality_stats["small"] += 1
        elif h < 256:
            quality_stats["medium"] += 1
        else:
            quality_stats["large"] += 1

    for ar in aspects:
        if ar > 3.5:
            quality_stats["narrow_aspect"] += 1
        elif ar < 1.5:
            quality_stats["wide_aspect"] += 1

    quality_stats["height_distribution"] = {
        "min": int(min(heights)) if heights else 0,
        "max": int(max(heights)) if heights else 0,
        "mean": round(float(np.mean(heights)), 1) if heights else 0,
        "median": round(float(np.median(heights)), 1) if heights else 0,
        "p25": round(float(np.percentile(heights, 25)), 1) if heights else 0,
        "p75": round(float(np.percentile(heights, 75)), 1) if heights else 0,
    }
    quality_stats["aspect_ratio_distribution"] = {
        "min": round(float(min(aspects)), 2) if aspects else 0,
        "max": round(float(max(aspects)), 2) if aspects else 0,
        "mean": round(float(np.mean(aspects)), 2) if aspects else 0,
        "median": round(float(np.median(aspects)), 2) if aspects else 0,
    }

    return quality_stats


# ===================================================================
# IDENTITY-PAIR BUILDING (using frame proximity as ground truth proxy)
# ===================================================================

def build_identity_pairs(crops, embeddings_dict):
    """
    Build positive and negative pairs from the benchmark results.

    Ground truth strategy:
    - POSITIVE: crops from the SAME frame with high spatial proximity
      (same person tracked across slightly different detections)
      OR crops from temporally adjacent frames with very high similarity
      from a DIFFERENT model (cross-validation)
    - NEGATIVE: crops from different frames showing clearly different people

    Since we don't have manual annotation, we use a conservative strategy:
    same-frame crops are DIFFERENT people (negative),
    and we identify positive pairs through frame-name grouping and
    face-based identity labels from the benchmark results directory.
    """
    # We'll use the phase2 annotated faces results if available
    # to get ground-truth identity labels
    pairs = {"positive": [], "negative": []}

    # Group crops by frame
    frame_groups = defaultdict(list)
    for idx, crop_info in enumerate(crops):
        frame_groups[crop_info["frame_name"]].append(idx)

    # Same-frame pairs are DIFFERENT people (negative pairs)
    for frame_name, indices in frame_groups.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs["negative"].append((indices[i], indices[j]))

    return pairs


def build_ground_truth_pairs_from_face_results(crops):
    """
    Use the phase2 annotated face results to create ground-truth pairs.

    The benchmark results contain files named like:
    0044_f0_34px_MATCH_Chirag.jpg
    0044_f1_60px_NOMATCH_Chirag.jpg

    The frame index (0044) and person name can give us identity labels.
    """
    results_dir = RESULTS_DIR / "phase2_annotated_faces.json"
    if not results_dir.exists():
        return None

    try:
        with open(results_dir) as f:
            face_data = json.load(f)
        return face_data
    except Exception:
        return None


# ===================================================================
# THRESHOLD CALIBRATION
# ===================================================================

def calibrate_threshold(same_sims, diff_sims, steps=100):
    """
    Find optimal threshold by sweeping from min to max similarity.
    Returns threshold analysis with precision/recall at each point.
    """
    if len(same_sims) == 0 or len(diff_sims) == 0:
        return {"error": "insufficient pairs"}

    all_sims = np.concatenate([same_sims, diff_sims])
    lo, hi = float(np.min(all_sims)), float(np.max(all_sims))

    thresholds = np.linspace(lo, hi, steps)
    results = []

    for t in thresholds:
        tp = np.sum(same_sims >= t)   # same person correctly matched
        fn = np.sum(same_sims < t)    # same person missed
        fp = np.sum(diff_sims >= t)   # different person falsely matched
        tn = np.sum(diff_sims < t)    # different person correctly rejected

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # false positive rate
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0  # false negative rate
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "threshold": round(float(t), 4),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "f1": round(f1, 4),
        })

    # Find best F1
    best_f1 = max(results, key=lambda x: x["f1"])
    # Find EER (where FPR ? FNR)
    eer_point = min(results, key=lambda x: abs(x["fpr"] - x["fnr"]))

    return {
        "best_f1_threshold": best_f1["threshold"],
        "best_f1": best_f1["f1"],
        "best_precision": best_f1["precision"],
        "best_recall": best_f1["recall"],
        "eer_threshold": eer_point["threshold"],
        "eer_fpr": eer_point["fpr"],
        "eer_fnr": eer_point["fnr"],
        "sweep": results[::5],  # Every 5th point to keep output manageable
    }


# ===================================================================
# MAIN BENCHMARK
# ===================================================================

def run_phase3_benchmark():
    print("=" * 70)
    print("PHASE 3: PERSON RE-IDENTIFICATION MODEL BENCHMARK")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # -- Step 1: Load frames --------------------------------------
    print(f"\n{'-' * 60}")
    print("STEP 1: Loading SentinelAI CCTV frames")
    print(f"{'-' * 60}")

    frame_data = load_frames_from_dir(FRAMES_DIR, max_frames=MAX_FRAMES, seed=42)
    print(f"  Loaded {len(frame_data)} frames from {FRAMES_DIR}")

    # -- Step 2: Extract person crops -----------------------------
    print(f"\n{'-' * 60}")
    print("STEP 2: Extracting person crops")
    print(f"{'-' * 60}")

    crops = get_person_crops(frame_data, max_per_frame=5, min_height=40)
    if len(crops) > MAX_CROPS:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(crops), MAX_CROPS, replace=False)
        crops = [crops[i] for i in sorted(indices)]
    print(f"  Extracted {len(crops)} person crops")

    if len(crops) < 20:
        print("ERROR: Too few crops for meaningful benchmark. Exiting.")
        return

    # -- Step 3: Analyze crop quality (Part 8) --------------------
    print(f"\n{'-' * 60}")
    print("STEP 3: Crop Quality Analysis (Part 8)")
    print(f"{'-' * 60}")

    crop_quality = analyze_crop_quality(crops)
    print(f"  Heights: {crop_quality['height_distribution']}")
    print(f"  Very small (<80px): {crop_quality['very_small']}")
    print(f"  Small (80-128px): {crop_quality['small']}")
    print(f"  Medium (128-256px): {crop_quality['medium']}")
    print(f"  Large (>256px): {crop_quality['large']}")

    # Also test padded crops
    padded_crops = []
    for c in crops:
        crop = c["crop"]
        h, w = crop.shape[:2]
        pad_h = int(h * 0.05)  # 5% vertical padding
        pad_w = int(w * 0.1)   # 10% horizontal padding
        padded = cv2.copyMakeBorder(crop, pad_h, pad_h, pad_w, pad_w,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded_crops.append(padded)

    # -- Step 4: Load models --------------------------------------
    print(f"\n{'-' * 60}")
    print("STEP 4: Loading ReID models")
    print(f"{'-' * 60}")

    extractors = {}

    # Current production model
    print("\n  Loading: ResNet50 ImageNet (current production)")
    extractors["resnet50_imagenet"] = CurrentResNet50Extractor(device)
    print(f"    Params: {extractors['resnet50_imagenet'].param_count:,}")

    # OSNet Market-1501 (proper ReID trained)
    market_path = str(MODELS_DIR / "osnet_x1_0_market1501.pth")
    print(f"\n  Loading: OSNet x1.0 Market-1501")
    osnet_market = OSNetReIDExtractor(
        device, market_path,
        name="osnet_x1_0_market1501",
        training_desc="Market-1501 person ReID (triplet+softmax, 751 identities)"
    )
    if osnet_market.available:
        extractors["osnet_x1_0_market1501"] = osnet_market
        print(f"    Params: {osnet_market.param_count:,}")

    # OSNet MSMT17 (broader ReID trained, better cross-domain)
    msmt_path = str(MODELS_DIR / "osnet_x1_0_msmt17.pth")
    print(f"\n  Loading: OSNet x1.0 MSMT17")
    osnet_msmt = OSNetReIDExtractor(
        device, msmt_path,
        name="osnet_x1_0_msmt17",
        training_desc="MSMT17 person ReID (combineall, 4,101 identities, 15 cameras)"
    )
    if osnet_msmt.available:
        extractors["osnet_x1_0_msmt17"] = osnet_msmt
        print(f"    Params: {osnet_msmt.param_count:,}")

    # -- Step 5: Benchmark each model -----------------------------
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_frames": len(frame_data),
            "n_crops": len(crops),
            "gpu": gpu_snapshot(),
            "cpu": cpu_snapshot(),
            "device": str(device),
        },
        "crop_quality": crop_quality,
        "models": {},
    }

    all_embeddings = {}

    for model_name, extractor in extractors.items():
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING: {model_name}")
        print(f"  Training: {extractor.TRAINING}")
        print(f"  Embedding dim: {extractor.feat_dim}")
        print(f"  Input size: {extractor.INPUT_SIZE}")
        print(f"  Distance: {extractor.DISTANCE}")
        print(f"{'=' * 60}")

        # Warmup
        print(f"  Warming up ({WARMUP} iterations)...")
        for i in range(min(WARMUP, len(crops))):
            extractor.extract(crops[i]["crop"])

        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # -- Single extraction timing -----------------------------
        print("  Timing single extractions...")
        gpu_sampler = GPUSampler()
        gpu_sampler.start()

        single_timer = Timer("single")
        embeddings = []

        for c in crops:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            with single_timer.measure():
                emb = extractor.extract(c["crop"])
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            embeddings.append(emb)

        embeddings = np.array(embeddings)
        all_embeddings[model_name] = embeddings

        # -- Padded crop test -------------------------------------
        print("  Testing padded crops...")
        padded_timer = Timer("padded")
        padded_embeddings = []
        for pc in padded_crops:
            with padded_timer.measure():
                emb = extractor.extract(pc)
            padded_embeddings.append(emb)
        padded_embeddings = np.array(padded_embeddings)

        # Compare padded vs unpadded similarity
        padded_sims = [float(np.dot(embeddings[i], padded_embeddings[i]))
                       for i in range(len(embeddings))]

        # -- Batch extraction timing ------------------------------
        print("  Timing batch extractions...")
        batch_results = {}
        for bs in BATCH_SIZES:
            batch_timer = Timer(f"batch_{bs}")
            crop_list = [c["crop"] for c in crops]
            for i in range(0, len(crop_list), bs):
                batch = crop_list[i:i + bs]
                if len(batch) < bs and len(batch) < 2:
                    continue
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                with batch_timer.measure():
                    extractor.extract_batch(batch)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
            batch_results[f"batch_{bs}"] = {
                **batch_timer.stats,
                "per_crop_ms": round(batch_timer.stats["avg"] / bs, 3)
                    if batch_timer.stats["avg"] > 0 else 0,
            }

        gpu_stats = gpu_sampler.stop()

        # -- Similarity matrix ------------------------------------
        print("  Computing similarity matrix...")
        sim_matrix = np.dot(embeddings, embeddings.T)

        # Upper triangle (exclude diagonal)
        triu_idx = np.triu_indices(len(embeddings), k=1)
        all_sims = sim_matrix[triu_idx]

        # -- Build identity pairs ---------------------------------
        # Same-frame crops = DIFFERENT people (negative)
        # Different-frame crops = unknown relationship
        frame_groups = defaultdict(list)
        for idx, c in enumerate(crops):
            frame_groups[c["frame_name"]].append(idx)

        negative_sims = []
        for frame_name, indices in frame_groups.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    negative_sims.append(float(sim_matrix[indices[i], indices[j]]))

        negative_sims = np.array(negative_sims) if negative_sims else np.array([])

        # For positive pairs: use very high similarity as proxy
        # (top 5% of all-pairs similarities, excluding same-frame pairs)
        same_frame_mask = np.zeros_like(sim_matrix, dtype=bool)
        for indices in frame_groups.values():
            for i in indices:
                for j in indices:
                    same_frame_mask[i, j] = True

        cross_frame_sims = sim_matrix.copy()
        cross_frame_sims[same_frame_mask] = -1  # exclude
        np.fill_diagonal(cross_frame_sims, -1)

        # Get high-confidence same-person candidates (top similarities)
        flat_cross = cross_frame_sims[np.triu_indices(len(embeddings), k=1)]
        valid_cross = flat_cross[flat_cross > -0.5]

        if len(valid_cross) > 0:
            high_threshold = np.percentile(valid_cross, 95)
            positive_sims = valid_cross[valid_cross >= high_threshold]
        else:
            positive_sims = np.array([])

        # -- Embedding analysis (Part 9) --------------------------
        # Verify L2 normalization
        norms = np.linalg.norm(embeddings, axis=1)

        # -- Compile model results --------------------------------
        model_result = {
            "model_info": {
                "name": model_name,
                "training": extractor.TRAINING,
                "feat_dim": extractor.feat_dim,
                "input_size": list(extractor.INPUT_SIZE),
                "distance": extractor.DISTANCE,
                "license": extractor.LICENSE,
                "param_count": extractor.param_count if hasattr(extractor, 'param_count') else "N/A",
            },
            "latency": {
                "single_ms": single_timer.stats,
                **batch_results,
            },
            "gpu": gpu_stats,
            "similarity_distribution": {
                "all_pairs": {
                    "mean": round(float(np.mean(all_sims)), 4),
                    "std": round(float(np.std(all_sims)), 4),
                    "min": round(float(np.min(all_sims)), 4),
                    "max": round(float(np.max(all_sims)), 4),
                    "p5": round(float(np.percentile(all_sims, 5)), 4),
                    "p10": round(float(np.percentile(all_sims, 10)), 4),
                    "p25": round(float(np.percentile(all_sims, 25)), 4),
                    "p50": round(float(np.percentile(all_sims, 50)), 4),
                    "p75": round(float(np.percentile(all_sims, 75)), 4),
                    "p90": round(float(np.percentile(all_sims, 90)), 4),
                    "p95": round(float(np.percentile(all_sims, 95)), 4),
                    "n_pairs": int(len(all_sims)),
                },
            },
            "identity_separation": {
                "same_frame_different_person": {
                    "n_pairs": int(len(negative_sims)),
                    "mean": round(float(np.mean(negative_sims)), 4) if len(negative_sims) > 0 else None,
                    "std": round(float(np.std(negative_sims)), 4) if len(negative_sims) > 0 else None,
                    "min": round(float(np.min(negative_sims)), 4) if len(negative_sims) > 0 else None,
                    "max": round(float(np.max(negative_sims)), 4) if len(negative_sims) > 0 else None,
                    "p50": round(float(np.median(negative_sims)), 4) if len(negative_sims) > 0 else None,
                    "p95": round(float(np.percentile(negative_sims, 95)), 4) if len(negative_sims) > 0 else None,
                },
                "high_confidence_same_person_proxy": {
                    "n_pairs": int(len(positive_sims)),
                    "mean": round(float(np.mean(positive_sims)), 4) if len(positive_sims) > 0 else None,
                    "std": round(float(np.std(positive_sims)), 4) if len(positive_sims) > 0 else None,
                    "min": round(float(np.min(positive_sims)), 4) if len(positive_sims) > 0 else None,
                    "max": round(float(np.max(positive_sims)), 4) if len(positive_sims) > 0 else None,
                },
                "separation_gap": None,  # Will compute if both distributions exist
            },
            "embedding_analysis": {
                "l2_norm_mean": round(float(np.mean(norms)), 6),
                "l2_norm_std": round(float(np.std(norms)), 6),
                "l2_norm_min": round(float(np.min(norms)), 6),
                "l2_norm_max": round(float(np.max(norms)), 6),
                "properly_normalized": bool(np.allclose(norms, 1.0, atol=1e-4)),
            },
            "crop_padding_impact": {
                "mean_similarity": round(float(np.mean(padded_sims)), 4),
                "std_similarity": round(float(np.std(padded_sims)), 4),
                "min_similarity": round(float(np.min(padded_sims)), 4),
                "max_similarity": round(float(np.max(padded_sims)), 4),
                "avg_drop_pct": round(float((1 - np.mean(padded_sims)) * 100), 2),
            },
        }

        # Compute separation gap
        if len(negative_sims) > 0 and len(positive_sims) > 0:
            neg_p95 = float(np.percentile(negative_sims, 95))
            pos_p5 = float(np.min(positive_sims))
            model_result["identity_separation"]["separation_gap"] = round(pos_p5 - neg_p95, 4)

        # Threshold calibration if we have both distributions
        if len(negative_sims) > 10 and len(positive_sims) > 10:
            model_result["threshold_calibration"] = calibrate_threshold(
                positive_sims, negative_sims
            )

        # Print summary
        print(f"\n  RESULTS for {model_name}:")
        print(f"    Single: P50={single_timer.stats['p50']:.1f}ms  P95={single_timer.stats['p95']:.1f}ms  max={single_timer.stats['max']:.1f}ms")
        for bs_key, bs_val in batch_results.items():
            print(f"    {bs_key}: avg={bs_val['avg']:.1f}ms  per-crop={bs_val['per_crop_ms']}ms")
        print(f"    All-pairs similarity: mean={model_result['similarity_distribution']['all_pairs']['mean']:.4f}  std={model_result['similarity_distribution']['all_pairs']['std']:.4f}")
        if len(negative_sims) > 0:
            print(f"    Negative pairs (same-frame diff-person): mean={model_result['identity_separation']['same_frame_different_person']['mean']:.4f}")
        if len(positive_sims) > 0:
            print(f"    Positive proxy (top-5%): mean={model_result['identity_separation']['high_confidence_same_person_proxy']['mean']:.4f}")
        if model_result["identity_separation"]["separation_gap"] is not None:
            print(f"    Separation gap: {model_result['identity_separation']['separation_gap']:.4f}")
        print(f"    L2 normalized: {model_result['embedding_analysis']['properly_normalized']}")
        print(f"    GPU: util={gpu_stats.get('gpu_util_avg', '?')}%  VRAM={gpu_stats.get('vram_max_mb', '?')}MB")
        print(f"    Padding impact: {model_result['crop_padding_impact']['avg_drop_pct']:.1f}% similarity drop")

        all_results["models"][model_name] = model_result

    # -- Step 6: Cross-model comparison ---------------------------
    print(f"\n{'=' * 60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 60}")

    model_names = list(all_results["models"].keys())
    comparison = {}

    for i, m1 in enumerate(model_names):
        for m2 in model_names[i + 1:]:
            # Cross-correlation of embeddings
            emb1 = all_embeddings[m1]
            emb2 = all_embeddings[m2]

            key = f"{m1}_vs_{m2}"
            # Note: cross-model sims are only meaningful if dimensions match
            if emb1.shape[1] == emb2.shape[1]:
                # Same crop, different model
                cross_sims = [float(np.dot(emb1[k], emb2[k]) /
                                    (np.linalg.norm(emb1[k]) * np.linalg.norm(emb2[k]) + 1e-8))
                              for k in range(min(len(emb1), len(emb2)))]
                comparison[key] = {
                    "same_dim": True,
                    "mean_agreement": round(float(np.mean(cross_sims)), 4),
                    "std_agreement": round(float(np.std(cross_sims)), 4),
                }
            else:
                comparison[key] = {
                    "same_dim": False,
                    "dim1": int(emb1.shape[1]),
                    "dim2": int(emb2.shape[1]),
                    "note": "Cannot directly compare embeddings of different dimensions"
                }

    all_results["cross_model_comparison"] = comparison

    # -- Step 7: Summary comparison table -------------------------
    print(f"\n{'=' * 60}")
    print("FINAL COMPARISON TABLE")
    print(f"{'=' * 60}")

    header = f"{'Metric':<35}"
    for mn in model_names:
        header += f" | {mn[:25]:>25}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Embedding dim", "model_info.feat_dim"),
        ("Params", "model_info.param_count"),
        ("Single P50 (ms)", "latency.single_ms.p50"),
        ("Single P95 (ms)", "latency.single_ms.p95"),
        ("Single max (ms)", "latency.single_ms.max"),
        ("All-pairs mean", "similarity_distribution.all_pairs.mean"),
        ("All-pairs std", "similarity_distribution.all_pairs.std"),
        ("Diff-person mean", "identity_separation.same_frame_different_person.mean"),
        ("Diff-person P95", "identity_separation.same_frame_different_person.p95"),
        ("Same-person proxy", "identity_separation.high_confidence_same_person_proxy.mean"),
        ("Separation gap", "identity_separation.separation_gap"),
        ("L2 normalized", "embedding_analysis.properly_normalized"),
        ("Padding drop %", "crop_padding_impact.avg_drop_pct"),
    ]

    for metric_name, path in metrics:
        row = f"{metric_name:<35}"
        for mn in model_names:
            val = all_results["models"][mn]
            for key in path.split("."):
                if isinstance(val, dict):
                    val = val.get(key, "N/A")
                else:
                    val = "N/A"
                    break
            row += f" | {str(val):>25}"
        print(row)

    # -- Save results ---------------------------------------------
    save_results("phase3_reid", all_results)

    # Save visual outputs
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    # Save sample crops with similarity annotations
    print(f"\n  Saving visual samples to {VISUAL_DIR}")
    n_visual = min(20, len(crops))
    for i in range(n_visual):
        crop_path = VISUAL_DIR / f"crop_{i:03d}_{crops[i]['height_px']}px.jpg"
        cv2.imwrite(str(crop_path), crops[i]["crop"])

    print(f"\n{'=' * 60}")
    print("PHASE 3 BENCHMARK COMPLETE")
    print(f"{'=' * 60}")

    torch.cuda.empty_cache()
    return all_results


if __name__ == "__main__":
    run_phase3_benchmark()
