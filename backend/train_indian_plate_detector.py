"""
Phase 3: Train a production-grade Indian license plate detector.

Uses the already-downloaded Roboflow Indian Number Plate dataset,
plus the existing HuggingFace keremberke dataset if available.

Architecture: YOLOv8m @ 1280px resolution.

Usage:
  py train_indian_plate_detector.py
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Check if indian_plates_2 source exists (already downloaded via Roboflow)
SOURCE_DIR = Path("./training/plate_indian_combined/sources/indian_plates_2")
COMBINED_DIR = Path("./training/plate_indian_combined")

# Also check existing plate_dataset_hf (from previous training script)
HF_SOURCE = Path("./training/plate_dataset_hf")

# Training config
TRAIN_CONFIG = {
    "model": "yolov8m.pt",
    "epochs": 100,
    "imgsz": 1280,
    "batch": 4,
    "patience": 20,
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "degrees": 10,
    "perspective": 0.001,
    "scale": 0.5,
    "shear": 5,
    "flipud": 0.0,
    "fliplr": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.4,
    "hsv_v": 0.4,
    "erasing": 0.3,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1: Build Combined Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def build_combined_dataset():
    """Combine all available plate datasets into a single training set."""
    combined_train_imgs = COMBINED_DIR / "train" / "images"
    combined_train_lbls = COMBINED_DIR / "train" / "labels"
    combined_val_imgs = COMBINED_DIR / "valid" / "images"
    combined_val_lbls = COMBINED_DIR / "valid" / "labels"

    for d in [combined_train_imgs, combined_train_lbls, combined_val_imgs, combined_val_lbls]:
        d.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0

    # Sources to merge
    sources = []

    # 1. Roboflow Indian plates (already downloaded)
    if SOURCE_DIR.exists():
        sources.append(("indian_rf", SOURCE_DIR))

    # 2. HuggingFace global plates (if previously downloaded)
    if HF_SOURCE.exists():
        sources.append(("global_hf", HF_SOURCE))

    if not sources:
        print("❌ No datasets found!")
        print(f"   Expected: {SOURCE_DIR}")
        print(f"   Or: {HF_SOURCE}")
        return None, 0

    for source_name, source_dir in sources:
        print(f"\n  Processing: {source_name} ({source_dir})")

        for split_type in ["train", "valid", "test"]:
            img_dir = source_dir / split_type / "images"
            lbl_dir = source_dir / split_type / "labels"

            if not img_dir.exists():
                continue

            # Determine destination (test → valid)
            if split_type == "train":
                dest_imgs = combined_train_imgs
                dest_lbls = combined_train_lbls
            else:
                dest_imgs = combined_val_imgs
                dest_lbls = combined_val_lbls

            count = 0
            for img_file in img_dir.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue

                # Find matching label
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if not lbl_file.exists():
                    continue

                # Validate and remap all class IDs to 0 (license-plate)
                try:
                    with open(lbl_file) as f:
                        lines = f.readlines()
                    if not lines:
                        continue

                    remapped = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            remapped.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                    if not remapped:
                        continue
                except Exception:
                    continue

                # Copy with unique prefix
                new_img_name = f"{source_name}_{img_file.name}"
                new_lbl_name = f"{source_name}_{img_file.stem}.txt"

                dest_img_path = dest_imgs / new_img_name
                dest_lbl_path = dest_lbls / new_lbl_name

                if not dest_img_path.exists():
                    shutil.copy2(img_file, dest_img_path)
                    with open(dest_lbl_path, "w") as f:
                        f.writelines(remapped)
                    count += 1

            if split_type == "train":
                total_train += count
            else:
                total_val += count
            
            if count > 0:
                print(f"    {split_type}: {count} images merged")

    # If we have too few validation images, move 20% of train to valid
    if total_val < 20 and total_train > 50:
        import random
        train_imgs = list(combined_train_imgs.glob("*.*"))
        random.shuffle(train_imgs)
        move_count = int(len(train_imgs) * 0.2)
        for img_file in train_imgs[:move_count]:
            lbl_file = combined_train_lbls / (img_file.stem + ".txt")
            shutil.move(str(img_file), str(combined_val_imgs / img_file.name))
            if lbl_file.exists():
                shutil.move(str(lbl_file), str(combined_val_lbls / lbl_file.name))
        total_train -= move_count
        total_val += move_count
        print(f"\n  Moved {move_count} images from train → valid for balance")

    # Count actual files
    actual_train = len(list(combined_train_imgs.glob("*.*")))
    actual_val = len(list(combined_val_imgs.glob("*.*")))
    print(f"\n  📊 Final dataset:")
    print(f"     Train: {actual_train} images")
    print(f"     Valid: {actual_val} images")
    print(f"     Total: {actual_train + actual_val} images")

    # Create data.yaml
    data_yaml = COMBINED_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {COMBINED_DIR.resolve()}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: license-plate\n")

    return data_yaml, actual_train + actual_val


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 2: Train
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(data_yaml, total_images):
    """Train YOLOv8m plate detector."""
    print("\n" + "=" * 60)
    print(f"  Training YOLOv8m Indian Plate Detector")
    print(f"  Dataset: {total_images} images @ {TRAIN_CONFIG['imgsz']}px")
    print(f"  Epochs: {TRAIN_CONFIG['epochs']} (patience={TRAIN_CONFIG['patience']})")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    model = YOLO(TRAIN_CONFIG["model"])

    results = model.train(
        data=str(data_yaml),
        epochs=TRAIN_CONFIG["epochs"],
        imgsz=TRAIN_CONFIG["imgsz"],
        batch=TRAIN_CONFIG["batch"],
        device=0,
        project="runs/detect/plates",
        name="indian_plate_v3",
        exist_ok=True,
        patience=TRAIN_CONFIG["patience"],
        lr0=TRAIN_CONFIG["lr0"],
        lrf=TRAIN_CONFIG["lrf"],
        warmup_epochs=TRAIN_CONFIG["warmup_epochs"],
        mosaic=TRAIN_CONFIG["mosaic"],
        mixup=TRAIN_CONFIG["mixup"],
        degrees=TRAIN_CONFIG["degrees"],
        perspective=TRAIN_CONFIG["perspective"],
        scale=TRAIN_CONFIG["scale"],
        shear=TRAIN_CONFIG["shear"],
        flipud=TRAIN_CONFIG["flipud"],
        fliplr=TRAIN_CONFIG["fliplr"],
        hsv_h=TRAIN_CONFIG["hsv_h"],
        hsv_s=TRAIN_CONFIG["hsv_s"],
        hsv_v=TRAIN_CONFIG["hsv_v"],
        erasing=TRAIN_CONFIG["erasing"],
        save=True,
        plots=True,
        verbose=True,
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 3: Deploy
# ═══════════════════════════════════════════════════════════════════════════════

def deploy_model():
    """Copy best weights and print instructions."""
    src = Path("runs/detect/plates/indian_plate_v3/weights/best.pt")
    dst = Path("./models/indian_plate_detector_v3.pt")

    if not src.exists():
        print(f"  ❌ Best weights not found at {src}")
        return False

    os.makedirs("./models", exist_ok=True)
    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)

    test_model = YOLO(str(dst))
    print(f"\n  ✅ Model deployed: {dst} ({size_mb:.1f} MB)")
    print(f"  Classes: {test_model.names}")
    print(f"\n  To use this model, update your .env:")
    print(f"     ANPR_MODEL_PATH=./models/indian_plate_detector_v3.pt")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PHASE 3: Indian License Plate Detector Training")
    print("=" * 60)

    # Step 1: Build dataset
    print("\n📦 Step 1: Building combined dataset...")
    data_yaml, total = build_combined_dataset()

    if not data_yaml or total < 50:
        print(f"\n❌ Only {total} images. Need at least 50.")
        sys.exit(1)

    # Step 2: Train
    print(f"\n🏋️ Step 2: Training on {total} images...")
    train_model(data_yaml, total)

    # Step 3: Deploy
    print("\n🚀 Step 3: Deploying best model...")
    success = deploy_model()

    if success:
        print("\n" + "=" * 60)
        print("  ✅ PHASE 3 COMPLETE!")
        print("  Restart backend to use the new plate detector.")
        print("=" * 60)
