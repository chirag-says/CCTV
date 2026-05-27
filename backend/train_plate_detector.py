"""
Download license plate dataset from HuggingFace and train YOLOv8s plate detector.
"""
import os, json, shutil
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from ultralytics import YOLO

DATASET_DIR = Path("./training/plate_dataset_hf")

# Step 1: Download from HuggingFace
print("Downloading license plate dataset from HuggingFace...")
ds = load_dataset("keremberke/license-plate-object-detection", name="full")

print(f"Train: {len(ds['train'])} images")
print(f"Val: {len(ds['validation'])} images")
print(f"Test: {len(ds['test'])} images")

# Step 2: Convert to YOLO format
for split_name, split_key in [("train", "train"), ("valid", "validation"), ("test", "test")]:
    img_dir = DATASET_DIR / split_name / "images"
    lbl_dir = DATASET_DIR / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    split_data = ds[split_key]
    for i, example in enumerate(split_data):
        # Save image
        img = example["image"]
        img_path = img_dir / f"{i:06d}.jpg"
        img.save(str(img_path))
        
        # Convert bbox to YOLO format
        w, h = img.size
        objects = example["objects"]
        lbl_path = lbl_dir / f"{i:06d}.txt"
        
        with open(lbl_path, "w") as f:
            for bbox, cat in zip(objects["bbox"], objects["category"]):
                # bbox is [x_min, y_min, x_max, y_max] in absolute pixels
                x_min, y_min, x_max, y_max = bbox
                # Convert to YOLO: [class, x_center, y_center, width, height] normalized
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                bw = (x_max - x_min) / w
                bh = (y_max - y_min) / h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
        
        if (i + 1) % 100 == 0:
            print(f"  {split_name}: {i+1}/{len(split_data)}")
    
    print(f"  {split_name}: {len(split_data)} images saved")

# Step 3: Create data.yaml
data_yaml = DATASET_DIR / "data.yaml"
with open(data_yaml, "w") as f:
    f.write(f"path: {DATASET_DIR.resolve()}\n")
    f.write("train: train/images\n")
    f.write("val: valid/images\n")
    f.write("test: test/images\n\n")
    f.write("nc: 1\n")
    f.write("names:\n")
    f.write("  0: license-plate\n")

print(f"\ndata.yaml created: {data_yaml}")

# Step 4: Train
print("\n" + "="*60)
print("  Training YOLOv8s Plate Detector")
print("  30 epochs, patience=10")
print("="*60)

model = YOLO("yolov8s.pt")

results = model.train(
    data=str(data_yaml),
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/detect/plates",
    name="plate_detector_v2",
    exist_ok=True,
    patience=10,
    lr0=0.01,
    lrf=0.001,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.1,
    save=True,
    plots=True,
    verbose=True,
)

print("\n" + "="*60)
print("  Training Complete!")
print("="*60)

# Step 5: Deploy
src = "runs/detect/plates/plate_detector_v2/weights/best.pt"
dst = "./models/plate_detector_v2.pt"

if os.path.exists(src):
    os.makedirs("./models", exist_ok=True)
    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"  ✅ Plate detector deployed: {dst} ({size_mb:.1f} MB)")
    
    # Quick test
    test_model = YOLO(dst)
    print(f"  Classes: {test_model.names}")
else:
    print(f"  ❌ Best weights not found at {src}")
