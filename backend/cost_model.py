"""
Per-camera compute cost model for SentinelAI.

Reads a bench_<gpu>.json produced by bench_pipeline.py and prints
per-camera cost per hour / day / month for each product tier on each
hosting option. Every input is listed at the top so it can be changed.

    python cost_model.py bench_NVIDIA_GeForce_GTX_1650.json
"""
import json, sys

INR_PER_USD = 94.5
UTIL = 0.70            # target GPU utilisation (headroom for bursts)
FPS = 5                # analysed frames per second per camera (substream)
FACES_PER_FRAME = 1.0  # avg ArcFace embeddings per analysed frame (recognition cache limits this)
ANPR_FPS = 2           # plate detect + OCR rate on gate cameras
WEAPON_FPS = 1         # heavy detector rate
REID_CROPS_PER_SEC = 2
HOURS_PER_MONTH = 730

# GPU speed relative to the benchmark GPU (GTX 1650, PyTorch/ORT, no TensorRT).
# Engineering estimates from public YOLO/CNN benchmarks; replace with measured
# ratios by running bench_pipeline.py on the target GPU.
GPU_FACTOR = {"GTX 1650": 1.0, "T4": 2.5, "L4": 5.0, "RTX 4060": 3.0, "RTX 4090": 8.0, "RTX 5070": 6.0}

CLOUD = [  # name, gpu, usd/hr, note
    ("AWS Mumbai g4dn.xlarge", "T4", 0.579, "on-demand, ap-south-1, data stays in India"),
    ("AWS Mumbai g6.xlarge", "L4", 0.966, "on-demand, ap-south-1, data stays in India"),
    ("RunPod secure cloud L4", "L4", 0.39, "no India region, no egress fee"),
    ("RunPod secure cloud RTX 4090", "RTX 4090", 0.69, "no India region, consumer card"),
]
EDGE = [  # name, gpu, hardware INR, amortise months, avg watts, INR per kWh
    ("RTX 4060 small-form-factor PC", "RTX 4060", 110_000, 36, 180, 9.0),
]

def tier_loads(bench):
    """GPU-milliseconds consumed per second of video, per camera, per tier, on the benchmark GPU."""
    S = {k: v["mean_ms"] for k, v in bench["stages"].items() if "mean_ms" in v}
    person = S["yolov8n_640_fp16_b8_per_frame"] * FPS
    face = (S["scrfd_det10g_640"] + S["arcface_r50_per_face"] * FACES_PER_FRAME) * FPS
    anpr = (S["yolov8n_640_fp16_b8_per_frame"] + S["easyocr_plate_crop"]) * ANPR_FPS
    weapon = S["yolo11m_640_fp16_b8_per_frame"] * WEAPON_FPS
    reid = S["reid_resnet50_b8_per_crop"] * REID_CROPS_PER_SEC
    return {
        "Basic  (people, tracking, zones, loitering, crowd)": person,
        "Plus   (Basic + face recognition)": person + face,
        "Pro    (Plus + ANPR + weapons + cross-camera ReID)": person + face + anpr + weapon + reid,
    }

def cams_per_gpu(ms_per_s, gpu):
    return 1000.0 * UTIL / (ms_per_s / GPU_FACTOR[gpu])

def main(path):
    b = json.load(open(path))
    print(f"Benchmark GPU: {b['gpu']}   (analysed {FPS} fps/camera, {int(UTIL*100)}% utilisation)\n")
    tiers = tier_loads(b)
    print("GPU load per camera on benchmark GPU (ms of GPU time per second of video):")
    for t, ms in tiers.items():
        print(f"  {t:55s} {ms:7.1f} ms/s")
    print()

    print("CLOUD tier (per camera, 24x7)")
    print(f"{'host':32s} {'tier':8s} {'cams/GPU':>9s} {'INR/hr':>8s} {'INR/day':>8s} {'INR/month':>10s}")
    for name, gpu, usd, note in CLOUD:
        for t, ms in tiers.items():
            n = cams_per_gpu(ms, gpu)
            hr = usd * INR_PER_USD / n
            print(f"{name:32s} {t[:6]:8s} {n:9.1f} {hr:8.2f} {hr*24:8.0f} {hr*HOURS_PER_MONTH:10.0f}")
        print(f"{'':32s} {note}")
    print()
    print("EDGE tier (box at customer site, per camera, 24x7)")
    print(f"{'box':32s} {'tier':8s} {'cams/box':>9s} {'INR/hr':>8s} {'INR/day':>8s} {'INR/month':>10s}")
    for name, gpu, hw, months, watts, tariff in EDGE:
        box_month = hw / months + watts / 1000 * 24 * 30 * tariff
        for t, ms in tiers.items():
            n = cams_per_gpu(ms, gpu)
            m = box_month / n
            print(f"{name:32s} {t[:6]:8s} {n:9.1f} {m/HOURS_PER_MONTH:8.2f} {m/30:8.0f} {m:10.0f}")
        print(f"{'':32s} box cost {box_month:,.0f} INR/month = hardware {hw/months:,.0f} + power {watts/1000*24*30*tariff:,.0f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench_NVIDIA_GeForce_GTX_1650.json")
