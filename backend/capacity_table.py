"""
Cameras per RTX 5070 by plan and analysed FPS, for three software states.
Reads the measured bench JSON (default: the RTX 5070 run of 2026-09-09).
    python capacity_table.py [bench.json]
"""
import json, sys

UTIL = 0.70
DECODE_CAP_720 = 80   # one NVDEC on the 5070, 720p30 streams
DECODE_CAP_1080 = 35
FACE_FPS, ANPR_FPS, WEAPON_FPS, REID_PER_S = 5, 2, 1, 2   # fixed rates, do not scale with tracking fps

# Speedups applied on top of the measured "today" numbers. On the 5070 the PyTorch YOLO path
# already uses fp16 tensor cores, so TensorRT gains less than the 5.3x measured on the 1650.
# The face stack runs in ONNX Runtime fp32 and gained only 3x from the 1650 to the 5070,
# so it has the most headroom.
TRT_GAIN = {"person": 1.92, "face": 4.0, "anpr": 2.5, "weapon": 2.0, "reid": 1.5}   # person measured on 5070 (2.09 -> 1.09 ms); rest estimates
RUST_GAIN = {"person": 1.5, "face": 1.25, "anpr": 1.25, "weapon": 1.25, "reid": 1.25}
CPU_MS = {"today": 4.0, "py_trt": 2.0, "rust_trt": 0.3}   # CPU ms per tracked frame

def load(path):
    b = json.load(open(path)); S = {k: v["mean_ms"] for k, v in b["stages"].items()}
    today = {"person": S["yolov8n_640_fp16_b8_per_frame"],
             "face": S["scrfd_det10g_640"] + S["arcface_r50_per_face"],
             "anpr": S["yolov8n_640_fp16_b8_per_frame"] + S["easyocr_plate_crop"],
             "weapon": S["yolo11m_640_fp16_b8_per_frame"],
             "reid": S["reid_resnet50_b8_per_crop"]}
    py_trt = {k: v / TRT_GAIN[k] for k, v in today.items()}
    rust = {k: v / RUST_GAIN[k] for k, v in py_trt.items()}
    return b["gpu"], {"today": today, "py_trt": py_trt, "rust_trt": rust}

def cams(s, tier, fps):
    load_ms = s["person"] * fps
    if tier in ("Plus", "Pro"): load_ms += s["face"] * FACE_FPS
    if tier == "Pro": load_ms += s["anpr"] * ANPR_FPS + s["weapon"] * WEAPON_FPS + s["reid"] * REID_PER_S
    return 1000 * UTIL / load_ms

def table(path="bench_NVIDIA_GeForce_RTX_5070.json", res=720):
    gpu, states = load(path)
    cap = DECODE_CAP_720 if res == 720 else DECODE_CAP_1080
    out = {}
    for tier in ("Basic", "Plus", "Pro"):
        for fps in (5, 10, 15, 25, 30):
            out[(tier, fps)] = {st: min(cams(states[st], tier, fps), cap) for st in states}
    return gpu, out

if __name__ == "__main__":
    gpu, out = table(sys.argv[1] if len(sys.argv) > 1 else "bench_NVIDIA_GeForce_RTX_5070.json")
    print(f"Cameras per {gpu} (70% util), decode cap {DECODE_CAP_720} @720p\n")
    print(f"{'plan':6s} {'fps':>4s} | {'today':>7s} {'Py+TRT':>7s} {'Rust+TRT':>8s} | {'CPU cores Py':>12s} {'Rust':>5s}")
    for (tier, fps), r in out.items():
        print(f"{tier:6s} {fps:4d} | {r['today']:7.1f} {r['py_trt']:7.1f} {r['rust_trt']:8.1f} | "
              f"{r['py_trt']*fps*CPU_MS['py_trt']/1000:12.1f} {r['rust_trt']*fps*CPU_MS['rust_trt']/1000:5.1f}")
        if fps == 30: print()
