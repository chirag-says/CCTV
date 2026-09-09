"""
Cameras per RTX 5070 by plan and analysed FPS, for three software states.
Basis: GTX 1650 measurements (bench_*.json + TensorRT run), 5070 = 6x GTX 1650.
    python capacity_table.py
"""
GPU_X = 6.0      # RTX 5070 vs GTX 1650 (to be replaced by tomorrow's measurement)
UTIL = 0.70
DECODE_CAP_720 = 80   # one NVDEC on the 5070, 720p30 streams
DECODE_CAP_1080 = 35
FACE_FPS, ANPR_FPS, WEAPON_FPS, REID_PER_S = 5, 2, 1, 2   # fixed rates, do not scale with tracking fps

# ms per frame on GTX 1650 -> 5070; three software states
# person: measured PyTorch b8 17.06, TensorRT b16 end-to-end 2.93; Rust/C++ removes Python pre/post (~1 ms of the 2.93)
today = {"person": 17.06, "face": 34.8, "anpr": 46.3, "weapon": 97.6, "reid": 27.4}
py_trt = {"person": 2.93, "face": 34.8/3, "anpr": 46.3/3, "weapon": 97.6/4, "reid": 27.4/3}
rust_trt = {k: v * (0.66 if k == "person" else 0.8) for k, v in py_trt.items()}
CPU_MS = {"today": 4.0, "py_trt": 2.0, "rust_trt": 0.3}   # CPU ms per tracked frame (decode copy, pre/post, tracker)

def cams(state, tier, fps):
    s = {"today": today, "py_trt": py_trt, "rust_trt": rust_trt}[state]
    load = s["person"] * fps
    if tier in ("Plus", "Pro"): load += s["face"] * FACE_FPS
    if tier == "Pro": load += s["anpr"] * ANPR_FPS + s["weapon"] * WEAPON_FPS + s["reid"] * REID_PER_S
    n = 1000 * UTIL / (load / GPU_X)
    return n

print(f"Cameras per RTX 5070 (70% util). Decode ceiling per GPU: {DECODE_CAP_720} @720p, {DECODE_CAP_1080} @1080p.\n")
print(f"{'plan':6s} {'fps':>4s} | {'today':>7s} {'Py+TRT':>7s} {'Rust+TRT':>8s} | {'CPU cores/GPU Py':>16s} {'Rust':>5s}")
for tier in ("Basic", "Plus", "Pro"):
    for fps in (5, 10, 15, 25, 30):
        a, b, c = (cams(s, tier, fps) for s in ("today", "py_trt", "rust_trt"))
        cap = lambda x: min(x, DECODE_CAP_720)
        py_cores = cap(b) * fps * CPU_MS["py_trt"] / 1000
        rs_cores = cap(c) * fps * CPU_MS["rust_trt"] / 1000
        print(f"{tier:6s} {fps:4d} | {a:7.1f} {cap(b):7.1f} {cap(c):8.1f} | {py_cores:16.1f} {rs_cores:5.1f}")
    print()
