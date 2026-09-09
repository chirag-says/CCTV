"""
TensorRT benchmark for the two stages that decide capacity: person detection (YOLOv8n)
and the face stack (SCRFD det_10g + ArcFace w600k_r50). Compares the pipeline's current
paths (PyTorch fp16 / ONNX Runtime CUDA fp32) against TensorRT fp16 on the same GPU.

    pip install "tensorrt-cu12==10.13.3.9"
    python bench_trt.py

First run builds engines (2 to 10 minutes). Writes bench_trt_<gpu>.json.
"""
import os, sys, time, json, glob, platform, statistics as st
import numpy as np, cv2, torch

def median_ms(fn, iters=40, warm=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
    return st.median(ts)

def add_trt_dlls():
    for name in ("tensorrt_libs", "tensorrt_cu12_libs"):
        try:
            m = __import__(name); d = os.path.dirname(m.__file__)
            os.add_dll_directory(d); os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]; return d
        except Exception: pass
    return None

def main():
    gpu = torch.cuda.get_device_name(0); out = {"gpu": gpu, "results": {}}; R = out["results"]
    print("GPU:", gpu, "| TensorRT libs:", add_trt_dlls())
    frame = np.random.randint(0, 255, (720, 1280, 3), np.uint8)

    # ── YOLOv8n: PyTorch fp16 vs TensorRT fp16, batch 1/8/16 ─────────────
    from ultralytics import YOLO
    if not os.path.exists("yolov8n.engine"):
        print("building yolov8n.engine ...")
        YOLO("yolov8n.pt").export(format="engine", half=True, dynamic=True, batch=16, imgsz=640, device=0, verbose=False)
    pt, eng = YOLO("yolov8n.pt"), YOLO("yolov8n.engine", task="detect")
    for b in (1, 8, 16):
        imgs = [frame] * b
        p = median_ms(lambda: pt.predict(imgs, imgsz=640, half=True, verbose=False, device=0)) / b
        e = median_ms(lambda: eng.predict(imgs, imgsz=640, half=True, verbose=False, device=0)) / b
        R[f"yolov8n_b{b}"] = {"pytorch_fp16_ms": round(p, 2), "tensorrt_fp16_ms": round(e, 2), "speedup": round(p / e, 2)}
        print(f"yolov8n batch {b:2d}: PyTorch {p:6.2f}  TensorRT {e:6.2f}  ({p/e:.1f}x)")

    # ── Face stack: ORT CUDA fp32 (current) vs ORT TensorRT EP fp16 ───────
    try:
        import onnxruntime as ort
        home = os.path.expanduser("~/.insightface/models/buffalo_l")
        det_path, rec_path = os.path.join(home, "det_10g.onnx"), os.path.join(home, "w600k_r50.onnx")
        cuda = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        trt = [("TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                              "trt_engine_cache_path": "./trt_cache", "trt_profile_min_shapes": "input.1:1x3x640x640",
                                              "trt_profile_max_shapes": "input.1:1x3x640x640", "trt_profile_opt_shapes": "input.1:1x3x640x640"}),
               "CUDAExecutionProvider", "CPUExecutionProvider"]
        os.makedirs("./trt_cache", exist_ok=True)
        det_in = np.random.rand(1, 3, 640, 640).astype(np.float32)
        rec_in = np.random.rand(1, 3, 112, 112).astype(np.float32)
        for label, path, x in (("scrfd_640", det_path, det_in), ("arcface", rec_path, rec_in)):
            s_cuda = ort.InferenceSession(path, providers=cuda)
            name = s_cuda.get_inputs()[0].name
            c = median_ms(lambda: s_cuda.run(None, {name: x}))
            try:
                opts = dict(trt[0][1]); 
                if label == "arcface": opts = {k: v.replace("1x3x640x640", "1x3x112x112").replace("input.1", name) for k, v in opts.items() if isinstance(v, str)} | {"trt_fp16_enable": True, "trt_engine_cache_enable": True}
                else: opts = {k: (v.replace("input.1", name) if isinstance(v, str) else v) for k, v in opts.items()}
                print(f"building TensorRT engine for {label} (first time is slow) ...")
                s_trt = ort.InferenceSession(path, providers=[("TensorrtExecutionProvider", opts)] + cuda)
                assert "TensorrtExecutionProvider" in s_trt.get_providers()
                t = median_ms(lambda: s_trt.run(None, {name: x}))
                R[label] = {"ort_cuda_fp32_ms": round(c, 2), "ort_tensorrt_fp16_ms": round(t, 2), "speedup": round(c / t, 2)}
                print(f"{label:10s}: ORT CUDA fp32 {c:6.2f}  TensorRT fp16 {t:6.2f}  ({c/t:.1f}x)")
            except Exception as ex:
                R[label] = {"ort_cuda_fp32_ms": round(c, 2), "tensorrt_error": str(ex)[:200]}
                print(f"{label:10s}: ORT CUDA fp32 {c:6.2f}  TensorRT failed: {str(ex)[:120]}")
    except Exception as ex:
        R["face_error"] = str(ex)[:300]; print("face stack benchmark failed:", str(ex)[:200])

    fn = f"bench_trt_{gpu.replace(' ', '_')}.json"
    json.dump(out, open(fn, "w"), indent=2); print("wrote", fn)

if __name__ == "__main__":
    main()
