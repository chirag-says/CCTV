"""
SentinelAI per-stage GPU benchmark.

Measures the inference cost of every model the pipeline can run, in
milliseconds per call, on the GPU in this machine. Output feeds the
per-camera cost model. Run on any machine with the backend env installed:

    cd backend && python bench_pipeline.py [--video uploads/videos/<file>.mp4]

Writes bench_<gpu-name>.json next to this file and prints a table.
"""
import argparse, json, os, sys, time, platform, statistics as st
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import numpy as np
import cv2
import torch  # preloads CUDA/cuDNN DLLs for onnxruntime on Windows

WARM, ITERS = 10, 60

def timeit(fn, iters=ITERS, warm=WARM):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": round(st.mean(ts), 2), "p50_ms": round(st.median(ts), 2),
            "p95_ms": round(sorted(ts)[int(0.95 * len(ts)) - 1], 2)}

def vram_mb():
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 2**20)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(0)
    res = {"gpu": gpu, "vram_total_mb": round(torch.cuda.mem_get_info()[1] / 2**20),
           "torch": torch.__version__, "python": platform.python_version(),
           "iters": ITERS, "stages": {}, "vram_after_mb": {}}
    S = res["stages"]
    print(f"GPU: {gpu}  VRAM {res['vram_total_mb']} MB  torch {torch.__version__}")

    # ── frames ────────────────────────────────────────────────────────
    vid = args.video
    if vid is None:
        import glob
        c = sorted(glob.glob("uploads/videos/*.mp4"))
        vid = c[0] if c else None
    frame720 = None
    if vid:
        cap = cv2.VideoCapture(vid)
        ok, f = cap.read()
        if ok:
            frame720 = cv2.resize(f, (1280, 720))
            # decode cost (CPU), ms per frame
            n, t0 = 0, time.perf_counter()
            while n < 150:
                ok, _ = cap.read()
                if not ok: break
                n += 1
            S["decode_720p_h264_cpu"] = {"mean_ms": round((time.perf_counter() - t0) * 1000 / max(n, 1), 2)}
        cap.release()
    if frame720 is None:
        frame720 = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
    frame1080 = cv2.resize(frame720, (1920, 1080))
    S["resize_1080_to_720_cpu"] = timeit(lambda: cv2.resize(frame1080, (1280, 720), interpolation=cv2.INTER_AREA))
    S["jpeg_encode_720p_cpu"] = timeit(lambda: cv2.imencode(".jpg", frame720, [cv2.IMWRITE_JPEG_QUALITY, 80]))

    # ── InsightFace: SCRFD detection + ArcFace embedding (ONNX Runtime) ─
    try:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        assert "CUDAExecutionProvider" in ort.get_available_providers(), "ORT has no CUDA provider"
        for det in (640, 1024):
            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                               allowed_modules=["detection"])
            app.prepare(ctx_id=0, det_size=(det, det))
            S[f"scrfd_det10g_{det}"] = timeit(lambda: app.det_model.detect(frame720, max_num=0, metric="default"))
        res["vram_after_mb"]["scrfd"] = vram_mb()
        from insightface.model_zoo import get_model
        rec = get_model(os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx"),
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        rec.prepare(ctx_id=0)
        face = np.random.randint(0, 255, (112, 112, 3), np.uint8)
        S["arcface_r50_per_face"] = timeit(lambda: rec.get_feat(face))
        res["vram_after_mb"]["arcface"] = vram_mb()
    except Exception as e:
        S["insightface_error"] = str(e)[:300]

    # ── Ultralytics YOLO (person / hazard / plate detector class) ─────
    try:
        from ultralytics import YOLO
        for w in ("yolov8n.pt", "yolo11m.pt"):
            if not os.path.exists(w): continue
            m = YOLO(w)
            m.predict(frame720, imgsz=640, half=True, verbose=False, device=0)
            S[f"{w[:-3]}_640_fp16_b1"] = timeit(lambda: m.predict(frame720, imgsz=640, half=True, verbose=False, device=0))
            batch = [frame720] * 8
            r = timeit(lambda: m.predict(batch, imgsz=640, half=True, verbose=False, device=0), iters=20)
            S[f"{w[:-3]}_640_fp16_b8_per_frame"] = {k: round(v / 8, 2) for k, v in r.items()}
            res["vram_after_mb"][w[:-3]] = vram_mb()
    except Exception as e:
        S["yolo_error"] = str(e)[:300]

    # ── RF-DETR (hazard / weapons detector) ───────────────────────────
    try:
        from rfdetr import RFDETRBase
        try:
            from rfdetr import RFDETRNano
        except ImportError:
            RFDETRNano = None
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(frame720, cv2.COLOR_BGR2RGB))
        for name, cls in (("rfdetr_nano", RFDETRNano), ("rfdetr_base", RFDETRBase)):
            if cls is None: continue
            try:
                m = cls()
                m.predict(pil, threshold=0.4)
                S[f"{name}_b1"] = timeit(lambda: m.predict(pil, threshold=0.4), iters=30)
                res["vram_after_mb"][name] = vram_mb()
                del m; torch.cuda.empty_cache()
            except Exception as e:
                S[f"{name}_error"] = str(e)[:200]
    except Exception as e:
        S["rfdetr_error"] = str(e)[:300]

    # ── ReID ResNet50 (as in app/vision/reid.py) ───────────────────────
    try:
        import torch.nn as nn
        from torchvision import models
        bb = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        net = nn.Sequential(*list(bb.children())[:-1]).cuda().eval().half()
        x1 = torch.randn(1, 3, 256, 128, device="cuda").half()
        x8 = torch.randn(8, 3, 256, 128, device="cuda").half()
        with torch.no_grad():
            S["reid_resnet50_b1"] = timeit(lambda: net(x1))
            r = timeit(lambda: net(x8))
            S["reid_resnet50_b8_per_crop"] = {k: round(v / 8, 2) for k, v in r.items()}
        res["vram_after_mb"]["reid"] = vram_mb()
    except Exception as e:
        S["reid_error"] = str(e)[:300]

    # ── EasyOCR plate read ─────────────────────────────────────────────
    try:
        import easyocr
        rd = easyocr.Reader(["en"], gpu=True, verbose=False)
        plate = np.full((60, 200, 3), 255, np.uint8)
        cv2.putText(plate, "KA01AB1234", (5, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        S["easyocr_plate_crop"] = timeit(lambda: rd.readtext(plate), iters=30)
        res["vram_after_mb"]["easyocr"] = vram_mb()
    except Exception as e:
        S["easyocr_error"] = str(e)[:300]

    res["vram_peak_mb"] = vram_mb()
    out = f"bench_{gpu.replace(' ', '_').replace('/', '_')}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n{'stage':40s} {'mean ms':>9s} {'p50':>8s} {'p95':>8s}")
    for k, v in S.items():
        if "mean_ms" in v:
            print(f"{k:40s} {v['mean_ms']:9.2f} {v.get('p50_ms', 0):8.2f} {v.get('p95_ms', 0):8.2f}")
        else:
            print(f"{k:40s} ERROR {v}")
    print("VRAM after each load (MB):", res["vram_after_mb"])
    print("wrote", out)

if __name__ == "__main__":
    main()
