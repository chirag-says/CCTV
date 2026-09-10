"""
SCRFD anchor decode and NMS in torch, on the GPU.

WHAT THIS REPLACES
insightface decodes SCRFD's nine output tensors in numpy, on the CPU, once per
image: build the anchor grid, turn 16,800 distance predictions into boxes,
threshold them, sort them, and run a Python-loop NMS. The engine call itself is
milliseconds on this card; the decode around it is the larger half, and it is
the part that holds the GIL while twenty camera threads wait for it.

Two things change here and nothing else:

  1. The nine outputs are bound straight into pre-allocated CUDA tensors with
     onnxruntime's IOBinding, so they are never copied to host memory. Before
     this, ORT copied roughly half a megabyte per frame to the CPU purely so
     numpy could look at it.
  2. Decode, threshold and NMS run in torch against those tensors. Only the
     surviving boxes and keypoints -- a handful of rows -- come back.

WHAT DOES NOT CHANGE
The arithmetic. This is a port, not a redesign, and it is deliberately a
literal one: same anchor construction, same distance2bbox and distance2kps,
same score threshold, same descending sort, same NMS threshold, and the same
`+1` box convention insightface's NMS uses (see _nms). A face this finds is a
face insightface found.

The letterbox stays on the CPU with cv2, bit-identical to insightface's, so
resampling cannot shift a detection by a pixel. That resize is the next thing
worth moving, not this one.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _nms(boxes, scores, thresh: float):
    """NMS with insightface's box convention.

    insightface measures a box as (x2 - x1 + 1) wide; torchvision measures it
    as (x2 - x1). On a 12-pixel face that is an 8 percent difference in every
    area and every intersection, which is enough to change whether a borderline
    pair suppresses. Shifting the far corner by one before the call makes the
    two identical rather than merely close, and costs an add.
    """
    from torchvision.ops import nms as tv_nms

    shifted = boxes.clone()
    shifted[:, 2:4] += 1.0
    return tv_nms(shifted, scores, thresh)


class GpuScrfd:
    """A drop-in replacement for `SCRFD.detect`, decoding on the GPU.

    Wraps the SCRFD model insightface already built, so the ONNX session, the
    execution provider, the thresholds and the input size are all whatever
    that model was prepared with. Only the code between "run the session" and
    "return boxes" is ours.
    """

    def __init__(self, model, device: str = "cuda:0"):
        import torch

        self.model = model
        self.device = torch.device(device)
        self.session = model.session
        self.input_name = model.input_name
        self.output_names = list(model.output_names)
        self.fmc = model.fmc
        self.strides = list(model._feat_stride_fpn)
        self.num_anchors = model._num_anchors
        self.use_kps = model.use_kps
        self.input_mean = model.input_mean
        self.input_std = model.input_std
        self.batched = model.batched

        # PER THREAD, not per decoder. In thread mode every camera calls this
        # same object concurrently, and both of these are written during a
        # call: an IOBinding accumulates the bindings it is given (two threads
        # binding nine outputs each produced "fetch names has 9 elements, but
        # fetch has 18"), and the output buffers are the memory ORT writes
        # results INTO -- sharing them means two cameras decoding each other's
        # frames, silently.
        self._local = threading.local()
        # Anchors are computed once and then only read, so they are shared. A
        # race here duplicates a grid, which is harmless.
        self._anchor_cache: Dict[Tuple[int, int, int], "torch.Tensor"] = {}

    # ── the pieces ────────────────────────────────────────────────────

    def _anchors(self, height: int, width: int, stride: int):
        """Anchor centres for one feature level, in input-image pixels.

        insightface builds these with np.mgrid and caches them; this is the
        same grid, on the device, cached the same way. `num_anchors` copies per
        cell, interleaved -- repeat_interleave, not repeat, because the
        predictions for a cell's anchors are adjacent rows.
        """
        import torch

        key = (height, width, stride)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached

        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self.device),
            torch.arange(width, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        centers = torch.stack((xs, ys), dim=-1).reshape(-1, 2) * float(stride)
        if self.num_anchors > 1:
            centers = centers.repeat_interleave(self.num_anchors, dim=0)
        self._anchor_cache[key] = centers
        return centers

    def _buffers(self, input_h: int, input_w: int):
        """CUDA tensors ORT writes its outputs into, allocated once per size."""
        import torch

        store = getattr(self._local, "buffers", None)
        if store is None:
            store = self._local.buffers = {}
        key = (input_h, input_w)
        bufs = store.get(key)
        if bufs is not None:
            return bufs

        # Output order is every stride's scores, then every stride's box
        # distances, then every stride's keypoint distances -- the order
        # `output_names` already has, and the order `_decode` indexes with
        # idx, idx+fmc, idx+2*fmc.
        cols_per_group = [1, 4] + ([10] if self.use_kps else [])
        bufs = []
        for cols in cols_per_group:
            for stride in self.strides:
                rows = (input_h // stride) * (input_w // stride) * self.num_anchors
                bufs.append(
                    torch.empty((rows, cols), dtype=torch.float32, device=self.device)
                )
        store[key] = bufs
        return bufs

    def _run_session(self, blob: np.ndarray):
        """One session call, outputs landing in CUDA memory.

        bind_cpu_input for the image (it comes from cv2 and has to be copied
        once either way) and bind_output straight into our tensors, so the nine
        result tensors never touch host memory.
        """
        input_h, input_w = blob.shape[2], blob.shape[3]
        bufs = self._buffers(input_h, input_w)

        io = getattr(self._local, "io", None)
        if io is None:
            io = self._local.io = self.session.io_binding()
        io.clear_binding_inputs()
        io.clear_binding_outputs()
        io.bind_cpu_input(self.input_name, blob)
        for name, buf in zip(self.output_names, bufs):
            io.bind_output(
                name,
                device_type="cuda",
                device_id=self.device.index or 0,
                element_type=np.float32,
                shape=tuple(buf.shape),
                buffer_ptr=buf.data_ptr(),
            )
        self.session.run_with_iobinding(io)
        return bufs

    def _decode(self, outs, input_h: int, input_w: int, det_scale: float, thresh: float):
        """Nine tensors in, (boxes, scores, kps) out. All on the device."""
        import torch

        fmc = self.fmc
        boxes_all, scores_all, kps_all = [], [], []

        for idx, stride in enumerate(self.strides):
            scores = outs[idx].reshape(-1)
            keep = scores >= thresh
            # No early `if not keep.any()` here on purpose. Reading that
            # predicate means synchronising with the device, three times a
            # frame, to skip work that is a handful of empty-tensor kernels.
            # The sync cost more than the work it avoided.

            height, width = input_h // stride, input_w // stride
            centers = self._anchors(height, width, stride)[keep]
            scores = scores[keep]

            # distance2bbox: the prediction is (left, top, right, bottom)
            # distances from the anchor centre, in stride units.
            dist = outs[idx + fmc][keep] * float(stride)
            boxes = torch.stack(
                (
                    centers[:, 0] - dist[:, 0],
                    centers[:, 1] - dist[:, 1],
                    centers[:, 0] + dist[:, 2],
                    centers[:, 1] + dist[:, 3],
                ),
                dim=-1,
            )
            boxes_all.append(boxes)
            scores_all.append(scores)

            if self.use_kps:
                # distance2kps: every keypoint is an offset from the SAME
                # centre, so the whole (N,5,2) block is one broadcast add.
                kps = outs[idx + fmc * 2][keep] * float(stride)
                kps = kps.reshape(-1, kps.shape[1] // 2, 2) + centers.unsqueeze(1)
                kps_all.append(kps)

        if not boxes_all:
            return None, None, None

        boxes = torch.cat(boxes_all) / det_scale
        scores = torch.cat(scores_all)
        kps = (torch.cat(kps_all) / det_scale) if kps_all else None

        # insightface sorts by score and then runs NMS. torchvision's NMS
        # sorts internally and returns its keep indices score-descending, so
        # the explicit sort is redundant -- and it was a full argsort over
        # every surviving anchor.
        keep = _nms(boxes, scores, self.model.nms_thresh)
        boxes, scores = boxes[keep], scores[keep]
        if kps is not None:
            kps = kps[keep]
        return boxes, scores, kps

    # ── the insightface entry point ───────────────────────────────────

    def detect(self, img, input_size=None, max_num: int = 0, metric: str = "default"):
        """Same signature, same return shapes, as SCRFD.detect."""
        import torch

        sizes = self.model._resolve_input_sizes(input_size)
        input_w, input_h = sizes[-1]

        # Letterbox exactly as insightface does: scale to fit, pad bottom-right
        # with zeros. Same cv2 call, same interpolation, same padding.
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_h) / input_w
        if im_ratio > model_ratio:
            new_height = input_h
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_w
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]

        resized = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self.input_std,
            (input_w, input_h),
            (self.input_mean,) * 3,
            swapRB=True,
        )

        outs = self._run_session(blob)

        with torch.no_grad():
            boxes, scores, kps = self._decode(
                outs, input_h, input_w, det_scale, self.model.det_thresh
            )

        if boxes is None or boxes.shape[0] == 0:
            empty_kps = np.empty((0, 5, 2), dtype=np.float32) if self.use_kps else None
            return np.empty((0, 5), dtype=np.float32), empty_kps

        # THE ONLY DEVICE-TO-HOST COPY: a handful of surviving rows.
        det = torch.cat((boxes, scores.unsqueeze(1)), dim=1).cpu().numpy()
        kpss = kps.cpu().numpy() if kps is not None else None

        if max_num > 0 and det.shape[0] > max_num:
            det, kpss = self._take_best(det, kpss, img, max_num, metric)
        return det, kpss

    def detect_gpu(self, frame):
        """detect(), for a frame that is already a CUDA tensor.

        (H, W, 3) uint8 RGB on the device, in, boxes and keypoints out. The
        frame is never copied to the host: the letterbox is a torch resize,
        the session input is bound from the tensor's own pointer, and the
        decode is the same GPU decode `detect` uses.

        Note the colour order. `detect` receives BGR from cv2 and lets
        blobFromImage(swapRB=True) flip it; NVDEC is asked for RGB directly,
        so the flip has already happened and doing it again would hand SCRFD
        inverted channels -- which does not error, it just detects worse.
        """
        import torch
        import torch.nn.functional as F

        sizes = self.model._resolve_input_sizes(None)
        input_w, input_h = sizes[-1]
        h, w = int(frame.shape[0]), int(frame.shape[1])

        # insightface's arithmetic, reproduced exactly -- including the
        # truncation. Rounding instead lands one pixel out, which shifts
        # det_scale and moves every decoded box by a fraction of a pixel.
        im_ratio = float(h) / float(w)
        model_ratio = float(input_h) / float(input_w)
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / h

        chw = frame.permute(2, 0, 1).unsqueeze(0).float()
        # antialias=False, deliberately. cv2.resize's INTER_LINEAR -- which is
        # what insightface uses -- does NOT prefilter on downscale, and torch's
        # antialiased bilinear approximates INTER_AREA instead. At the 0.24x
        # scale a 4 MP frame needs, that difference is not cosmetic: it blurs
        # away small faces and invents others, measured here as 20 detections
        # against the CPU path's 32.
        resized = F.interpolate(
            chw, size=(new_h, new_w), mode="bilinear",
            align_corners=False, antialias=False,
        )
        blob = torch.zeros(
            (1, 3, input_h, input_w), dtype=torch.float32, device=frame.device
        )
        blob[:, :, :new_h, :new_w] = resized
        blob = blob.sub_(self.input_mean).div_(self.input_std).contiguous()

        outs = self._run_session_gpu(blob)
        with torch.no_grad():
            boxes, scores, kps = self._decode(
                outs, input_h, input_w, det_scale, self.model.det_thresh
            )
        if boxes is None or boxes.shape[0] == 0:
            empty_kps = np.empty((0, 5, 2), dtype=np.float32) if self.use_kps else None
            return np.empty((0, 5), dtype=np.float32), empty_kps

        det = torch.cat((boxes, scores.unsqueeze(1)), dim=1).cpu().numpy()
        kpss = kps.cpu().numpy() if kps is not None else None
        return det, kpss

    def _run_session_gpu(self, blob):
        """Like _run_session, with the INPUT bound from device memory too."""
        import torch

        input_h, input_w = int(blob.shape[2]), int(blob.shape[3])
        bufs = self._buffers(input_h, input_w)

        io = getattr(self._local, "io", None)
        if io is None:
            io = self._local.io = self.session.io_binding()
        io.clear_binding_inputs()
        io.clear_binding_outputs()
        io.bind_input(
            self.input_name, "cuda", self.device.index or 0, np.float32,
            tuple(blob.shape), blob.data_ptr(),
        )
        for name, buf in zip(self.output_names, bufs):
            io.bind_output(
                name, device_type="cuda", device_id=self.device.index or 0,
                element_type=np.float32, shape=tuple(buf.shape),
                buffer_ptr=buf.data_ptr(),
            )
        self.session.run_with_iobinding(io)
        return bufs

    @staticmethod
    def _take_best(det, kpss, img, max_num: int, metric: str):
        """insightface's max_num selection, unchanged: biggest, or biggest and
        nearest the centre. Runs on a few rows, so it stays in numpy."""
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        img_center = img.shape[0] // 2, img.shape[1] // 2
        offsets = np.vstack([
            (det[:, 0] + det[:, 2]) / 2 - img_center[1],
            (det[:, 1] + det[:, 3]) / 2 - img_center[0],
        ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        values = area if metric == "max" else area - offset_dist_squared * 2.0
        bindex = np.argsort(values)[::-1][:max_num]
        return det[bindex, :], (kpss[bindex, :] if kpss is not None else None)


def attach(app, device: str = "cuda:0") -> bool:
    """Point a prepared FaceAnalysis at the GPU decoder. Never raises.

    Replaces the bound `detect` method on the detection model, so every caller
    -- FaceAnalysis.get, the pipeline, the inference server -- keeps working
    against the same interface and does not learn that anything changed.
    """
    model = app.models.get("detection") if hasattr(app, "models") else None
    if model is None:
        logger.warning("FACE_GPU_POSTPROCESS: no detection model to attach to")
        return False
    try:
        import torch
        from torchvision.ops import nms  # noqa: F401

        if not torch.cuda.is_available():
            logger.warning("FACE_GPU_POSTPROCESS: no CUDA device; keeping the numpy decode")
            return False

        if getattr(model, "batched", False):
            # The output buffers here are 2D, one row per anchor. A batched
            # SCRFD emits (1, N, C) and would need a third dimension bound.
            # buffalo_l's det_10g is not batched; refuse rather than guess.
            logger.warning(
                "FACE_GPU_POSTPROCESS: this SCRFD emits batched outputs, which "
                "the GPU decode does not bind; keeping the numpy path."
            )
            return False

        decoder = GpuScrfd(model, device=device)
        model.detect = decoder.detect
        app.det_model = model
        logger.info(
            f"FACE_GPU_POSTPROCESS: SCRFD decode and NMS on {device} "
            f"(strides={decoder.strides}, anchors={decoder.num_anchors}, "
            f"kps={decoder.use_kps})"
        )
        return True
    except Exception as e:                                      # noqa: BLE001
        logger.warning(
            f"FACE_GPU_POSTPROCESS: could not attach the GPU decode ({e}); "
            f"keeping insightface's numpy path."
        )
        return False
