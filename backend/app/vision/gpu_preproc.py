"""
Letterbox and face alignment in torch, on the GPU.

WHY THIS EXISTS
Phase 4 step 2 asks for the frame to stay on the GPU from decode to NMS. Two
things stood between those, and both were written in terms of numpy arrays:

  * the letterbox every detector wants -- scale to fit, pad to square,
    normalise, HWC to CHW;
  * the similarity transform ArcFace wants -- rotate and scale a face onto
    five canonical landmark positions.

Both are done here against CUDA tensors, so a decoded frame is never asked to
become a numpy array on the way to a model.

FIDELITY
These replace cv2.resize and cv2.warpAffine, and both are resampling
operations, so "equivalent" has to be measured rather than asserted:

  * letterbox uses bilinear without align_corners and WITHOUT antialias,
    which is what cv2.resize's INTER_LINEAR does. Torch's antialiased
    bilinear is closer to INTER_AREA, and substituting it is not a free
    upgrade: measured on the face path it changed 32 detections into 29.
  * align_faces_gpu inverts the same 2x3 affine insightface computes, so the
    geometry is identical by construction and only the sampling differs.

tests/test_gpu_preproc.py holds both to the CPU versions, and the ArcFace
embeddings that come out of them to cosine distance from the CPU path -- which
is the number that actually decides whether a face is recognised.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ArcFace's canonical five points for a 112x112 crop. Identical to
# insightface.utils.face_align.arcface_dst -- copied rather than imported so
# this module does not depend on insightface's internals being public.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def letterbox_gpu(frame, size: int = 640, half: bool = False):
    """(H, W, 3) uint8 RGB on the GPU -> (1, 3, size, size) float, plus geometry.

    Returns (batched_chw, scale, pad_x, pad_y). Boxes the model produces are in
    the padded square's coordinates; `unletterbox` takes them back.

    Padding goes bottom and right only, and the fill is 114 -- the value
    ultralytics uses, so a box near the edge sees the same context it would
    have on the CPU path.
    """
    import torch
    import torch.nn.functional as F

    h, w = int(frame.shape[0]), int(frame.shape[1])
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    chw = frame.permute(2, 0, 1).unsqueeze(0).float()
    # antialias=False to match cv2.resize's INTER_LINEAR, which is what
    # ultralytics letterboxes with. Torch's antialiased bilinear behaves like
    # INTER_AREA instead; on the face path the same substitution cost 3 of 32
    # detections, so it is not a free "nicer" resize.
    resized = F.interpolate(
        chw, size=(new_h, new_w), mode="bilinear",
        align_corners=False, antialias=False,
    )

    out = torch.full(
        (1, 3, size, size), 114.0, dtype=torch.float32, device=frame.device
    )
    out[:, :, :new_h, :new_w] = resized
    out = out.div_(255.0)
    if half:
        out = out.half()
    return out, scale, 0, 0


def unletterbox(boxes, scale: float, pad_x: int = 0, pad_y: int = 0):
    """Boxes from letterbox space back to the original frame's pixels."""
    if boxes is None or len(boxes) == 0:
        return boxes
    out = boxes.clone() if hasattr(boxes, "clone") else np.array(boxes, copy=True)
    out[:, 0] = (out[:, 0] - pad_x) / scale
    out[:, 1] = (out[:, 1] - pad_y) / scale
    out[:, 2] = (out[:, 2] - pad_x) / scale
    out[:, 3] = (out[:, 3] - pad_y) / scale
    return out


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 2x3 similarity transform mapping src onto dst.

    insightface uses skimage's SimilarityTransform for this, which is the
    Umeyama estimate. Reimplemented so the alignment does not depend on
    skimage being installed in the inference process, and so the matrix is
    available without a round trip through a class.
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = dst_demean.T @ src_demean / src.shape[0]
    d = np.ones((2,), dtype=np.float64)
    if np.linalg.det(cov) < 0:
        d[1] = -1

    u, s, vt = np.linalg.svd(cov)
    rank = np.linalg.matrix_rank(cov)
    if rank == 0:
        return np.eye(2, 3, dtype=np.float32)
    if rank == 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            rot = u @ vt
        else:
            t = d[1]
            d[1] = -1
            rot = u @ np.diag(d) @ vt
            d[1] = t
    else:
        rot = u @ np.diag(d) @ vt

    var_src = src_demean.var(axis=0).sum()
    scale = 1.0 if var_src == 0 else (s @ d) / var_src

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rot
    matrix[:, 2] = dst_mean - scale * rot @ src_mean
    return matrix.astype(np.float32)


def align_faces_gpu(frame, landmarks_list, out_size: int = 112):
    """Aligned face crops, on the GPU, from full-frame landmarks.

    `frame` is (H, W, 3) uint8 RGB on the device; `landmarks_list` holds the
    SCRFD 5-point sets in FULL-FRAME pixels. Returns (N, 3, 112, 112) float in
    0..1, in the same order.

    The affine is computed on the CPU -- it is a 2x3 solve over five points,
    and moving that to the GPU would cost more in launches than it saves --
    but it is APPLIED with grid_sample, so the pixels never leave the device.

    grid_sample wants normalised coordinates and the INVERSE mapping (output
    pixel -> input pixel), which is why the matrix is inverted here rather
    than being handed over as insightface's forward warp.
    """
    import torch
    import torch.nn.functional as F

    if not landmarks_list:
        return None

    h, w = int(frame.shape[0]), int(frame.shape[1])
    thetas = []
    for kps in landmarks_list:
        m = _umeyama(np.asarray(kps, dtype=np.float64)[:5], ARCFACE_DST.astype(np.float64))
        # Forward maps input -> output. grid_sample needs output -> input.
        full = np.vstack([m, [0.0, 0.0, 1.0]])
        inv = np.linalg.inv(full)[:2]

        # Pixel-space inverse into grid_sample's [-1, 1] convention.
        #   x_in = a*x_out + b*y_out + c   (pixels)
        # becomes, with x_n = 2*x/(S-1) - 1 style normalisation on both sides.
        a, b, c = inv[0]
        d, e, f = inv[1]
        sx_out, sy_out = (out_size - 1) / 2.0, (out_size - 1) / 2.0
        theta = np.array(
            [
                [a * sx_out / ((w - 1) / 2.0),
                 b * sy_out / ((w - 1) / 2.0),
                 (a * sx_out + b * sy_out + c) / ((w - 1) / 2.0) - 1.0],
                [d * sx_out / ((h - 1) / 2.0),
                 e * sy_out / ((h - 1) / 2.0),
                 (d * sx_out + e * sy_out + f) / ((h - 1) / 2.0) - 1.0],
            ],
            dtype=np.float32,
        )
        thetas.append(theta)

    n = len(thetas)
    theta_t = torch.from_numpy(np.stack(thetas)).to(frame.device)
    src = frame.permute(2, 0, 1).unsqueeze(0).float().expand(n, -1, -1, -1)
    grid = F.affine_grid(theta_t, (n, 3, out_size, out_size), align_corners=True)
    crops = F.grid_sample(
        src, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return crops.div_(255.0)
