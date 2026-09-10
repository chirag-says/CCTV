"""
GPU letterbox and face alignment agree with the cv2 versions they replace.

WHY EMBEDDINGS, NOT PIXELS, ARE THE ASSERTION
Both of these are resampling operations, so pixel-exact equality with cv2 is
not available and not the right bar. What matters is whether a face still
recognises as the same person, and that is decided by the cosine distance
between ArcFace embeddings -- against a match threshold of 0.62, with
OPTIMIZATION_BRIEF.md allowing 0.02 of drift from any change to this path.

So the alignment is held to the embedding it produces, not to its pixels. The
affine matrix is checked separately against insightface's own estimate,
because that is geometry and CAN be exact -- if it drifts, the sampling
comparison downstream is measuring the wrong thing.

Skipped without CUDA.
"""

import glob
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

cv2 = pytest.importorskip("cv2")
face_align = pytest.importorskip("insightface.utils.face_align")

from app.vision.gpu_preproc import (  # noqa: E402
    ARCFACE_DST,
    _umeyama,
    align_faces_gpu,
    letterbox_gpu,
    unletterbox,
)

FRAMES = sorted(glob.glob(os.path.join("snapshots", "frames", "*.jpg")))[:8]


@pytest.fixture(scope="module")
def detected():
    """(bgr image, faces) for frames that actually contain a face."""
    if not FRAMES:
        pytest.skip("no sample frames in snapshots/frames")
    from app.vision.detector import _get_insightface_app

    app = _get_insightface_app()
    if app is None:
        pytest.skip("insightface unavailable")
    out = []
    for path in FRAMES:
        img = cv2.imread(path)
        if img is None:
            continue
        faces = app.get(img)
        if faces:
            out.append((img, faces))
    if not out:
        pytest.skip("no faces in the sample frames")
    return app, out


def test_affine_matches_insightface(detected):
    """Geometry is exact, so it is asserted exactly (to float tolerance)."""
    _app, samples = detected
    worst = 0.0
    for _img, faces in samples:
        for f in faces:
            ours = _umeyama(np.asarray(f.kps, dtype=np.float64), ARCFACE_DST.astype(np.float64))
            theirs = face_align.estimate_norm(f.kps, 112)
            worst = max(worst, float(np.abs(ours - theirs).max()))
    assert worst < 1e-3, f"affine differs from insightface by {worst}"


def test_aligned_crops_produce_the_same_embedding(detected):
    """The assertion that decides whether recognition still works."""
    app, samples = detected
    rec = app.models["recognition"]
    name = rec.session.get_inputs()[0].name

    def embed(chw01):
        x = chw01.cpu().numpy() * 255.0
        x = (x - rec.input_mean) / rec.input_std
        return rec.session.run(None, {name: x.astype(np.float32)})[0]

    distances = []
    for img, faces in samples:
        rgb = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cuda()
        crops = align_faces_gpu(rgb, [f.kps for f in faces])
        assert crops.shape[1:] == (3, 112, 112)
        gpu_emb = embed(crops)
        for i, f in enumerate(faces):
            a = f.embedding / np.linalg.norm(f.embedding)
            b = gpu_emb[i] / np.linalg.norm(gpu_emb[i])
            distances.append(1.0 - float(np.dot(a, b)))

    d = np.array(distances)
    assert len(d) > 0
    # 0.02 is what the brief allows a change to this path to move a match
    # distance by. Measured here: about 0.002 at worst.
    assert d.max() < 0.02, (
        f"GPU alignment moves embeddings by up to {d.max():.4f} cosine, "
        f"which is past the 0.02 an accuracy-neutral change may spend"
    )


def test_letterbox_geometry_round_trips():
    """A box in letterbox space comes back to where it started."""
    frame = torch.randint(0, 255, (720, 1280, 3), dtype=torch.uint8, device="cuda")
    out, scale, pad_x, pad_y = letterbox_gpu(frame, 640)

    assert out.shape == (1, 3, 640, 640)
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0

    original = np.array([[100.0, 50.0, 300.0, 400.0]])
    scaled = original * scale
    back = unletterbox(np.array(scaled), scale, pad_x, pad_y)
    assert np.abs(back - original).max() < 1e-3


def test_letterbox_preserves_aspect_and_pads_bottom_right():
    """Padding is 114 and goes bottom-right, matching ultralytics.

    A box near the right edge otherwise sees black where the CPU path shows
    grey, which changes its score.
    """
    frame = torch.full((360, 1280, 3), 200, dtype=torch.uint8, device="cuda")
    out, scale, _px, _py = letterbox_gpu(frame, 640)
    img = out[0].permute(1, 2, 0).cpu().numpy() * 255.0

    new_h = int(round(360 * scale))
    assert abs(img[: new_h - 2, :, :].mean() - 200) < 2.0, "content region altered"
    assert abs(img[new_h + 2 :, :, :].mean() - 114) < 1.0, "pad is not 114"


def test_align_handles_no_faces():
    frame = torch.zeros((64, 64, 3), dtype=torch.uint8, device="cuda")
    assert align_faces_gpu(frame, []) is None
