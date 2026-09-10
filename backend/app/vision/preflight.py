"""
Refuse to start on a machine that cannot do the work.

WHY THIS IS FATAL BY DEFAULT
Every acceleration in this system degrades SILENTLY. A missing CUDA runtime
turns SCRFD from 34 ms into 3,648 ms and reports healthy. A TensorRT provider
that will not register logs one line and continues on CUDA. A missing NVDEC
falls back to cv2 and burns a core per camera. None of them raise, so a
deployment can be a hundred times slower than intended and the only symptom is
that the cameras seem slow -- which reads as "we need a bigger GPU", not "one
DLL is in the wrong place".

Each of those has already happened in this codebase, and each was a
five-minute fix once it had a name. So the answer is to check at startup, name
what is missing, and stop.

ALLOW_DEGRADED=true is for the deployments where this is genuinely fine: an
API host with no cameras, or CI. It logs what it is giving up.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


class DegradedRuntime(RuntimeError):
    """A required accelerator is missing and ALLOW_DEGRADED is not set."""


def _check_cuda() -> Tuple[bool, str]:
    try:
        import torch
    except Exception as e:                                      # noqa: BLE001
        return False, f"torch will not import ({e})"
    if not torch.cuda.is_available():
        return False, (
            "torch.cuda.is_available() is False -- no driver, no device, or a "
            "CPU-only torch build"
        )
    return True, f"CUDA on {torch.cuda.get_device_name(0)}"


def _check_tensorrt() -> Tuple[bool, str]:
    """TensorRT means BOTH: the package imports, and onnxruntime can use it.

    Checked together because they fail apart. `ort.get_available_providers()`
    lists TensorrtExecutionProvider whenever onnxruntime-gpu is installed,
    which says nothing about whether nvinfer can be found -- see
    detector._ensure_tensorrt_loaded for the failure that hides behind that.
    """
    if str(getattr(settings, "FACE_BACKEND", "cuda")).lower() != "trt":
        return True, "not required (FACE_BACKEND is not trt)"
    try:
        import tensorrt
    except Exception as e:                                      # noqa: BLE001
        return False, f"the tensorrt package will not import ({e})"
    try:
        import onnxruntime as ort
    except Exception as e:                                      # noqa: BLE001
        return False, f"onnxruntime will not import ({e})"
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        return False, (
            f"onnxruntime has no TensorrtExecutionProvider "
            f"(has {ort.get_available_providers()})"
        )
    return True, f"TensorRT {tensorrt.__version__}"


def _check_nvdec() -> Tuple[bool, str]:
    if str(getattr(settings, "DECODE_BACKEND", "cpu")).lower() != "nvdec":
        return True, "not required (DECODE_BACKEND is not nvdec)"
    from app.vision import nvdecode

    if not nvdecode.available():
        return False, "PyNvVideoCodec will not import"
    return True, "NVDEC ready"


CHECKS = (
    ("CUDA", _check_cuda),
    ("TensorRT", _check_tensorrt),
    ("NVDEC", _check_nvdec),
)


def run() -> List[str]:
    """Check everything, then decide. Returns the lines worth logging.

    Every check runs even after the first failure: being told about CUDA, then
    TensorRT on the next boot, then NVDEC on the one after is three restarts
    to learn what one message could have said.
    """
    allow = bool(getattr(settings, "ALLOW_DEGRADED", False))
    notes, missing = [], []

    for name, check in CHECKS:
        try:
            ok, detail = check()
        except Exception as e:                                  # noqa: BLE001
            ok, detail = False, f"check itself failed ({e})"
        notes.append(f"  {name:<9} {'OK  ' if ok else 'MISSING'}  {detail}")
        if not ok:
            missing.append(f"{name}: {detail}")

    if not missing:
        logger.info("Accelerator preflight:\n" + "\n".join(notes))
        return notes

    summary = "\n".join(notes)
    if allow:
        logger.error(
            "Accelerator preflight FAILED, continuing because "
            "ALLOW_DEGRADED=true:\n" + summary +
            "\n  Expect roughly a hundredfold slowdown on the missing paths."
        )
        return notes

    raise DegradedRuntime(
        "\n" + "=" * 70 + "\n"
        "  This machine is missing accelerators the pipeline is configured\n"
        "  to use. Every one of these degrades silently rather than failing,\n"
        "  so the system will not start on them.\n\n"
        + summary + "\n\n"
        "  Fix the above, or set ALLOW_DEGRADED=true to run anyway (an API\n"
        "  host with no cameras, or CI).\n"
        + "=" * 70
    )
