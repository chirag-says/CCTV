"""
Image utility functions for the CCTV pipeline.

- Snapshot saving (face crops, context frames)
- Frame resizing (aspect-ratio preserving)
- JPEG encoding (TurboJPEG with OpenCV fallback)
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ── JPEG Encoder Selection ────────────────────────────────────────
# TurboJPEG is 2-4x faster than OpenCV's libjpeg for JPEG encoding.
# We try to use it first, and fall back to OpenCV if not installed.

_turbojpeg_encoder = None
_jpeg_encoder_name = "OpenCV"

try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    _turbojpeg_encoder = TurboJPEG()
    _jpeg_encoder_name = "TurboJPEG"
    logger.info("JPEG Encoder: TurboJPEG (libturbojpeg) — fast path active")
except ImportError:
    logger.info("JPEG Encoder: OpenCV (libjpeg) — install PyTurboJPEG for 2-4x faster encoding")
except (RuntimeError, OSError) as e:
    logger.info(f"JPEG Encoder: OpenCV (libjpeg) — TurboJPEG native lib not found: {e}")
except Exception as e:
    logger.warning(f"TurboJPEG init failed, using OpenCV: {e}")


def get_jpeg_encoder_name() -> str:
    """Return the active JPEG encoder name for startup summary."""
    return _jpeg_encoder_name


def save_snapshot(
    image: np.ndarray,
    name: str,
    subdirectory: str = "",
    quality: int = 90,
) -> Tuple[str, str]:
    """
    Save an image as a JPEG snapshot.

    Args:
        image: BGR numpy array
        name: Filename (without extension)
        subdirectory: Optional subdirectory under snapshots/
        quality: JPEG quality (1-100)

    Returns:
        Tuple of (filepath, url_path)
    """
    try:
        base_dir = os.path.join("snapshots", subdirectory) if subdirectory else "snapshots"
        os.makedirs(base_dir, exist_ok=True)

        filename = f"{name}.jpg"
        filepath = os.path.join(base_dir, filename)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(filepath, image, encode_params)

        url_parts = ["snapshots"]
        if subdirectory:
            url_parts.append(subdirectory)
        url_parts.append(filename)
        url_path = "/" + "/".join(url_parts)

        logger.debug(f"Snapshot saved: {filepath} → {url_path}")
        return filepath, url_path

    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")
        return "", ""


def resize_frame(
    frame: np.ndarray,
    max_width: int = 640,
    max_height: int = 480,
) -> np.ndarray:
    """
    Resize a frame maintaining aspect ratio.

    Args:
        frame: Input BGR image
        max_width: Maximum output width
        max_height: Maximum output height

    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]

    if w <= max_width and h <= max_height:
        return frame

    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_jpeg_bytes(
    frame: np.ndarray,
    quality: int = 65,
) -> bytes:
    """
    Convert a BGR frame to JPEG bytes for WebSocket streaming.

    Uses TurboJPEG if available (2-4x faster), otherwise OpenCV.

    Args:
        frame: BGR numpy array
        quality: JPEG quality (lower = smaller payload, faster encode)

    Returns:
        JPEG-encoded bytes
    """
    if _turbojpeg_encoder is not None:
        try:
            return _turbojpeg_encoder.encode(frame, quality=quality)
        except Exception:
            pass  # Fall through to OpenCV

    # OpenCV fallback with optimized params
    encode_params = [
        cv2.IMWRITE_JPEG_QUALITY, quality,
        cv2.IMWRITE_JPEG_OPTIMIZE, 0,       # Skip Huffman optimization (faster)
        cv2.IMWRITE_JPEG_RST_INTERVAL, 0,    # No restart markers (smaller)
    ]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return buffer.tobytes()


def validate_image_file(content_type: str, max_size_mb: int = 10) -> bool:
    """Check if an uploaded file has a valid image content type."""
    valid_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    return content_type in valid_types
