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
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── JPEG Encoder Selection ────────────────────────────────────────
# Priority: simplejpeg (bundled libjpeg-turbo) > PyTurboJPEG > OpenCV
# Benchmarked on 1920×1080 real RTSP frame (500 runs, quality=80):
#   simplejpeg:  5.4 ms/frame  (libjpeg-turbo, SIMD-optimized)
#   OpenCV:     19.1 ms/frame  (built-in libjpeg, no SIMD)
#   Speedup:    3.5×

_simplejpeg_available = False
_turbojpeg_encoder = None
_jpeg_encoder_name = "OpenCV"

# Try simplejpeg first (bundles its own libjpeg-turbo, no external DLL needed)
try:
    import simplejpeg as _simplejpeg
    _simplejpeg_available = True
    _jpeg_encoder_name = "simplejpeg"
    logger.info("JPEG Encoder: simplejpeg (libjpeg-turbo, bundled) — 3.5× faster than OpenCV")
except ImportError:
    _simplejpeg = None
    logger.debug("simplejpeg not installed, trying PyTurboJPEG...")

# Try PyTurboJPEG second (requires external libjpeg-turbo DLL)
if not _simplejpeg_available:
    try:
        from turbojpeg import TurboJPEG, TJPF_BGR
        _turbojpeg_encoder = TurboJPEG()
        _jpeg_encoder_name = "TurboJPEG"
        logger.info("JPEG Encoder: TurboJPEG (libturbojpeg) — fast path active")
    except ImportError:
        logger.info("JPEG Encoder: OpenCV (libjpeg) — install simplejpeg for 3.5x faster encoding")
    except (RuntimeError, OSError) as e:
        logger.info(f"JPEG Encoder: OpenCV (libjpeg) — TurboJPEG native lib not found: {e}")
    except Exception as e:
        logger.warning(f"TurboJPEG init failed, using OpenCV: {e}")


def get_jpeg_encoder_name() -> str:
    """Return the active JPEG encoder name for startup summary."""
    return _jpeg_encoder_name


def save_jpeg_bytes(
    jpeg_bytes: bytes,
    name: str,
    subdirectory: str = "",
) -> Tuple[str, str]:
    """
    Write already-encoded JPEG bytes straight to disk.

    Used for security-alert evidence, where the pipeline has already encoded
    the annotated frame for the live stream (pipeline._last_jpeg). Reusing
    those bytes skips a decode/re-encode round trip entirely, and the saved
    image carries the detection boxes the operator saw — better evidence than
    a clean frame.

    fsync before returning, so the file is on disk before the URL reaches the
    database row that points at it.

    Returns:
        (filepath, url_path), or ("", "") on failure — never raises, because
        losing an evidence image must not lose the incident record.
    """
    if not jpeg_bytes:
        return "", ""
    try:
        base_dir = os.path.join("snapshots", subdirectory) if subdirectory else "snapshots"
        os.makedirs(base_dir, exist_ok=True)

        filename = f"{name}.jpg"
        filepath = os.path.join(base_dir, filename)
        with open(filepath, "wb") as f:
            f.write(jpeg_bytes)
            f.flush()
            os.fsync(f.fileno())

        url_parts = ["snapshots"]
        if subdirectory:
            url_parts.append(subdirectory)
        url_parts.append(filename)
        return filepath, "/" + "/".join(url_parts)
    except Exception as e:
        logger.error(f"Failed to save JPEG bytes: {e}")
        return "", ""


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

        # Use simplejpeg when available (3.5× faster than cv2.imwrite)
        if _simplejpeg_available:
            # simplejpeg expects RGB; our images are BGR
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            jpeg_bytes = _simplejpeg.encode_jpeg(rgb, quality=quality, colorspace='RGB')
            with open(filepath, 'wb') as f:
                f.write(jpeg_bytes)
                f.flush()
                os.fsync(f.fileno())  # Guarantee flush before DB commit
        else:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            cv2.imwrite(filepath, image, encode_params)
            # fsync after cv2.imwrite to guarantee flush
            try:
                fd = os.open(filepath, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except OSError:
                pass  # Best-effort on Windows

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


def resolve_snapshot_path(url_or_path: str) -> Optional[str]:
    """
    Map a stored snapshot URL to a real file on disk.

    save_snapshot() writes to ``snapshots/<subdir>/<name>.jpg`` and stores the
    URL form ``/snapshots/<subdir>/<name>.jpg``. Naively joining that onto
    SNAPSHOT_DIR ("./snapshots") produces ``snapshots/snapshots/<subdir>/...``,
    which never exists — and on Windows the leading slash does not make the
    path absolute either, since there is no drive letter, so the usual
    is_absolute() guard does not catch it.

    Tries, in order: the path as given; the URL relative to the working
    directory; the URL with its leading "snapshots/" segment stripped and
    re-joined to SNAPSHOT_DIR; and finally the bare filename in SNAPSHOT_DIR.

    Returns None when the file cannot be found anywhere.
    """
    if not url_or_path:
        return None

    from app.config import settings

    snap_dir = str(settings.SNAPSHOT_DIR)
    rel = url_or_path.lstrip("/").replace("\\", "/")
    base = os.path.basename(rel)
    leaf = os.path.basename(os.path.normpath(snap_dir))

    # Strip a duplicated leading "snapshots/" so the join does not double it.
    trimmed = rel[len(leaf) + 1:] if rel.startswith(leaf + "/") else rel

    for candidate in (
        url_or_path,
        rel,
        os.path.join(snap_dir, trimmed),
        os.path.join(snap_dir, base),
    ):
        try:
            if candidate and os.path.isfile(candidate):
                return candidate
        except (OSError, ValueError):
            continue

    return None


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
    quality: int = 80,
) -> bytes:
    """
    Convert a BGR frame to JPEG bytes for WebSocket streaming.

    Encoder priority:
      1. simplejpeg (bundled libjpeg-turbo) — 5.4 ms @ 1080p
      2. PyTurboJPEG (external DLL)        — similar speed
      3. OpenCV imencode (fallback)         — 19.1 ms @ 1080p

    Args:
        frame: BGR numpy array (must be contiguous)
        quality: JPEG quality 1-100 (default 80)

    Returns:
        JPEG-encoded bytes
    """
    # Fast path: simplejpeg (bundled libjpeg-turbo, no external DLL)
    if _simplejpeg_available:
        try:
            # simplejpeg requires C-contiguous array
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            return _simplejpeg.encode_jpeg(frame, quality=quality, colorspace="BGR")
        except Exception:
            pass  # Fall through

    # Second path: PyTurboJPEG (external DLL)
    if _turbojpeg_encoder is not None:
        try:
            return _turbojpeg_encoder.encode(frame, quality=quality)
        except Exception:
            pass  # Fall through to OpenCV

    # Final fallback: OpenCV libjpeg
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
