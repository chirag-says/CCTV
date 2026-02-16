"""
Image processing utilities for snapshot management, compression,
and URL path generation.
"""

import cv2
import numpy as np
import os
import time
import logging
from typing import Optional, Tuple
from app.config import settings

logger = logging.getLogger(__name__)


def compress_image(
    image: np.ndarray,
    max_size_kb: int = None,
    quality: int = 85,
) -> bytes:
    """
    Compress an image to JPEG with optional max size constraint.

    Args:
        image: BGR numpy array
        max_size_kb: Maximum size in KB (iteratively reduces quality)
        quality: Initial JPEG quality (0-100)

    Returns:
        Compressed JPEG bytes
    """
    max_size = (max_size_kb or settings.MAX_SNAPSHOT_SIZE_KB) * 1024

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", image, encode_params)

    # Iteratively reduce quality if too large
    while len(buffer) > max_size and quality > 20:
        quality -= 10
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode(".jpg", image, encode_params)

    return buffer.tobytes()


def save_snapshot(
    image: np.ndarray,
    prefix: str,
    subdirectory: str = "",
    quality: int = 85,
) -> Tuple[str, str]:
    """
    Save a snapshot image to disk and return both file path and URL path.

    Args:
        image: BGR numpy array
        prefix: Filename prefix (e.g., 'unknown_<uuid>')
        subdirectory: Optional subdirectory under SNAPSHOT_DIR
        quality: JPEG quality

    Returns:
        Tuple of (filesystem_path, url_path)
    """
    try:
        timestamp = int(time.time() * 1000)  # Millisecond precision
        filename = f"{prefix}_{timestamp}.jpg"

        save_dir = settings.SNAPSHOT_DIR
        if subdirectory:
            save_dir = os.path.join(save_dir, subdirectory)
            os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, filename)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(filepath, image, encode_params)

        # Generate URL-accessible path
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
    quality: int = 70,
) -> bytes:
    """
    Convert a BGR frame to JPEG bytes for WebSocket streaming.

    Args:
        frame: BGR numpy array
        quality: JPEG quality (lower = smaller payload)

    Returns:
        JPEG-encoded bytes
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return buffer.tobytes()


def validate_image_file(content_type: str, max_size_mb: int = 10) -> bool:
    """Check if an uploaded file has a valid image content type."""
    valid_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    return content_type in valid_types
