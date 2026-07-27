"""
Application entry point.
Run with: python run.py
"""

import os
# Force CUDA to use the NVIDIA GPU (not Intel iGPU) on hybrid GPU laptops
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import uvicorn
from app.config import settings


def main():
    """Start the FastAPI application server."""

    # ── Suppress noisy third-party loggers ─────────────────────
    # These produce hundreds of DEBUG lines per second:
    #   DEBUG > BINARY ff d8 ff e0 ... [214745 bytes]
    #   INFO  connection closed
    # Force them to WARNING regardless of app debug mode.
    for noisy_logger in (
        "websockets",
        "websockets.protocol",
        "websockets.server",
        "uvicorn.access",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,  # Single worker due to in-memory state (camera workers)
        log_level="warning",  # Suppress uvicorn INFO/DEBUG (our app does its own logging)
        access_log=False,     # Disable per-request access log (middleware handles slow requests)
    )


if __name__ == "__main__":
    main()

