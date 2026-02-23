"""
Application entry point.
Run with: python run.py
"""

import os
# Force CUDA to use the NVIDIA GPU (not Intel iGPU) on hybrid GPU laptops
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import uvicorn
from app.config import settings


def main():
    """Start the FastAPI application server."""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1,  # Single worker due to in-memory state (camera workers)
        log_level="debug" if settings.DEBUG else "info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
