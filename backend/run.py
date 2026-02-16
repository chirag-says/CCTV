"""
Application entry point.
Run with: python run.py
"""

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
