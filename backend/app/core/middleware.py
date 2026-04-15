"""
Request Logging Middleware — logs method, path, status, response time, and client IP.
Each request gets a unique X-Request-ID for traceability.
"""

import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("cctv.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs each HTTP request with timing and a unique ID."""

    # Paths to skip logging (high-frequency, noisy endpoints)
    SKIP_PATHS = {
        "/health",
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
    }

    # Paths that are logged at DEBUG level (frequent polling)
    DEBUG_PATHS_PREFIXES = (
        "/api/cameras/active-persons",
        "/snapshots/",
    )

    async def dispatch(self, request: Request, call_next):
        # Skip logging for noisy endpoints
        path = request.url.path
        if path in self.SKIP_PATHS:
            return await call_next(request)

        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]

        # Extract client IP
        client_ip = request.client.host if request.client else "unknown"

        # Start timer
        start = time.perf_counter()

        # Add request ID to request state for use in error handlers
        request.state.request_id = request_id

        response: Response = None
        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                f"[REQ-{request_id}] {request.method} {path} → 500 ({elapsed:.0f}ms) "
                f"IP={client_ip} ERROR={type(exc).__name__}: {exc}"
            )
            raise

        elapsed = (time.perf_counter() - start) * 1000

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Choose log level based on path and status
        status = response.status_code
        is_debug_path = any(path.startswith(p) for p in self.DEBUG_PATHS_PREFIXES)

        msg = (
            f"[REQ-{request_id}] {request.method} {path} → {status} ({elapsed:.0f}ms) "
            f"IP={client_ip}"
        )

        if is_debug_path:
            logger.debug(msg)
        elif status >= 500:
            logger.error(msg)
        elif status >= 400:
            logger.warning(msg)
        else:
            logger.info(msg)

        return response
