"""
Request Logging Middleware — logs only slow (>100ms) or error requests.

Normal fast requests are silently processed to reduce log noise.
Each logged request gets a unique X-Request-ID for traceability.

NOTE: Implemented as a pure ASGI middleware (NOT BaseHTTPMiddleware)
because BaseHTTPMiddleware wraps responses in StreamingResponse,
which strips headers set by outer middleware like CORSMiddleware.
See: https://github.com/encode/starlette/issues/1600
"""

import time
import uuid
import logging
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("cctv.requests")

# Requests slower than this threshold are logged as warnings
SLOW_REQUEST_THRESHOLD_MS = 100.0

# Paths to always skip logging (health checks, docs)
SKIP_PATHS = {
    "/health",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class RequestLoggingMiddleware:
    """
    Pure ASGI middleware that logs only slow or error HTTP requests.

    Does NOT use BaseHTTPMiddleware to avoid the response-header-stripping
    bug that breaks CORSMiddleware.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())[:8]

        # Extract client IP from scope
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"

        method = scope.get("method", "???")
        start = time.perf_counter()
        status_code = 0

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                # Inject X-Request-ID header
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                f"[REQ-{request_id}] {method} {path} → 500 ({elapsed:.0f}ms) "
                f"IP={client_ip} ERROR={type(exc).__name__}: {exc}"
            )
            raise

        elapsed = (time.perf_counter() - start) * 1000

        # Only log: errors (≥400) or slow requests (>100ms)
        if status_code >= 500:
            logger.error(
                f"[REQ-{request_id}] {method} {path} → {status_code} ({elapsed:.0f}ms) IP={client_ip}"
            )
        elif status_code >= 400:
            logger.warning(
                f"[REQ-{request_id}] {method} {path} → {status_code} ({elapsed:.0f}ms) IP={client_ip}"
            )
        elif elapsed > SLOW_REQUEST_THRESHOLD_MS:
            logger.warning(
                f"[SLOW] {method} {path} → {status_code} ({elapsed:.0f}ms) IP={client_ip}"
            )
        # else: fast + successful → silently skip
