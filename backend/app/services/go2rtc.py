"""
go2rtc as the live-view relay.

WHAT THIS REPLACES
The old live view was a JPEG loop: every camera decoded, drew, resized and
JPEG-encoded a frame per displayed frame, then pushed it down a WebSocket as
a still image. That put the video path through Python, through the GIL, and
through an encoder that cost more CPU per camera than the tracker did.

go2rtc takes the video path out of the application entirely. It pulls RTSP
straight from the camera and hands the browser WebRTC, so frames never enter
this process at all. The pipeline still reads the same RTSP URL for analysis;
the two are independent consumers of the same source.

WHAT THIS MODULE OWNS
  * generating go2rtc.yaml from the cameras table at startup
  * starting go2rtc as a managed child process
  * checking it is alive, and restarting it when it is not
  * saying where a browser should signal for a given camera

FILE CAMERAS
Development runs on .mp4 files, not on cameras. go2rtc cannot serve a file as
a live stream on its own, so a file source is wrapped in ffmpeg with
`-stream_loop -1 -re`: loop forever, at real time. That makes a file behave
exactly like a camera all the way to the browser, which is the only way the
live-view feature can be developed or demonstrated without a camera wall.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional
from urllib.parse import quote
from urllib.request import urlopen

from app.config import settings

logger = logging.getLogger(__name__)

# Extensions we treat as "a file pretending to be a camera".
_VIDEO_FILE_EXT = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".ts")


def _yaml_quote(value: str) -> str:
    """Single-quoted YAML scalar. go2rtc sources are full of `#`, `?` and `:`,
    every one of which changes meaning unquoted."""
    return "'" + str(value).replace("'", "''") + "'"


def is_file_source(stream_url: str) -> bool:
    """A local video file, as opposed to a network camera or a webcam index."""
    if not stream_url:
        return False
    url = str(stream_url)
    if "://" in url:
        return False
    return url.lower().endswith(_VIDEO_FILE_EXT)


def source_for(stream_url: str) -> Optional[str]:
    """The go2rtc source string for one camera's stream_url.

    Returns None for sources go2rtc cannot serve -- a browser-pushed session,
    or a local webcam index, which belongs to whoever opened it first and is
    already held by the analysis pipeline.
    """
    if not stream_url:
        return None
    url = str(stream_url).strip()

    if url.startswith("browser:"):
        return None
    if url.isdigit():
        # A webcam index. cv2 already holds this device; a second reader
        # would either fail or steal it. Live view falls back to MJPEG.
        return None

    if is_file_source(url):
        # Forward slashes even on Windows: this string is handed to ffmpeg
        # inside a #-delimited go2rtc source, and a backslash there is one
        # more escaping layer than either parser needs.
        path = os.path.abspath(url).replace(chr(92), "/")
        # `-stream_loop -1` before `-i` loops the input forever; `-re` paces
        # it at the file's own frame rate instead of decoding it as fast as
        # the disk allows. Without -re the whole clip is relayed in seconds
        # and the browser sees a burst, not a live stream.
        return (
            f"ffmpeg:{path}"
            "#input=-re -stream_loop -1 -i {input}"
            "#video=h264"
        )

    if url.startswith(("rtsp://", "rtsps://", "rtmp://", "http://", "https://")):
        return url

    # Anything else is passed through: go2rtc understands more source schemes
    # than this function needs to enumerate, and refusing them here would be
    # this module deciding what a camera is.
    return url


def build_config(cameras: List[dict]) -> str:
    """Render go2rtc.yaml. One stream per camera, keyed by camera id.

    Keyed by id rather than name because the name is user-editable and the
    signalling URL is not: renaming a camera in the dashboard must not break
    the live view of every browser holding it open.
    """
    lines = [
        "# GENERATED FILE -- written by app/services/go2rtc.py at startup.",
        "# Edits are overwritten on the next boot; change the cameras table.",
        "",
        "api:",
        f"  listen: {_yaml_quote(settings.GO2RTC_API_LISTEN)}",
        f"  origin: {_yaml_quote('*')}",
        "",
        "rtsp:",
        f"  listen: {_yaml_quote(settings.GO2RTC_RTSP_LISTEN)}",
        "",
        "webrtc:",
        f"  listen: {_yaml_quote(settings.GO2RTC_WEBRTC_LISTEN)}",
        "  candidates:",
    ]
    for candidate in settings.go2rtc_candidates_list:
        lines.append(f"    - {_yaml_quote(candidate)}")

    lines += ["", "log:", f"  level: {_yaml_quote(settings.GO2RTC_LOG_LEVEL)}", "", "streams:"]

    served = 0
    for camera in cameras:
        camera_id = camera.get("id")
        source = source_for(camera.get("stream_url"))
        if not camera_id or not source:
            continue
        lines.append(f"  {_yaml_quote(camera_id)}: {_yaml_quote(source)}")
        served += 1

    if served == 0:
        # An empty `streams:` key is valid YAML but reads as a mistake in the
        # file. Say why it is empty instead.
        lines.append("  # no relayable cameras registered at startup")

    lines.append("")
    return "\n".join(lines)


class Go2RtcSupervisor:
    """Starts go2rtc, watches it, restarts it when it dies.

    Deliberately not a systemd unit or a compose-only concern: in development
    the backend is the only thing anybody starts, and a live view that needs a
    second manual command is a live view nobody sees. In compose the sidecar
    is its own service and GO2RTC_MANAGED=false switches this off.
    """

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._config_path: Optional[str] = None
        self._streams: Dict[str, str] = {}
        self._restarts = 0
        self._last_error: Optional[str] = None

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self, cameras: List[dict]) -> bool:
        """Write the config and bring go2rtc up. Never raises."""
        if not settings.GO2RTC_ENABLED:
            logger.info("go2rtc disabled (GO2RTC_ENABLED=false)")
            return False

        self._streams = {
            c["id"]: source_for(c.get("stream_url"))
            for c in cameras
            if c.get("id") and source_for(c.get("stream_url"))
        }

        try:
            self._config_path = self.write_config(cameras)
        except Exception as e:                                  # noqa: BLE001
            self._last_error = f"config write failed: {e}"
            logger.error(f"go2rtc: {self._last_error}")
            return False

        if not settings.GO2RTC_MANAGED:
            logger.info(
                "go2rtc config written to %s; process not managed here "
                "(GO2RTC_MANAGED=false)", self._config_path
            )
            return True

        binary = self._resolve_binary()
        if not binary:
            self._last_error = (
                f"binary not found (looked for {settings.GO2RTC_BINARY!r} on PATH "
                f"and at GO2RTC_BINARY). Live view falls back to MJPEG."
            )
            logger.warning(f"go2rtc: {self._last_error}")
            return False

        if not self._spawn(binary):
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._supervise, name="go2rtc-supervisor", daemon=True
        )
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        with self._lock:
            proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        logger.info("go2rtc stopped")

    # ── config ────────────────────────────────────────────────────────

    def write_config(self, cameras: List[dict]) -> str:
        path = os.path.abspath(settings.GO2RTC_CONFIG_PATH)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(build_config(cameras))
        logger.info(
            f"go2rtc config written: {path} "
            f"({sum(1 for c in cameras if source_for(c.get('stream_url')))} stream(s))"
        )
        return path

    # ── process ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_binary() -> Optional[str]:
        configured = settings.GO2RTC_BINARY
        if os.path.isfile(configured):
            return os.path.abspath(configured)
        return shutil.which(configured)

    def _spawn(self, binary: str) -> bool:
        try:
            proc = subprocess.Popen(
                [binary, "-config", self._config_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(self._config_path) or None,
            )
        except Exception as e:                                  # noqa: BLE001
            self._last_error = f"spawn failed: {e}"
            logger.error(f"go2rtc: {self._last_error}")
            return False

        with self._lock:
            self._proc = proc
        logger.info(f"go2rtc started (pid={proc.pid}) with {self._config_path}")

        if self.wait_healthy(settings.GO2RTC_STARTUP_TIMEOUT_SEC):
            self._last_error = None
            logger.info(f"go2rtc healthy at {settings.GO2RTC_INTERNAL_URL}")
            return True

        self._last_error = "started but never became healthy"
        logger.warning(f"go2rtc: {self._last_error}")
        # Left running regardless: it may still come up, and killing it here
        # would guarantee it does not.
        return False

    def _supervise(self):
        """Restart go2rtc if it dies. Backed off, so a binary that cannot run
        at all does not spin."""
        backoff = 1.0
        while not self._stop.wait(2.0):
            with self._lock:
                proc = self._proc
            if proc is None or proc.poll() is None:
                backoff = 1.0
                continue

            self._restarts += 1
            logger.warning(
                f"go2rtc exited (code={proc.returncode}); "
                f"restart #{self._restarts} in {backoff:.0f}s"
            )
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, 60.0)

            binary = self._resolve_binary()
            if binary:
                self._spawn(binary)

    # ── health ────────────────────────────────────────────────────────

    def healthy(self) -> bool:
        try:
            with urlopen(f"{settings.GO2RTC_INTERNAL_URL}/api/streams", timeout=1.5) as r:
                return 200 <= r.status < 300
        except Exception:
            return False

    def wait_healthy(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.healthy():
                return True
            if self._stop.wait(0.25):
                return False
        return False

    # ── what the browser needs ────────────────────────────────────────

    def has_stream(self, camera_id: str) -> bool:
        return camera_id in self._streams

    def signalling_url(self, camera_id: str) -> str:
        """The WebSocket the browser exchanges SDP with, for this camera."""
        base = settings.GO2RTC_PUBLIC_URL.rstrip("/")
        ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_base}/api/ws?src={quote(camera_id, safe='')}"

    def status(self) -> dict:
        with self._lock:
            proc = self._proc
        return {
            "enabled": settings.GO2RTC_ENABLED,
            "managed": settings.GO2RTC_MANAGED,
            "running": bool(proc and proc.poll() is None),
            "pid": proc.pid if proc and proc.poll() is None else None,
            "restarts": self._restarts,
            "config_path": self._config_path,
            "streams": sorted(self._streams),
            "last_error": self._last_error,
        }


go2rtc = Go2RtcSupervisor()
