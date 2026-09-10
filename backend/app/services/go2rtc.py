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
Development runs on .mp4 files, not on cameras. Left alone go2rtc plays a file
once and stops, so the FILE INPUT TEMPLATE is overridden to add
`-stream_loop -1`: loop forever, paced by `-re` at the file's own frame rate.
A file then behaves exactly like a camera all the way to the browser, which is
the only way the live-view feature can be developed or demonstrated without a
camera wall.

That override is global rather than per-source on purpose. go2rtc 1.9 accepts
a per-source `#input=` and then does not use it -- the default template still
wins and the leftover tokens are appended AFTER `-i`, where `-stream_loop` is
not an input option any more and ffmpeg exits on it. The `ffmpeg.file`
template is the lever that actually works, and "a file camera loops" is a
property of file cameras, not of one of them.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote
from urllib.request import urlopen

from app.config import settings

logger = logging.getLogger(__name__)

# Extensions we treat as "a file pretending to be a camera".
_VIDEO_FILE_EXT = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".ts")

# go2rtc's own file template, with looping added and the initial burst raised.
# Every token before `-i` is an INPUT option, which is the whole reason this
# has to be a template and not something appended to a source string.
#   -readrate_initial_burst  seconds of media to read flat out before `-re`
#                            starts pacing. go2rtc ships 0.001, which is fine
#                            for a stream it copies and far too small for one
#                            it TRANSCODES: the encoder then produces its first
#                            frames at real time, and go2rtc gives up waiting
#                            for the producer before any arrive. Two file
#                            cameras cold-starting together on a box that is
#                            also running the analysis pipelines is enough to
#                            hit that, and it surfaces as "NO RELAY" in the UI.
#                            One second of front-loaded video costs one second
#                            of encoding and removes the race.
#   -re                      play at the clip's frame rate, not disk speed
#   -stream_loop -1          start over at EOF, forever
FFMPEG_FILE_TEMPLATE = "-readrate_initial_burst 1 -re -stream_loop -1 -i {input}"


def find_executable(name: str) -> Optional[str]:
    """Locate a helper binary: as given, in bin/, or on PATH.

    WHY bin/ IS SEARCHED AT ALL
    go2rtc and ffmpeg are single files that get dropped next to the app rather
    than installed -- there is no `pip install go2rtc`. Requiring an absolute
    path in the environment means the live view silently does not exist for
    anyone who has not set one, and the only symptom is a browser stuck on
    "Negotiating WebRTC with the relay" while the backend logs one warning at
    startup that has long since scrolled away.

    So the obvious place is checked. `backend/bin/go2rtc.exe` is where these
    land, and finding them there is not a guess -- it is the layout.

    The .exe suffix is added on Windows so the same setting works on both.
    """
    if not name:
        return None

    candidates = [name]
    if os.name == "nt" and not name.lower().endswith(".exe"):
        candidates.append(name + ".exe")

    here = Path(__file__).resolve().parent.parent.parent      # backend/
    for candidate in candidates:
        # As given, relative to the working directory or absolute.
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        # Beside the app, which is where a dropped-in binary lives.
        beside = here / "bin" / os.path.basename(candidate)
        if beside.is_file():
            return str(beside)
        found = shutil.which(candidate)
        if found:
            return found
    return None


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
        # Looping comes from FFMPEG_FILE_TEMPLATE below, not from here. `h264`
        # is what WebRTC needs; go2rtc copies rather than transcodes when the
        # file already carries it.
        return f"ffmpeg:{path}#video=h264"

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
    ]
    # No `candidates:` key at all when none are configured. An empty list is
    # not the same as an absent one: go2rtc only gathers its own host
    # candidates when nothing overrides them, and those are the ones a local
    # or LAN browser can actually reach.
    candidates = settings.go2rtc_candidates_list
    if candidates:
        lines.append("  candidates:")
        for candidate in candidates:
            lines.append(f"    - {_yaml_quote(candidate)}")

    # The input template every FILE source is built from. See the module
    # docstring: this is where a dev clip becomes an endless camera.
    lines += ["", "ffmpeg:", f"  file: {_yaml_quote(FFMPEG_FILE_TEMPLATE)}"]
    # ffmpeg is found the same way, and only written when it is actually
    # there: pointing go2rtc at a path that does not exist is worse than
    # leaving it to its own PATH lookup.
    ffmpeg = find_executable(settings.GO2RTC_FFMPEG_BIN or "ffmpeg")
    if ffmpeg:
        lines.append(f"  bin: {_yaml_quote(ffmpeg.replace(chr(92), '/'))}")

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
        self._log_path: Optional[str] = None

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
            # "falls back to MJPEG" is only true in thread mode. In server
            # mode the cameras live in other processes, CameraManager._workers
            # is empty, and both /feed and the MJPEG socket look themselves up
            # in it -- so there is no fallback at all, and saying there is
            # sends someone looking for a JPEG path that cannot exist.
            server_mode = str(
                getattr(settings, "INFERENCE_MODE", "thread")
            ).lower() == "server"
            consequence = (
                "INFERENCE_MODE=server has no MJPEG fallback (the cameras are "
                "in other processes), so there will be NO live video until "
                "go2rtc runs."
                if server_mode else
                "Live view falls back to MJPEG."
            )
            self._last_error = (
                f"binary not found (looked for {settings.GO2RTC_BINARY!r} on "
                f"PATH and at GO2RTC_BINARY). {consequence}"
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
        return find_executable(settings.GO2RTC_BINARY)

    def _spawn(self, binary: str) -> bool:
        # go2rtc's output goes to a file beside its config, NOT to DEVNULL.
        # Everything that can go wrong with a relayed camera -- a source that
        # will not open, credentials the camera rejects, an ffmpeg that cannot
        # start a transcode before the producer times out -- is reported by
        # go2rtc and by nothing else. Discarding that stream makes every one of
        # those failures show up as an unexplained "NO RELAY" in the UI with
        # no way to find out why.
        try:
            self._log_path = os.path.join(
                os.path.dirname(self._config_path) or ".", "go2rtc.log"
            )
            log = open(self._log_path, "ab", buffering=0)
        except Exception:
            self._log_path = None
            log = subprocess.DEVNULL

        try:
            proc = subprocess.Popen(
                [binary, "-config", self._config_path],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(self._config_path) or None,
            )
        except Exception as e:                                  # noqa: BLE001
            self._last_error = f"spawn failed: {e}"
            logger.error(f"go2rtc: {self._last_error}")
            return False

        with self._lock:
            self._proc = proc
        logger.info(
            f"go2rtc started (pid={proc.pid}) with {self._config_path}"
            + (f", log {self._log_path}" if self._log_path else "")
        )

        if self.wait_healthy(settings.GO2RTC_STARTUP_TIMEOUT_SEC):
            self._last_error = None
            logger.info(f"go2rtc healthy at {settings.GO2RTC_INTERNAL_URL}")
            return True

        if not self._last_error:
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
                f"{self._log_tail()}"
            )
            if self._stop.wait(backoff):
                return
            backoff = min(backoff * 2, 60.0)

            binary = self._resolve_binary()
            if binary:
                self._spawn(binary)

    def _log_tail(self, lines: int = 3) -> str:
        """The last few lines go2rtc wrote, for the message that reports a death.

        Nothing is logged at the default `warn` level while go2rtc is happy, so
        an empty tail is the normal case and reads correctly as one.
        """
        if not self._log_path:
            return ""
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="replace") as fh:
                tail = [ln.strip() for ln in fh.readlines()[-lines:] if ln.strip()]
        except Exception:
            return ""
        return ("; last output: " + " | ".join(tail)) if tail else ""

    # ── health ────────────────────────────────────────────────────────

    def api_responds(self) -> bool:
        """Is SOMETHING answering go2rtc's API on the configured address?"""
        try:
            with urlopen(f"{settings.GO2RTC_INTERNAL_URL}/api/streams", timeout=1.5) as r:
                return 200 <= r.status < 300
        except Exception:
            return False

    def healthy(self) -> bool:
        """Is OUR go2rtc up?

        Both halves are required, and the second one is the one that bites.
        Asking only whether :1984 answers makes any other process on that port
        -- most often a go2rtc orphaned by a previous hard-killed backend --
        look like a healthy child. The supervisor then reports "healthy"
        immediately after spawning a process that in fact exited because the
        port was already taken, and restarts it forever, two seconds at a time,
        while the UI is served by a relay holding a stale camera list.

        When this process is not the one managing go2rtc there is no child to
        check and the API answering is the whole of the question.
        """
        if settings.GO2RTC_MANAGED:
            with self._lock:
                proc = self._proc
            if proc is None or proc.poll() is not None:
                return False
        return self.api_responds()

    def wait_healthy(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                proc = self._proc
            if settings.GO2RTC_MANAGED and proc is not None and proc.poll() is not None:
                # Died during startup. Waiting out the rest of the timeout
                # would only delay a diagnosis we already have.
                self._last_error = (
                    f"exited during startup (code={proc.returncode})"
                    + (f"; see {self._log_path}" if self._log_path else "")
                )
                if self.api_responds():
                    self._last_error += (
                        f" -- and something else is already answering on "
                        f"{settings.GO2RTC_INTERNAL_URL}, most likely an "
                        f"orphaned go2rtc holding the port"
                    )
                return False
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
            "log_path": self._log_path,
            "streams": sorted(self._streams),
            "last_error": self._last_error,
        }


go2rtc = Go2RtcSupervisor()
