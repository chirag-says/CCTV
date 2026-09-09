"""Vision pipeline package.

Silencing ultralytics telemetry happens here because this is the only place
every YOLO user in this codebase passes through: nine modules under
app/vision import `from ultralytics import YOLO`, and none of them is
imported before this package's __init__.
"""

import logging

logger = logging.getLogger(__name__)


def _disable_ultralytics_telemetry():
    """Stop ultralytics posting usage analytics to Google.

    ultralytics ships with `sync: true` and posts an event to
    google-analytics.com from inside predict() (see its
    utils/events.py). Nothing in this repo asked for that. It leaves the
    site on every inference batch, it is a third party we never chose, and
    it fails the "no network calls" acceptance test on its own -- the
    20-camera baseline recorded 12 of these and no other outbound traffic.

    Written to the user-level settings file by ultralytics itself, so this
    is a no-op after the first run.
    """
    try:
        from app.config import settings
        from ultralytics import settings as ul_settings

        wanted = bool(getattr(settings, "ULTRALYTICS_TELEMETRY", False))
        if bool(ul_settings.get("sync")) != wanted:
            ul_settings.update({"sync": wanted})
            logger.info(f"ultralytics telemetry sync={wanted}")
    except Exception as e:  # never let telemetry config break startup
        logger.debug(f"could not set ultralytics telemetry: {e}")


_disable_ultralytics_telemetry()
