"""
Async Event Queue — Offloads all blocking I/O from the vision thread.

The vision pipeline publishes events (fire-and-forget) and immediately
continues processing the next frame. A dedicated consumer thread handles:
  - Database writes (DetectionEvent, TrackingSession, UnknownFace)
  - Snapshot saving to disk
  - WebSocket broadcast notifications
  - Embedding suggestion evaluation

Priority system:
  CRITICAL events (entry, exit, security alerts) → never dropped
  NORMAL events (snapshots, suggestions, unknowns) → dropped if queue is full

This eliminates 50-170ms of blocking I/O per detection frame.
"""

import logging
import queue
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels."""
    CRITICAL = 0   # entry, exit, security_alert — never dropped
    NORMAL = 1     # unknown_face, snapshot, suggestion — droppable


class EventQueue:
    """
    Thread-safe event queue with priority-based overflow handling.

    Critical events bypass the queue size limit.
    Normal events are dropped if the queue is full.
    """

    def __init__(self, max_size: int = 500):
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._max_size = max_size
        self._consumer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._handlers: Dict[str, Callable] = {}
        self._running = False

        # Stats
        self._processed = 0
        self._dropped = 0
        self._errors = 0
        self._lock = threading.Lock()

        # In-memory cache of known person IDs (for FK guard)
        # Refreshed on startup and whenever persons are added/removed
        self._known_person_ids: Set[str] = set()
        self._person_ids_lock = threading.Lock()

    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler function for a specific event type."""
        self._handlers[event_type] = handler
        logger.debug(f"EventQueue: registered handler for '{event_type}'")

    def refresh_person_ids(self):
        """
        Reload the set of known person IDs from the database.
        Called on startup and after person add/delete operations.
        """
        try:
            from app.db.session import SessionLocal
            from app.db.models import Person
            db = SessionLocal()
            try:
                ids = {row[0] for row in db.query(Person.id).all()}
                with self._person_ids_lock:
                    self._known_person_ids = ids
                logger.info(f"EventQueue: refreshed person ID cache ({len(ids)} persons)")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"EventQueue: failed to refresh person IDs: {e}")

    def is_known_person(self, person_id: str) -> bool:
        """Check if a person_id exists in the in-memory cache."""
        if not person_id:
            return False
        with self._person_ids_lock:
            return person_id in self._known_person_ids

    def add_person_id(self, person_id: str):
        """Add a person_id to the cache (call after enrollment)."""
        with self._person_ids_lock:
            self._known_person_ids.add(person_id)

    def remove_person_id(self, person_id: str):
        """Remove a person_id from the cache (call after deletion)."""
        with self._person_ids_lock:
            self._known_person_ids.discard(person_id)

    def put(self, event_type: str, data: Any, priority: EventPriority = EventPriority.NORMAL):
        """
        Enqueue an event for async processing.

        Critical events always go through (bypass queue size limit).
        Normal events are silently dropped if the queue is full.
        """
        event = {"type": event_type, "data": data, "priority": priority, "enqueued_at": time.time()}

        if priority == EventPriority.CRITICAL:
            # Critical events are never dropped — force into queue even if "full"
            # We use put with block=False and catch Full to use a secondary path
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                # Queue is full, but this is critical — force it by expanding
                # Use a put with a short timeout; if it blocks, log but still try
                try:
                    self._queue.put(event, timeout=1.0)
                except queue.Full:
                    # Absolute last resort — log and process synchronously
                    logger.error(
                        f"EventQueue CRITICAL: queue full, processing {event_type} synchronously"
                    )
                    self._dispatch(event)
        else:
            # Normal events — drop if full
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                with self._lock:
                    self._dropped += 1
                # Notify profiler
                try:
                    from app.core.profiler import profiler
                    profiler.record_event_dropped()
                except Exception:
                    pass
                if self._dropped % 50 == 1:
                    logger.warning(
                        f"EventQueue: dropped {event_type} event (queue full, "
                        f"total dropped: {self._dropped})"
                    )

    def start(self):
        """Start the consumer thread."""
        if self._running:
            return

        self._stop_event.clear()
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            name="event-queue-consumer",
            daemon=True,
        )
        self._consumer_thread.start()
        self._running = True
        logger.info(f"EventQueue started (max_size={self._max_size})")

    def stop(self):
        """Stop the consumer thread, draining remaining events."""
        if not self._running:
            return

        logger.info("EventQueue: stopping, draining remaining events...")
        self._stop_event.set()

        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=10)

        # Drain any remaining events
        drained = 0
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                self._dispatch(event)
                drained += 1
            except queue.Empty:
                break

        self._running = False
        logger.info(
            f"EventQueue stopped. Processed: {self._processed}, "
            f"Dropped: {self._dropped}, Errors: {self._errors}, "
            f"Drained on shutdown: {drained}"
        )

    def _consumer_loop(self):
        """Main consumer loop — runs in a background thread."""
        logger.info("EventQueue consumer thread started")

        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
                self._dispatch(event)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"EventQueue consumer error: {e}", exc_info=True)
                with self._lock:
                    self._errors += 1

        logger.info("EventQueue consumer thread exiting")

    def _dispatch(self, event: dict):
        """Dispatch an event to its registered handler."""
        event_type = event["type"]
        handler = self._handlers.get(event_type)

        if handler is None:
            logger.warning(f"EventQueue: no handler for event type '{event_type}'")
            return

        # Record wait time in profiler
        try:
            from app.core.profiler import profiler
            enqueued_at = event.get("enqueued_at", 0)
            if enqueued_at > 0:
                wait_ms = (time.time() - enqueued_at) * 1000
                profiler.record_event_dispatched(wait_ms)
        except Exception:
            pass

        try:
            handler(event["data"])
            with self._lock:
                self._processed += 1
        except Exception as e:
            with self._lock:
                self._errors += 1
            logger.error(
                f"EventQueue: handler error for '{event_type}': {e}",
                exc_info=True,
            )

    @property
    def stats(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            return {
                "queue_size": self._queue.qsize(),
                "max_size": self._max_size,
                "processed": self._processed,
                "dropped": self._dropped,
                "errors": self._errors,
                "running": self._running,
            }


# ── Singleton ─────────────────────────────────────────────────────
_event_queue: Optional[EventQueue] = None


def get_event_queue() -> EventQueue:
    """Get or create the singleton EventQueue."""
    global _event_queue
    if _event_queue is None:
        from app.config import settings
        _event_queue = EventQueue(max_size=settings.EVENT_QUEUE_SIZE)
    return _event_queue
