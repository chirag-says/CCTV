"""
Shared (cross-camera) UnknownRegistry: thread safety and the singleton.

The registry used to be one-per-pipeline and camera-thread-only. Sharing one
instance across 13 camera threads is what makes cross-camera stranger
agreement instant, and these tests pin the properties that sharing depends on:

  * concurrent resolve() calls on the same person yield ONE identity
  * concurrent resolve() calls on different people don't corrupt the index
  * the sighting cooldown is per (identity, camera) -- a shared registry with
    a global cooldown would let camera A's report suppress camera B's, hiding
    working cross-camera re-ID from detection_events
  * the DB refresh loader runs OUTSIDE the lock, so a slow query never stalls
    resolve() on every camera at once
  * exactly one thread runs the loader per refresh interval

No DB required.
"""

import threading
import time

import numpy as np
import pytest

import app.vision.unknown_registry as ur
from app.vision.unknown_registry import UnknownRegistry, get_shared_registry


DIM = 512


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _jitter(base: np.ndarray, seed: int, scale: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base + rng.normal(size=DIM).astype(np.float32) * scale
    return v / np.linalg.norm(v)


# ── Singleton ────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_shared_registry():
    """Isolate the module-level singleton from other tests and restore it."""
    saved = ur._shared_registry
    ur._shared_registry = None
    yield
    ur._shared_registry = saved


def test_shared_factory_returns_one_instance(clean_shared_registry):
    a = get_shared_registry()
    b = get_shared_registry()
    assert a is b


def test_shared_factory_reads_settings(clean_shared_registry):
    from app.config import settings
    reg = get_shared_registry()
    assert reg.match_threshold == settings.UNKNOWN_SIMILARITY_THRESHOLD
    assert reg.max_entries == settings.UNKNOWN_REGISTRY_MAX_ENTRIES


# ── Concurrency ──────────────────────────────────────────────────────────────

def test_concurrent_resolve_same_person_mints_once():
    """13 'cameras' seeing the same face at once must agree on one identity."""
    reg = UnknownRegistry(match_threshold=0.70)
    base = _vec(42)
    n_threads, per_thread = 13, 20
    barrier = threading.Barrier(n_threads)
    ids = [set() for _ in range(n_threads)]
    errors = []

    def camera(i):
        try:
            barrier.wait(timeout=10)
            for k in range(per_thread):
                uid, label, _ = reg.resolve(_jitter(base, seed=i * 1000 + k))
                ids[i].add(uid)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=camera, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    all_ids = set().union(*ids)
    assert len(all_ids) == 1, f"same person split into {len(all_ids)} identities"
    assert reg.stat_minted == 1


def test_concurrent_resolve_distinct_people_stay_distinct():
    reg = UnknownRegistry(match_threshold=0.50)
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    results = {}
    errors = []

    def camera(i):
        try:
            base = _vec(i)
            barrier.wait(timeout=10)
            uid_first, _, _ = reg.resolve(base)
            for k in range(10):
                uid, _, _ = reg.resolve(_jitter(base, seed=i * 100 + k, scale=0.01))
                assert uid == uid_first, "identity flapped under concurrency"
            results[i] = uid_first
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=camera, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert len(set(results.values())) == n_threads
    assert reg.size == n_threads
    # Index survived concurrent appends: every identity still resolves home.
    for i, uid in results.items():
        again, _, is_new = reg.resolve(_vec(i))
        assert again == uid and not is_new


# ── Per-camera sighting cooldown ─────────────────────────────────────────────

def test_sighting_cooldown_is_per_camera():
    reg = UnknownRegistry(sighting_cooldown=60.0)
    uid, _, _ = reg.resolve(_vec(7))

    assert reg.should_report_sighting(uid, "cam-a") is True
    # A different camera seeing the same person must still get its report --
    # that per-camera event row IS the cross-camera re-ID evidence.
    assert reg.should_report_sighting(uid, "cam-b") is True
    assert reg.should_report_sighting(uid, "cam-a") is False
    assert reg.should_report_sighting(uid, "cam-b") is False


def test_sighting_cooldown_default_camera_arg():
    """Legacy single-arg call keeps throttling (used by older call sites/tests)."""
    reg = UnknownRegistry(sighting_cooldown=60.0)
    uid, _, _ = reg.resolve(_vec(8))
    assert reg.should_report_sighting(uid) is True
    assert reg.should_report_sighting(uid) is False


# ── Refresh: loader outside the lock, once per interval ──────────────────────

def test_resolve_not_blocked_while_loader_runs():
    reg = UnknownRegistry(refresh_interval=0.0)
    in_loader = threading.Event()
    release = threading.Event()

    def slow_loader():
        in_loader.set()
        release.wait(timeout=5)
        return []

    t = threading.Thread(target=lambda: reg.maybe_refresh(slow_loader))
    t.start()
    assert in_loader.wait(timeout=5)

    start = time.monotonic()
    _, _, is_new = reg.resolve(_vec(9))
    elapsed = time.monotonic() - start

    release.set()
    t.join(timeout=5)
    assert is_new
    assert elapsed < 2.0, "resolve() blocked behind the refresh loader's DB query"


def test_refresh_runs_once_per_interval_across_threads():
    reg = UnknownRegistry(refresh_interval=1000.0)
    calls = []
    barrier = threading.Barrier(6)
    outcomes = []

    def loader():
        calls.append(1)
        return []

    def worker():
        barrier.wait(timeout=10)
        outcomes.append(reg.maybe_refresh(loader))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(calls) == 1, "shared registry must not run one DB refresh per camera"
    assert outcomes.count(True) == 1
