"""
Pipeline Benchmark — Compatibility wrapper.

Delegates to the new PerformanceProfiler (app.core.profiler).
Kept for backward compatibility with any remaining references.

Usage (unchanged):
    from app.vision.benchmarks import pipeline_bench

    with pipeline_bench.measure("scrfd_detect"):
        face_locations, embeddings, rgb_frame = self.detector.detect_faces_with_embeddings(frame)

    with pipeline_bench.measure("arcface_match"):
        result = self.recognizer.recognize_face(embedding)
"""

from app.core.profiler import profiler


class PipelineBenchmark:
    """
    Thin compatibility wrapper around the new PerformanceProfiler.

    All timing and reporting is delegated to the profiler singleton.
    """

    def __init__(self, log_interval: int = 100):
        self._profiler = profiler

    @property
    def enabled(self) -> bool:
        return self._profiler.enabled

    def measure(self, stage_name: str):
        """Context manager to measure a pipeline stage."""
        return self._profiler.measure(stage_name)

    def tick_frame(self):
        """Called once per detection frame (now handled by camera_worker)."""
        # Tick is now called in camera_worker._run_loop for every frame.
        # This is kept as a no-op for backward compatibility.
        pass


# -- Singleton (compatibility) -------------------------------------------------
pipeline_bench = PipelineBenchmark(log_interval=30)
