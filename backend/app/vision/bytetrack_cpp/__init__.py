"""ByteTrack's association core, compiled.

The extension is built out of tree (see setup.py) rather than on import, so a
missing build is an ImportError here and a logged fallback to the Python path
in bytetrack.py -- never a compiler invocation on a camera's first frame.
"""

from ._core import iou_batch, linear_assignment  # noqa: F401

__all__ = ["iou_batch", "linear_assignment"]
