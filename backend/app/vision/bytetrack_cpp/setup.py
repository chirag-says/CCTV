"""
Build the ByteTrack association extension.

    cd backend/app/vision/bytetrack_cpp
    python setup.py build_ext --inplace

The result lands beside this file as _core*.pyd (Windows) or .so, and
app/vision/bytetrack.py picks it up on import. It is NOT built automatically:
compiling on first import means a camera's first frame pays a toolchain
invocation, and a machine without a compiler would fail at the worst moment
rather than at install time. TRACKER_BACKEND falls back to the Python path
with a warning when the module is absent.
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "_core",
        ["_core.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="bytetrack_cpp",
    version="0.1.0",
    description="ByteTrack association core (IoU + linear assignment) in C++",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
