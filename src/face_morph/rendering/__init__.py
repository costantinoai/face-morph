"""Mesh rendering subsystem.

Supports multiple rendering backends:
- PyTorch3D: GPU-accelerated rendering (requires CUDA)
- PyRender: CPU-based OpenGL rendering (50-100x faster than PyTorch3D on CPU)

The factory module automatically selects the appropriate renderer based on
available hardware and software.
"""

# These will be implemented later
__all__ = []
