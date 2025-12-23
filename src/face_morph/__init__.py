"""Face Morph - Production-ready 3D face morphing with GPU acceleration.

This package provides tools for morphing 3D face meshes with advanced heatmap
visualization, GPU acceleration, and comprehensive analysis capabilities.

Quick Start:
    >>> from face_morph.core import load_mesh, create_morpher
    >>> from face_morph.utils.logging import setup_logger
    >>> import torch
    >>>
    >>> # Setup
    >>> logger = setup_logger(verbose=True)
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>>
    >>> # Load meshes
    >>> mesh1, aux1 = load_mesh('face1.obj', device)
    >>> mesh2, aux2 = load_mesh('face2.obj', device)
    >>>
    >>> # Morph
    >>> morpher = create_morpher(device)
    >>> morphed = morpher.morph_mesh(mesh1, mesh2, ratio=0.5)

Modules:
    core: Core morphing algorithms (morpher, mesh I/O, validation)
    rendering: Mesh rendering (PyTorch3D, PyRender)
    visualization: Heatmaps and video generation
    pipeline: High-level orchestration and configuration
    cli: Command-line interface
    utils: Logging, parallelization, context managers
"""

__version__ = "1.0.0"
__author__ = "Face Morph Contributors"
__license__ = "MIT"

# Expose key classes and functions at package level
from .core.exceptions import (
    FaceMorphError,
    TopologyMismatchError,
    TextureMissingError,
    RendererError,
    CUDANotAvailableError,
    MeshLoadError,
    MeshSaveError,
    ValidationError,
    ConversionError,
)

from .utils.logging import setup_logger, get_logger

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Exceptions
    "FaceMorphError",
    "TopologyMismatchError",
    "TextureMissingError",
    "RendererError",
    "CUDANotAvailableError",
    "MeshLoadError",
    "MeshSaveError",
    "ValidationError",
    "ConversionError",
    # Logging
    "setup_logger",
    "get_logger",
]
