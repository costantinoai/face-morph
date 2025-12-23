"""
Face Morphing Library
=====================

A modular library for 3D face mesh morphing with GPU acceleration.

Components:
    - mesh_io: Mesh loading and saving
    - texture_io: Texture loading and saving
    - morpher: Core morphing algorithms
    - validator: Input validation
    - converter: Format conversion utilities
    - renderer: Mesh rendering to 2D images
    - heatmap: Variance computation and visualization
    - video_utils: Video creation from frames
"""

__version__ = "1.0.0"
__author__ = "@costantinoai"

from .utils import set_verbose, vprint
from .mesh_io import load_mesh, save_mesh
from .texture_io import load_texture, save_texture, has_texture
from .morpher import create_morpher
from .validator import validate_input_file, validate_device, validate_ratios
from .converter import convert_fbx_to_obj, convert_obj_to_fbx
from .video_utils import create_video_from_frames, check_ffmpeg_available
from .renderer import create_renderer, MeshRenderer3D
from .heatmap import (
    compute_shape_displacement_components,
    compute_texture_difference_components,
    create_shape_displacement_visualization,
    create_texture_difference_components_visualization,
)

__all__ = [
    'set_verbose',
    'vprint',
    'load_mesh',
    'save_mesh',
    'load_texture',
    'save_texture',
    'has_texture',
    'create_morpher',
    'validate_input_file',
    'validate_device',
    'validate_ratios',
    'convert_fbx_to_obj',
    'convert_obj_to_fbx',
    'create_video_from_frames',
    'check_ffmpeg_available',
    'create_renderer',
    'MeshRenderer3D',
    'compute_shape_displacement_components',
    'compute_texture_difference_components',
    'create_shape_displacement_visualization',
    'create_texture_difference_components_visualization',
]
