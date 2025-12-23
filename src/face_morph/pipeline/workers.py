"""
Pipeline Worker Functions
==========================

Single responsibility: Parallel I/O and helper functions for morphing pipeline.

Worker functions must be module-level for multiprocessing.Pool to pickle them.
"""

from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import torch
from PIL import Image

from face_morph.core.converter import convert_fbx_to_obj, convert_obj_to_fbx
from face_morph.core.mesh_io import load_mesh, save_mesh
from face_morph.core.texture_io import load_texture
from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Helper Functions
# ==============================================================================

def ensure_obj_format(
    mesh_path: Path,
    blender_path: str
) -> Path:
    """
    Convert FBX to OBJ if needed.

    Single responsibility: Handle format conversion.

    PyTorch3D and PyRender work best with OBJ format, so we convert
    FBX files to OBJ before loading. The OBJ file is cached to avoid
    repeated conversions.

    Args:
        mesh_path: Path to mesh file (FBX or OBJ)
        blender_path: Path to Blender executable

    Returns:
        Path to OBJ file (original if already OBJ, converted if FBX)

    Raises:
        RuntimeError: If FBX to OBJ conversion fails

    Example:
        >>> fbx_path = Path("face.fbx")
        >>> obj_path = ensure_obj_format(fbx_path, "blender")
        >>> obj_path.suffix
        '.obj'
    """
    if mesh_path.suffix.lower() == '.fbx':
        obj_path = mesh_path.with_suffix('.obj')

        if not obj_path.exists():
            logger.info(f"Converting {mesh_path.name} to OBJ format...")

            success = convert_fbx_to_obj(mesh_path, obj_path, blender_path)
            if not success:
                raise RuntimeError(
                    f"Failed to convert {mesh_path.name} to OBJ format\n"
                    f"Ensure Blender is installed and accessible at: {blender_path}"
                )

            logger.debug(f"Converted {mesh_path.name} → {obj_path.name}")

        return obj_path

    return mesh_path


def load_mesh_with_texture(
    mesh_path: Path,
    device: torch.device
) -> Tuple:
    """
    Load mesh and optional texture from file.

    Single responsibility: Load mesh data with all auxiliary information.

    Args:
        mesh_path: Path to mesh file (OBJ format)
        device: PyTorch device to load mesh onto

    Returns:
        Tuple of (mesh, aux_data, texture, has_texture)
        - mesh: PyTorch3D Meshes object
        - aux_data: Dict with num_vertices, num_faces, verts_uvs, faces_uvs
        - texture: Texture tensor (H, W, 3) or None
        - has_texture: Boolean indicating if texture was loaded

    Example:
        >>> mesh, aux, tex, has_tex = load_mesh_with_texture(
        ...     Path("face.obj"), torch.device('cpu')
        ... )
        >>> aux['num_vertices']
        10000
        >>> has_tex
        True
    """
    # Load mesh with all auxiliary data
    mesh, aux = load_mesh(mesh_path, device)

    # Attempt to load texture
    texture = load_texture(aux, device)
    has_texture = texture is not None

    logger.debug(
        f"Loaded {mesh_path.name}: {aux['num_vertices']:,} vertices, "
        f"{aux['num_faces']:,} faces, texture={'Yes' if has_texture else 'No'}"
    )

    return mesh, aux, texture, has_texture


def create_pair_output_structure(
    output_dir: Path,
    stim1_name: str,
    stim2_name: str,
    timestamp: Optional[str] = None
) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Create output directory structure for a stimulus pair.

    Structure:
        results/<timestamp>/<stim1>_<stim2>/
        ├── session.log
        ├── png/
        ├── mesh/
        ├── shape_displacement_components.png
        ├── texture_difference_components.png
        └── animation.mp4

    Args:
        output_dir: Base output directory
        stim1_name: First stimulus name (filename without extension)
        stim2_name: Second stimulus name (filename without extension)
        timestamp: Optional timestamp string (generated if None)
                   Used for batch mode to group all pairs under same timestamp

    Returns:
        Tuple of (pair_dir, png_dir, mesh_dir, log_file, video_file)

    Example:
        >>> pair_dir, png_dir, mesh_dir, log, video = create_pair_output_structure(
        ...     Path("results"), "face1", "face2", "20231215_143022"
        ... )
        >>> pair_dir
        Path('results/20231215_143022/face1_face2')
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create pair directory: results/<timestamp>/<stim1>_<stim2>/
    pair_dir = output_dir / timestamp / f"{stim1_name}_{stim2_name}"

    # Create subdirectories
    png_dir = pair_dir / 'png'
    mesh_dir = pair_dir / 'mesh'

    png_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    log_file = pair_dir / 'session.log'
    video_file = pair_dir / 'animation.mp4'

    logger.debug(f"Created output structure: {pair_dir}")

    return pair_dir, png_dir, mesh_dir, log_file, video_file


def generate_morph_filename(
    stim1_name: str,
    stim2_name: str,
    ratio1: float,
    ratio2: float
) -> str:
    """
    Create permille notation filename for morph.

    Single responsibility: Format filenames consistently.

    Uses permille (parts per thousand) notation for precision.
    Format: "<stim1>-<permille1>_<stim2>-<permille2>"

    Args:
        stim1_name: First stimulus name
        stim2_name: Second stimulus name
        ratio1: Ratio for stimulus 1 (0.0 to 1.0)
        ratio2: Ratio for stimulus 2 (0.0 to 1.0)

    Returns:
        Filename string (without extension)

    Example:
        >>> generate_morph_filename("face1", "face2", 0.75, 0.25)
        'face1-750_face2-250'
        >>> generate_morph_filename("face1", "face2", 0.0, 1.0)
        'face1-000_face2-1000'
        >>> generate_morph_filename("face1", "face2", 0.5, 0.5)
        'face1-500_face2-500'
    """
    # Convert to permille (parts per thousand) for integer representation
    pct1 = int(round(ratio1 * 1000))
    pct2 = int(round(ratio2 * 1000))

    return f"{stim1_name}-{pct1:03d}_{stim2_name}-{pct2:03d}"


# ==============================================================================
# Parallel Worker Functions (must be module-level for pickling)
# ==============================================================================

def _save_png_worker(task: Tuple) -> Tuple[str, bool]:
    """
    Save single PNG image (worker for parallel processing).

    Must be module-level for multiprocessing.Pool to pickle it.

    Args:
        task: Tuple of (img_array, path, name)
              - img_array: Numpy array (H, W, 3) in [0, 255]
              - path: Output path for PNG file
              - name: Morph name for logging

    Returns:
        Tuple of (name, success)

    Example:
        >>> import numpy as np
        >>> img = np.zeros((512, 512, 3), dtype=np.uint8)
        >>> task = (img, Path("output.png"), "face1-500_face2-500")
        >>> name, success = _save_png_worker(task)
        >>> success
        True
    """
    img_array, path, name = task

    try:
        Image.fromarray(img_array).save(path)
        return (name, True)
    except Exception as e:
        logger.error(f"Failed to save {name}.png: {e}")
        return (name, False)


def _save_obj_worker(task: Tuple) -> Tuple[str, bool]:
    """
    Save single OBJ mesh (worker for parallel processing).

    Must be module-level for multiprocessing.Pool to pickle it.

    Note: Currently not used due to PyTorch3D Meshes pickling issues.
          OBJ files are saved sequentially in main process.

    Args:
        task: Tuple of (mesh, path, verts_uvs, texture, name)

    Returns:
        Tuple of (name, success)
    """
    mesh, path, verts_uvs, texture, name = task

    try:
        save_mesh(mesh, path, verts_uvs, texture)
        return (name, True)
    except Exception as e:
        logger.error(f"Failed to save {name}.obj: {e}")
        return (name, False)


def _convert_single_fbx_worker(args: Tuple) -> Tuple[str, bool]:
    """
    Convert single OBJ file to FBX (worker for parallel processing).

    Must be module-level for multiprocessing.Pool to pickle it.

    This worker enables parallel FBX conversion for significant speedup:
    - Sequential: ~1.5s per file × 41 files = ~60s
    - Parallel (8 workers): ~12s total (5x speedup)

    Args:
        args: Tuple of (obj_path, fbx_path, name, blender_path)
              - obj_path: Input OBJ file path
              - fbx_path: Output FBX file path
              - name: Morph name for logging
              - blender_path: Path to Blender executable

    Returns:
        Tuple of (name, success)

    Example:
        >>> args = (
        ...     Path("mesh.obj"),
        ...     Path("mesh.fbx"),
        ...     "face1-500_face2-500",
        ...     "blender"
        ... )
        >>> name, success = _convert_single_fbx_worker(args)
        >>> success
        True
    """
    obj_path, fbx_path, name, blender_path = args

    success = convert_obj_to_fbx(
        obj_path,
        fbx_path,
        blender_path=blender_path
    )

    if not success:
        logger.error(f"FBX conversion failed for {name}")

    return (name, success)
