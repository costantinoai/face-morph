#!/usr/bin/env python3
"""
3D Face Morphing - Production Pipeline
=======================================

Dual-mode script: Run from IDE with defaults or CLI with arguments.

## Morphing Algorithm

The pipeline performs linear interpolation (LERP) between two 3D face meshes:

    V_morph = (1 - α) * V₁ + α * V₂
    T_morph = (1 - α) * T₁ + α * T₂

Where:
- V₁, V₂: Vertex positions of input meshes
- T₁, T₂: Texture maps (H×W×3 tensors)
- α ∈ [0, 1]: Interpolation weight
- 41 frames generated: α = [0.000, 0.025, 0.050, ..., 1.000] (steps of 0.025)

## Reproducibility

For deterministic results:
1. Set PyTorch seed: `torch.manual_seed(42)`
2. Use same precision mode (FP16 vs FP32)
3. Use same device (CPU vs GPU)
4. Use same PyTorch version

## CPU vs GPU Implementation

### GPU Mode (Recommended):
- Uses `torch.autocast(device_type='cuda', dtype=torch.float16)` for mixed precision
- 2-3x faster computation via FP16
- Small numerical differences vs CPU due to reduced precision (ε < 1e-4)
- Batch processes all 41 morphs on GPU in ~0.5s
- FBX conversion: Sequential (CPU-bound via Blender subprocess)

### CPU Mode:
- Uses full FP32 precision
- No mixed precision autocast
- Slower computation (~4-5s for 41 morphs)
- Deterministic within FP32 precision
- FBX conversion: Sequential (same as GPU)

### Performance Optimization:
FBX conversion via Blender is CPU-bound (~1.5s per file, ~60s total for 41 frames).
**Parallel FBX conversion** using multiprocessing.Pool with 8 workers:
- Sequential: ~60s for 41 files
- Parallel (8 workers): ~12s for 41 files (**5x speedup**)
- Configurable via `parallel_fbx` and `num_workers` settings

## Dependencies

- PyTorch >= 2.0 with CUDA 12.4
- PyTorch3D >= 0.7.8
- Blender >= 4.0 (for FBX export)
- ffmpeg (for video creation)

## Usage

IDE Mode:
    1. Edit DEFAULT_CONFIG below
    2. Run: python run.py

CLI Mode:
    ./morph -i1 face1.fbx -i2 face2.fbx --gpu
    ./morph --batch data/ --gpu
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from itertools import combinations
from multiprocessing import Pool, cpu_count
import torch
import numpy as np

from lib import (
    set_verbose,
    vprint,
    validate_input_file,
    validate_device,
    validate_ratios,
    convert_fbx_to_obj,
    convert_obj_to_fbx,
    load_mesh,
    save_mesh,
    load_texture,
    save_texture,
    create_morpher,
    create_video_from_frames,
    check_ffmpeg_available,
    create_renderer,
    create_shape_displacement_visualization,
    create_texture_difference_components_visualization,
)
from lib.renderer_pyrender import create_pyrender_renderer


# ==============================================================================
# DEFAULT CONFIGURATION (for IDE usage)
# ==============================================================================

def generate_video_ratios(step: int = 25) -> List[Tuple[float, float]]:
    """
    Generate fine-grained ratios for video frames.

    Creates ratios from 0-1000 to 1000-0 in specified steps.

    Args:
        step: Step size in permille (default: 25)

    Returns:
        List of (ratio1, ratio2) tuples
    """
    ratios = []
    for r1_permille in range(0, 1001, step):
        r2_permille = 1000 - r1_permille
        ratios.append((r1_permille / 1000.0, r2_permille / 1000.0))
    return ratios


DEFAULT_CONFIG = {
    'input_mesh_1': 'data/male1.fbx',
    'input_mesh_2': 'data/male2.fbx',
    'output_dir': 'results',
    'morph_ratios': generate_video_ratios(step=25),  # 41 frames (0-1000 in steps of 25)
    'use_gpu': True,
    'use_mixed_precision': True,
    'blender_path': 'blender',
    'verbose': True,
    'video_fps': 30,
    'parallel_fbx': True,  # Parallelize FBX conversion
    'num_workers': max(1, cpu_count() - 1),  # Use all cores except one for system responsiveness
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

class Logger:
    """Simple logger for progress tracking."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.step_num = 0

    def header(self, text: str):
        """Print header."""
        print(f"\n{'='*70}")
        print(f"{text:^70}")
        print(f"{'='*70}\n")

    def step(self, description: str):
        """Print step header."""
        if self.verbose:
            self.step_num += 1
            print(f"STEP {self.step_num}: {description}...")

    def info(self, message: str, indent: int = 2):
        """Print info message."""
        if self.verbose:
            print(f"{' '*indent}{message}")

    def success(self, message: str, indent: int = 2):
        """Print success message."""
        if self.verbose:
            print(f"{' '*indent}✓ {message}")

    def warning(self, message: str, indent: int = 2):
        """Print warning message."""
        if self.verbose:
            print(f"{' '*indent}⚠ {message}")

    def blank(self):
        """Print blank line."""
        if self.verbose:
            print()


def ensure_obj_format(
    mesh_path: Path,
    blender_path: str,
    logger: Logger
) -> Path:
    """
    Convert FBX to OBJ if needed.

    Single responsibility: Handle format conversion.

    Args:
        mesh_path: Path to mesh file
        blender_path: Path to Blender executable
        logger: Logger instance

    Returns:
        Path to OBJ file
    """
    if mesh_path.suffix.lower() == '.fbx':
        obj_path = mesh_path.with_suffix('.obj')

        if not obj_path.exists():
            logger.info(f"Converting {mesh_path.name} to OBJ...")

            if not convert_fbx_to_obj(mesh_path, obj_path, blender_path):
                raise RuntimeError(f"Failed to convert {mesh_path.name}")

        return obj_path

    return mesh_path


def load_mesh_with_texture(
    mesh_path: Path,
    device: torch.device,
    logger: Logger
) -> Tuple:
    """
    Load mesh and optional texture.

    Single responsibility: Load mesh data.

    Returns:
        Tuple of (mesh, aux_data, texture, has_texture)
    """
    mesh, aux = load_mesh(mesh_path, device)
    texture = load_texture(aux, device)
    has_texture = texture is not None

    logger.success(
        f"Mesh: {aux['num_vertices']:,} vertices, {aux['num_faces']:,} faces"
    )

    return mesh, aux, texture, has_texture


# Worker functions for parallel I/O (must be module-level for pickling)
def _save_png_worker(task):
    """Save single PNG image."""
    from PIL import Image
    img_array, path, name = task
    try:
        Image.fromarray(img_array).save(path)
        return (name, True)
    except Exception as e:
        print(f"Warning: Failed to save {name}.png: {e}")
        return (name, False)


def _save_obj_worker(task):
    """Save single OBJ mesh."""
    mesh, path, verts_uvs, texture, name = task
    try:
        save_mesh(mesh, path, verts_uvs, texture)
        return (name, True)
    except Exception as e:
        print(f"Warning: Failed to save {name}.obj: {e}")
        return (name, False)


def create_pair_output_structure(
    output_dir: Path,
    stim1_name: str,
    stim2_name: str,
    timestamp: str = None
) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Create output structure for a stimulus pair.

    Structure:
        results/<timestamp>/<stim1>_<stim2>/
        ├── session.log
        ├── png/
        ├── mesh/
        └── animation.mp4

    Args:
        output_dir: Base output directory
        stim1_name: First stimulus name
        stim2_name: Second stimulus name
        timestamp: Optional timestamp (generated if None)

    Returns:
        Tuple of (pair_dir, png_dir, mesh_dir, log_file, video_file)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create pair directory
    pair_dir = output_dir / timestamp / f"{stim1_name}_{stim2_name}"

    # Create subdirectories
    png_dir = pair_dir / 'png'
    mesh_dir = pair_dir / 'mesh'

    png_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    log_file = pair_dir / 'session.log'
    video_file = pair_dir / 'animation.mp4'

    return pair_dir, png_dir, mesh_dir, log_file, video_file


def setup_session_logging(log_file: Path, verbose: bool = True):
    """
    Setup logging to both file and console.

    Args:
        log_file: Path to log file
        verbose: If True, also log to console

    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger('morphing_session')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (if verbose)
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def generate_morph_filename(
    stim1_name: str,
    stim2_name: str,
    ratio1: float,
    ratio2: float
) -> str:
    """
    Create permille notation filename.

    Single responsibility: Format filenames.

    Returns:
        Filename like "stim1-750_stim2-250"
    """
    pct1 = int(round(ratio1 * 1000))
    pct2 = int(round(ratio2 * 1000))
    return f"{stim1_name}-{pct1:03d}_{stim2_name}-{pct2:03d}"


def _convert_single_fbx_worker(args):
    """
    Helper function for parallel FBX conversion.

    Must be at module level for multiprocessing.Pool to pickle it.

    Args:
        args: Tuple of (obj_path, fbx_path, name, blender_path)

    Returns:
        Tuple of (name, success)
    """
    obj_path, fbx_path, name, blender_path = args
    success = convert_obj_to_fbx(obj_path, fbx_path, blender_path=blender_path)
    return (name, success)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='3D Face Morphing with GPU Acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s -i1 face1.fbx -i2 face2.fbx

  # Batch mode (all unique pairs in folder)
  %(prog)s --batch data/

  # Custom ratios (comma-separated pairs)
  %(prog)s -i1 face1.fbx -i2 face2.fbx --ratios "0.9,0.1 0.5,0.5 0.1,0.9"

  # CPU mode
  %(prog)s -i1 face1.fbx -i2 face2.fbx --no-gpu

  # Custom output directory
  %(prog)s -i1 face1.fbx -i2 face2.fbx -o my_morphs
        """
    )

    # Input/Output
    parser.add_argument(
        '-i1', '--input1',
        type=str,
        help='First mesh file (FBX or OBJ)'
    )
    parser.add_argument(
        '-i2', '--input2',
        type=str,
        help='Second mesh file (FBX or OBJ)'
    )
    parser.add_argument(
        '--batch',
        type=str,
        metavar='FOLDER',
        help='Batch mode: Process all unique pairs in folder (excludes self-pairs)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory (default: results)'
    )

    # Morph ratios
    parser.add_argument(
        '--ratios',
        type=str,
        help='Morph ratios as space-separated pairs (e.g., "0.9,0.1 0.5,0.5")'
    )

    # GPU settings
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration (use CPU)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable mixed precision (FP16)'
    )

    # Misc
    parser.add_argument(
        '--blender',
        type=str,
        help='Path to Blender executable (default: blender)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Disable verbose output'
    )

    return parser.parse_args()


def discover_mesh_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    """
    Discover all unique mesh pairs in a folder.

    Excludes self-pairs (a-a) and treats pairs as unordered (a-b == b-a).
    Prefers FBX over OBJ when both exist for same mesh.

    Args:
        folder: Directory containing mesh files

    Returns:
        List of unique mesh file pairs
    """
    # Find all mesh files
    all_files = list(folder.iterdir())

    # Group by stem (filename without extension)
    mesh_dict = {}
    for f in all_files:
        if f.suffix.lower() in {'.fbx', '.obj'}:
            stem = f.stem
            # Prefer FBX over OBJ
            if stem not in mesh_dict or f.suffix.lower() == '.fbx':
                mesh_dict[stem] = f

    mesh_files = sorted(mesh_dict.values())

    if len(mesh_files) < 2:
        raise ValueError(
            f"Need at least 2 unique mesh files in {folder}\n"
            f"Found: {len(mesh_files)} files"
        )

    # Generate unique combinations (no self-pairs, unordered)
    pairs = list(combinations(mesh_files, 2))

    return pairs


def merge_config(args: argparse.Namespace) -> dict:
    """
    Merge CLI arguments with defaults.

    CLI arguments override defaults.
    """
    config = DEFAULT_CONFIG.copy()

    # Check for batch mode
    if args.batch:
        folder = Path(args.batch)
        if not folder.is_dir():
            raise ValueError(f"Batch folder not found: {folder}")

        config['batch_mode'] = True
        config['batch_folder'] = folder
    else:
        config['batch_mode'] = False

    # Override with CLI args if provided
    if args.input1:
        config['input_mesh_1'] = args.input1
    if args.input2:
        config['input_mesh_2'] = args.input2
    if args.output:
        config['output_dir'] = args.output
    if args.blender:
        config['blender_path'] = args.blender

    # Parse ratios if provided
    if args.ratios:
        pairs = []
        for pair_str in args.ratios.split():
            r1, r2 = map(float, pair_str.split(','))
            pairs.append((r1, r2))
        config['morph_ratios'] = pairs

    # GPU flags
    if args.no_gpu:
        config['use_gpu'] = False
    elif args.gpu:
        config['use_gpu'] = True

    if args.no_amp:
        config['use_mixed_precision'] = False

    # Verbose flags
    if args.quiet:
        config['verbose'] = False
    elif args.verbose:
        config['verbose'] = True

    return config


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_morphing_pipeline(config: dict):
    """
    Execute the complete morphing pipeline with video generation.

    Single responsibility: Coordinate all steps.

    Output structure:
        results/<timestamp>/<stim1>_<stim2>/
        ├── session.log
        ├── png/
        │   ├── stim1-0000_stim2-1000.png
        │   ├── stim1-0025_stim2-0975.png
        │   └── ...
        ├── mesh/
        │   ├── stim1-0000_stim2-1000.fbx
        │   ├── stim1-0025_stim2-0975.fbx
        │   └── ...
        └── animation.mp4
    """
    # Set global verbose mode
    set_verbose(config['verbose'])

    # Get stimulus names
    stim1_name = Path(config['input_mesh_1']).stem
    stim2_name = Path(config['input_mesh_2']).stem

    # Create output structure
    output_dir = Path(config['output_dir'])
    timestamp = config.get('timestamp')  # Allow shared timestamp in batch mode
    pair_dir, png_dir, mesh_dir, log_file, video_file = create_pair_output_structure(
        output_dir, stim1_name, stim2_name, timestamp
    )

    # Setup session logging
    session_logger = setup_session_logging(log_file, verbose=config['verbose'])
    logger = Logger(verbose=config['verbose'])

    def log(message):
        """Log to both session log and console."""
        session_logger.info(message)

    log("="*70)
    log("3D FACE MORPHING PIPELINE")
    log("="*70)
    log("")

    # -------------------------------------------------------------------------
    # STEP 1: Validate inputs
    # -------------------------------------------------------------------------

    logger.step("Validating inputs")
    log("STEP 1: Validating inputs...")

    mesh1_path = validate_input_file(config['input_mesh_1'])
    mesh2_path = validate_input_file(config['input_mesh_2'])
    validate_ratios(config['morph_ratios'])

    device_str = 'cuda' if config['use_gpu'] else 'cpu'
    device = validate_device(device_str)

    log(f"  Input 1: {mesh1_path.name}")
    log(f"  Input 2: {mesh2_path.name}")
    log(f"  Device: {device}")
    log(f"  Morph count: {len(config['morph_ratios'])}")
    log("")

    logger.success(f"Input 1: {mesh1_path.name}")
    logger.success(f"Input 2: {mesh2_path.name}")
    logger.success(f"Device: {device}")
    logger.success(f"Morph count: {len(config['morph_ratios'])}")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 2: Convert to OBJ format
    # -------------------------------------------------------------------------

    logger.step("Preparing mesh files")
    log("STEP 2: Preparing mesh files...")

    mesh1_path = ensure_obj_format(mesh1_path, config['blender_path'], logger)
    mesh2_path = ensure_obj_format(mesh2_path, config['blender_path'], logger)

    log(f"  Mesh 1: {mesh1_path.name}")
    log(f"  Mesh 2: {mesh2_path.name}")
    log("")

    logger.success(f"Mesh 1: {mesh1_path.name}")
    logger.success(f"Mesh 2: {mesh2_path.name}")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 3: Load meshes and textures
    # -------------------------------------------------------------------------

    logger.step("Loading meshes and textures")
    log("STEP 3: Loading meshes and textures...")

    mesh1, aux1, texture1, has_texture1 = load_mesh_with_texture(
        mesh1_path, device, logger
    )
    mesh2, aux2, texture2, has_texture2 = load_mesh_with_texture(
        mesh2_path, device, logger
    )

    has_textures = has_texture1 and has_texture2

    log(f"  Mesh 1: {aux1['num_vertices']:,} vertices, {aux1['num_faces']:,} faces")
    log(f"  Mesh 2: {aux2['num_vertices']:,} vertices, {aux2['num_faces']:,} faces")

    if has_textures:
        log(f"  Textures: Both meshes have textures")
        logger.success("Textures: Both meshes have textures - will morph textures")
    elif has_texture1 or has_texture2:
        log(f"  Warning: Only one mesh has texture - shape only")
        logger.warning("Textures: Only one mesh has texture - will morph shapes only")
    else:
        log(f"  Warning: No textures - shape only")
        logger.warning("Textures: No textures found - will morph shapes only")

    log("")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 4: Validate topology
    # -------------------------------------------------------------------------

    logger.step("Validating topology")
    log("STEP 4: Validating topology...")

    if aux1['num_vertices'] != aux2['num_vertices']:
        error_msg = (
            f"Meshes have different vertex counts:\n"
            f"  Mesh 1: {aux1['num_vertices']:,} vertices\n"
            f"  Mesh 2: {aux2['num_vertices']:,} vertices\n"
        )
        log(f"  ERROR: {error_msg}")
        raise ValueError(error_msg)

    if aux1['num_faces'] != aux2['num_faces']:
        log(f"  Warning: Different face counts but same vertex count")
        logger.warning("Different face counts but same vertex count")

    log(f"  Topology: IDENTICAL ({aux1['num_vertices']:,} vertices each)")
    log("")

    logger.success(f"Topology: IDENTICAL ({aux1['num_vertices']:,} vertices each)")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 5: Create morpher
    # -------------------------------------------------------------------------

    logger.step("Initializing morpher")
    log("STEP 5: Initializing morpher...")

    use_amp = config['use_mixed_precision'] and config['use_gpu']
    morpher = create_morpher(device, use_amp=use_amp)

    if config['use_gpu'] and use_amp:
        log("  GPU acceleration: ENABLED (mixed precision)")
        logger.success("GPU acceleration: ENABLED (mixed precision)")
    elif config['use_gpu']:
        log("  GPU acceleration: ENABLED (full precision)")
        logger.success("GPU acceleration: ENABLED (full precision)")
    else:
        log("  CPU mode: ENABLED")
        logger.success("CPU mode: ENABLED")

    log("")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 5.5: Initialize renderer
    # -------------------------------------------------------------------------

    logger.step("Initializing renderer")
    log("STEP 5.5: Initializing renderer...")

    # Choose renderer based on device
    if device.type == 'cuda':
        # GPU: Use PyTorch3D (CUDA-accelerated)
        renderer = create_renderer(device, image_size=512)
        renderer_type = 'pytorch3d'
        log("  Renderer ready for mesh visualization (GPU mode - PyTorch3D)")
        logger.success("Renderer initialized (GPU mode - PyTorch3D accelerated)")
    else:
        # CPU: Use PyRender (OpenGL/OSMesa - 50-100x faster than PyTorch3D on CPU)
        renderer = create_pyrender_renderer(image_size=512)
        renderer_type = 'pyrender'
        log("  Renderer ready for mesh visualization (CPU mode - PyRender/OpenGL)")
        logger.success("Renderer initialized (CPU mode - PyRender 50-100x faster)")
    log("")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 6: Generate morphs
    # -------------------------------------------------------------------------

    logger.step("Generating morphs")
    log("STEP 6: Generating morphs...")

    morphed_results = morpher.batch_morph(
        mesh1, mesh2,
        texture1 if has_textures else None,
        texture2 if has_textures else None,
        config['morph_ratios']
    )

    log(f"  Generated {len(morphed_results)} morphs")
    log("")

    logger.success(f"Generated {len(morphed_results)} morphs")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 7: Save results (PNG + OBJ + FBX)
    # -------------------------------------------------------------------------

    logger.step("Saving frames and meshes")
    log("STEP 7: Saving frames and meshes...")

    # Temporary directory for OBJ files
    import tempfile
    import shutil
    temp_dir = Path(tempfile.mkdtemp())

    # Save all PNG and OBJ files first
    # Also collect meshes and textures for heatmap generation
    obj_fbx_pairs = []
    morphed_meshes = []  # For shape heatmap
    morphed_textures = []  # For texture heatmap

    from PIL import Image

    # === RENDERING + PARALLEL I/O ===
    # Render sequentially (GPU limitation), prepare for parallel saves
    log("  Rendering meshes and preparing save tasks...")
    logger.info("Rendering meshes...")

    # Free GPU memory from morphing
    torch.cuda.empty_cache()

    rendered_images = []
    verts_uvs = aux1['verts_uvs'] if has_textures and aux1.get('verts_uvs') is not None else None
    faces_uvs = aux1.get('faces_uvs') if has_textures else None

    for idx, (mesh, texture) in enumerate(morphed_results, 1):
        if renderer_type == 'pytorch3d':
            # PyTorch3D renderer (GPU)
            if has_textures and texture is not None:
                rendered_img = renderer.render_mesh(mesh, texture=texture, verts_uvs=verts_uvs, faces_uvs=faces_uvs)
            else:
                rendered_img = renderer.render_mesh(mesh)
        else:
            # PyRender renderer (CPU) - needs numpy arrays
            vertices = mesh.verts_packed().cpu().numpy()
            faces = mesh.faces_packed().cpu().numpy()

            if has_textures and texture is not None and verts_uvs is not None:
                # Convert texture and UV coords to numpy
                texture_np = texture.cpu().numpy() if torch.is_tensor(texture) else texture
                uv_coords_np = verts_uvs.cpu().numpy() if torch.is_tensor(verts_uvs) else verts_uvs
                uv_faces_np = faces_uvs.cpu().numpy() if torch.is_tensor(faces_uvs) else faces_uvs if faces_uvs is not None else None

                # Handle batched UV coordinates
                if uv_coords_np.ndim == 3:
                    uv_coords_np = uv_coords_np[0]  # Take first batch
                if uv_faces_np is not None and uv_faces_np.ndim == 3:
                    uv_faces_np = uv_faces_np[0]  # Take first batch

                # Render with texture (renderer handles vertex splitting internally)
                rendered_img = renderer.render_mesh(
                    vertices, faces,
                    texture=texture_np,
                    uv_coords=uv_coords_np,
                    uv_faces=uv_faces_np
                )
            else:
                # Render without texture
                rendered_img = renderer.render_mesh(vertices, faces)

        rendered_images.append(rendered_img)

    log(f"  ✓ Rendered {len(rendered_images)} images")
    logger.info(f"Rendered {len(rendered_images)} images")

    # === PARALLEL PNG SAVES + SEQUENTIAL OBJ SAVES ===
    log("  Saving rendered images and meshes...")

    # Prepare PNG save tasks
    png_save_tasks = []

    for idx, ((mesh, texture), (r1, r2), rendered_img) in enumerate(
        zip(morphed_results, config['morph_ratios'], rendered_images), 1
    ):
        morph_name = generate_morph_filename(stim1_name, stim2_name, r1, r2)

        # Collect for heatmaps
        morphed_meshes.append(mesh.verts_packed())
        if has_textures and texture is not None:
            morphed_textures.append(texture)

        # Queue PNG save task
        png_path = png_dir / f"{morph_name}.png"
        png_save_tasks.append((rendered_img, png_path, morph_name))

        # Save OBJ sequentially (PyTorch Mesh objects don't pickle well)
        obj_path = temp_dir / f"{morph_name}.obj"
        save_mesh(mesh, obj_path, aux1['verts_uvs'], texture if has_textures else None)

        # Queue for FBX conversion
        fbx_path = mesh_dir / f"{morph_name}.fbx"
        obj_fbx_pairs.append((obj_path, fbx_path, morph_name))

    # Save PNGs in parallel
    num_workers = max(1, cpu_count() - 1)
    log(f"  Saving {len(png_save_tasks)} PNGs in parallel ({num_workers} workers)...")

    with Pool(processes=num_workers) as pool:
        png_results = pool.map(_save_png_worker, png_save_tasks)

    png_success = sum(1 for _, success in png_results if success)
    log(f"  ✓ Saved {png_success}/{len(png_save_tasks)} PNGs and 41 OBJ files")
    logger.info(f"Saved {png_success} PNGs and 41 OBJ files")

    log("")
    log("  Converting OBJ to FBX...")

    # Prepare arguments for parallel conversion (add blender_path to each tuple)
    fbx_args = [(obj_path, fbx_path, name, config['blender_path'])
                for obj_path, fbx_path, name in obj_fbx_pairs]

    if config.get('parallel_fbx', True) and len(obj_fbx_pairs) > 1:
        # Parallel conversion
        num_workers = min(config.get('num_workers', cpu_count()), len(obj_fbx_pairs))
        log(f"  Using {num_workers} parallel workers")

        with Pool(processes=num_workers) as pool:
            results = pool.map(_convert_single_fbx_worker, fbx_args)

        for idx, (name, success) in enumerate(results, 1):
            status = "✓" if success else "✗"
            if not success:
                log(f"  Warning: FBX conversion failed for {name}")
            log(f"  [{idx}/{len(results)}] {name} {status}")
            logger.info(f"[{idx}/{len(results)}] {name}")
    else:
        # Sequential conversion (fallback)
        log("  Using sequential conversion")
        for idx, (obj_path, fbx_path, name) in enumerate(obj_fbx_pairs, 1):
            success = convert_obj_to_fbx(obj_path, fbx_path, blender_path=config['blender_path'])
            status = "✓" if success else "✗"
            if not success:
                log(f"  Warning: FBX conversion failed for {name}")
            log(f"  [{idx}/{len(obj_fbx_pairs)}] {name} {status}")
            logger.info(f"[{idx}/{len(obj_fbx_pairs)}] {name}")

    # Cleanup temporary OBJ files
    shutil.rmtree(temp_dir, ignore_errors=True)

    log("")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 7.5: Generate variance heatmaps
    # -------------------------------------------------------------------------

    logger.step("Generating variance heatmaps")
    log("STEP 7.5: Generating variance heatmaps...")

    # Shape displacement heatmap (always generated)
    # Compute signed normal displacement between the two original input stimuli (A and B)
    log("  Computing shape displacement between stimulus A and B...")

    # Move meshes to renderer's device for heatmap rendering
    mesh1_for_heatmap = mesh1.to(renderer.device) if hasattr(mesh1, 'to') else mesh1
    mesh2_for_heatmap = mesh2.to(renderer.device) if hasattr(mesh2, 'to') else mesh2
    shape_heatmap_path = pair_dir / "shape_displacement_components.png"

    try:
        success = create_shape_displacement_visualization(
            mesh1_for_heatmap,  # Mesh A
            mesh2_for_heatmap,  # Mesh B
            renderer,
            shape_heatmap_path,
        )

        if success:
            log(f"  Shape displacement heatmap saved: shape_displacement_components.png")
            logger.success("Shape displacement components heatmap created")
        else:
            log(f"  Warning: Shape heatmap generation failed")
            logger.warning("Shape heatmap generation failed")
    except Exception as e:
        log(f"  Warning: Shape heatmap error: {e}")
        logger.warning(f"Shape heatmap error: {e}")

    # Texture difference heatmap (only if textures available)
    if has_textures and texture1 is not None and texture2 is not None:
        log("  Computing texture difference between stimulus A and B...")
        texture_heatmap_path = pair_dir / "texture_difference_components.png"

        try:
            # Move mesh to renderer's device for heatmap rendering
            mesh1_for_heatmap = mesh1.to(renderer.device) if hasattr(mesh1, 'to') else mesh1

            success = create_texture_difference_components_visualization(
                [texture1, texture2],  # Only the two original stimuli
                mesh1_for_heatmap,
                renderer,
                aux1['verts_uvs'],
                aux1.get('faces_uvs'),  # Pass faces_uvs from aux
                texture_heatmap_path,
            )

            if success:
                log(f"  Texture difference heatmap saved: texture_difference_components.png")
                logger.success("Texture difference components heatmap created")
            else:
                log(f"  Warning: Texture heatmap generation failed")
                logger.warning("Texture heatmap generation failed")
        except Exception as e:
            log(f"  Warning: Texture heatmap error: {e}")
            logger.warning(f"Texture heatmap error: {e}")
    else:
        log("  Skipped texture variance: No textures available")

    log("")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 8: Create video animation
    # -------------------------------------------------------------------------

    logger.step("Creating video animation")
    log("STEP 8: Creating video animation...")

    # Always create video from rendered PNG frames
    ffmpeg_available = check_ffmpeg_available()

    if ffmpeg_available:
        video_created = create_video_from_frames(
            png_dir,
            video_file,
            fps=config.get('video_fps', 30)
        )

        if video_created:
            content_type = "textured meshes" if has_textures else "shape-only meshes"
            log(f"  Video created: animation.mp4 ({content_type}, {config.get('video_fps', 30)} fps)")
            logger.success(f"Video created: animation.mp4 ({content_type})")
        else:
            log(f"  Warning: Video creation failed")
            logger.warning("Video creation failed")
    else:
        log(f"  Warning: ffmpeg not available - skipping video creation")
        logger.warning("ffmpeg not available - skipping video creation")

    log("")
    logger.blank()

    # -------------------------------------------------------------------------
    # STEP 9: Cleanup & Summary
    # -------------------------------------------------------------------------

    if config['use_gpu']:
        torch.cuda.empty_cache()

    log("="*70)
    log("RESULTS")
    log("="*70)
    log(f"Pair: {stim1_name} + {stim2_name}")
    log(f"Frames generated: {len(config['morph_ratios'])}")
    log(f"Textures: {'Yes' if has_textures else 'No (shape only)'}")
    log(f"")
    log(f"Output directory: {pair_dir}/")
    log(f"  PNG frames (rendered): {png_dir}/")
    log(f"  FBX meshes: {mesh_dir}/")
    if video_file.exists():
        log(f"  Video: {video_file}")
    if shape_heatmap_path.exists():
        log(f"  Shape displacement heatmap: {shape_heatmap_path.name}")
    if has_textures and texture_heatmap_path.exists():
        log(f"  Texture difference heatmap: {texture_heatmap_path.name}")
    log(f"  Session log: {log_file}")
    log("="*70)
    log("SUCCESS!")
    log("="*70)

    logger.header("RESULTS")
    print(f"Pair: {stim1_name} + {stim2_name}")
    print(f"Frames generated: {len(config['morph_ratios'])}")
    print(f"Textures: {'Yes' if has_textures else 'No (shape only)'}")
    print(f"\nOutput directory: {pair_dir}/")
    print(f"  PNG frames (rendered): {png_dir}/")
    print(f"  FBX meshes: {mesh_dir}/")
    if video_file.exists():
        print(f"  Video: {video_file}")
    if shape_heatmap_path.exists():
        print(f"  Shape displacement heatmap: {shape_heatmap_path.name}")
    if has_textures and texture_heatmap_path.exists():
        print(f"  Texture difference heatmap: {texture_heatmap_path.name}")
    print(f"  Session log: {log_file}")

    logger.header("SUCCESS!")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    # Check if CLI arguments provided
    if len(sys.argv) > 1:
        args = parse_arguments()
        config = merge_config(args)

        # Batch mode: process all pairs
        if config.get('batch_mode'):
            logger = Logger(verbose=config['verbose'])
            logger.header("BATCH MODE: PROCESSING ALL PAIRS")

            pairs = discover_mesh_pairs(config['batch_folder'])
            print(f"Found {len(pairs)} unique pairs to process\n")

            # Shared timestamp for all pairs in this batch
            batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            for idx, (mesh1, mesh2) in enumerate(pairs, 1):
                print(f"\n{'='*70}")
                print(f"PAIR {idx}/{len(pairs)}: {mesh1.name} + {mesh2.name}")
                print(f"{'='*70}\n")

                # Override config for this pair
                pair_config = config.copy()
                pair_config['input_mesh_1'] = str(mesh1)
                pair_config['input_mesh_2'] = str(mesh2)
                pair_config['timestamp'] = batch_timestamp  # Share timestamp

                try:
                    run_morphing_pipeline(pair_config)
                except Exception as e:
                    print(f"\n❌ ERROR processing pair: {e}\n")
                    continue

            logger.header(f"BATCH COMPLETE: {len(pairs)} PAIRS PROCESSED")

        else:
            # Single pair mode
            run_morphing_pipeline(config)

    else:
        # Use defaults for IDE execution
        config = DEFAULT_CONFIG
        run_morphing_pipeline(config)
