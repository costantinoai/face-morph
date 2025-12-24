"""
Morphing Pipeline Orchestrator
===============================

Single responsibility: Coordinate the complete morphing pipeline.

This module orchestrates all steps of the morphing process, from
loading meshes to generating final output (PNG frames, heatmaps,
meshes, video, and CSV files).

Refactored to use PipelineStages for better SRP adherence and testability.
"""

from pathlib import Path
import torch

from face_morph.pipeline.config import MorphConfig
from face_morph.pipeline.workers import create_pair_output_structure
from face_morph.pipeline.stages import PipelineStages
from face_morph.core.morpher import create_morpher
from face_morph.rendering.factory import create_optimal_renderer
from face_morph.utils.logging import setup_logger, get_logger

logger = get_logger(__name__)


def run_morphing_pipeline(config: MorphConfig) -> Path:
    """
    Execute the complete morphing pipeline with video generation.

    This function orchestrates all steps of the morphing process by delegating
    to specialized PipelineStages methods. Each stage is now a focused,
    testable component following the Single Responsibility Principle.

    Pipeline stages:
    1. Validate inputs and prepare meshes (PipelineStages.prepare_meshes)
    2. Initialize morpher and renderer (PipelineStages.initialize_rendering)
    3. Generate morphed meshes and textures (PipelineStages.generate_morphs)
    4. Render PNG frames and save meshes (PipelineStages.render_and_save)
    5. Generate heatmaps, videos, CSV (PipelineStages.generate_visualizations)

    Args:
        config: MorphConfig instance with all pipeline settings

    Returns:
        Path to output directory containing all results

    Raises:
        ValidationError: If inputs are invalid
        TopologyMismatchError: If meshes have different vertex counts
        RuntimeError: If renderer or morpher initialization fails

    Output structure (default mode):
        results/<timestamp>/<stim1>_<stim2>/
        ├── session.log
        ├── png/
        │   ├── stim1-0000_stim2-1000.png
        │   └── ...
        ├── shape_displacement_components.png
        └── texture_difference_components.png (if textures available)

    Output structure (full mode):
        results/<timestamp>/<stim1>_<stim2>/
        ├── session.log
        ├── png/
        │   └── ... (41 PNG files)
        ├── mesh/
        │   ├── stim1-0000_stim2-1000.fbx
        │   └── ... (41 FBX files)
        ├── shape_displacement_components.png
        ├── texture_difference_components.png
        ├── animation.mp4
        └── statistics.csv

    Example:
        >>> config = MorphConfig(
        ...     input_mesh_1=Path("face1.fbx"),
        ...     input_mesh_2=Path("face2.fbx"),
        ...     output_mode="full",
        ...     device=torch.device('cuda')
        ... )
        >>> output_dir = run_morphing_pipeline(config)
        >>> output_dir.exists()
        True
    """
    # -------------------------------------------------------------------------
    # SETUP: Get stimulus names and create output structure
    # -------------------------------------------------------------------------

    stim1_name = config.input_mesh_1.stem
    stim2_name = config.input_mesh_2.stem

    pair_dir, png_dir, mesh_dir, log_file, video_file = create_pair_output_structure(
        config.output_dir,
        stim1_name,
        stim2_name,
        config.timestamp
    )

    # Setup session logging (file + console)
    session_logger = setup_logger(
        name='morphing_session',
        verbose=config.verbose,
        log_file=log_file,
        log_level=config.log_level
    )

    def log(message: str):
        """Log to both module logger and session logger."""
        logger.info(message)
        session_logger.info(message)

    log("=" * 70)
    log("3D FACE MORPHING PIPELINE")
    log("=" * 70)
    log(f"Output mode: {config.output_mode.upper()}")
    log("")

    # -------------------------------------------------------------------------
    # STEP 1: Validate inputs
    # -------------------------------------------------------------------------

    log("STEP 1: Validating inputs...")

    log(f"  Input 1: {config.input_mesh_1.name}")
    log(f"  Input 2: {config.input_mesh_2.name}")
    log(f"  Device: {config.device}")
    log(f"  Morph count: {len(config.morph_ratios)}")
    log("")

    # -------------------------------------------------------------------------
    # STAGE 1: Prepare meshes (Steps 2-4)
    # -------------------------------------------------------------------------

    (
        mesh1_path, mesh2_path,
        mesh1, mesh2,
        aux1, aux2,
        texture1, texture2,
        has_textures
    ) = PipelineStages.prepare_meshes(config, log)

    # -------------------------------------------------------------------------
    # STEP 5: Initialize morpher
    # -------------------------------------------------------------------------

    log("STEP 5: Initializing morpher...")

    use_amp = config.use_mixed_precision and config.device.type == 'cuda'
    morpher = create_morpher(config.device, use_amp=use_amp)

    if config.device.type == 'cuda' and use_amp:
        log("  GPU acceleration: ENABLED (mixed precision)")
    elif config.device.type == 'cuda':
        log("  GPU acceleration: ENABLED (full precision)")
    else:
        log("  CPU mode: ENABLED")

    log("")

    # -------------------------------------------------------------------------
    # STAGE 2: Initialize rendering
    # -------------------------------------------------------------------------

    renderer = create_optimal_renderer(config.device, image_size=512)

    device_manager, mesh1, mesh2 = PipelineStages.initialize_rendering(
        config, renderer, mesh1, mesh2, texture1, texture2, aux1, has_textures, log
    )

    # -------------------------------------------------------------------------
    # STAGE 3: Generate morphs
    # -------------------------------------------------------------------------

    morphed_results = PipelineStages.generate_morphs(
        morpher, mesh1, mesh2, texture1, texture2, config.morph_ratios, has_textures, log
    )

    # -------------------------------------------------------------------------
    # STAGE 4: Render and save results
    # -------------------------------------------------------------------------

    rendered_images, mesh_export_count = PipelineStages.render_and_save(
        config, renderer, morphed_results, aux1, has_textures,
        stim1_name, stim2_name, png_dir, mesh_dir, log
    )

    # -------------------------------------------------------------------------
    # STAGE 5: Generate visualizations
    # -------------------------------------------------------------------------

    (
        normal_disp, tangent_disp, total_disp,
        luminance_diff, chroma_diff, delta_e
    ) = PipelineStages.generate_visualizations(
        config, mesh1, mesh2, texture1, texture2, renderer, aux1,
        has_textures, pair_dir, video_file, png_dir, log
    )

    # -------------------------------------------------------------------------
    # CLEANUP & SUMMARY
    # -------------------------------------------------------------------------

    if config.device.type == 'cuda':
        torch.cuda.empty_cache()

    log("=" * 70)
    log("RESULTS")
    log("=" * 70)
    log(f"Pair: {stim1_name} + {stim2_name}")
    log(f"Frames generated: {len(config.morph_ratios)}")
    log(f"Textures: {'Yes' if has_textures else 'No (shape only)'}")
    log("")
    log(f"Output directory: {pair_dir}/")
    log(f"  PNG frames: {png_dir}/")

    if config.should_export_meshes:
        log(f"  OBJ meshes: {mesh_dir}/")

    shape_heatmap_path = pair_dir / "shape_displacement_components.png"
    if shape_heatmap_path.exists():
        log(f"  Shape heatmap: {shape_heatmap_path.name}")

    if has_textures:
        texture_heatmap_path = pair_dir / "texture_difference_components.png"
        if texture_heatmap_path.exists():
            log(f"  Texture heatmap: {texture_heatmap_path.name}")

    if config.should_create_video and video_file.exists():
        log(f"  Video: {video_file}")

    if config.should_export_csv:
        stats_csv_path = pair_dir / "statistics.csv"
        vertex_csv_path = pair_dir / "vertex_displacements.csv"
        texture_csv_path = pair_dir / "texture_differences.csv"

        if stats_csv_path.exists():
            log(f"  Statistics CSV: {stats_csv_path.name}")
        if vertex_csv_path.exists():
            log(f"  Vertex data CSV: {vertex_csv_path.name}")
        if has_textures and texture_csv_path.exists():
            log(f"  Texture data CSV: {texture_csv_path.name}")

    log(f"  Session log: {log_file}")

    # Log device transfer optimization stats
    transfer_count = device_manager.get_transfer_count()
    log(f"  Device transfers: {transfer_count} (optimized)")

    log("=" * 70)
    log("SUCCESS!")
    log("=" * 70)

    return pair_dir
