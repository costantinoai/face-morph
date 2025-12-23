"""
Morphing Pipeline Orchestrator
===============================

Single responsibility: Coordinate the complete morphing pipeline.

This module orchestrates all steps of the morphing process, from
loading meshes to generating final output (PNG frames, heatmaps,
meshes, video, and CSV files).
"""

from pathlib import Path
from multiprocessing import Pool
import tempfile
import shutil
import torch

from face_morph.pipeline.config import MorphConfig
from face_morph.pipeline.workers import (
    ensure_obj_format,
    load_mesh_with_texture,
    create_pair_output_structure,
    generate_morph_filename,
    _save_png_worker,
    _convert_single_fbx_worker,
)
from face_morph.core.morpher import create_morpher
from face_morph.core.mesh_io import save_mesh
from face_morph.core.exceptions import TopologyMismatchError
from face_morph.rendering.factory import create_optimal_renderer, get_renderer_type
from face_morph.visualization.heatmap import (
    create_shape_displacement_visualization,
    create_texture_difference_components_visualization,
    compute_shape_displacement_components,
    compute_texture_difference_components,
)
from face_morph.visualization.export import (
    export_statistics_csv,
    export_vertex_data_csv,
    export_texture_data_csv,
)
from face_morph.visualization.video import create_video_from_frames, check_ffmpeg_available
from face_morph.utils.logging import setup_logger, get_logger

logger = get_logger(__name__)


def run_morphing_pipeline(config: MorphConfig) -> Path:
    """
    Execute the complete morphing pipeline with video generation.

    This function orchestrates all steps of the morphing process:
    1. Validate inputs and prepare meshes
    2. Load meshes and textures
    3. Validate topology (matching vertex counts)
    4. Initialize morpher and renderer
    5. Generate morphed meshes and textures
    6. Render PNG frames
    7. Save meshes (if full mode)
    8. Generate heatmap visualizations
    9. Create video animation (if full mode)
    10. Export CSV data (if full mode)

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
        ...     output_mode="full",  # Default: full output
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
    # STEP 2: Convert to OBJ format
    # -------------------------------------------------------------------------

    log("STEP 2: Preparing mesh files...")

    mesh1_path = ensure_obj_format(config.input_mesh_1, config.blender_path)
    mesh2_path = ensure_obj_format(config.input_mesh_2, config.blender_path)

    log(f"  Mesh 1: {mesh1_path.name}")
    log(f"  Mesh 2: {mesh2_path.name}")
    log("")

    # -------------------------------------------------------------------------
    # STEP 3: Load meshes and textures
    # -------------------------------------------------------------------------

    log("STEP 3: Loading meshes and textures...")

    mesh1, aux1, texture1, has_texture1 = load_mesh_with_texture(
        mesh1_path, config.device
    )
    mesh2, aux2, texture2, has_texture2 = load_mesh_with_texture(
        mesh2_path, config.device
    )

    has_textures = has_texture1 and has_texture2

    log(f"  Mesh 1: {aux1['num_vertices']:,} vertices, {aux1['num_faces']:,} faces")
    log(f"  Mesh 2: {aux2['num_vertices']:,} vertices, {aux2['num_faces']:,} faces")

    if has_textures:
        log("  Textures: Both meshes have textures - will morph textures")
    elif has_texture1 or has_texture2:
        log("  Warning: Only one mesh has texture - shape only")
    else:
        log("  Warning: No textures - shape only")

    log("")

    # -------------------------------------------------------------------------
    # STEP 4: Validate topology
    # -------------------------------------------------------------------------

    log("STEP 4: Validating topology...")

    if aux1['num_vertices'] != aux2['num_vertices']:
        error_msg = (
            f"Meshes have different vertex counts:\n"
            f"  Mesh 1: {aux1['num_vertices']:,} vertices\n"
            f"  Mesh 2: {aux2['num_vertices']:,} vertices"
        )
        log(f"  ERROR: {error_msg}")
        raise TopologyMismatchError(aux1['num_vertices'], aux2['num_vertices'])

    if aux1['num_faces'] != aux2['num_faces']:
        log("  Warning: Different face counts but same vertex count")

    log(f"  Topology: IDENTICAL ({aux1['num_vertices']:,} vertices each)")
    log("")

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
    # STEP 5.5: Initialize renderer
    # -------------------------------------------------------------------------

    log("STEP 5.5: Initializing renderer...")

    renderer = create_optimal_renderer(config.device, image_size=512)
    renderer_type = get_renderer_type(renderer)

    if renderer_type == 'pytorch3d':
        log("  Renderer: PyTorch3D (GPU-accelerated)")
    else:
        log("  Renderer: PyRender (CPU mode)")

    # Initialize device manager to track and minimize transfers
    from face_morph.utils.device import DeviceManager, optimize_texture_device
    device_manager = DeviceManager(renderer.device)

    # Optimize: Move meshes to renderer device once (not in render loop)
    mesh1 = device_manager.ensure_mesh_device(mesh1, renderer.device)
    mesh2 = device_manager.ensure_mesh_device(mesh2, renderer.device)

    # Optimize: Move textures/UVs to renderer device once (not in render loop)
    if has_textures and renderer_type == 'pytorch3d':
        texture1, aux1['verts_uvs'], aux1['faces_uvs'] = optimize_texture_device(
            texture1, aux1.get('verts_uvs'), aux1.get('faces_uvs'), renderer.device
        )
        texture2, _, _ = optimize_texture_device(
            texture2, None, None, renderer.device  # UVs already moved
        )
        log("  Optimized textures/UVs to renderer device")

    log("")

    # -------------------------------------------------------------------------
    # STEP 6: Generate morphs
    # -------------------------------------------------------------------------

    log("STEP 6: Generating morphs...")

    morphed_results = morpher.batch_morph(
        mesh1, mesh2,
        texture1 if has_textures else None,
        texture2 if has_textures else None,
        config.morph_ratios
    )

    log(f"  Generated {len(morphed_results)} morphs")
    log("")

    # -------------------------------------------------------------------------
    # STEP 7: Render and save results
    # -------------------------------------------------------------------------

    log("STEP 7: Rendering and saving frames...")

    # Create temporary directory for OBJ files
    temp_dir = Path(tempfile.mkdtemp())

    # Free GPU memory from morphing
    if config.device.type == 'cuda':
        torch.cuda.empty_cache()

    # Prepare for rendering
    rendered_images = []
    obj_fbx_pairs = []  # For FBX conversion (full mode only)

    verts_uvs = aux1['verts_uvs'] if has_textures and aux1.get('verts_uvs') is not None else None
    faces_uvs = aux1.get('faces_uvs') if has_textures else None

    # === RENDERING ===
    log("  Rendering meshes...")

    # Use batch rendering for PyTorch3D (10-20x faster), fallback to sequential for PyRender
    if renderer_type == 'pytorch3d' and hasattr(renderer, 'batch_render_meshes'):
        log(f"  Using batch rendering (chunks of {config.chunk_size})...")

        # Extract meshes and textures into separate lists
        meshes_list = [mesh for mesh, _ in morphed_results]
        textures_list = [texture for _, texture in morphed_results] if has_textures else None

        # Batch render all meshes efficiently
        rendered_images = renderer.batch_render_meshes(
            meshes_list,
            textures_list,
            verts_uvs,
            faces_uvs,
            chunk_size=config.chunk_size
        )
    else:
        # Sequential rendering (PyRender or fallback)
        log("  Using sequential rendering (PyRender)...")
        for idx, (mesh, texture) in enumerate(morphed_results, 1):
            if renderer_type == 'pytorch3d':
                # PyTorch3D renderer (GPU) - sequential fallback
                if has_textures and texture is not None:
                    rendered_img = renderer.render_mesh(
                        mesh, texture=texture, verts_uvs=verts_uvs, faces_uvs=faces_uvs
                    )
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

                    # Render with texture
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

    log(f"  Rendered {len(rendered_images)} images")

    # === SAVE PNG FILES (parallel) + OBJ FILES (sequential) ===
    log("  Saving PNG images and meshes...")

    png_save_tasks = []

    for idx, ((mesh, texture), (r1, r2), rendered_img) in enumerate(
        zip(morphed_results, config.morph_ratios, rendered_images), 1
    ):
        morph_name = generate_morph_filename(stim1_name, stim2_name, r1, r2)

        # Queue PNG save task (parallel)
        png_path = png_dir / f"{morph_name}.png"
        png_save_tasks.append((rendered_img, png_path, morph_name))

        # Save OBJ if full mode (sequential - PyTorch Meshes don't pickle well)
        if config.should_export_meshes:
            obj_path = temp_dir / f"{morph_name}.obj"
            save_mesh(mesh, obj_path, aux1['verts_uvs'], texture if has_textures else None)

            # Queue for FBX conversion
            fbx_path = mesh_dir / f"{morph_name}.fbx"
            obj_fbx_pairs.append((obj_path, fbx_path, morph_name))

    # Save PNGs in parallel
    with Pool(processes=config.num_workers) as pool:
        png_results = pool.map(_save_png_worker, png_save_tasks)

    png_success = sum(1 for _, success in png_results if success)
    log(f"  Saved {png_success}/{len(png_save_tasks)} PNG files")

    # === CONVERT OBJ TO FBX (parallel, full mode only) ===
    if config.should_export_meshes:
        log("  Converting OBJ to FBX...")

        # Prepare arguments for parallel conversion
        fbx_args = [
            (obj_path, fbx_path, name, config.blender_path)
            for obj_path, fbx_path, name in obj_fbx_pairs
        ]

        if config.parallel_fbx and len(obj_fbx_pairs) > 1:
            # Parallel conversion (5x speedup)
            num_workers = min(config.num_workers, len(obj_fbx_pairs))
            log(f"    Using {num_workers} parallel workers")

            with Pool(processes=num_workers) as pool:
                fbx_results = pool.map(_convert_single_fbx_worker, fbx_args)

            fbx_success = sum(1 for _, success in fbx_results if success)
            log(f"  Converted {fbx_success}/{len(obj_fbx_pairs)} FBX files")
        else:
            # Sequential conversion (fallback)
            log("    Using sequential conversion")
            from face_morph.core.converter import convert_obj_to_fbx
            fbx_success = 0
            for obj_path, fbx_path, name in obj_fbx_pairs:
                if convert_obj_to_fbx(obj_path, fbx_path, blender_path=config.blender_path):
                    fbx_success += 1
            log(f"  Converted {fbx_success}/{len(obj_fbx_pairs)} FBX files")

        # Cleanup temporary OBJ files
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # Default mode: just cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    log("")

    # -------------------------------------------------------------------------
    # STEP 7.5: Generate variance heatmaps
    # -------------------------------------------------------------------------

    log("STEP 7.5: Generating variance heatmaps...")

    # Compute displacement components (needed for both heatmaps and CSV export)
    log("  Computing shape displacement between stimulus A and B...")
    normal_disp, tangent_disp, total_disp = compute_shape_displacement_components(
        mesh1, mesh2
    )

    # Meshes are already on renderer device (optimized earlier), no transfer needed
    mesh1_for_heatmap = mesh1
    mesh2_for_heatmap = mesh2

    # Compute texture difference components if available
    luminance_diff = None
    chroma_diff = None
    delta_e = None

    if has_textures and texture1 is not None and texture2 is not None:
        log("  Computing texture difference between stimulus A and B...")
        luminance_diff, chroma_diff, delta_e = compute_texture_difference_components(
            [texture1, texture2]
        )

    # Define heatmap paths
    shape_heatmap_path = pair_dir / "shape_displacement_components.png"
    texture_heatmap_path = pair_dir / "texture_difference_components.png"

    # Generate heatmaps in parallel using ThreadPoolExecutor (3-6x faster)
    from concurrent.futures import ThreadPoolExecutor

    def generate_shape_heatmap():
        """Worker function for shape heatmap generation."""
        try:
            success = create_shape_displacement_visualization(
                mesh1_for_heatmap,
                mesh2_for_heatmap,
                renderer,
                shape_heatmap_path,
            )
            return ("shape", success, None)
        except Exception as e:
            return ("shape", False, str(e))

    def generate_texture_heatmap():
        """Worker function for texture heatmap generation."""
        try:
            success = create_texture_difference_components_visualization(
                [texture1, texture2],
                mesh1_for_heatmap,
                renderer,
                aux1['verts_uvs'],
                aux1.get('faces_uvs'),
                texture_heatmap_path,
            )
            return ("texture", success, None)
        except Exception as e:
            return ("texture", False, str(e))

    # Prepare tasks for parallel execution
    tasks = [generate_shape_heatmap]
    if has_textures and texture1 is not None and texture2 is not None:
        tasks.append(generate_texture_heatmap)

    # Execute heatmap generation in parallel
    log(f"  Generating {len(tasks)} heatmap(s) in parallel...")
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        results = list(executor.map(lambda f: f(), tasks))

    # Process results
    for heatmap_type, success, error in results:
        if success:
            log(f"  ✓ {heatmap_type.capitalize()} displacement components heatmap created")
        elif error:
            log(f"  Warning: {heatmap_type.capitalize()} heatmap error: {error}")
        else:
            log(f"  Warning: {heatmap_type.capitalize()} heatmap generation failed")

    if not has_textures:
        log("  Skipped texture heatmap: No textures available")

    log("")

    # -------------------------------------------------------------------------
    # STEP 8: Create video animation (full mode only)
    # -------------------------------------------------------------------------

    if config.should_create_video:
        log("STEP 8: Creating video animation...")

        ffmpeg_available = check_ffmpeg_available()

        if ffmpeg_available:
            video_created = create_video_from_frames(
                png_dir,
                video_file,
                fps=config.video_fps
            )

            if video_created:
                content_type = "textured meshes" if has_textures else "shape-only meshes"
                log(f"  ✓ Video created: animation.mp4 ({content_type}, {config.video_fps} fps)")
            else:
                log("  Warning: Video creation failed")
        else:
            log("  Warning: ffmpeg not available - skipping video creation")

        log("")

    # -------------------------------------------------------------------------
    # STEP 9: Export CSV data (full mode only)
    # -------------------------------------------------------------------------

    if config.should_export_csv:
        log("STEP 9: Exporting CSV data for quantitative analysis...")

        # Export summary statistics
        stats_csv_path = pair_dir / "statistics.csv"
        export_statistics_csv(
            normal_disp, tangent_disp, total_disp,
            luminance_diff, chroma_diff, delta_e,
            stats_csv_path
        )
        log("  ✓ Statistics exported to statistics.csv")

        # Export per-vertex displacement data
        vertex_csv_path = pair_dir / "vertex_displacements.csv"
        export_vertex_data_csv(
            mesh1, mesh2,
            normal_disp, tangent_disp, total_disp,
            vertex_csv_path
        )
        log("  ✓ Vertex data exported to vertex_displacements.csv")

        # Export per-pixel texture data (if textures available)
        if has_textures and luminance_diff is not None:
            texture_csv_path = pair_dir / "texture_differences.csv"
            export_texture_data_csv(
                luminance_diff, chroma_diff, delta_e,
                texture_csv_path,
                downsample_factor=4  # Reduce CSV size
            )
            log("  ✓ Texture data exported to texture_differences.csv")

        log("")

    # -------------------------------------------------------------------------
    # STEP 10: Cleanup & Summary
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
        log(f"  FBX meshes: {mesh_dir}/")

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
