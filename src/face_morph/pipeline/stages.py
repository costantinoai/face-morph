"""
Pipeline Stages
===============

Single responsibility: Break down the morphing pipeline into discrete, testable stages.

This module extracts the orchestrator logic into focused, reusable components
following the Single Responsibility Principle. Each stage handles one phase of
the pipeline with clear inputs and outputs.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Callable
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil
import torch
import numpy as np

from pytorch3d.structures import Meshes

from face_morph.pipeline.config import MorphConfig
from face_morph.pipeline.workers import (
    ensure_obj_format,
    load_mesh_with_texture,
    generate_morph_filename,
    _save_png_worker,
    _convert_single_fbx_worker,
)
from face_morph.core.morpher import MeshMorpher
from face_morph.core.mesh_io import save_mesh
from face_morph.core.exceptions import TopologyMismatchError
from face_morph.rendering.base import BaseRenderer
from face_morph.rendering.factory import get_renderer_type
from face_morph.visualization.heatmap import (
    compute_shape_displacement_components,
    compute_texture_difference_components,
    create_shape_displacement_visualization,
    create_texture_difference_components_visualization,
)
from face_morph.visualization.export import (
    export_statistics_csv,
    export_vertex_data_csv,
    export_texture_data_csv,
)
from face_morph.visualization.video import create_video_from_frames, check_ffmpeg_available
from face_morph.utils.device import DeviceManager, optimize_texture_device

from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


class PipelineStages:
    """Encapsulates individual stages of the morphing pipeline."""

    @staticmethod
    def prepare_meshes(
        config: MorphConfig,
        log: Callable[[str], None]
    ) -> Tuple[Path, Path, Meshes, Meshes, dict, dict, Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """
        Prepare meshes by converting to OBJ, loading, and validating topology.

        Args:
            config: Pipeline configuration
            log: Logging function

        Returns:
            Tuple of (mesh1_path, mesh2_path, mesh1, mesh2, aux1, aux2, texture1, texture2, has_textures)

        Raises:
            TopologyMismatchError: If meshes have different vertex counts
        """
        log("STEP 2: Preparing mesh files...")

        mesh1_path = ensure_obj_format(config.input_mesh_1, config.blender_path)
        mesh2_path = ensure_obj_format(config.input_mesh_2, config.blender_path)

        log(f"  Mesh 1: {mesh1_path.name}")
        log(f"  Mesh 2: {mesh2_path.name}")
        log("")

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

        return (
            mesh1_path, mesh2_path,
            mesh1, mesh2,
            aux1, aux2,
            texture1, texture2,
            has_textures
        )

    @staticmethod
    def initialize_rendering(
        config: MorphConfig,
        renderer: BaseRenderer,
        mesh1: Meshes,
        mesh2: Meshes,
        texture1: Optional[torch.Tensor],
        texture2: Optional[torch.Tensor],
        aux1: dict,
        has_textures: bool,
        log: Callable[[str], None]
    ) -> Tuple[DeviceManager, Meshes, Meshes]:
        """
        Initialize renderer and optimize device transfers.

        Args:
            config: Pipeline configuration
            renderer: Renderer instance
            mesh1, mesh2: Mesh objects to optimize
            texture1, texture2: Optional texture tensors
            aux1: Auxiliary data for mesh1 (UVs, etc.)
            has_textures: Whether both meshes have textures
            log: Logging function

        Returns:
            Tuple of (device_manager, optimized_mesh1, optimized_mesh2)
        """
        log("STEP 5.5: Initializing renderer...")

        renderer_type = get_renderer_type(renderer)

        if renderer_type == 'pytorch3d':
            log("  Renderer: PyTorch3D (GPU-accelerated)")
        else:
            log("  Renderer: PyRender (CPU mode)")

        # Initialize device manager to track and minimize transfers
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

        return device_manager, mesh1, mesh2

    @staticmethod
    def generate_morphs(
        morpher: MeshMorpher,
        mesh1: Meshes,
        mesh2: Meshes,
        texture1: Optional[torch.Tensor],
        texture2: Optional[torch.Tensor],
        ratios: List[Tuple[float, float]],
        has_textures: bool,
        log: Callable[[str], None]
    ) -> List[Tuple[Meshes, Optional[torch.Tensor]]]:
        """
        Generate morphed meshes and textures.

        Args:
            morpher: MeshMorpher instance
            mesh1, mesh2: Source meshes
            texture1, texture2: Optional source textures
            ratios: List of (ratio1, ratio2) tuples
            has_textures: Whether to morph textures
            log: Logging function

        Returns:
            List of (morphed_mesh, morphed_texture) tuples
        """
        log("STEP 6: Generating morphs...")

        morphed_results = morpher.batch_morph(
            mesh1, mesh2,
            texture1 if has_textures else None,
            texture2 if has_textures else None,
            ratios
        )

        log(f"  Generated {len(morphed_results)} morphs")
        log("")

        return morphed_results

    @staticmethod
    def render_and_save(
        config: MorphConfig,
        renderer: BaseRenderer,
        morphed_results: List[Tuple[Meshes, Optional[torch.Tensor]]],
        aux1: dict,
        has_textures: bool,
        stim1_name: str,
        stim2_name: str,
        png_dir: Path,
        mesh_dir: Path,
        log: Callable[[str], None]
    ) -> Tuple[List[np.ndarray], int]:
        """
        Render morphed meshes and save PNG/FBX files.

        Args:
            config: Pipeline configuration
            renderer: Renderer instance
            morphed_results: List of (mesh, texture) tuples
            aux1: Auxiliary data (UVs, etc.)
            has_textures: Whether textures are available
            stim1_name: Name of first stimulus
            stim2_name: Name of second stimulus
            png_dir: Directory for PNG output
            mesh_dir: Directory for mesh output
            log: Logging function

        Returns:
            Tuple of (rendered_images, fbx_success_count)
        """
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

        renderer_type = get_renderer_type(renderer)

        # Use batch rendering for PyTorch3D (10-20x faster), fallback to sequential for PyRender
        if renderer_type == 'pytorch3d' and hasattr(renderer, 'batch_render_meshes'):
            # Calculate optimal chunk size based on GPU memory (if CUDA)
            if config.device.type == 'cuda' and morphed_results:
                first_mesh = morphed_results[0][0]
                optimal_chunk_size = config.get_optimal_chunk_size(
                    num_vertices=first_mesh.verts_packed().shape[0],
                    num_faces=first_mesh.faces_packed().shape[0],
                    has_texture=has_textures
                )
                log(f"  Using batch rendering (dynamic chunks of {optimal_chunk_size})...")
            else:
                optimal_chunk_size = config.chunk_size
                log(f"  Using batch rendering (chunks of {optimal_chunk_size})...")

            # Extract meshes and textures into separate lists
            meshes_list = [mesh for mesh, _ in morphed_results]
            textures_list = [texture for _, texture in morphed_results] if has_textures else None

            # Batch render all meshes efficiently
            rendered_images = renderer.batch_render_meshes(
                meshes_list,
                textures_list,
                verts_uvs,
                faces_uvs,
                chunk_size=optimal_chunk_size
            )
        else:
            # Sequential rendering (PyRender or fallback)
            log("  Using sequential rendering (PyRender)...")
            for mesh, texture in morphed_results:
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

        # === CONVERT OBJ TO FBX ===
        fbx_success = 0
        if config.should_export_meshes:
            log("  Converting OBJ to FBX...")

            if config.parallel_fbx and len(obj_fbx_pairs) > 1:
                # Async parallel conversion (20-40% faster than multiprocessing)
                from face_morph.utils.io_optimizer import AsyncFBXConverter

                max_concurrent = min(4, len(obj_fbx_pairs))
                log(f"    Using async conversion (max {max_concurrent} concurrent)")

                converter = AsyncFBXConverter(
                    blender_path=config.blender_path,
                    max_concurrent=max_concurrent
                )

                # Convert using async I/O (overlaps subprocess waits)
                fbx_results = converter.convert_batch_sync(obj_fbx_pairs)

                fbx_success = sum(1 for _, success in fbx_results if success)
                log(f"  Converted {fbx_success}/{len(obj_fbx_pairs)} FBX files")
            else:
                # Sequential conversion (fallback for single file or disabled parallel)
                log("    Using sequential conversion")
                from face_morph.core.converter import convert_obj_to_fbx
                for obj_path, fbx_path, name in obj_fbx_pairs:
                    if convert_obj_to_fbx(obj_path, fbx_path, blender_path=config.blender_path):
                        fbx_success += 1
                log(f"  Converted {fbx_success}/{len(obj_fbx_pairs)} FBX files")

        # Cleanup temporary OBJ files
        shutil.rmtree(temp_dir, ignore_errors=True)

        log("")

        return rendered_images, fbx_success

    @staticmethod
    def generate_visualizations(
        config: MorphConfig,
        mesh1: Meshes,
        mesh2: Meshes,
        texture1: Optional[torch.Tensor],
        texture2: Optional[torch.Tensor],
        renderer: BaseRenderer,
        aux1: dict,
        has_textures: bool,
        pair_dir: Path,
        video_file: Path,
        png_dir: Path,
        log: Callable[[str], None]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate heatmaps, videos, and CSV exports.

        Args:
            config: Pipeline configuration
            mesh1, mesh2: Source meshes
            texture1, texture2: Optional source textures
            renderer: Renderer instance
            aux1: Auxiliary data
            has_textures: Whether textures are available
            pair_dir: Output directory for visualizations
            video_file: Path for video output
            png_dir: Directory containing PNG frames
            log: Logging function

        Returns:
            Tuple of displacement/difference components for CSV export
        """
        log("STEP 7.5: Generating variance heatmaps...")

        # ✅ OPTIMIZATION: Check for cached heatmap computation results
        import hashlib
        cache_file = pair_dir / ".heatmap_cache.pt"

        # Generate cache key from mesh vertices (only vertices change between different meshes)
        # Use first 1000 vertices for hash to balance accuracy vs speed
        verts1_sample = mesh1.verts_packed()[:1000].cpu().numpy()
        verts2_sample = mesh2.verts_packed()[:1000].cpu().numpy()
        mesh1_hash = hashlib.md5(verts1_sample.tobytes()).hexdigest()[:16]
        mesh2_hash = hashlib.md5(verts2_sample.tobytes()).hexdigest()[:16]
        cache_key = f"{mesh1_hash}_{mesh2_hash}"

        # Try to load from cache
        cache_hit = False
        if cache_file.exists():
            try:
                cached_data = torch.load(cache_file)
                if cached_data.get('cache_key') == cache_key:
                    # Cache is valid - use cached computation results
                    log("  ✓ Using cached heatmap computation data")

                    # Move cached tensors to correct device (GPU if available)
                    device = mesh1.device
                    normal_disp = cached_data['normal_disp']
                    tangent_disp = cached_data['tangent_disp']
                    total_disp = cached_data['total_disp']

                    # Move torch tensors to device if needed
                    if torch.is_tensor(normal_disp) and normal_disp.device != device:
                        normal_disp = normal_disp.to(device)
                    if torch.is_tensor(tangent_disp) and tangent_disp.device != device:
                        tangent_disp = tangent_disp.to(device)
                    if torch.is_tensor(total_disp) and total_disp.device != device:
                        total_disp = total_disp.to(device)

                    luminance_diff = cached_data.get('luminance_diff')
                    chroma_diff = cached_data.get('chroma_diff')
                    delta_e = cached_data.get('delta_e')
                    cache_hit = True
                else:
                    log("  Cache key mismatch - recomputing")
            except Exception as e:
                log(f"  Cache load failed ({e}) - recomputing")

        # If no cache hit, compute displacement components
        if not cache_hit:
            # Compute displacement components (needed for both heatmaps and CSV export)
            log("  Computing shape displacement between stimulus A and B...")
            normal_disp, tangent_disp, total_disp = compute_shape_displacement_components(
                mesh1, mesh2
            )

            # Compute texture difference components if available
            luminance_diff = None
            chroma_diff = None
            delta_e = None

            if has_textures and texture1 is not None and texture2 is not None:
                log("  Computing texture difference between stimulus A and B...")
                luminance_diff, chroma_diff, delta_e = compute_texture_difference_components(
                    [texture1, texture2]
                )

            # ✅ OPTIMIZATION: Save computation results to cache
            try:
                cache_data = {
                    'cache_key': cache_key,
                    'normal_disp': normal_disp.cpu() if torch.is_tensor(normal_disp) else normal_disp,
                    'tangent_disp': tangent_disp.cpu() if torch.is_tensor(tangent_disp) else tangent_disp,
                    'total_disp': total_disp.cpu() if torch.is_tensor(total_disp) else total_disp,
                    'luminance_diff': luminance_diff,  # Already numpy
                    'chroma_diff': chroma_diff,        # Already numpy
                    'delta_e': delta_e,                # Already numpy
                }
                torch.save(cache_data, cache_file)
                log("  ✓ Saved heatmap computation to cache")
            except Exception as e:
                log(f"  Warning: Failed to save cache ({e})")

        # Define heatmap paths
        shape_heatmap_path = pair_dir / "shape_displacement_components.png"
        texture_heatmap_path = pair_dir / "texture_difference_components.png"

        # Generate heatmaps in parallel using ThreadPoolExecutor
        def generate_shape_heatmap():
            """Worker function for shape heatmap generation."""
            try:
                success = create_shape_displacement_visualization(
                    mesh1, mesh2, renderer, shape_heatmap_path,
                )
                return ("shape", success, None)
            except Exception as e:
                return ("shape", False, str(e))

        def generate_texture_heatmap():
            """Worker function for texture heatmap generation."""
            try:
                success = create_texture_difference_components_visualization(
                    [texture1, texture2],
                    mesh1,
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

        # === CREATE VIDEO ===
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

        # === EXPORT CSV ===
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

        return normal_disp, tangent_disp, total_disp, luminance_diff, chroma_diff, delta_e
