"""Pyrender-based Mesh Renderer - Fast CPU rendering with OpenGL.

Performance:
- CPU (Mesa OSMesa): ~1-2s per frame
- CPU (OpenGL): ~0.5s per frame
- GPU (OpenGL): ~0.2s per frame
- 50-100x faster than PyTorch3D on CPU

Note: PyRender requires 1:1 mapping between vertices and UVs.
Meshes with texture seams need vertex duplication.
Reference: https://github.com/mmatl/pyrender/issues/126
"""

import numpy as np
import pyrender
import trimesh
from pathlib import Path
from typing import Optional, Tuple, List

from face_morph.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def split_vertices_at_seams(vertices, faces, uv_coords, uv_faces):
    """
    Split vertices at texture seams to create 1:1 vertex-UV mapping.

    PyRender requires each vertex to have exactly one UV coordinate.
    When a vertex has multiple UV coordinates (texture seams), we duplicate it.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face vertex indices
        uv_coords: (U, 2) unique UV coordinates
        uv_faces: (F, 3) face UV indices

    Returns:
        new_vertices: (V', 3) duplicated vertices
        new_faces: (F, 3) updated face indices
        new_uvs: (V', 2) per-vertex UVs
    """
    logger.debug("Splitting vertices at texture seams...")
    logger.debug("  Original: {len(vertices)} verts, {len(uv_coords)} UVs")

    # Create a mapping from (vertex_idx, uv_idx) -> new_vertex_idx
    vertex_uv_map = {}
    new_vertices = []
    new_uvs = []
    new_faces = []

    # Process each face
    for face_idx, (face, uv_face) in enumerate(zip(faces, uv_faces)):
        new_face = []

        for vert_idx, uv_idx in zip(face, uv_face):
            # Create unique key for this vertex-UV pair
            key = (int(vert_idx), int(uv_idx))

            # If we haven't seen this combination, create a new vertex
            if key not in vertex_uv_map:
                new_vert_idx = len(new_vertices)
                vertex_uv_map[key] = new_vert_idx
                new_vertices.append(vertices[vert_idx])
                new_uvs.append(uv_coords[uv_idx])

            new_face.append(vertex_uv_map[key])

        new_faces.append(new_face)

    new_vertices = np.array(new_vertices, dtype=np.float32)
    new_uvs = np.array(new_uvs, dtype=np.float32)
    new_faces = np.array(new_faces, dtype=np.int32)

    logger.debug("  After split: {len(new_vertices)} verts (1:1 with UVs)")
    logger.debug("  Shapes - vertices: {new_vertices.shape}, uvs: {new_uvs.shape}, faces: {new_faces.shape}")
    logger.debug("  Dtypes - vertices: {new_vertices.dtype}, uvs: {new_uvs.dtype}")

    return new_vertices, new_faces, new_uvs


class PyRenderMeshRenderer:
    """OpenGL-based mesh renderer using pyrender (fast CPU rendering).

    Uses:
    - OSMesa for pure CPU rendering
    - OpenGL for GPU/integrated GPU acceleration
    - Much faster than PyTorch3D on CPU (50-100x)
    """

    def __init__(
        self,
        image_size: int = 512,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        use_offscreen: bool = True,
    ):
        """Initialize pyrender renderer.

        Args:
            image_size: Output image resolution (square)
            background_color: RGB background color (0-1 range)
            use_offscreen: Use offscreen rendering (required for headless)
        """
        self.image_size = image_size
        self.background_color = np.array(background_color) * 255  # pyrender uses 0-255
        self.use_offscreen = use_offscreen
        self.device = 'cpu'  # PyRender always runs on CPU (uses OpenGL/OSMesa)

        # Create renderer (will auto-detect OSMesa/OpenGL)
        if use_offscreen:
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=image_size,
                viewport_height=image_size,
            )
            logger.debug("Using offscreen renderer (OSMesa/OpenGL)")
        else:
            self.renderer = None  # Will create on-demand
            logger.debug("Using interactive renderer")

    def render_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        texture: Optional[np.ndarray] = None,
        uv_coords: Optional[np.ndarray] = None,
        uv_faces: Optional[np.ndarray] = None,
        vertex_colors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render mesh to 2D image.

        Args:
            vertices: Vertex positions (V, 3)
            faces: Face indices (F, 3)
            texture: Optional texture image (H, W, 3) in range [0, 1]
            uv_coords: Optional UV coordinates (U, 2) - can be unique UVs
            uv_faces: Optional UV face indices (F, 3) - maps faces to UVs
            vertex_colors: Optional per-vertex colors (V, 3) in range [0, 1]

        Returns:
            RGB image as numpy array (H, W, 3) in range [0, 255], uint8
        """

        # Apply texture or colors
        if texture is not None and uv_coords is not None:
            # Convert texture from [0,1] float to [0,255] uint8
            if texture.dtype == np.float32 or texture.dtype == np.float64:
                texture_uint8 = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
            else:
                texture_uint8 = texture

            # Ensure texture has correct shape (H, W, 3) or (H, W, 4)
            if texture_uint8.ndim == 2:
                # Grayscale to RGB
                texture_uint8 = np.stack([texture_uint8] * 3, axis=-1)
            elif texture_uint8.shape[2] == 4:
                # RGBA to RGB
                texture_uint8 = texture_uint8[:, :, :3]

            # Create PIL Image for trimesh
            from PIL import Image
            texture_image = Image.fromarray(texture_uint8, mode='RGB')

            # Handle texture seams: Split vertices if needed
            if uv_coords.ndim != 2 or uv_coords.shape[1] != 2:
                logger.debug("Warning: UV coords have unexpected shape {uv_coords.shape}, skipping texture")
                # Create mesh and fall back to default color
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                mesh.visual.vertex_colors = np.ones((len(vertices), 4)) * [180, 180, 180, 255]

            elif uv_coords.shape[0] != vertices.shape[0]:
                # UV coordinates don't match vertex count - texture has seams
                # Need to split vertices at seams to create 1:1 mapping
                if uv_faces is not None:
                    logger.debug("UV count {uv_coords.shape[0]} != vertex count {vertices.shape[0]}")
                    logger.debug("Splitting vertices at texture seams...")

                    # Split vertices to create 1:1 vertex-UV mapping
                    new_vertices, new_faces, new_uvs = split_vertices_at_seams(
                        vertices, faces, uv_coords, uv_faces
                    )

                    # Create mesh with split vertices
                    # IMPORTANT: process=False to prevent vertex merging/cleanup
                    logger.debug("Creating trimesh: verts={new_vertices.shape}, faces={new_faces.shape}")
                    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
                    logger.debug("Trimesh created, adding texture with UV={new_uvs.shape}")
                    logger.debug("UV range: [{new_uvs.min():.3f}, {new_uvs.max():.3f}]")
                    mesh.visual = trimesh.visual.TextureVisuals(
                        uv=new_uvs,
                        image=texture_image
                    )
                    logger.debug("Trimesh.visual.uv shape: {mesh.visual.uv.shape}")
                    logger.debug("Trimesh.vertices shape: {mesh.vertices.shape}")
                    logger.debug("About to convert to pyrender...")
                else:
                    logger.debug("Warning: No UV faces provided, cannot split vertices")
                    # Create mesh and fall back to default color
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                    mesh.visual.vertex_colors = np.ones((len(vertices), 4)) * [180, 180, 180, 255]

            else:
                # UV coordinates already match vertices - direct mapping
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uv_coords,
                    image=texture_image
                )

        elif vertex_colors is not None:
            # Use vertex colors
            logger.debug("Applying vertex colors: shape={vertex_colors.shape}, min={vertex_colors.min():.3f}, max={vertex_colors.max():.3f}")

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            vertex_colors_uint8 = (vertex_colors * 255).astype(np.uint8)

            # Add alpha channel if missing (trimesh expects RGBA)
            if vertex_colors_uint8.shape[1] == 3:
                alpha = np.full((len(vertex_colors_uint8), 1), 255, dtype=np.uint8)
                vertex_colors_uint8 = np.hstack([vertex_colors_uint8, alpha])
                logger.debug("Added alpha channel: shape={vertex_colors_uint8.shape}")

            logger.debug("Vertex colors uint8: min={vertex_colors_uint8.min()}, max={vertex_colors_uint8.max()}")
            mesh.visual.vertex_colors = vertex_colors_uint8

        else:
            # Default gray color
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.visual.vertex_colors = np.ones((len(vertices), 4)) * [180, 180, 180, 255]

        # Create pyrender mesh from trimesh
        # Use smooth=True to preserve our 1:1 vertex-UV mapping from split_vertices_at_seams()
        # If we use smooth=False, pyrender duplicates vertices per-face, breaking the UV mapping
        mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        # Create scene
        scene = pyrender.Scene(
            bg_color=list(self.background_color) + [255.0],
            ambient_light=[0.6, 0.6, 0.6],
        )
        scene.add(mesh_pr)

        # Add camera (front view)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.5],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=camera_pose)

        # Add light (frontal lighting)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)

        # Render
        if self.use_offscreen:
            color, depth = self.renderer.render(scene)
        else:
            # Interactive viewer (not supported in headless)
            raise NotImplementedError("Interactive rendering not implemented")

        return color

    def render_with_heatmap(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_values: np.ndarray,
        colormap: str = 'viridis',
        add_colorbar: bool = False,
        title: str = "Heatmap",
        normalize_colorbar: bool = True,
        normalize_values: bool = True,
        symmetric_diverging: bool = False,
    ) -> np.ndarray:
        """Render mesh with heatmap coloring.

        Args:
            vertices: Vertex positions (V, 3)
            faces: Face indices (F, 3)
            vertex_values: Per-vertex scalar values (V,) for heatmap
            colormap: Matplotlib colormap name
            add_colorbar: If True, add a colorbar to the rendered image
            title: Title for the heatmap (used when colorbar is added)
            normalize_colorbar: If True, show colorbar as 0-1 instead of actual values
            normalize_values: If True, normalize values to [0, 1] using percentile
            symmetric_diverging: If True, use symmetric range around 0 for diverging colormap

        Returns:
            RGB image as numpy array (H, W, 3) in range [0, 255], uint8
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
        from matplotlib.colors import Normalize, TwoSlopeNorm


        logger.debug("{title}: Raw values: min={vertex_values.min():.6f}, max={vertex_values.max():.6f}, mean={vertex_values.mean():.6f}")

        if normalize_values:
            # Normalize vertex values to [0, 1] using 95th percentile
            percentile = 95
            vmin = vertex_values.min()
            vmax = np.percentile(vertex_values, percentile)

            logger.debug("{title}: Normalizing with p{percentile}={vmax:.6f}")

            if vmax > vmin:
                # Clip values above percentile
                clipped = np.clip(vertex_values, vmin, vmax)
                normalized = (clipped - vmin) / (vmax - vmin)
                logger.debug("{title}: After norm: min={normalized.min():.6f}, max={normalized.max():.6f}, mean={normalized.mean():.6f}")
            else:
                normalized = np.zeros_like(vertex_values)
        elif symmetric_diverging:
            # Use symmetric range around 0 for diverging colormap
            vmax_abs = max(abs(vertex_values.min()), abs(vertex_values.max()))
            vmin = -vmax_abs
            vmax = vmax_abs
            logger.debug("{title}: Symmetric diverging: range=[{vmin:.6f}, {vmax:.6f}]")

            # Normalize to [-1, 1] then shift to [0, 1]
            if vmax_abs > 0:
                normalized = (vertex_values / vmax_abs + 1.0) / 2.0  # Map [-vmax_abs, vmax_abs] to [0, 1]
            else:
                normalized = np.full_like(vertex_values, 0.5)  # All zeros -> center of colormap
        else:
            # No normalization - use raw values
            vmin = vertex_values.min()
            vmax = vertex_values.max()
            logger.debug("{title}: No normalization, using raw range=[{vmin:.6f}, {vmax:.6f}]")

            if vmax > vmin:
                normalized = (vertex_values - vmin) / (vmax - vmin)
            else:
                normalized = np.full_like(vertex_values, 0.5)

        # Map to colors using matplotlib colormap
        cmap_func = get_cmap(colormap)
        colors = cmap_func(normalized)[:, :3]  # RGB only (drop alpha)

        # Render with vertex colors
        rendered_img = self.render_mesh(vertices, faces, vertex_colors=colors)

        # Add colorbar if requested
        if add_colorbar:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            ax.imshow(rendered_img)
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

            # Add colorbar with normalized or actual values
            if symmetric_diverging:
                # Use TwoSlopeNorm for diverging colormap centered at 0
                if normalize_colorbar:
                    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
                else:
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                if normalize_colorbar:
                    norm = Normalize(vmin=0.0, vmax=1.0)
                else:
                    norm = Normalize(vmin=vmin, vmax=vmax)

            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

            # Determine label based on title (Shape vs Texture)
            if 'Shape' in title or 'Displacement' in title:
                if symmetric_diverging:
                    label = 'Displacement (B - A)'
                else:
                    label = 'Displacement (normalized)' if normalize_colorbar else 'Displacement'
            else:
                if symmetric_diverging:
                    label = 'Difference (B - A)'
                else:
                    label = 'Difference (normalized)' if normalize_colorbar else 'Difference'
            cbar.set_label(label, rotation=270, labelpad=20, fontsize=12)

            # Convert figure to numpy array
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = buf.reshape(h, w, 4)[:, :, :3]  # Drop alpha channel
            plt.close(fig)
            return img_array

        return rendered_img

    def __del__(self):
        """Clean up renderer resources."""
        if self.use_offscreen and self.renderer is not None:
            self.renderer.delete()


def create_pyrender_renderer(
    image_size: int = 512,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PyRenderMeshRenderer:
    """Factory function to create a pyrender-based renderer.

    Args:
        image_size: Output image resolution (square)
        background_color: RGB background color (0-1 range)

    Returns:
        Configured PyRenderMeshRenderer instance
    """
    return PyRenderMeshRenderer(image_size, background_color)
