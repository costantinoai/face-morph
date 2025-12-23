"""Mesh Rendering Utilities - Single responsibility: Render 3D meshes to 2D images.

Optimizations:
- Batch rendering for multiple meshes at once
- GPU-accelerated texture processing
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV,
)
from face_morph.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


class MeshRenderer3D:
    """PyTorch3D-based mesh renderer for creating 2D images from 3D meshes.

    Supports:
    - Textured meshes (with UV maps)
    - Vertex-colored meshes (for heatmaps)
    - Configurable camera, lighting, resolution
    - GPU acceleration (if PyTorch3D compiled with GPU support)
    """

    def __init__(
        self,
        device: torch.device,
        image_size: int = 512,
        background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """Initialize renderer with camera and lighting setup.

        Args:
            device: PyTorch device (cuda or cpu)
            image_size: Output image resolution (square)
            background_color: RGB background color (0-1 range)
        """
        self.device = device
        self.image_size = image_size
        self.background_color = background_color

        # Camera setup - front view of face
        # R, T = look_at_view_transform(dist=2.5, elev=0, azim=0)
        R, T = look_at_view_transform(
            dist=2.5,      # Distance from object
            elev=0,        # Elevation angle (0 = horizontal)
            azim=0,        # Azimuth angle (0 = front view)
            device=device
        )

        self.cameras = FoVPerspectiveCameras(
            device=device,
            R=R,
            T=T,
            fov=60,  # Field of view in degrees
        )

        # Lighting setup - soft frontal lighting with minimal specular
        self.lights = PointLights(
            device=device,
            location=[[0.0, 0.0, 3.0]],  # Front light
            ambient_color=((0.6, 0.6, 0.6),),  # Increased ambient for better base lighting
            diffuse_color=((0.5, 0.5, 0.5),),  # Moderate diffuse
            specular_color=((0.05, 0.05, 0.05),),  # Minimal specular to reduce artifacts
        )

        # Rasterization settings
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )

        # Create renderer
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=self.cameras,
                lights=self.lights,
            ),
        )

    def render_mesh(
        self,
        mesh: Meshes,
        texture: Optional[torch.Tensor] = None,
        verts_uvs: Optional[torch.Tensor] = None,
        faces_uvs: Optional[torch.Tensor] = None,
        vertex_colors: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Render mesh to 2D image.

        Args:
            mesh: PyTorch3D Meshes object
            texture: Optional texture image (H, W, 3) in range [0, 1]
            verts_uvs: Optional UV coordinates for texture mapping
            faces_uvs: Optional face UV indices for texture mapping
            vertex_colors: Optional per-vertex colors (V, 3) in range [0, 1]
                          Used for heatmaps or when no texture available

        Returns:
            RGB image as numpy array (H, W, 3) in range [0, 255], uint8
        """
        # Create textured mesh
        if texture is not None and verts_uvs is not None:
            # Use UV texture mapping
            if texture.dim() == 3:
                texture = texture.unsqueeze(0)  # Add batch dimension

            # Clamp UV coordinates to valid range [0, 1] to avoid sampling outside texture
            verts_uvs_clamped = torch.clamp(verts_uvs, 0.0, 1.0)

            # Ensure verts_uvs has batch dimension
            if verts_uvs_clamped.dim() == 2:
                verts_uvs_clamped = verts_uvs_clamped.unsqueeze(0)

            # Get faces_uvs from mesh or parameter
            if faces_uvs is None:
                # Try to get from existing mesh textures
                if hasattr(mesh, 'textures') and mesh.textures is not None and hasattr(mesh.textures, 'faces_uvs_list'):
                    try:
                        faces_uvs_list = mesh.textures.faces_uvs_list()
                        if len(faces_uvs_list) > 0:
                            faces_uvs = faces_uvs_list[0].unsqueeze(0)  # First mesh, add batch dim
                        else:
                            faces_uvs = mesh.faces_padded()  # Fallback
                    except:
                        faces_uvs = mesh.faces_padded()  # Fallback
                else:
                    faces_uvs = mesh.faces_padded()  # Fallback

            # Ensure faces_uvs has correct shape and device
            if faces_uvs.dim() == 2:
                faces_uvs = faces_uvs.unsqueeze(0)
            faces_uvs = faces_uvs.to(self.device)

            # Ensure texture and verts_uvs are on correct device
            texture = texture.to(self.device)
            verts_uvs_clamped = verts_uvs_clamped.to(self.device)

            # Fix for cyan artifacts: texture is mostly black (background), but some UVs sample from it
            # When very dark texture values (~0) interact with Phong lighting, numerical issues
            # can cause cyan artifacts. Solution: Fill black regions with neutral skin tone BEFORE rendering

            # Identify "background" pixels (very dark, likely unused texture space)
            is_background = (texture < 0.1).all(dim=-1, keepdim=True)  # Nearly black

            # Compute mean color from non-background pixels only
            non_bg_mask = ~is_background
            if non_bg_mask.any():
                mean_color = texture[non_bg_mask.expand_as(texture)].reshape(-1, 3).mean(dim=0)
                # Clamp to reasonable skin tone range
                mean_color = torch.clamp(mean_color, 0.4, 0.9)
            else:
                # Fallback if entire texture is dark
                mean_color = torch.tensor([0.75, 0.65, 0.55], device=self.device)

            # Replace background regions with mean color to prevent rendering artifacts
            texture = torch.where(
                non_bg_mask.expand_as(texture),
                texture,
                mean_color.view(1, 1, 1, 3).expand_as(texture)
            )

            # Create TexturesUV with explicit sampling parameters
            textures = TexturesUV(
                maps=texture,
                faces_uvs=faces_uvs,
                verts_uvs=verts_uvs_clamped,
                padding_mode='border',  # Use border padding (safe since we filled background)
                sampling_mode='bilinear',  # Smooth interpolation
            )
            mesh.textures = textures

        elif vertex_colors is not None:
            # Use vertex colors (for heatmaps or shape-only rendering)
            if vertex_colors.dim() == 2:
                vertex_colors = vertex_colors.unsqueeze(0)  # Add batch dimension

            textures = TexturesVertex(verts_features=vertex_colors.to(self.device))
            mesh.textures = textures

        else:
            # Default gray color
            verts = mesh.verts_packed()
            vertex_colors = torch.ones_like(verts) * 0.7  # Gray
            vertex_colors = vertex_colors.unsqueeze(0)
            textures = TexturesVertex(verts_features=vertex_colors.to(self.device))
            mesh.textures = textures

        # Render on the device specified during initialization
        with torch.inference_mode():
            images = self.renderer(mesh.to(self.device))

        # Extract RGB (remove alpha channel if present)
        image = images[0, ..., :3].cpu().numpy()

        # Convert from [0, 1] to [0, 255]
        image = (image * 255).astype(np.uint8)

        return image

    def batch_render_meshes(
        self,
        meshes_list: List[Meshes],
        textures_list: Optional[List[torch.Tensor]] = None,
        verts_uvs: Optional[torch.Tensor] = None,
        faces_uvs: Optional[torch.Tensor] = None,
        chunk_size: int = 10,
    ) -> List[np.ndarray]:
        """
        Render multiple meshes using chunked batch processing.

        Memory-efficient: Renders meshes in chunks to avoid OOM errors.

        Args:
            meshes_list: List of PyTorch3D Meshes objects
            textures_list: Optional list of texture images, one per mesh
            verts_uvs: UV coordinates (shared across all meshes)
            faces_uvs: Face UV indices (shared across all meshes)
            chunk_size: Number of meshes to render per batch (default: 10)

        Returns:
            List of RGB images as numpy arrays (H, W, 3) in range [0, 255], uint8
        """
        logger.debug(f"Batch rendering {len(meshes_list)} meshes in chunks of {chunk_size}...")

        all_results = []
        num_chunks = (len(meshes_list) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(meshes_list))

            logger.debug(f"Processing chunk {chunk_idx + 1}/{num_chunks} (meshes {start_idx}-{end_idx-1})")

            chunk_meshes = meshes_list[start_idx:end_idx]
            chunk_textures = textures_list[start_idx:end_idx] if textures_list else None

            chunk_results = self._render_mesh_chunk(chunk_meshes, chunk_textures, verts_uvs, faces_uvs)
            all_results.extend(chunk_results)

            # Free GPU memory after each chunk
            torch.cuda.empty_cache()

        logger.debug(f"Batch rendering complete: {len(all_results)} images")
        return all_results

    def _render_mesh_chunk(
        self,
        meshes_list: List[Meshes],
        textures_list: Optional[List[torch.Tensor]] = None,
        verts_uvs: Optional[torch.Tensor] = None,
        faces_uvs: Optional[torch.Tensor] = None,
    ) -> List[np.ndarray]:
        """Render a single chunk of meshes."""

        # Combine all meshes into a single batched Meshes object
        all_verts = [m.verts_list()[0] for m in meshes_list]
        all_faces = [m.faces_list()[0] for m in meshes_list]

        batched_mesh = Meshes(verts=all_verts, faces=all_faces).to(self.device)

        # Handle textures
        if textures_list is not None and len(textures_list) > 0:
            # Stack all textures into a batch: (N, H, W, 3)
            texture_batch = torch.stack([t if t.dim() == 3 else t.squeeze(0) for t in textures_list], dim=0)

            # Fix for cyan artifacts: Replace background with mean color
            is_background = (texture_batch < 0.1).all(dim=-1, keepdim=True)
            non_bg_mask = ~is_background

            if non_bg_mask.any():
                mean_color = texture_batch[non_bg_mask.expand_as(texture_batch)].reshape(-1, 3).mean(dim=0)
                mean_color = torch.clamp(mean_color, 0.4, 0.9)
            else:
                mean_color = torch.tensor([0.75, 0.65, 0.55], device=self.device)

            texture_batch = torch.where(
                non_bg_mask.expand_as(texture_batch),
                texture_batch,
                mean_color.view(1, 1, 1, 3).expand_as(texture_batch)
            )

            # Create batched TexturesUV
            if verts_uvs is not None:
                verts_uvs_clamped = torch.clamp(verts_uvs, 0.0, 1.0)
                if verts_uvs_clamped.dim() == 2:
                    verts_uvs_clamped = verts_uvs_clamped.unsqueeze(0)

                # Get faces_uvs
                if faces_uvs is None:
                    faces_uvs = batched_mesh.faces_padded()
                elif faces_uvs.dim() == 2:
                    faces_uvs = faces_uvs.unsqueeze(0)

                # Replicate UVs for all meshes in batch
                verts_uvs_batch = verts_uvs_clamped.expand(len(meshes_list), -1, -1).to(self.device)
                faces_uvs_batch = faces_uvs.expand(len(meshes_list), -1, -1).to(self.device)

                textures = TexturesUV(
                    maps=texture_batch.to(self.device),
                    faces_uvs=faces_uvs_batch,
                    verts_uvs=verts_uvs_batch,
                    padding_mode='border',
                    sampling_mode='bilinear',
                )
                batched_mesh.textures = textures
        else:
            # Use default gray color for all meshes
            num_meshes = len(meshes_list)
            for idx, m in enumerate(meshes_list):
                verts = m.verts_packed()
                vertex_colors = torch.ones_like(verts) * 0.7
                if idx == 0:
                    all_colors = vertex_colors.unsqueeze(0)
                else:
                    all_colors = torch.cat([all_colors, vertex_colors.unsqueeze(0)], dim=0)

            textures = TexturesVertex(verts_features=all_colors.to(self.device))
            batched_mesh.textures = textures

        # Render all meshes at once
        with torch.inference_mode():
            images_batch = self.renderer(batched_mesh.to(self.device))

        # Split batch into individual images
        results = []
        for i in range(len(meshes_list)):
            image = images_batch[i, ..., :3].cpu().numpy()
            image = (image * 255).astype(np.uint8)
            results.append(image)

        return results

    def render_with_heatmap(
        self,
        mesh: Meshes,
        vertex_values: torch.Tensor,
        colormap: str = 'viridis',
        add_colorbar: bool = False,
        title: str = "Heatmap",
        normalize_colorbar: bool = True,
        normalize_values: bool = True,
        symmetric_diverging: bool = False,
    ) -> np.ndarray:
        """Render mesh with heatmap coloring based on vertex values.

        Args:
            mesh: PyTorch3D Meshes object
            vertex_values: Per-vertex scalar values (V,) to visualize as heatmap
            colormap: Matplotlib colormap name ('viridis', 'jet', 'hot', etc.)
            add_colorbar: If True, add colorbar overlay to the rendered image
            title: Title for the visualization (used if add_colorbar=True)
            normalize_colorbar: If True, show colorbar as 0-1 instead of actual values
            normalize_values: If True, normalize values to [0, 1] using percentile
            symmetric_diverging: If True, use symmetric range around 0 for diverging colormap

        Returns:
            RGB image as numpy array (H, W, 3) in range [0, 255], uint8
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap

        if normalize_values:
            # Normalize vertex values to [0, 1] using 95th percentile
            percentile = 0.95
            vmin = vertex_values.min().item()
            vmax = torch.quantile(vertex_values, percentile).item()

            if vmax > vmin:
                # Clip values above percentile
                clipped = torch.clamp(vertex_values, vmin, vmax)
                normalized = (clipped - vmin) / (vmax - vmin)
            else:
                normalized = torch.zeros_like(vertex_values)
        elif symmetric_diverging:
            # Use symmetric range around 0 for diverging colormap
            vmax_abs = max(abs(vertex_values.min().item()), abs(vertex_values.max().item()))
            vmin = -vmax_abs
            vmax = vmax_abs

            # Normalize to [-1, 1] then shift to [0, 1]
            if vmax_abs > 0:
                normalized = (vertex_values / vmax_abs + 1.0) / 2.0  # Map [-vmax_abs, vmax_abs] to [0, 1]
            else:
                normalized = torch.full_like(vertex_values, 0.5)  # All zeros -> center of colormap
        else:
            # No normalization - use raw values
            vmin = vertex_values.min().item()
            vmax = vertex_values.max().item()

            if vmax > vmin:
                normalized = (vertex_values - vmin) / (vmax - vmin)
            else:
                normalized = torch.full_like(vertex_values, 0.5)

        # Map to colors using matplotlib colormap
        cmap = get_cmap(colormap)
        colors = cmap(normalized.cpu().numpy())[:, :3]  # RGB only (drop alpha)
        vertex_colors = torch.from_numpy(colors).float().to(self.device)

        # Render with vertex colors using unlit mode to preserve exact colors
        # Save current lights
        saved_lights = self.lights

        # Create "unlit" lights (full white ambient, no diffuse/specular)
        from pytorch3d.renderer import PointLights
        unlit_lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 3.0]],
            ambient_color=((1.0, 1.0, 1.0),),  # Full white ambient = no color change
            diffuse_color=((0.0, 0.0, 0.0),),  # No diffuse lighting
            specular_color=((0.0, 0.0, 0.0),),  # No specular
        )

        # Temporarily replace lights
        self.lights = unlit_lights
        self.renderer.shader.lights = unlit_lights

        # Render with vertex colors
        rendered_img = self.render_mesh(mesh, vertex_colors=vertex_colors)

        # Restore original lights
        self.lights = saved_lights
        self.renderer.shader.lights = saved_lights

        # Add colorbar if requested
        if add_colorbar:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import Normalize, TwoSlopeNorm

            # Create figure with rendered image
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

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
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
            # Get canvas size: get_width_height() returns (width, height)
            w, h = fig.canvas.get_width_height()
            # buffer_rgba() returns RGBA data
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            # Reshape to (height, width, 4) and drop alpha
            img_array = buf.reshape(h, w, 4)[:, :, :3]
            plt.close(fig)

            return img_array

        return rendered_img


def create_renderer(
    device: torch.device,
    image_size: int = 512,
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> MeshRenderer3D:
    """Factory function to create a mesh renderer.

    Args:
        device: PyTorch device (cuda or cpu)
        image_size: Output image resolution (square)
        background_color: RGB background color (0-1 range)

    Returns:
        Configured MeshRenderer3D instance
    """
    return MeshRenderer3D(device, image_size, background_color)
