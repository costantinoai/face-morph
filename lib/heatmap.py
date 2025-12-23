"""Heatmap Generation Utilities - Single responsibility: Compute and visualize variance maps."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def compute_shape_displacement_components(
    mesh_a,
    mesh_b,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-vertex displacement decomposed into normal, tangential, and total components.

    Args:
        mesh_a: PyTorch3D Meshes object for first mesh
        mesh_b: PyTorch3D Meshes object for second mesh

    Returns:
        Tuple of (normal_displacement, tangent_displacement, total_displacement):
        - normal_displacement (V,): Signed displacement along surface normal
          Positive = outward (expansion), Negative = inward (contraction)
        - tangent_displacement (V,): Unsigned displacement along surface tangent
          Magnitude of movement parallel to surface
        - total_displacement (V,): Unsigned total Euclidean displacement
          Total 3D distance moved
    """
    # Get vertex positions
    verts_a = mesh_a.verts_packed().cpu()  # (V, 3)
    verts_b = mesh_b.verts_packed().cpu()  # (V, 3)

    # Compute displacement vectors
    displacement_vectors = verts_b - verts_a  # (V, 3)

    # Get surface normals from mesh A
    normals_a = mesh_a.verts_normals_packed().cpu()  # (V, 3)

    # 1. Normal component (signed): projection onto normal
    normal_displacement = (displacement_vectors * normals_a).sum(dim=-1)  # (V,)

    # 2. Tangent component (magnitude): perpendicular to normal
    # Tangent vector = total displacement - normal component
    normal_vector = normal_displacement.unsqueeze(-1) * normals_a  # (V, 3)
    tangent_vector = displacement_vectors - normal_vector  # (V, 3)
    tangent_displacement = torch.norm(tangent_vector, dim=-1)  # (V,)

    # 3. Total displacement (magnitude): Euclidean distance
    total_displacement = torch.norm(displacement_vectors, dim=-1)  # (V,)

    return normal_displacement, tangent_displacement, total_displacement


def compute_texture_difference_components(
    textures: List[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pixel-wise texture difference decomposed into luminance, chrominance, and total components.

    Args:
        textures: List of texture tensors (H, W, 3) in range [0, 1]
                  Should contain exactly 2 textures (A and B)

    Returns:
        Tuple of (luminance_diff, chroma_diff, delta_e):
        - luminance_diff (H, W): Signed luminance difference (L_B - L_A)
          Positive = B brighter, Negative = A darker
        - chroma_diff (H, W): Unsigned chrominance difference
          Magnitude of color change (hue/saturation)
        - delta_e (H, W): Unsigned perceptual color difference (CIEDE2000)
          Industry-standard perceptual metric
    """
    if not textures or len(textures) < 2:
        # Not enough textures to compute difference
        return np.zeros((512, 512)), np.zeros((512, 512)), np.zeros((512, 512))

    # Get the two textures (A and B)
    texture_a = textures[0].cpu().numpy()  # (H, W, 3)
    texture_b = textures[1].cpu().numpy()  # (H, W, 3)

    # Identify background pixels (very dark, likely unused texture space)
    texture_a_brightness = texture_a.mean(axis=-1)  # (H, W)
    texture_b_brightness = texture_b.mean(axis=-1)  # (H, W)
    is_background = (texture_a_brightness < 0.1) & (texture_b_brightness < 0.1)

    # Convert RGB to LAB color space (perceptually uniform)
    from skimage.color import rgb2lab, deltaE_ciede2000

    # Ensure proper range [0, 1] for skimage
    texture_a_clipped = np.clip(texture_a, 0, 1)
    texture_b_clipped = np.clip(texture_b, 0, 1)

    lab_a = rgb2lab(texture_a_clipped)  # (H, W, 3) - L in [0, 100], a,b in [-128, 127]
    lab_b = rgb2lab(texture_b_clipped)  # (H, W, 3)

    # 1. Luminance difference (signed): L channel
    luminance_diff = lab_b[:, :, 0] - lab_a[:, :, 0]  # (H, W) - range ~[-100, 100]

    # 2. Chrominance difference (magnitude): a and b channels
    a_diff = lab_b[:, :, 1] - lab_a[:, :, 1]  # Green-red axis
    b_diff = lab_b[:, :, 2] - lab_a[:, :, 2]  # Blue-yellow axis
    chroma_diff = np.sqrt(a_diff**2 + b_diff**2)  # (H, W) - magnitude

    # 3. Total perceptual difference: CIEDE2000
    delta_e = deltaE_ciede2000(lab_a, lab_b)  # (H, W) - perceptually uniform distance

    # Mask background pixels
    luminance_diff[is_background] = 0.0
    chroma_diff[is_background] = 0.0
    delta_e[is_background] = 0.0

    return luminance_diff, chroma_diff, delta_e


def create_heatmap_image(
    data: np.ndarray,
    output_path: Path,
    title: str = "Heatmap",
    colormap: str = 'hot',
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
) -> bool:
    """Create and save heatmap visualization.

    Args:
        data: 2D array (H, W) with values to visualize
        output_path: Path to save image (PNG recommended)
        title: Plot title
        colormap: Matplotlib colormap name ('hot', 'viridis', 'jet', etc.)
        figsize: Figure size in inches
        dpi: Dots per inch for output image

    Returns:
        True if successful, False otherwise
    """
    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Create heatmap
        im = ax.imshow(data, cmap=colormap, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Variance', rotation=270, labelpad=20)

        # Set title
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path.exists()

    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return False


def vertices_to_heatmap_colors(
    vertex_values: torch.Tensor,
    colormap: str = 'hot',
) -> torch.Tensor:
    """Convert per-vertex scalar values to RGB colors for heatmap visualization.

    Args:
        vertex_values: Per-vertex values (V,) to visualize
        colormap: Matplotlib colormap name

    Returns:
        RGB colors (V, 3) in range [0, 1]
    """
    # Normalize to [0, 1]
    vmin = vertex_values.min()
    vmax = vertex_values.max()

    if vmax > vmin:
        normalized = (vertex_values - vmin) / (vmax - vmin)
    else:
        normalized = torch.zeros_like(vertex_values)

    # Map to colors
    cmap = get_cmap(colormap)
    colors = cmap(normalized.cpu().numpy())[:, :3]  # RGB only

    return torch.from_numpy(colors).float()


def create_shape_displacement_visualization(
    mesh_a,
    mesh_b,
    renderer,
    output_path: Path,
) -> bool:
    """Create and save 1x3 shape displacement component heatmaps (normal, tangent, total).

    Args:
        mesh_a: PyTorch3D Meshes object for first mesh
        mesh_b: PyTorch3D Meshes object for second mesh
        renderer: MeshRenderer3D instance
        output_path: Path to save rendered heatmap image (1x3 composite)

    Returns:
        True if successful, False otherwise
    """
    import logging
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    logger = logging.getLogger(__name__)

    # Compute all three displacement components
    normal_disp, tangent_disp, total_disp = compute_shape_displacement_components(mesh_a, mesh_b)

    # Print statistics for all components
    logger.info("="*70)
    logger.info("SHAPE DISPLACEMENT STATISTICS")
    logger.info("="*70)

    logger.info("")
    logger.info("NORMAL COMPONENT (signed - depth changes):")
    logger.info(f"  Min:     {normal_disp.min().item():>10.6f} (inward)")
    logger.info(f"  Max:     {normal_disp.max().item():>10.6f} (outward)")
    logger.info(f"  Mean:    {normal_disp.mean().item():>10.6f}")
    logger.info(f"  Median:  {normal_disp.median().item():>10.6f}")
    logger.info(f"  Std Dev: {normal_disp.std().item():>10.6f}")
    logger.info(f"  P90:     {torch.quantile(normal_disp, 0.90).item():>10.6f}")

    logger.info("")
    logger.info("TANGENTIAL COMPONENT (magnitude - positional changes):")
    logger.info(f"  Min:     {tangent_disp.min().item():>10.6f}")
    logger.info(f"  Max:     {tangent_disp.max().item():>10.6f}")
    logger.info(f"  Mean:    {tangent_disp.mean().item():>10.6f}")
    logger.info(f"  Median:  {tangent_disp.median().item():>10.6f}")
    logger.info(f"  Std Dev: {tangent_disp.std().item():>10.6f}")
    logger.info(f"  P90:     {torch.quantile(tangent_disp, 0.90).item():>10.6f}")

    logger.info("")
    logger.info("TOTAL DISPLACEMENT (magnitude - all movement):")
    logger.info(f"  Min:     {total_disp.min().item():>10.6f}")
    logger.info(f"  Max:     {total_disp.max().item():>10.6f}")
    logger.info(f"  Mean:    {total_disp.mean().item():>10.6f}")
    logger.info(f"  Median:  {total_disp.median().item():>10.6f}")
    logger.info(f"  Std Dev: {total_disp.std().item():>10.6f}")
    logger.info(f"  P90:     {torch.quantile(total_disp, 0.90).item():>10.6f}")

    logger.info("")
    logger.info("Interpretation:")
    logger.info("  Normal:    Red=outward (fuller), Blue=inward (thinner), White=no depth change")
    logger.info("  Tangential: Magnitude of movement parallel to surface (repositioning)")
    logger.info("  Total:     Complete 3D displacement magnitude")
    logger.info("="*70)

    # Prepare components for rendering
    # Colormap convention:
    # - coolwarm (diverging) for signed values, centered at 0
    # - viridis (sequential) for magnitudes, starting at 0
    components = [
        ('Normal', normal_disp, 'coolwarm', True),       # Signed
        ('Tangential', tangent_disp, 'viridis', False),  # Magnitude
        ('Total', total_disp, 'viridis', False),         # Magnitude
    ]

    # Render each component (without colorbar)
    rendered_images = []

    vertices_np = mesh_a.verts_packed().cpu().numpy()
    faces_np = mesh_a.faces_packed().cpu().numpy()
    is_cpu_renderer = hasattr(renderer, 'device') and renderer.device == 'cpu'

    for name, disp_data, cmap, is_diverging in components:
        if is_cpu_renderer:
            # PyRender renderer
            disp_np = disp_data.cpu().numpy()
            img = renderer.render_with_heatmap(
                vertices_np,
                faces_np,
                disp_np,
                colormap=cmap,
                add_colorbar=False,  # No colorbar - we'll add manually
                normalize_values=False,
                symmetric_diverging=is_diverging,
            )
        else:
            # PyTorch3D renderer
            img = renderer.render_with_heatmap(
                mesh_a,
                disp_data,
                colormap=cmap,
                add_colorbar=False,  # No colorbar - we'll add manually
                normalize_values=False,
                symmetric_diverging=is_diverging,
            )
        rendered_images.append((name, img, disp_data, cmap, is_diverging))

    # Create 1x3 composite figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Shape Displacement Components', fontsize=16, fontweight='bold')

    for idx, (name, img, disp_data, cmap, is_diverging) in enumerate(rendered_images):
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(name, fontsize=14, fontweight='bold')

        # Add colorbar
        from matplotlib.colors import Normalize, TwoSlopeNorm
        from matplotlib import cm

        if is_diverging:
            # Symmetric diverging colorbar centered at 0
            vmax_abs = max(abs(disp_data.min().item()), abs(disp_data.max().item()))
            norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)
            label = 'Displacement'
        else:
            # Sequential colorbar starting at 0
            norm = Normalize(vmin=0.0, vmax=disp_data.max().item())
            label = 'Magnitude'

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, rotation=270, labelpad=15, fontsize=10)

    plt.tight_layout()

    # Save composite figure
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path.exists()


def map_texture_difference_to_vertices(
    difference_map: np.ndarray,
    verts_uvs: torch.Tensor,
    faces_uvs: Optional[torch.Tensor],
    num_vertices: int,
) -> torch.Tensor:
    """Map 2D texture difference to per-vertex values using UV coordinates.

    Args:
        difference_map: 2D difference map (H, W)
        verts_uvs: UV coordinates (U, 2) - may be unique UVs
        faces_uvs: Face UV indices (F, 3) - maps faces to UVs
        num_vertices: Number of vertices in mesh

    Returns:
        Per-vertex difference values (num_vertices,)
    """
    import torch.nn.functional as F
    import logging
    logger = logging.getLogger(__name__)

    # Convert difference map to tensor
    difference_tensor = torch.from_numpy(difference_map).float()

    # Prepare for grid_sample: need (1, 1, H, W) and (1, U, 1, 2)
    difference_grid = difference_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Convert UV coords from [0, 1] to [-1, 1] for grid_sample
    uvs = verts_uvs.cpu().clone()  # (U, 2)

    # Handle batch dimension
    if uvs.dim() == 3:  # Already batched
        uvs = uvs[0]  # Take first batch

    # Convert [0, 1] to [-1, 1]
    uvs = uvs * 2.0 - 1.0

    # Flip V coordinate (texture V is top-to-bottom, grid_sample V is bottom-to-top)
    uvs[:, 1] = -uvs[:, 1]

    # Reshape for grid_sample: (1, U, 1, 2)
    uvs_grid = uvs.unsqueeze(0).unsqueeze(2)  # (1, U, 1, 2)

    # Sample difference at UV coordinates - gives us per-UV difference
    sampled = F.grid_sample(
        difference_grid,
        uvs_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # Extract per-UV values: (1, 1, U, 1) -> (U,)
    uv_difference = sampled[0, 0, :, 0]

    logger.debug(f"UV difference: shape={uv_difference.shape}, min={uv_difference.min().item():.6f}, max={uv_difference.max().item():.6f}")

    # Now expand per-UV difference to per-vertex difference
    # If we have faces_uvs, use it to map UVs to vertices
    if faces_uvs is not None and uv_difference.shape[0] != num_vertices:
        logger.debug(f"Expanding {uv_difference.shape[0]} UV values to {num_vertices} vertices using faces_uvs")

        # faces_uvs maps faces to UV indices: (F, 3)
        # We need to create a vertex-to-UV mapping
        # For each vertex, find which UV index it uses (via faces)

        # This is complex, so let's use a simpler approach:
        # Sample directly at each vertex's UV coordinate by duplicating UVs
        # Actually, just duplicate the difference values according to which vertices share UVs

        # Create a vertex difference array initialized to zero
        vertex_difference = torch.zeros(num_vertices, dtype=uv_difference.dtype, device=uv_difference.device)

        # For meshes with texture seams, verts_uvs is unique UVs (U, 2)
        # We need to duplicate/expand to match all vertices
        # The simplest approach: just return the UV difference and let the caller handle the mismatch
        # OR: pad with zeros or repeat the mean
        logger.warning(f"UV-to-vertex expansion not fully implemented. Using mean difference for unmapped vertices.")

        # For now, use the mean of UV difference for all vertices
        vertex_difference[:] = uv_difference.mean()

        return vertex_difference
    else:
        # UV count matches vertex count - direct mapping
        return uv_difference


def create_texture_difference_visualization(
    textures: List[torch.Tensor],
    mesh,
    renderer,
    verts_uvs: torch.Tensor,
    faces_uvs: Optional[torch.Tensor],
    output_path: Path,
    colormap: str = 'coolwarm',
) -> bool:
    """Create and save texture difference heatmap rendered on 3D mesh with colorbar.

    Args:
        textures: List of texture images
        mesh: PyTorch3D Meshes object to render
        renderer: MeshRenderer3D instance
        verts_uvs: UV coordinates for vertices
        faces_uvs: Face UV indices (optional)
        output_path: Path to save heatmap
        colormap: Matplotlib colormap name (diverging recommended)

    Returns:
        True if successful, False otherwise
    """
    # Compute 2D texture difference
    difference_map = compute_texture_difference(textures)

    # Map variance from 2D texture space to per-vertex values
    # This avoids showing UV seam artifacts
    import logging
    logger = logging.getLogger(__name__)

    # Get number of vertices from mesh
    num_vertices = mesh.verts_packed().shape[0]
    num_faces = mesh.faces_packed().shape[0]

    # PyTorch3D uses indexed UVs: unique UVs + faces_uvs to index them
    # We need to expand to per-vertex UVs by using the faces
    logger.info(f"Checking mesh.textures: hasattr={hasattr(mesh, 'textures')}, is not None={mesh.textures is not None if hasattr(mesh, 'textures') else 'N/A'}")

    # Check if mesh has TexturesUV (not TexturesVertex)
    from pytorch3d.renderer import TexturesUV
    has_texture_uv = (hasattr(mesh, 'textures') and mesh.textures is not None and
                      isinstance(mesh.textures, TexturesUV))

    if has_texture_uv:
        logger.info("Using mesh.textures (TexturesUV) to expand UVs")
        # Get unique UVs and face UV indices
        unique_uvs = mesh.textures.verts_uvs_padded()[0]  # (U, 2) - 10,583 unique UVs
        mesh_faces_uvs = mesh.textures.faces_uvs_padded()[0]  # (F, 3) - face UV indices
        mesh_faces = mesh.faces_packed()  # (F, 3) - face vertex indices

        logger.info(f"Mesh: {num_vertices} vertices, {num_faces} faces")
        logger.info(f"Unique UVs: {unique_uvs.shape}, faces_uvs: {mesh_faces_uvs.shape}")

        # Create per-vertex UVs by mapping vertex indices to UV indices via faces
        # Initialize with zeros
        per_vertex_uvs = torch.zeros(num_vertices, 2, dtype=unique_uvs.dtype, device=unique_uvs.device)

        # For each face, assign the corresponding UV to each vertex
        # Note: vertices on seams will have their UV overwritten multiple times
        # We'll use the last assignment (arbitrary choice for seam vertices)
        for face_idx in range(num_faces):
            for corner in range(3):
                vertex_idx = mesh_faces[face_idx, corner].item()
                uv_idx = mesh_faces_uvs[face_idx, corner].item()
                per_vertex_uvs[vertex_idx] = unique_uvs[uv_idx]

        logger.info(f"Created per-vertex UVs: {per_vertex_uvs.shape}")
        mesh_verts_uvs = per_vertex_uvs
    else:
        logger.info("Fallback: using provided verts_uvs and faces_uvs to expand")
        # Mesh doesn't have textures attached - use provided UVs and faces_uvs
        unique_uvs = verts_uvs
        if unique_uvs.dim() == 3:
            unique_uvs = unique_uvs[0]

        logger.info(f"Provided UVs: {unique_uvs.shape}")

        # If we have faces_uvs and UV count doesn't match vertex count, expand
        if faces_uvs is not None and unique_uvs.shape[0] != num_vertices:
            logger.info(f"Expanding {unique_uvs.shape[0]} UVs to {num_vertices} vertices using faces_uvs")

            mesh_faces = mesh.faces_packed()
            if faces_uvs.dim() == 3:
                faces_uvs = faces_uvs[0]

            # Create per-vertex UVs
            per_vertex_uvs = torch.zeros(num_vertices, 2, dtype=unique_uvs.dtype, device=unique_uvs.device)

            # Map vertex indices to UV indices via faces
            for face_idx in range(num_faces):
                for corner in range(3):
                    vertex_idx = mesh_faces[face_idx, corner].item()
                    uv_idx = faces_uvs[face_idx, corner].item()
                    per_vertex_uvs[vertex_idx] = unique_uvs[uv_idx]

            logger.info(f"Expanded to per-vertex UVs: {per_vertex_uvs.shape}")
            mesh_verts_uvs = per_vertex_uvs
        else:
            logger.info(f"UVs already match vertices: {unique_uvs.shape}")
            mesh_verts_uvs = unique_uvs

    logger.info(f"Calling map_texture_difference_to_vertices with UVs shape: {mesh_verts_uvs.shape}, num_vertices: {num_vertices}")

    vertex_difference = map_texture_difference_to_vertices(
        difference_map,
        mesh_verts_uvs,
        None,  # Don't need faces_uvs anymore since we expanded
        num_vertices
    )

    logger.info(f"Returned vertex_difference shape: {vertex_difference.shape}")

    # Print statistics
    logger.info("="*70)
    logger.info("TEXTURE DIFFERENCE STATISTICS")
    logger.info("="*70)
    logger.info(f"  Min:     {vertex_difference.min().item():>10.6f} (A brighter)")
    logger.info(f"  Max:     {vertex_difference.max().item():>10.6f} (B brighter)")
    logger.info(f"  Mean:    {vertex_difference.mean().item():>10.6f}")
    logger.info(f"  Median:  {vertex_difference.median().item():>10.6f}")
    logger.info(f"  Std Dev: {vertex_difference.std().item():>10.6f}")
    logger.info(f"  P10:     {torch.quantile(vertex_difference, 0.10).item():>10.6f}")
    logger.info(f"  P90:     {torch.quantile(vertex_difference, 0.90).item():>10.6f}")
    logger.info(f"  P95:     {torch.quantile(vertex_difference, 0.95).item():>10.6f}")
    logger.info(f"  P99:     {torch.quantile(vertex_difference, 0.99).item():>10.6f}")
    logger.info("")
    logger.info(f"Interpretation:")
    logger.info(f"  Positive (red)  = texture B brighter")
    logger.info(f"  Zero (white)    = no difference")
    logger.info(f"  Negative (blue) = texture A brighter")
    logger.info("="*70)

    # Render mesh with difference as vertex colors (like shape displacement)
    # Handle both PyTorch3D and PyRender renderers
    if hasattr(renderer, 'device') and renderer.device == 'cpu':
        # PyRender renderer - convert to numpy
        vertices_np = mesh.verts_packed().cpu().numpy()
        faces_np = mesh.faces_packed().cpu().numpy()
        difference_np = vertex_difference.cpu().numpy()

        logger.debug(f"Texture heatmap: vertices={vertices_np.shape}, faces={faces_np.shape}, difference={difference_np.shape}")
        logger.debug(f"Texture difference: min={difference_np.min():.6f}, max={difference_np.max():.6f}, mean={difference_np.mean():.6f}")

        # Check for vertex count mismatch
        if difference_np.shape[0] != vertices_np.shape[0]:
            error_msg = f"Vertex count mismatch: mesh has {vertices_np.shape[0]} vertices but difference has {difference_np.shape[0]} values"
            logger.error(error_msg)
            raise ValueError(error_msg)

        image = renderer.render_with_heatmap(
            vertices_np,
            faces_np,
            difference_np,
            colormap=colormap,
            add_colorbar=True,
            title="Texture Difference Heatmap",
            normalize_values=False,  # Don't normalize - show raw signed differences
            symmetric_diverging=True,  # Use symmetric range around 0
            normalize_colorbar=False  # Show actual values on colorbar
        )
    else:
        # PyTorch3D renderer - use original parameters
        image = renderer.render_with_heatmap(
            mesh,
            vertex_difference,
            colormap=colormap,
            add_colorbar=True,
            title="Texture Difference Heatmap",
            normalize_values=False,  # Don't normalize - show raw signed differences
            symmetric_diverging=True,  # Use symmetric range around 0
            normalize_colorbar=False  # Show actual values on colorbar
        )

    # Save image
    from PIL import Image
    img = Image.fromarray(image)
    img.save(output_path)

    return output_path.exists()


def create_texture_difference_components_visualization(
    textures: List[torch.Tensor],
    mesh,
    renderer,
    verts_uvs: torch.Tensor,
    faces_uvs: Optional[torch.Tensor],
    output_path: Path,
) -> bool:
    """Create and save 1x3 texture difference component heatmaps (luminance, chrominance, ΔE).

    Args:
        textures: List of texture images
        mesh: PyTorch3D Meshes object to render
        renderer: MeshRenderer3D instance
        verts_uvs: UV coordinates for vertices
        faces_uvs: Face UV indices (optional)
        output_path: Path to save heatmap (1x3 composite)

    Returns:
        True if successful, False otherwise
    """
    import logging
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    logger = logging.getLogger(__name__)

    # Compute all three texture components
    luminance_diff, chroma_diff, delta_e = compute_texture_difference_components(textures)

    # Map each component from 2D texture space to per-vertex values
    num_vertices = mesh.verts_packed().shape[0]
    num_faces = mesh.faces_packed().shape[0]

    # Get per-vertex UVs (same logic as before)
    from pytorch3d.renderer import TexturesUV
    has_texture_uv = (hasattr(mesh, 'textures') and mesh.textures is not None and
                      isinstance(mesh.textures, TexturesUV))

    if has_texture_uv:
        unique_uvs = mesh.textures.verts_uvs_padded()[0]
        mesh_faces_uvs = mesh.textures.faces_uvs_padded()[0]
        mesh_faces = mesh.faces_packed()

        per_vertex_uvs = torch.zeros(num_vertices, 2, dtype=unique_uvs.dtype, device=unique_uvs.device)
        for face_idx in range(num_faces):
            for corner in range(3):
                vertex_idx = mesh_faces[face_idx, corner].item()
                uv_idx = mesh_faces_uvs[face_idx, corner].item()
                per_vertex_uvs[vertex_idx] = unique_uvs[uv_idx]
        mesh_verts_uvs = per_vertex_uvs
    else:
        unique_uvs = verts_uvs
        if unique_uvs.dim() == 3:
            unique_uvs = unique_uvs[0]

        if faces_uvs is not None and unique_uvs.shape[0] != num_vertices:
            mesh_faces = mesh.faces_packed()
            if faces_uvs.dim() == 3:
                faces_uvs = faces_uvs[0]

            per_vertex_uvs = torch.zeros(num_vertices, 2, dtype=unique_uvs.dtype, device=unique_uvs.device)
            for face_idx in range(num_faces):
                for corner in range(3):
                    vertex_idx = mesh_faces[face_idx, corner].item()
                    uv_idx = faces_uvs[face_idx, corner].item()
                    per_vertex_uvs[vertex_idx] = unique_uvs[uv_idx]
            mesh_verts_uvs = per_vertex_uvs
        else:
            mesh_verts_uvs = unique_uvs

    # Map each component to vertices
    vertex_luminance = map_texture_difference_to_vertices(luminance_diff, mesh_verts_uvs, None, num_vertices)
    vertex_chroma = map_texture_difference_to_vertices(chroma_diff, mesh_verts_uvs, None, num_vertices)
    vertex_delta_e = map_texture_difference_to_vertices(delta_e, mesh_verts_uvs, None, num_vertices)

    # Print statistics
    logger.info("="*70)
    logger.info("TEXTURE DIFFERENCE STATISTICS")
    logger.info("="*70)

    logger.info("")
    logger.info("LUMINANCE (signed - brightness):")
    logger.info(f"  Min:     {vertex_luminance.min().item():>10.6f} (A brighter)")
    logger.info(f"  Max:     {vertex_luminance.max().item():>10.6f} (B brighter)")
    logger.info(f"  Mean:    {vertex_luminance.mean().item():>10.6f}")
    logger.info(f"  Median:  {vertex_luminance.median().item():>10.6f}")
    logger.info(f"  P90:     {torch.quantile(vertex_luminance, 0.90).item():>10.6f}")

    logger.info("")
    logger.info("CHROMINANCE (magnitude - color/hue):")
    logger.info(f"  Min:     {vertex_chroma.min().item():>10.6f}")
    logger.info(f"  Max:     {vertex_chroma.max().item():>10.6f}")
    logger.info(f"  Mean:    {vertex_chroma.mean().item():>10.6f}")
    logger.info(f"  Median:  {vertex_chroma.median().item():>10.6f}")
    logger.info(f"  P90:     {torch.quantile(vertex_chroma, 0.90).item():>10.6f}")

    logger.info("")
    logger.info("DELTA E (magnitude - perceptual difference):")
    logger.info(f"  Min:     {vertex_delta_e.min().item():>10.6f}")
    logger.info(f"  Max:     {vertex_delta_e.max().item():>10.6f}")
    logger.info(f"  Mean:    {vertex_delta_e.mean().item():>10.6f}")
    logger.info(f"  Median:  {vertex_delta_e.median().item():>10.6f}")
    logger.info(f"  P90:     {torch.quantile(vertex_delta_e, 0.90).item():>10.6f}")

    logger.info("")
    logger.info("Interpretation:")
    logger.info("  Luminance:   Red=B brighter, Blue=A brighter, White=same brightness")
    logger.info("  Chrominance: Magnitude of color difference (hue/saturation)")
    logger.info("  ΔE:          Perceptual color difference (<1=imperceptible, 1-2=barely noticeable, >10=very different)")
    logger.info("="*70)

    # Prepare components for rendering
    # Colormap convention:
    # - coolwarm (diverging) for signed values, centered at 0
    # - viridis (sequential) for magnitudes, starting at 0
    components = [
        ('Luminance', vertex_luminance, 'coolwarm', True),    # Signed
        ('Chrominance', vertex_chroma, 'viridis', False),     # Magnitude
        ('ΔE (Total)', vertex_delta_e, 'viridis', False),     # Magnitude
    ]

    # Render each component (without colorbar)
    rendered_images = []

    vertices_np = mesh.verts_packed().cpu().numpy()
    faces_np = mesh.faces_packed().cpu().numpy()
    is_cpu_renderer = hasattr(renderer, 'device') and renderer.device == 'cpu'

    for name, data, cmap, is_diverging in components:
        if is_cpu_renderer:
            data_np = data.cpu().numpy()
            img = renderer.render_with_heatmap(
                vertices_np,
                faces_np,
                data_np,
                colormap=cmap,
                add_colorbar=False,
                normalize_values=False,
                symmetric_diverging=is_diverging,
            )
        else:
            img = renderer.render_with_heatmap(
                mesh,
                data,
                colormap=cmap,
                add_colorbar=False,
                normalize_values=False,
                symmetric_diverging=is_diverging,
            )
        rendered_images.append((name, img, data, cmap, is_diverging))

    # Create 1x3 composite figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Texture Difference Components', fontsize=16, fontweight='bold')

    for idx, (name, img, data, cmap, is_diverging) in enumerate(rendered_images):
        ax = axes[idx]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(name, fontsize=14, fontweight='bold')

        # Add colorbar
        from matplotlib.colors import Normalize, TwoSlopeNorm
        from matplotlib import cm

        if is_diverging:
            # Symmetric diverging colorbar centered at 0
            vmax_abs = max(abs(data.min().item()), abs(data.max().item()))
            norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)
            label = 'Difference'
        else:
            # Sequential colorbar starting at 0
            norm = Normalize(vmin=0.0, vmax=data.max().item())
            label = 'Magnitude'

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, rotation=270, labelpad=15, fontsize=10)

    plt.tight_layout()

    # Save composite figure
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path.exists()
