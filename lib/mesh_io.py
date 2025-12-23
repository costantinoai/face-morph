"""
Mesh I/O Operations
===================

Single responsibility: Load and save 3D meshes.
"""

from pathlib import Path
from typing import Tuple
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes


def load_mesh(
    filepath: Path,
    device: torch.device
) -> Tuple[Meshes, dict]:
    """
    Load mesh from OBJ file.

    Single responsibility: Load mesh geometry.

    Args:
        filepath: Path to OBJ file
        device: PyTorch device

    Returns:
        Tuple of (Meshes object, auxiliary data dict)

    Raises:
        ValueError: If mesh is invalid
    """
    # Load OBJ
    verts, faces, aux = load_obj(
        str(filepath),
        load_textures=True,
        device=device
    )

    # Validate
    if verts.shape[0] == 0:
        raise ValueError(f"Mesh has no vertices: {filepath}")

    if faces.verts_idx.shape[0] == 0:
        raise ValueError(f"Mesh has no faces: {filepath}")

    # Create mesh
    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx]
    ).to(device)

    # Package aux data
    aux_data = {
        'verts_uvs': aux.verts_uvs,
        'faces_uvs': faces.textures_idx if hasattr(faces, 'textures_idx') else None,
        'texture_images': aux.texture_images,
        'normals': aux.normals,
        'num_vertices': verts.shape[0],
        'num_faces': faces.verts_idx.shape[0]
    }

    return mesh, aux_data


def save_mesh(
    mesh: Meshes,
    filepath: Path,
    verts_uvs: torch.Tensor = None,
    texture_map: torch.Tensor = None
) -> None:
    """
    Save mesh to OBJ file.

    Single responsibility: Save mesh geometry.

    Args:
        mesh: Meshes object
        filepath: Output path
        verts_uvs: UV coordinates (optional)
        texture_map: Texture map (optional)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    verts = mesh.verts_packed().cpu()
    faces = mesh.faces_packed().cpu()

    save_obj(
        str(filepath),
        verts=verts,
        faces=faces,
        verts_uvs=verts_uvs.cpu() if verts_uvs is not None else None,
        texture_map=texture_map.cpu() if texture_map is not None else None,
        decimal_places=6
    )
