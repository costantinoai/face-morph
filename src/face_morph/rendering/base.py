"""
Base renderer interface for polymorphic rendering.

This module defines the abstract base class that all renderers must implement,
ensuring a consistent API across PyTorch3D and PyRender backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import torch
from pytorch3d.structures import Meshes


class BaseRenderer(ABC):
    """
    Abstract base class for mesh renderers.

    Defines the interface that all renderer implementations must provide.
    This enables polymorphic rendering without type checking in orchestrator.

    Attributes:
        device: Torch device ('cpu' or 'cuda:0')
        renderer_type: String identifier ('pytorch3d' or 'pyrender')
    """

    def __init__(self, device: torch.device):
        """
        Initialize base renderer.

        Args:
            device: Torch device for rendering
        """
        self.device = device
        self.renderer_type = self.__class__.__name__.lower().replace('renderer', '')

    @abstractmethod
    def render_mesh(
        self,
        mesh: Meshes,
        texture: Optional[torch.Tensor] = None,
        verts_uvs: Optional[torch.Tensor] = None,
        faces_uvs: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Render a single mesh to an image.

        Args:
            mesh: PyTorch3D Meshes object (single mesh)
            texture: Optional texture image (H, W, 3) in range [0, 1]
            verts_uvs: Optional UV coordinates (V, 2)
            faces_uvs: Optional UV face indices (F, 3)

        Returns:
            RGB image as numpy array (H, W, 3) in range [0, 255], uint8

        Note:
            Subclasses should handle both textured and untextured rendering.
            PyRender subclass should convert PyTorch3D mesh to numpy internally.
        """
        pass

    @abstractmethod
    def batch_render_meshes(
        self,
        meshes_list: List[Meshes],
        textures_list: Optional[List[torch.Tensor]] = None,
        verts_uvs: Optional[torch.Tensor] = None,
        faces_uvs: Optional[torch.Tensor] = None,
        chunk_size: int = 10,
    ) -> List[np.ndarray]:
        """
        Render multiple meshes efficiently.

        Args:
            meshes_list: List of PyTorch3D Meshes objects
            textures_list: Optional list of texture images, one per mesh
            verts_uvs: UV coordinates (shared across all meshes)
            faces_uvs: Face UV indices (shared across all meshes)
            chunk_size: Number of meshes to render per batch

        Returns:
            List of RGB images as numpy arrays (H, W, 3) in range [0, 255], uint8

        Note:
            PyTorch3D: Uses GPU batching for efficiency
            PyRender: Falls back to sequential rendering (no batching support)
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up renderer resources.

        Releases GPU memory, closes windows, etc.
        Called when renderer is no longer needed.
        """
        pass

    def __repr__(self) -> str:
        """String representation of renderer."""
        return f"{self.__class__.__name__}(device={self.device}, type='{self.renderer_type}')"
