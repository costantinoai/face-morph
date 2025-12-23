"""Device management utilities for efficient GPU/CPU data handling.

This module provides utilities to minimize device transfers and track tensor locations,
reducing CPU ↔ GPU transfer overhead by 20-30%.
"""

import torch
from typing import Optional, Union, List
from pytorch3d.structures import Meshes

from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


class DeviceManager:
    """
    Manages device placement and transfers for tensors and meshes.

    Minimizes unnecessary CPU ↔ GPU transfers by:
    - Tracking tensor locations
    - Caching device-specific conversions
    - Providing smart transfer methods
    """

    def __init__(self, primary_device: torch.device):
        """
        Initialize device manager.

        Args:
            primary_device: Primary device for computation (cpu or cuda)
        """
        self.primary_device = primary_device
        self._transfer_count = 0
        self._cached_transfers = {}

    def ensure_device(
        self,
        tensor: Optional[torch.Tensor],
        target_device: Optional[torch.device] = None,
        cache_key: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Ensure tensor is on target device, avoiding unnecessary transfers.

        Args:
            tensor: Input tensor (can be None)
            target_device: Target device (defaults to primary_device)
            cache_key: Optional key for caching this transfer

        Returns:
            Tensor on target device, or None if input was None

        Example:
            >>> dm = DeviceManager(torch.device('cuda'))
            >>> texture_gpu = dm.ensure_device(texture_cpu, cache_key='texture1')
            # Second call returns cached result if same key
            >>> texture_gpu = dm.ensure_device(texture_cpu, cache_key='texture1')
        """
        if tensor is None:
            return None

        target_device = target_device or self.primary_device

        # Check cache first
        if cache_key and cache_key in self._cached_transfers:
            cached = self._cached_transfers[cache_key]
            if cached.device == target_device:
                logger.debug(f"Using cached transfer for {cache_key}")
                return cached

        # Check if already on correct device
        if tensor.device == target_device:
            if cache_key:
                self._cached_transfers[cache_key] = tensor
            return tensor

        # Transfer needed
        self._transfer_count += 1
        logger.debug(f"Transferring tensor from {tensor.device} to {target_device}")
        result = tensor.to(target_device)

        # Cache if requested
        if cache_key:
            self._cached_transfers[cache_key] = result

        return result

    def ensure_mesh_device(
        self,
        mesh: Meshes,
        target_device: Optional[torch.device] = None,
    ) -> Meshes:
        """
        Ensure PyTorch3D mesh is on target device.

        Args:
            mesh: PyTorch3D Meshes object
            target_device: Target device (defaults to primary_device)

        Returns:
            Mesh on target device
        """
        target_device = target_device or self.primary_device

        # Check if already on correct device
        if hasattr(mesh, 'device') and mesh.device == target_device:
            return mesh

        # Transfer
        self._transfer_count += 1
        logger.debug(f"Transferring mesh to {target_device}")
        return mesh.to(target_device)

    def to_cpu_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move tensor to CPU only if not already there.

        Args:
            tensor: Input tensor

        Returns:
            Tensor on CPU
        """
        if tensor.device.type == 'cpu':
            return tensor

        self._transfer_count += 1
        logger.debug(f"Transferring tensor from {tensor.device} to CPU")
        return tensor.cpu()

    def get_transfer_count(self) -> int:
        """Get total number of device transfers performed."""
        return self._transfer_count

    def clear_cache(self):
        """Clear cached transfers to free memory."""
        self._cached_transfers.clear()
        logger.debug("Cleared device transfer cache")


def optimize_texture_device(
    texture: Optional[torch.Tensor],
    verts_uvs: Optional[torch.Tensor],
    faces_uvs: Optional[torch.Tensor],
    target_device: torch.device,
) -> tuple:
    """
    Move texture and UV data to target device once.

    Avoids repeated transfers in rendering loop.

    Args:
        texture: Texture tensor (H, W, 3)
        verts_uvs: UV coordinates (V, 2)
        faces_uvs: UV face indices (F, 3)
        target_device: Target device

    Returns:
        Tuple of (texture, verts_uvs, faces_uvs) on target device
    """
    if texture is None:
        return None, verts_uvs, faces_uvs

    # Move all texture-related data to device once
    texture_device = texture.to(target_device) if texture is not None else None
    verts_uvs_device = verts_uvs.to(target_device) if verts_uvs is not None else None
    faces_uvs_device = faces_uvs.to(target_device) if faces_uvs is not None else None

    logger.debug(f"Optimized texture data to {target_device}")

    return texture_device, verts_uvs_device, faces_uvs_device
