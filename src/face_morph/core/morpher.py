"""
Core Morphing Logic
===================

Single responsibility: Implement morphing algorithms.

Optimizations:
- torch.compile for JIT compilation (PyTorch 2.0+)
- Vectorized batch processing
- Minimal memory allocation
"""

from typing import List, Tuple, Optional
import torch
from pytorch3d.structures import Meshes
from .texture_io import resize_texture
from face_morph.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


# ============================================================================
# JIT-Compiled Helper Functions (torch.compile for max performance)
# ============================================================================

# TODO: torch.compile disabled due to CUDA Graphs memory usage (1.94 GiB)
# Re-enable once GPU memory is optimized
# @torch.compile(mode="reduce-overhead", dynamic=True)
def _vectorized_vertex_lerp(verts1: torch.Tensor, verts2: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled vectorized vertex interpolation.

    Args:
        verts1: (V, 3) vertex positions of first mesh
        verts2: (V, 3) vertex positions of second mesh
        ratios: (N,) interpolation ratios

    Returns:
        (N, V, 3) morphed vertices for all ratios
    """
    verts1_expanded = verts1.unsqueeze(0)  # (1, V, 3)
    verts2_expanded = verts2.unsqueeze(0)  # (1, V, 3)
    ratios_expanded = ratios.view(-1, 1, 1)  # (N, 1, 1)
    return torch.lerp(verts1_expanded, verts2_expanded, ratios_expanded)


# @torch.compile(mode="reduce-overhead", dynamic=True)
def _vectorized_texture_lerp(tex1: torch.Tensor, tex2: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled vectorized texture interpolation.

    Args:
        tex1: (H, W, C) first texture
        tex2: (H, W, C) second texture
        ratios: (N,) interpolation ratios

    Returns:
        (N, H, W, C) morphed textures for all ratios
    """
    tex1_expanded = tex1.unsqueeze(0).contiguous()  # (1, H, W, C)
    tex2_expanded = tex2.unsqueeze(0).contiguous()  # (1, H, W, C)
    ratios_expanded = ratios.view(-1, 1, 1, 1)  # (N, 1, 1, 1)
    return torch.lerp(tex1_expanded, tex2_expanded, ratios_expanded)


class MeshMorpher:
    """
    3D mesh morphing with GPU acceleration.

    Single responsibility: Morph meshes and textures.
    """

    def __init__(self, device: torch.device, use_amp: bool = True):
        """
        Initialize morpher.

        Args:
            device: PyTorch device
            use_amp: Use automatic mixed precision (FP16)
        """
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'

    def morph_mesh(
        self,
        mesh1: Meshes,
        mesh2: Meshes,
        ratio: float
    ) -> Meshes:
        """
        Morph two meshes with linear interpolation.

        Single responsibility: Geometric morphing.

        Args:
            mesh1: First mesh
            mesh2: Second mesh
            ratio: Interpolation ratio (0=mesh1, 1=mesh2)

        Returns:
            Morphed mesh
        """
        verts1 = mesh1.verts_packed()
        verts2 = mesh2.verts_packed()

        # Validate topology
        if verts1.shape != verts2.shape:
            raise ValueError(
                f"Meshes have different topology:\n"
                f"  Mesh 1: {verts1.shape[0]} vertices\n"
                f"  Mesh 2: {verts2.shape[0]} vertices\n"
                f"Morphing requires identical vertex counts."
            )

        # GPU-optimized interpolation
        if self.use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                morphed_verts = torch.lerp(verts1, verts2, ratio)
        else:
            morphed_verts = torch.lerp(verts1, verts2, ratio)

        # Create morphed mesh
        return Meshes(
            verts=[morphed_verts],
            faces=[mesh1.faces_packed()]
        ).to(self.device)

    def morph_texture(
        self,
        texture1: Optional[torch.Tensor],
        texture2: Optional[torch.Tensor],
        ratio: float
    ) -> Optional[torch.Tensor]:
        """
        Morph two textures with linear interpolation.

        Single responsibility: Texture morphing.

        Args:
            texture1: First texture (H, W, C) or None
            texture2: Second texture (H, W, C) or None
            ratio: Interpolation ratio (0=texture1, 1=texture2)

        Returns:
            Morphed texture or None
        """
        # Handle missing textures
        if texture1 is None or texture2 is None:
            return None

        # Resize if needed
        if texture1.shape != texture2.shape:
            texture2 = resize_texture(
                texture2,
                (texture1.shape[0], texture1.shape[1])
            )

        # GPU-optimized interpolation
        if self.use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                morphed = torch.lerp(
                    texture1.contiguous(),
                    texture2.contiguous(),
                    ratio
                )
        else:
            morphed = torch.lerp(
                texture1.contiguous(),
                texture2.contiguous(),
                ratio
            )

        return morphed

    def morph(
        self,
        mesh1: Meshes,
        mesh2: Meshes,
        texture1: Optional[torch.Tensor],
        texture2: Optional[torch.Tensor],
        ratio1: float,
        ratio2: float
    ) -> Tuple[Meshes, Optional[torch.Tensor]]:
        """
        Morph meshes and textures together.

        Single responsibility: Coordinate mesh+texture morphing.

        Args:
            mesh1: First mesh
            mesh2: Second mesh
            texture1: First texture or None
            texture2: Second texture or None
            ratio1: Weight for first mesh/texture
            ratio2: Weight for second mesh/texture

        Returns:
            Tuple of (morphed mesh, morphed texture or None)
        """
        # Normalize ratios
        total = ratio1 + ratio2
        ratio1, ratio2 = ratio1 / total, ratio2 / total

        # Morph mesh
        morphed_mesh = self.morph_mesh(mesh1, mesh2, ratio2)

        # Morph texture (if available)
        morphed_texture = self.morph_texture(texture1, texture2, ratio2)

        return morphed_mesh, morphed_texture

    def batch_morph(
        self,
        mesh1: Meshes,
        mesh2: Meshes,
        texture1: Optional[torch.Tensor],
        texture2: Optional[torch.Tensor],
        ratios: List[Tuple[float, float]]
    ) -> List[Tuple[Meshes, Optional[torch.Tensor]]]:
        """
        Generate multiple morphs efficiently using vectorized GPU operations.

        Optimizations:
        - Vectorized vertex interpolation (all morphs at once)
        - Vectorized texture interpolation (batch processing)
        - Pre-allocated result list
        - Minimal tensor copying

        Args:
            mesh1: First mesh
            mesh2: Second mesh
            texture1: First texture or None
            texture2: Second texture or None
            ratios: List of (ratio1, ratio2) tuples

        Returns:
            List of (morphed mesh, morphed texture or None) tuples
        """
        # Normalize ratios and convert to tensors for vectorized ops
        normalized_ratios = [(r2 / (r1 + r2)) for r1, r2 in ratios]
        ratio_tensor = torch.tensor(normalized_ratios, device=self.device, dtype=torch.float32)

        verts1 = mesh1.verts_packed()
        verts2 = mesh2.verts_packed()
        faces = mesh1.faces_packed()

        # Pre-allocate results list for efficiency
        num_morphs = len(ratios)
        results = [None] * num_morphs

        # Use inference mode for optimal performance
        with torch.inference_mode():
            # === JIT-COMPILED VECTORIZED MESH MORPHING ===
            # Use compiled function for automatic kernel fusion and optimization
            logger.debug("Compiling vertex interpolation (first run only)...")
            if self.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    all_morphed_verts = _vectorized_vertex_lerp(verts1, verts2, ratio_tensor).clone()
            else:
                all_morphed_verts = _vectorized_vertex_lerp(verts1, verts2, ratio_tensor).clone()

            # === JIT-COMPILED VECTORIZED TEXTURE MORPHING ===
            all_morphed_textures = None
            if texture1 is not None and texture2 is not None:
                # Resize if needed
                if texture1.shape != texture2.shape:
                    texture2 = resize_texture(texture2, (texture1.shape[0], texture1.shape[1]))

                # Use compiled function for texture batch processing
                logger.debug("Compiling texture interpolation (first run only)...")
                if self.use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        all_morphed_textures = _vectorized_texture_lerp(texture1, texture2, ratio_tensor).clone()
                else:
                    all_morphed_textures = _vectorized_texture_lerp(texture1, texture2, ratio_tensor).clone()

            # === ASSEMBLE RESULTS ===
            # Efficiently create mesh objects without copying vertex data
            for i in range(num_morphs):
                mesh = Meshes(verts=[all_morphed_verts[i]], faces=[faces]).to(self.device)
                texture = all_morphed_textures[i] if all_morphed_textures is not None else None
                results[i] = (mesh, texture)

        return results


def create_morpher(device: torch.device, use_amp: bool = True) -> MeshMorpher:
    """
    Factory function to create morpher.

    Args:
        device: PyTorch device
        use_amp: Use automatic mixed precision

    Returns:
        MeshMorpher instance
    """
    return MeshMorpher(device, use_amp)
