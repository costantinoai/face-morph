"""
Texture I/O Operations
=======================

Single responsibility: Load and save textures.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def has_texture(aux_data: dict) -> bool:
    """
    Check if auxiliary data contains valid texture.

    Args:
        aux_data: Auxiliary data from mesh loading

    Returns:
        True if texture available
    """
    if aux_data['texture_images'] is None:
        return False

    if len(aux_data['texture_images']) == 0:
        return False

    return True


def load_texture(aux_data: dict, device: torch.device) -> Optional[torch.Tensor]:
    """
    Extract texture from auxiliary data.

    Single responsibility: Extract and prepare texture.

    Args:
        aux_data: Auxiliary data from mesh loading
        device: PyTorch device

    Returns:
        Texture tensor (H, W, C) or None
    """
    if not has_texture(aux_data):
        return None

    # Get first texture
    texture = list(aux_data['texture_images'].values())[0]

    if texture is None:
        return None

    return texture.to(device)


def save_texture(
    texture: torch.Tensor,
    filepath: Path,
    format: str = 'png'
) -> None:
    """
    Save texture to image file.

    Single responsibility: Save texture as image.

    Args:
        texture: Texture tensor (H, W, C) in [0, 1]
        filepath: Output path
        format: Image format ('png', 'jpg')
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL required for texture saving. Install: pip install pillow")

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    texture_np = texture.cpu().numpy()

    # Clip to valid range
    texture_np = np.clip(texture_np, 0, 1)

    # Convert to uint8
    texture_uint8 = (texture_np * 255).astype(np.uint8)

    # Save
    Image.fromarray(texture_uint8).save(filepath, format=format.upper())


def resize_texture(
    texture: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Resize texture to match target dimensions.

    Args:
        texture: Input texture (H, W, C)
        target_size: (height, width)
        mode: Interpolation mode

    Returns:
        Resized texture
    """
    import torch.nn.functional as F

    # Permute to (C, H, W) and add batch dim
    texture_chw = texture.permute(2, 0, 1).unsqueeze(0)

    # Resize
    resized = F.interpolate(
        texture_chw,
        size=target_size,
        mode=mode,
        align_corners=False
    )

    # Permute back to (H, W, C)
    return resized.squeeze(0).permute(1, 2, 0)
