"""
Renderer Factory
================

Single responsibility: Auto-select optimal renderer based on device.
"""

import torch

from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


def create_optimal_renderer(device: torch.device, image_size: int = 512):
    """
    Create optimal renderer based on device type.

    Strategy:
    - GPU (CUDA): Use PyTorch3D for GPU-accelerated rendering
    - CPU: Use PyRender (OpenGL/OSMesa)

    This function abstracts renderer selection so the pipeline
    doesn't need to know about implementation details.

    Args:
        device: PyTorch device (cpu or cuda)
        image_size: Output image resolution in pixels

    Returns:
        Renderer instance (PyTorch3DRenderer or PyRenderRenderer)

    Raises:
        RuntimeError: If renderer initialization fails

    Example:
        >>> device = torch.device('cuda')
        >>> renderer = create_optimal_renderer(device, image_size=512)
        >>> type(renderer).__name__
        'PyTorch3DRenderer'

        >>> device = torch.device('cpu')
        >>> renderer = create_optimal_renderer(device, image_size=512)
        >>> type(renderer).__name__
        'PyRenderRenderer'
    """
    if device.type == 'cuda':
        # GPU: Use PyTorch3D (CUDA-accelerated)
        logger.info("Creating PyTorch3D renderer (GPU mode)")
        try:
            from face_morph.rendering.pytorch3d import create_renderer
            renderer = create_renderer(device, image_size=image_size)
            logger.debug(f"PyTorch3D renderer initialized: {image_size}x{image_size}")
            return renderer
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch3D renderer: {e}")
            raise RuntimeError(
                f"PyTorch3D renderer initialization failed: {e}\n"
                f"Ensure PyTorch3D is installed with CUDA support"
            ) from e

    else:
        # CPU: Use PyRender (OpenGL/OSMesa)
        logger.info("Creating PyRender renderer (CPU mode)")
        try:
            from face_morph.rendering.pyrender import create_pyrender_renderer
            renderer = create_pyrender_renderer(image_size=image_size)
            logger.debug(f"PyRender renderer initialized: {image_size}x{image_size}")
            return renderer
        except Exception as e:
            logger.error(f"Failed to initialize PyRender renderer: {e}")
            raise RuntimeError(
                f"PyRender renderer initialization failed: {e}\n"
                f"Ensure pyrender is installed and OpenGL/OSMesa is available"
            ) from e


def get_renderer_type(renderer) -> str:
    """
    Get renderer type from instance.

    This allows pipeline code to adapt rendering logic
    based on renderer capabilities.

    Args:
        renderer: Renderer instance

    Returns:
        Renderer type string: 'pytorch3d' or 'pyrender'

    Example:
        >>> renderer = create_optimal_renderer(torch.device('cpu'))
        >>> get_renderer_type(renderer)
        'pyrender'
    """
    class_name = type(renderer).__name__.lower()

    # Check class name patterns
    if 'pytorch3d' in class_name or 'meshrenderer3d' in class_name or 'meshrasterizer' in class_name:
        return 'pytorch3d'
    elif 'pyrender' in class_name or 'offscreen' in class_name:
        return 'pyrender'
    else:
        # Fallback: check for specific attributes
        if hasattr(renderer, 'device') and hasattr(renderer, 'rasterizer'):
            return 'pytorch3d'
        elif hasattr(renderer, 'renderer') and hasattr(renderer, 'scene'):
            return 'pyrender'
        else:
            logger.warning(f"Unknown renderer type: {type(renderer).__name__}")
            # Use renderer_type attribute if available (from BaseRenderer)
            if hasattr(renderer, 'renderer_type'):
                return renderer.renderer_type
            return 'unknown'
