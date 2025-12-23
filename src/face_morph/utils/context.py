"""
Context Manager Utilities
==========================

Single responsibility: Provide reusable context managers for common patterns.
"""

import torch
from contextlib import contextmanager
from typing import Optional

from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def amp_autocast(device: torch.device, enabled: bool = True, dtype: Optional[torch.dtype] = None):
    """
    Context manager for Automatic Mixed Precision (AMP) with proper device handling.

    This abstracts away the complexity of AMP configuration and provides
    a clean API for enabling/disabling mixed precision based on device type.

    AMP provides 2-3x speedup on GPU with minimal accuracy loss (ε < 1e-4).
    Not supported on CPU (silently disabled).

    Args:
        device: PyTorch device (cpu or cuda)
        enabled: Whether to enable AMP (automatically disabled on CPU)
        dtype: Data type for autocast (default: torch.float16 for CUDA)

    Yields:
        Context for mixed precision operations

    Example:
        >>> device = torch.device('cuda')
        >>> with amp_autocast(device, enabled=True):
        ...     # Operations here use FP16 on GPU
        ...     output = model(input)
        ...
        >>> device = torch.device('cpu')
        >>> with amp_autocast(device, enabled=True):
        ...     # AMP automatically disabled on CPU (not supported)
        ...     output = model(input)
    """
    # AMP only supported on CUDA
    if device.type != 'cuda':
        if enabled:
            logger.debug("AMP requested but disabled (CPU mode)")
        # No-op context for CPU
        yield
        return

    # Determine dtype
    if dtype is None:
        dtype = torch.float16  # Default for CUDA

    # Apply autocast
    if enabled:
        logger.debug(f"AMP enabled: device={device}, dtype={dtype}")
        with torch.autocast(device_type='cuda', dtype=dtype):
            yield
    else:
        logger.debug("AMP disabled (full precision)")
        yield


@contextmanager
def torch_inference_mode():
    """
    Context manager for PyTorch inference mode.

    Disables gradient tracking and autograd for better performance
    during inference (rendering, morphing).

    Inference mode provides:
    - Reduced memory usage (no gradient buffers)
    - Faster operations (no autograd overhead)
    - Clear intent (this is inference, not training)

    Yields:
        Context for inference operations

    Example:
        >>> with torch_inference_mode():
        ...     output = model(input)  # No gradients tracked
    """
    logger.debug("Entering inference mode (no gradient tracking)")
    with torch.inference_mode():
        yield
    logger.debug("Exited inference mode")
