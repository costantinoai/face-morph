"""
Input Validation
================

Single responsibility: Validate inputs before processing.
"""

from pathlib import Path
from typing import Union
import torch


def validate_input_file(filepath: Union[str, Path]) -> Path:
    """
    Validate that input file exists and has supported format.

    Args:
        filepath: Path to mesh file

    Returns:
        Path object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format not supported
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}\n"
            f"Please check the path."
        )

    supported_formats = {'.fbx', '.obj'}
    if filepath.suffix.lower() not in supported_formats:
        raise ValueError(
            f"Unsupported format: {filepath.suffix}\n"
            f"Supported: {', '.join(supported_formats)}"
        )

    return filepath


def validate_device(device: str) -> torch.device:
    """
    Validate and create torch device.

    Args:
        device: Device string ('cpu', 'cuda', etc.)

    Returns:
        torch.device object

    Raises:
        RuntimeError: If CUDA requested but not available
    """
    device = torch.device(device)

    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available.\n"
            "Options:\n"
            "  1. Use CPU: set USE_GPU = False\n"
            "  2. Install CUDA: https://developer.nvidia.com/cuda-downloads\n"
            "  3. Reinstall PyTorch with CUDA support"
        )

    return device


def validate_ratios(ratios: list) -> list:
    """
    Validate morph ratio list.

    Args:
        ratios: List of (ratio1, ratio2) tuples

    Returns:
        Validated ratios

    Raises:
        ValueError: If ratios invalid
    """
    if not ratios:
        raise ValueError("Ratios list is empty")

    for r1, r2 in ratios:
        if r1 < 0 or r2 < 0:
            raise ValueError(f"Negative ratio: ({r1}, {r2})")

        if r1 + r2 == 0:
            raise ValueError(f"Both ratios are zero: ({r1}, {r2})")

    return ratios
