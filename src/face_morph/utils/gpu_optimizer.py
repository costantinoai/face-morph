"""
GPU Optimization Utilities
===========================

Single responsibility: Provide GPU optimization utilities (CUDA streams, dynamic batching).

This module implements advanced GPU optimization techniques to maximize throughput:
1. CUDA streams for overlapping compute and data transfer
2. Dynamic batch sizing based on available GPU memory
3. Memory profiling for optimal resource utilization

Best practices from NVIDIA CUDA Programming Guide and PyTorch documentation.
"""

from typing import Optional, List, Tuple
import torch
from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


class CUDAStreamManager:
    """
    Manages CUDA streams for overlapping computation and data transfers.

    Pattern: Resource Pool
    - Maintains a pool of CUDA streams for concurrent operations
    - Automatically allocates and manages stream lifecycle
    - Enables pipelining of compute and memory transfer operations

    Benefits:
    - 10-30% performance improvement through overlap
    - Better GPU utilization
    - Reduced idle time between operations
    """

    def __init__(self, num_streams: int = 2):
        """
        Initialize CUDA stream manager.

        Args:
            num_streams: Number of concurrent streams (default: 2 for compute/transfer overlap)
        """
        self.num_streams = num_streams
        self.streams: List[torch.cuda.Stream] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.device.type == 'cuda':
            # Create CUDA streams for pipelining
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
            logger.debug(f"Created {num_streams} CUDA streams for pipelining")
        else:
            logger.debug("CUDA not available, stream management disabled")

    def get_stream(self, index: int = 0) -> Optional[torch.cuda.Stream]:
        """Get CUDA stream by index."""
        if self.streams and 0 <= index < len(self.streams):
            return self.streams[index]
        return None

    def synchronize_all(self):
        """Synchronize all streams (wait for all operations to complete)."""
        if self.device.type == 'cuda':
            for stream in self.streams:
                stream.synchronize()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - synchronize all streams."""
        self.synchronize_all()


def get_available_gpu_memory() -> float:
    """
    Get available GPU memory in GB.

    Returns:
        Available memory in GB, or 0.0 if CUDA unavailable
    """
    if not torch.cuda.is_available():
        return 0.0

    # Get memory stats
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)

    # Available = total - max(allocated, reserved)
    # Use reserved as it includes cached memory
    available_memory = total_memory - reserved_memory
    available_gb = available_memory / (1024 ** 3)

    return available_gb


def estimate_tensor_memory(tensor: torch.Tensor) -> float:
    """
    Estimate memory usage of a tensor in GB.

    Args:
        tensor: PyTorch tensor

    Returns:
        Memory usage in GB
    """
    num_elements = tensor.numel()
    bytes_per_element = tensor.element_size()
    total_bytes = num_elements * bytes_per_element
    total_gb = total_bytes / (1024 ** 3)

    return total_gb


def calculate_optimal_batch_size(
    sample_tensor: torch.Tensor,
    min_batch_size: int = 1,
    max_batch_size: int = 50,
    safety_margin: float = 0.7,
    default_batch_size: int = 10
) -> int:
    """
    Dynamically calculate optimal batch size based on available GPU memory.

    This function estimates how many items can fit in GPU memory by:
    1. Measuring memory required for a single item
    2. Checking available GPU memory
    3. Calculating max batch size with safety margin
    4. Clamping to min/max bounds

    Args:
        sample_tensor: Representative tensor (e.g., single mesh vertices)
        min_batch_size: Minimum allowed batch size (default: 1)
        max_batch_size: Maximum allowed batch size (default: 50)
        safety_margin: Fraction of available memory to use (default: 0.7 = 70%)
        default_batch_size: Fallback if GPU unavailable (default: 10)

    Returns:
        Optimal batch size (clamped to [min_batch_size, max_batch_size])

    Example:
        >>> vertices = torch.randn(10523, 3)  # Sample mesh
        >>> batch_size = calculate_optimal_batch_size(vertices)
        >>> print(f"Optimal batch size: {batch_size}")
    """
    if not torch.cuda.is_available():
        logger.debug(f"CUDA unavailable, using default batch size: {default_batch_size}")
        return default_batch_size

    # Get available memory
    available_gb = get_available_gpu_memory()
    usable_gb = available_gb * safety_margin

    if available_gb < 0.1:  # Less than 100MB available
        logger.warning(f"Very low GPU memory available ({available_gb:.2f}GB), using minimum batch size")
        return min_batch_size

    # Estimate memory per item
    # Assume rendering requires 3x memory (vertices + faces + textures)
    memory_per_item = estimate_tensor_memory(sample_tensor) * 3.0

    if memory_per_item < 1e-6:  # Essentially zero
        logger.debug(f"Sample tensor too small to estimate, using default: {default_batch_size}")
        return default_batch_size

    # Calculate how many items fit
    estimated_batch_size = int(usable_gb / memory_per_item)

    # Clamp to bounds
    optimal_batch_size = max(min_batch_size, min(estimated_batch_size, max_batch_size))

    logger.info(
        f"Dynamic batching: available={available_gb:.2f}GB, "
        f"usable={usable_gb:.2f}GB, "
        f"per_item={memory_per_item:.4f}GB, "
        f"optimal_batch={optimal_batch_size}"
    )

    return optimal_batch_size


def estimate_render_batch_size(
    num_vertices: int,
    num_faces: int,
    has_texture: bool = False,
    image_size: int = 512,
    min_batch_size: int = 1,
    max_batch_size: int = 10,  # Conservative: PyTorch3D batch rendering scales linearly
    safety_margin: float = 0.6  # Conservative for rendering (60%)
) -> int:
    """
    Estimate optimal batch size for mesh rendering based on mesh complexity.

    IMPORTANT: PyTorch3D batch rendering scales LINEARLY with batch size
    (see https://github.com/facebookresearch/pytorch3d/issues/1120).
    Therefore, smaller batch sizes (4-10) are often optimal to balance
    memory usage vs minimal actual speedup from batching.

    Rendering is memory-intensive due to:
    - Rasterization buffers
    - Texture atlases
    - Fragment shader intermediate results

    Args:
        num_vertices: Number of vertices in mesh
        num_faces: Number of faces in mesh
        has_texture: Whether mesh has textures
        image_size: Output image resolution (e.g., 512)
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size (default 10, PyTorch3D limitation)
        safety_margin: Fraction of memory to use (conservative for rendering)

    Returns:
        Optimal rendering batch size (typically 4-10 for PyTorch3D)
    """
    if not torch.cuda.is_available():
        return 10  # Default for CPU

    # Estimate memory per mesh
    # Vertices: (V, 3) float32 = V * 3 * 4 bytes
    vertex_memory_mb = (num_vertices * 3 * 4) / (1024 ** 2)

    # Faces: (F, 3) int64 = F * 3 * 8 bytes
    face_memory_mb = (num_faces * 3 * 8) / (1024 ** 2)

    # Texture: (H, W, 3) float32 (if present)
    texture_memory_mb = 0
    if has_texture:
        # Assume 1024x1024 texture typical
        texture_memory_mb = (1024 * 1024 * 3 * 4) / (1024 ** 2)

    # Render buffer: image_size x image_size x 4 (RGBA) float32
    render_buffer_mb = (image_size * image_size * 4 * 4) / (1024 ** 2)

    # Total per mesh (vertices + faces + texture + render buffer)
    # Add 50% overhead for intermediate calculations
    total_per_mesh_mb = (vertex_memory_mb + face_memory_mb + texture_memory_mb + render_buffer_mb) * 1.5
    total_per_mesh_gb = total_per_mesh_mb / 1024

    # Get available memory
    available_gb = get_available_gpu_memory()
    usable_gb = available_gb * safety_margin

    if usable_gb < 0.1:
        return min_batch_size

    # Calculate batch size
    estimated_batch = int(usable_gb / total_per_mesh_gb) if total_per_mesh_gb > 0 else max_batch_size
    optimal_batch = max(min_batch_size, min(estimated_batch, max_batch_size))

    logger.info(
        f"Render batch sizing: vertices={num_vertices:,}, faces={num_faces:,}, "
        f"texture={'yes' if has_texture else 'no'}, "
        f"available={available_gb:.2f}GB, "
        f"per_mesh={total_per_mesh_gb:.4f}GB, "
        f"optimal_batch={optimal_batch}"
    )

    return optimal_batch


class GPUMemoryMonitor:
    """
    Monitor GPU memory usage for profiling and debugging.

    Usage:
        with GPUMemoryMonitor("Morphing phase") as monitor:
            # GPU operations here
            pass
        # Automatically logs memory stats
    """

    def __init__(self, phase_name: str):
        """
        Initialize memory monitor.

        Args:
            phase_name: Name of the operation phase for logging
        """
        self.phase_name = phase_name
        self.start_allocated = 0
        self.start_reserved = 0

    def __enter__(self):
        """Record starting memory state."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            self.start_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.debug(f"{self.phase_name} - Start: {self.start_allocated:.2f}GB allocated, {self.start_reserved:.2f}GB reserved")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record ending memory state and log delta."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            end_reserved = torch.cuda.memory_reserved() / (1024 ** 3)

            delta_allocated = end_allocated - self.start_allocated
            delta_reserved = end_reserved - self.start_reserved

            logger.info(
                f"{self.phase_name} - Memory: "
                f"allocated={end_allocated:.2f}GB (Δ{delta_allocated:+.2f}GB), "
                f"reserved={end_reserved:.2f}GB (Δ{delta_reserved:+.2f}GB)"
            )
