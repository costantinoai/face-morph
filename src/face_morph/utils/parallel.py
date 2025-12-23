"""
Parallelization Utilities
==========================

Single responsibility: Optimize CPU core usage for parallel tasks.
"""

import multiprocessing as mp
from typing import Optional

from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


def get_optimal_workers(task_type: str = "cpu") -> int:
    """
    Determine optimal worker count for parallel tasks.

    Strategy:
    - CPU-bound tasks: cpu_count - 1 (reserve 1 core for system)
    - I/O-bound tasks: cpu_count * 2 (can oversubscribe)

    Args:
        task_type: "cpu" (CPU-bound) or "io" (I/O-bound)

    Returns:
        Optimal worker count

    Example:
        >>> get_optimal_workers("cpu")  # On 8-core system
        7
        >>> get_optimal_workers("io")
        16
    """
    cpu_count_val = mp.cpu_count()

    if task_type == "cpu":
        # CPU-bound: Leave 1 core for OS to maintain responsiveness
        workers = max(1, cpu_count_val - 1)
        logger.debug(f"CPU-bound task: {workers} workers ({cpu_count_val} cores total)")
        return workers

    elif task_type == "io":
        # I/O-bound: Can use 2x cores (threads wait for I/O)
        workers = cpu_count_val * 2
        logger.debug(f"I/O-bound task: {workers} workers ({cpu_count_val} cores total)")
        return workers

    else:
        # Unknown task type: use all cores
        logger.warning(f"Unknown task_type '{task_type}', using all cores")
        return cpu_count_val


def create_worker_pool(task_type: str = "cpu", max_workers: Optional[int] = None):
    """
    Create optimally-sized multiprocessing pool.

    Args:
        task_type: "cpu" or "io" for optimal worker count
        max_workers: Optional maximum worker limit (overrides optimal count)

    Returns:
        multiprocessing.Pool instance (use with context manager)

    Example:
        >>> with create_worker_pool("cpu") as pool:
        ...     results = pool.map(heavy_function, data)
        ...
        >>> with create_worker_pool("io", max_workers=4) as pool:
        ...     results = pool.map(io_function, data)
    """
    workers = get_optimal_workers(task_type)

    if max_workers is not None:
        workers = min(workers, max_workers)
        logger.debug(f"Worker count limited to {workers} (max_workers={max_workers})")

    logger.debug(f"Creating worker pool with {workers} processes")
    return mp.Pool(processes=workers)
