"""High-level pipeline orchestration.

Contains configuration, workflow orchestration, and parallel worker functions.
"""

from .config import MorphConfig, generate_video_ratios
from .orchestrator import run_morphing_pipeline

__all__ = [
    'MorphConfig',
    'generate_video_ratios',
    'run_morphing_pipeline',
]
