"""
Video Creation Utilities
=========================

Single responsibility: Create videos from image frames.
"""

import subprocess
from pathlib import Path
from typing import Optional

from face_morph.utils.logging import get_logger

# Get logger for this module
logger = get_logger(__name__)


def create_video_from_frames(
    frame_dir: Path,
    output_video: Path,
    fps: int = 30,
    pattern: str = "*.png"
) -> bool:
    """
    Create MP4 video from PNG frames using ffmpeg.

    Single responsibility: Video encoding.

    Args:
        frame_dir: Directory containing frame images
        output_video: Output video path
        fps: Frames per second
        pattern: Glob pattern for frames

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if ffmpeg is available
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )

        if result.returncode != 0:
            return False

        # Create video using ffmpeg
        # Sort frames naturally by filename
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', str(frame_dir / pattern),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',  # Compatibility
            '-crf', '23',  # Quality (lower = better)
            '-preset', 'medium',
            str(output_video)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=300
        )

        return result.returncode == 0 and output_video.exists()

    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_ffmpeg_available() -> bool:
    """
    Check if ffmpeg is installed and available.

    Returns:
        True if ffmpeg is available
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
