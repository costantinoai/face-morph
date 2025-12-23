"""
Video Creation Utilities
=========================

Single responsibility: Create videos from image frames.
"""

import subprocess
from pathlib import Path
from typing import Optional

from face_morph.utils.logging import get_logger
from face_morph.utils.platform_utils import (
    get_ffmpeg_executable,
    get_ffmpeg_install_instructions,
)

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
    # Get ffmpeg executable (auto-detect)
    ffmpeg_path = get_ffmpeg_executable()

    if ffmpeg_path is None:
        logger.warning("ffmpeg not found - cannot create video")
        return False

    try:
        # Verify ffmpeg is working
        result = subprocess.run(
            [ffmpeg_path, '-version'],
            capture_output=True,
            timeout=5
        )

        if result.returncode != 0:
            logger.warning(f"ffmpeg not working: {ffmpeg_path}")
            return False

        # Create video using ffmpeg
        # Sort frames naturally by filename
        cmd = [
            ffmpeg_path,
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

    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"ffmpeg execution failed: {e}")
        return False


def check_ffmpeg_available() -> bool:
    """
    Check if ffmpeg is installed and available.

    Logs platform-specific installation instructions if not found.

    Returns:
        True if ffmpeg is available
    """
    ffmpeg_path = get_ffmpeg_executable()

    if ffmpeg_path is None:
        logger.warning("ffmpeg not found - video creation will be skipped")
        logger.info("To enable video creation:")
        logger.info(get_ffmpeg_install_instructions())
        return False

    try:
        result = subprocess.run(
            [ffmpeg_path, '-version'],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.debug(f"ffmpeg found at: {ffmpeg_path}")
            return True
        else:
            logger.warning(f"ffmpeg found but not working: {ffmpeg_path}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("ffmpeg check failed")
        return False
