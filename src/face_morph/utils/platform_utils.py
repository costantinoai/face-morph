"""
Platform-Specific Utilities
============================

Cross-platform compatibility helpers for Windows, macOS, and Linux.
"""

import platform
import shutil
from pathlib import Path
from typing import Optional, Tuple

from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


def get_platform_info() -> Tuple[str, str, str]:
    """
    Get platform information.

    Returns:
        Tuple of (system, release, machine)
        - system: 'Windows', 'Darwin' (macOS), or 'Linux'
        - release: OS version
        - machine: CPU architecture (x86_64, arm64, etc.)

    Example:
        >>> system, release, machine = get_platform_info()
        >>> system in ['Windows', 'Darwin', 'Linux']
        True
    """
    return platform.system(), platform.release(), platform.machine()


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system() == 'Windows'


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == 'Darwin'


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == 'Linux'


def supports_cuda() -> bool:
    """
    Check if platform supports CUDA.

    macOS does not support CUDA (Apple dropped NVIDIA support).

    Returns:
        True if platform can support CUDA (Windows/Linux)
    """
    return not is_macos()


def get_blender_executable() -> Optional[str]:
    """
    Auto-detect Blender executable path for current platform.

    Checks common installation locations for each platform:
    - Linux: blender (in PATH), snap installation
    - macOS: /Applications/Blender.app/Contents/MacOS/Blender
    - Windows: Program Files, common version directories

    Returns:
        Path to Blender executable or 'blender' fallback

    Example:
        >>> blender_path = get_blender_executable()
        >>> blender_path is not None
        True
    """
    # First check if 'blender' is in PATH (works on all platforms)
    if shutil.which('blender'):
        logger.debug("Found Blender in PATH")
        return 'blender'

    system = platform.system()

    if system == 'Darwin':  # macOS
        # Check standard macOS application location
        macos_paths = [
            '/Applications/Blender.app/Contents/MacOS/Blender',
            Path.home() / 'Applications/Blender.app/Contents/MacOS/Blender',
        ]
        for path in macos_paths:
            if Path(path).exists():
                logger.debug(f"Found Blender at: {path}")
                return str(path)

    elif system == 'Windows':
        # Check common Windows installation paths
        # Try multiple versions (newest first)
        base_paths = [
            Path('C:/Program Files/Blender Foundation'),
            Path('C:/Program Files (x86)/Blender Foundation'),
        ]

        for base in base_paths:
            if not base.exists():
                continue

            # Find Blender directories (e.g., "Blender 4.0", "Blender 3.6")
            blender_dirs = sorted(base.glob('Blender*'), reverse=True)
            for blender_dir in blender_dirs:
                blender_exe = blender_dir / 'blender.exe'
                if blender_exe.exists():
                    logger.debug(f"Found Blender at: {blender_exe}")
                    return str(blender_exe)

    elif system == 'Linux':
        # Check snap installation
        snap_path = '/snap/bin/blender'
        if Path(snap_path).exists():
            logger.debug(f"Found Blender at: {snap_path}")
            return snap_path

    # Fallback: assume 'blender' is in PATH
    logger.debug("Blender not found in standard locations, using 'blender' (must be in PATH)")
    return 'blender'


def get_ffmpeg_executable() -> Optional[str]:
    """
    Auto-detect ffmpeg executable path for current platform.

    Returns:
        Path to ffmpeg executable or None if not found

    Example:
        >>> ffmpeg_path = get_ffmpeg_executable()
        >>> ffmpeg_path is not None  # If ffmpeg installed
        True
    """
    # Check if ffmpeg is in PATH
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        logger.debug(f"Found ffmpeg at: {ffmpeg}")
        return ffmpeg

    logger.debug("ffmpeg not found in PATH")
    return None


def get_ffmpeg_install_instructions() -> str:
    """
    Get platform-specific ffmpeg installation instructions.

    Returns:
        Installation command/instructions for current platform

    Example:
        >>> instructions = get_ffmpeg_install_instructions()
        >>> 'install' in instructions.lower()
        True
    """
    system = platform.system()

    if system == 'Linux':
        return (
            "Install ffmpeg:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  Fedora/RHEL:   sudo dnf install ffmpeg\n"
            "  Arch:          sudo pacman -S ffmpeg"
        )
    elif system == 'Darwin':  # macOS
        return (
            "Install ffmpeg:\n"
            "  Homebrew:      brew install ffmpeg\n"
            "  MacPorts:      sudo port install ffmpeg"
        )
    elif system == 'Windows':
        return (
            "Install ffmpeg:\n"
            "  Chocolatey:    choco install ffmpeg\n"
            "  Scoop:         scoop install ffmpeg\n"
            "  Manual:        Download from https://ffmpeg.org/download.html\n"
            "                 and add to PATH"
        )
    else:
        return "Install ffmpeg and ensure it's in your PATH"


def get_blender_install_instructions() -> str:
    """
    Get platform-specific Blender installation instructions.

    Returns:
        Installation instructions for current platform

    Example:
        >>> instructions = get_blender_install_instructions()
        >>> 'Blender' in instructions
        True
    """
    system = platform.system()

    if system == 'Linux':
        return (
            "Install Blender:\n"
            "  Snap:          sudo snap install blender --classic\n"
            "  apt:           sudo apt install blender\n"
            "  Manual:        Download from https://www.blender.org/download/"
        )
    elif system == 'Darwin':  # macOS
        return (
            "Install Blender:\n"
            "  Homebrew:      brew install --cask blender\n"
            "  Manual:        Download from https://www.blender.org/download/\n"
            "                 and drag to Applications folder"
        )
    elif system == 'Windows':
        return (
            "Install Blender:\n"
            "  Microsoft Store: Search for 'Blender'\n"
            "  Chocolatey:      choco install blender\n"
            "  Manual:          Download installer from https://www.blender.org/download/"
        )
    else:
        return "Download Blender from https://www.blender.org/download/"


def check_opengl_support() -> Tuple[bool, Optional[str]]:
    """
    Check if OpenGL is supported and warn about macOS deprecation.

    Returns:
        Tuple of (supported, warning_message)
        - supported: Always True (OpenGL still works)
        - warning_message: Deprecation warning for macOS, None otherwise

    Example:
        >>> supported, warning = check_opengl_support()
        >>> supported
        True
    """
    if is_macos():
        warning = (
            "macOS has deprecated OpenGL (replaced by Metal).\n"
            "PyRender rendering may show warnings and could break in future macOS versions.\n"
            "For now, it still works but consider using CPU mode for stability."
        )
        return True, warning

    return True, None


def get_pytorch3d_install_instructions() -> str:
    """
    Get platform-specific PyTorch3D installation instructions.

    Returns:
        Installation instructions for current platform with warnings

    Example:
        >>> instructions = get_pytorch3d_install_instructions()
        >>> 'PyTorch3D' in instructions
        True
    """
    system = platform.system()

    if system == 'Linux':
        return (
            "PyTorch3D installation (Linux with CUDA):\n\n"
            "1. Install PyTorch with CUDA:\n"
            "   conda install pytorch==2.4.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia\n\n"
            "2. Install PyTorch3D:\n"
            "   FORCE_CUDA=1 pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'\n\n"
            "3. Verify installation:\n"
            "   python -c 'import pytorch3d; print(\"PyTorch3D OK\")'"
        )
    elif system == 'Darwin':  # macOS
        return (
            "PyTorch3D installation (macOS - CPU ONLY):\n\n"
            "⚠️  WARNING: macOS does NOT support CUDA (Apple dropped NVIDIA support).\n"
            "    You can only use CPU mode on macOS.\n\n"
            "1. Install PyTorch (CPU version):\n"
            "   conda install pytorch==2.4.1 torchvision -c pytorch\n\n"
            "2. Install PyTorch3D (CPU version):\n"
            "   pip install 'git+https://github.com/facebookresearch/pytorch3d.git'\n\n"
            "3. Verify installation:\n"
            "   python -c 'import pytorch3d; print(\"PyTorch3D OK\")'\n\n"
            "Note: Use --cpu flag when running face-morph commands."
        )
    elif system == 'Windows':
        return (
            "PyTorch3D installation (Windows - ADVANCED):\n\n"
            "⚠️  WARNING: PyTorch3D installation on Windows is complex and requires:\n"
            "    - Visual Studio 2019/2022 with C++ tools\n"
            "    - CUDA Toolkit matching your PyTorch version\n"
            "    - Several GB of build tools\n\n"
            "Recommended for beginners: Use CPU-only mode (skip PyTorch3D).\n\n"
            "For advanced users with CUDA:\n"
            "1. Install Visual Studio Build Tools:\n"
            "   https://visualstudio.microsoft.com/downloads/\n"
            "   (Select 'Desktop development with C++')\n\n"
            "2. Install PyTorch with CUDA:\n"
            "   conda install pytorch==2.4.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia\n\n"
            "3. Install PyTorch3D (requires compilation):\n"
            "   pip install 'git+https://github.com/facebookresearch/pytorch3d.git'\n\n"
            "Alternative: Check for pre-built wheels:\n"
            "   https://github.com/facebookresearch/pytorch3d/issues\n\n"
            "For CPU-only (easier):\n"
            "   conda install pytorch==2.4.1 torchvision -c pytorch\n"
            "   (Skip PyTorch3D installation, use --cpu flag)"
        )
    else:
        return "See https://github.com/facebookresearch/pytorch3d for installation instructions"


def validate_cuda_platform() -> Tuple[bool, Optional[str]]:
    """
    Validate if CUDA can be used on current platform.

    Returns:
        Tuple of (can_use_cuda, error_message)
        - can_use_cuda: True if platform supports CUDA
        - error_message: Explanation if CUDA not supported

    Example:
        >>> can_use, msg = validate_cuda_platform()
        >>> isinstance(can_use, bool)
        True
    """
    if is_macos():
        error_msg = (
            "CUDA is not supported on macOS.\n\n"
            "Apple dropped NVIDIA GPU support in macOS 10.14+ (2018).\n"
            "macOS only supports Metal (Apple's GPU API), not CUDA.\n\n"
            "Please use CPU mode instead:\n"
            "  face-morph morph input1.fbx input2.fbx --cpu\n\n"
            "CPU mode uses PyRender with OpenGL acceleration and is still reasonably fast."
        )
        return False, error_msg

    return True, None
