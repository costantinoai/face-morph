"""
Pipeline Configuration
======================

Single responsibility: Configure morphing pipeline with validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
from multiprocessing import cpu_count
import torch

from face_morph.core.exceptions import ValidationError


def generate_video_ratios(step: int = 25) -> List[Tuple[float, float]]:
    """
    Generate fine-grained ratios for video frames.

    Creates ratios from 0-1000 to 1000-0 in specified steps.
    This generates a smooth morph sequence for video animation.

    Args:
        step: Step size in permille (default: 25)
              - step=25 generates 41 frames (0, 25, 50, ..., 1000)
              - step=50 generates 21 frames (faster processing)
              - step=10 generates 101 frames (smoother video)

    Returns:
        List of (ratio1, ratio2) tuples where ratio1 + ratio2 = 1.0

    Example:
        >>> ratios = generate_video_ratios(step=25)
        >>> len(ratios)
        41
        >>> ratios[0]
        (0.0, 1.0)
        >>> ratios[-1]
        (1.0, 0.0)
        >>> ratios[20]  # Middle frame
        (0.5, 0.5)
    """
    ratios = []
    for r1_permille in range(0, 1001, step):
        r2_permille = 1000 - r1_permille
        ratios.append((r1_permille / 1000.0, r2_permille / 1000.0))
    return ratios


@dataclass
class MorphConfig:
    """
    Configuration for face morphing pipeline.

    This dataclass encapsulates all settings needed for morphing,
    with validation in __post_init__ to catch errors early.

    Attributes:
        input_mesh_1: Path to first input mesh (FBX or OBJ)
        input_mesh_2: Path to second input mesh (FBX or OBJ)
        output_dir: Base output directory (default: 'results')
        output_mode: Output mode - 'default' (PNG + heatmaps) or 'full' (+ meshes + video)
        morph_ratios: List of (ratio1, ratio2) tuples for interpolation
        device: PyTorch device (cpu or cuda)
        use_mixed_precision: Enable FP16 mixed precision (GPU only)
        blender_path: Path to Blender executable for FBX conversion
        ffmpeg_path: Path to ffmpeg for video creation
        verbose: Enable verbose output
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        video_fps: Frames per second for video output
        parallel_fbx: Enable parallel FBX conversion
        num_workers: Number of parallel workers for I/O tasks
        chunk_size: Batch size for GPU rendering (meshes per batch, default: 10)
        timestamp: Optional shared timestamp for batch mode

    Example:
        >>> config = MorphConfig(
        ...     input_mesh_1=Path("face1.fbx"),
        ...     input_mesh_2=Path("face2.fbx"),
        ...     output_mode="full",
        ...     device=torch.device('cuda')
        ... )
        >>> config.should_export_meshes
        True
        >>> config.should_create_video
        True
    """

    # Input/Output
    input_mesh_1: Path
    input_mesh_2: Path
    output_dir: Path = Path("results")

    # Output control (NEW - Week 2 feature)
    output_mode: str = "default"  # "default" or "full"

    # Morphing parameters
    morph_ratios: List[Tuple[float, float]] = field(
        default_factory=lambda: generate_video_ratios(step=25)
    )

    # Hardware
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    use_mixed_precision: bool = True

    # External tools
    blender_path: str = "blender"
    ffmpeg_path: str = "ffmpeg"

    # Logging
    verbose: bool = True
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

    # Video settings
    video_fps: int = 30

    # Performance
    parallel_fbx: bool = True
    num_workers: int = field(default_factory=lambda: max(1, cpu_count() - 1))
    chunk_size: int = 10  # Batch size for GPU rendering (number of meshes per batch)

    # Batch mode support
    timestamp: Optional[str] = None

    def __post_init__(self):
        """
        Validate and normalize configuration after initialization.

        This method runs automatically after __init__ to ensure
        all settings are valid before pipeline execution.

        Raises:
            ValidationError: If any configuration is invalid
        """
        # Convert string paths to Path objects
        self.input_mesh_1 = Path(self.input_mesh_1)
        self.input_mesh_2 = Path(self.input_mesh_2)
        self.output_dir = Path(self.output_dir)

        # Validate input files exist
        if not self.input_mesh_1.exists():
            raise ValidationError(f"Input mesh 1 not found: {self.input_mesh_1}")
        if not self.input_mesh_2.exists():
            raise ValidationError(f"Input mesh 2 not found: {self.input_mesh_2}")

        # Validate file extensions
        valid_exts = {'.fbx', '.obj'}
        if self.input_mesh_1.suffix.lower() not in valid_exts:
            raise ValidationError(
                f"Invalid file format for mesh 1: {self.input_mesh_1.suffix}\n"
                f"Supported formats: {valid_exts}"
            )
        if self.input_mesh_2.suffix.lower() not in valid_exts:
            raise ValidationError(
                f"Invalid file format for mesh 2: {self.input_mesh_2.suffix}\n"
                f"Supported formats: {valid_exts}"
            )

        # Validate output mode
        if self.output_mode not in ('default', 'full'):
            raise ValidationError(
                f"output_mode must be 'default' or 'full', got '{self.output_mode}'"
            )

        # Disable AMP on CPU (not supported)
        if self.device.type == 'cpu':
            self.use_mixed_precision = False

        # Validate morph ratios
        if not self.morph_ratios:
            raise ValidationError("morph_ratios cannot be empty")

        for idx, (r1, r2) in enumerate(self.morph_ratios):
            if not (0.0 <= r1 <= 1.0 and 0.0 <= r2 <= 1.0):
                raise ValidationError(
                    f"Invalid ratio at index {idx}: ({r1}, {r2})\n"
                    f"Ratios must be in range [0.0, 1.0]"
                )

        # Validate log level
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR'}
        if self.log_level.upper() not in valid_levels:
            raise ValidationError(
                f"Invalid log_level: {self.log_level}\n"
                f"Must be one of: {valid_levels}"
            )
        self.log_level = self.log_level.upper()

        # Validate workers count
        if self.num_workers < 1:
            raise ValidationError(f"num_workers must be >= 1, got {self.num_workers}")

    @property
    def should_export_meshes(self) -> bool:
        """
        Whether to export OBJ/FBX meshes.

        In default mode, meshes are not exported to save time.
        In full mode, all 41 meshes are exported as OBJ and FBX.

        Returns:
            True if output_mode is 'full'
        """
        return self.output_mode == "full"

    @property
    def should_create_video(self) -> bool:
        """
        Whether to create MP4 video animation.

        In default mode, video is not created to save time.
        In full mode, video is created from PNG frames.

        Returns:
            True if output_mode is 'full'
        """
        return self.output_mode == "full"

    @property
    def should_export_csv(self) -> bool:
        """
        Whether to export CSV statistics and vertex data.

        In default mode, CSV files are not exported.
        In full mode, generates:
        - statistics.csv: Summary metrics for all displacement components
        - vertex_displacements.csv: Per-vertex displacement data
        - texture_differences.csv: Per-pixel texture metrics (if textures available)

        Returns:
            True if output_mode is 'full'
        """
        return self.output_mode == "full"

    def __repr__(self) -> str:
        """
        Create readable string representation of config.

        Returns:
            Formatted configuration summary
        """
        return (
            f"MorphConfig(\n"
            f"  input_1={self.input_mesh_1.name},\n"
            f"  input_2={self.input_mesh_2.name},\n"
            f"  output_mode={self.output_mode},\n"
            f"  frames={len(self.morph_ratios)},\n"
            f"  device={self.device},\n"
            f"  amp={self.use_mixed_precision}\n"
            f")"
        )
