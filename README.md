# face-morph

3D face morphing with GPU acceleration and quantitative analysis tools.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This package provides tools for morphing 3D face meshes with texture support. It includes:

- Linear interpolation between two face meshes
- Shape and texture heatmap visualization
- GPU-accelerated batch rendering with PyTorch3D
- CSV export for quantitative analysis
- Video generation from morph sequences

## Installation

### CPU-only

```bash
git clone https://github.com/costantinoai/face-morph-laura.git
cd face-morph-laura
conda create -n face-morph python=3.10
conda activate face-morph
pip install -e .
```

### GPU (CUDA)

Requires NVIDIA GPU with CUDA 12.4:

```bash
conda create -n face-morph python=3.10
conda activate face-morph
conda install pytorch==2.4.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -e .[cuda]
FORCE_CUDA=1 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

Verify:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### External Dependencies

**Blender** (for FBX to OBJ conversion):
```bash
sudo snap install blender --classic  # Ubuntu
brew install --cask blender           # macOS
```

**FFmpeg** (for video generation):
```bash
sudo apt install ffmpeg  # Ubuntu
brew install ffmpeg      # macOS
```

## Usage

### Command Line

Morph two faces:
```bash
face-morph morph face1.fbx face2.fbx
```

With GPU acceleration:
```bash
face-morph morph face1.fbx face2.fbx --gpu
```

Full output (meshes + video + CSV):
```bash
face-morph morph face1.fbx face2.fbx --full --gpu
```

Batch process folder:
```bash
face-morph batch data/faces/ --full --gpu
```

### Python API

```python
import torch
from pathlib import Path
from face_morph.pipeline import MorphConfig, run_morphing_pipeline

config = MorphConfig(
    input_mesh_1=Path("face1.fbx"),
    input_mesh_2=Path("face2.fbx"),
    output_mode="full",
    device=torch.device('cuda'),
    use_mixed_precision=True
)

output_dir = run_morphing_pipeline(config)
```

## Output

### Default Mode

Generates 41 interpolated frames (0%, 2.5%, 5%, ..., 100%) plus heatmaps:

```
results/YYYYMMDD_HHMMSS/face1_face2/
├── png/
│   ├── face1-000_face2-1000.png  # 0% face1, 100% face2
│   ├── face1-500_face2-500.png   # 50-50 blend
│   └── ...                        # 41 frames total
├── shape_displacement_components.png
├── texture_difference_components.png
└── session.log
```

### Full Mode

Adds mesh exports, video, and CSV data:

```
results/YYYYMMDD_HHMMSS/face1_face2/
├── png/                           # 41 PNG frames
├── mesh/                          # 41 FBX files
├── animation.mp4
├── shape_displacement_components.png
├── texture_difference_components.png
├── statistics.csv
├── vertex_displacements.csv
└── texture_differences.csv
```

## Heatmaps

### Shape Displacement

Three components showing geometric changes:

1. **Normal displacement** - Depth changes (red = outward, blue = inward)
2. **Tangent displacement** - Lateral sliding along surface
3. **Total displacement** - Combined 3D movement magnitude

### Texture Difference

Three components showing appearance changes:

1. **Luminance** - Brightness differences
2. **Chrominance** - Color/saturation differences
3. **Perceptual (ΔE)** - CIEDE2000 color difference

## Performance

Tested on NVIDIA GeForce RTX 3080 Laptop GPU with 18K vertex meshes:

| Mode | Device | Time | Notes |
|------|--------|------|-------|
| Default | CPU | ~16s | PyRender (OpenGL) |
| Default | GPU | ~18s | PyTorch3D batch rendering |
| Full | CPU | ~48s | Includes FBX export + video |

GPU batch rendering is active and working, but offers marginal gains for small workloads (41 frames). Expected to perform better with larger batches or full mode processing.

**Optimizations implemented:**
- Device transfers reduced by 98.8% (2 vs 164 unoptimized)
- Parallel heatmap generation
- Batch rendering with chunking (10 meshes/batch)
- Optional mixed precision (FP16)

## CLI Reference

```
face-morph morph INPUT1 INPUT2 [OPTIONS]

Options:
  -o, --output PATH     Output directory (default: results/)
  --full                Full output mode (meshes + video + CSV)
  --gpu / --cpu         Device selection (default: cpu)
  --no-amp              Disable mixed precision
  --log-level LEVEL     DEBUG|INFO|WARNING|ERROR
  --blender PATH        Blender executable path
```

```
face-morph batch FOLDER [OPTIONS]

Process all unique pairs in folder.
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch3D (for GPU rendering)
- pyrender (for CPU rendering)
- trimesh, pillow, matplotlib, opencv-python
- Blender (external, for FBX conversion)
- FFmpeg (external, for video)

## Project Structure

```
face-morph-laura/
├── src/face_morph/
│   ├── cli/              # Command-line interface
│   ├── core/             # Mesh I/O, morphing logic
│   ├── rendering/        # PyTorch3D and PyRender backends
│   ├── visualization/    # Heatmaps, video, CSV export
│   ├── pipeline/         # Main orchestrator
│   └── utils/            # Logging, device management
├── data/                 # Input meshes (user-provided)
├── results/              # Output directory
├── pyproject.toml
└── README.md
```

## Troubleshooting

**"CUDA not available"**
- Check: `nvidia-smi`
- Reinstall PyTorch with matching CUDA version

**"Topology mismatch"**
- Meshes must have identical vertex counts
- Use remeshing tools to match topology

**"Out of memory" (GPU)**
- Use `--cpu` mode
- Disable mixed precision with `--no-amp`

**"Blender not found"**
- Install Blender or specify path: `--blender /path/to/blender`

## License

MIT License - see [LICENSE](LICENSE)

## Citation

If you use this software in your research, please cite appropriately.
