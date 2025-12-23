# face-morph

3D face morphing with GPU acceleration and quantitative analysis tools.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Morphing Animation](assets/demo_morph.gif)

## Features

This package provides tools for morphing 3D face meshes with texture support:

- Linear interpolation between two face meshes
- Shape and texture heatmap visualization
- GPU-accelerated batch rendering with PyTorch3D
- CSV export for quantitative analysis
- Video generation from morph sequences

## Quick Start

```bash
# Install
git clone https://github.com/costantinoai/face-morph.git
cd face-morph
conda create -n face-morph python=3.10
conda activate face-morph
pip install -e .

# Run
face-morph morph face1.fbx face2.fbx
```

Results saved to `results/YYYYMMDD_HHMMSS/face1_face2/`

## Installation

### Prerequisites

- **Python 3.9+**
- **Conda** (recommended) or pip/venv
- **Blender** - for FBX to OBJ conversion
- **FFmpeg** - for video generation (optional)

### CPU-Only Installation

Suitable for most users. Uses PyRender (OpenGL-based) for fast CPU rendering.

```bash
# 1. Clone repository
git clone https://github.com/costantinoai/face-morph.git
cd face-morph

# 2. Create environment
conda create -n face-morph python=3.10
conda activate face-morph

# 3. Install package
pip install -e .

# 4. Verify
face-morph --help
```

### GPU Installation (CUDA)

For GPU-accelerated rendering with PyTorch3D. Requires NVIDIA GPU with CUDA 12.4.

```bash
# 1. Create environment
conda create -n face-morph python=3.10
conda activate face-morph

# 2. Install PyTorch with CUDA
conda install pytorch==2.4.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# 3. Install package with CUDA extras
pip install -e .[cuda]

# 4. Install PyTorch3D
FORCE_CUDA=1 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# 5. Verify
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import pytorch3d; print('PyTorch3D OK')"
```

### Install External Tools

**Blender** (required for FBX conversion):

```bash
# Ubuntu/Debian
sudo snap install blender --classic

# macOS
brew install --cask blender

# Windows
# Download from https://www.blender.org/download/
```

**FFmpeg** (optional, for video generation):

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Usage

### Command Line

**Basic morphing:**
```bash
face-morph morph face1.fbx face2.fbx
```

**With GPU acceleration:**
```bash
face-morph morph face1.fbx face2.fbx --gpu
```

**Full output (meshes + video + CSV):**
```bash
face-morph morph face1.fbx face2.fbx --full --gpu
```

**Batch process folder:**
```bash
face-morph batch data/faces/ --full --gpu
```

**Custom output directory:**
```bash
face-morph morph face1.fbx face2.fbx -o my_results/
```

### Python API

```python
import torch
from pathlib import Path
from face_morph.pipeline import MorphConfig, run_morphing_pipeline

config = MorphConfig(
    input_mesh_1=Path("face1.fbx"),
    input_mesh_2=Path("face2.fbx"),
    output_mode="full",               # "default" or "full"
    device=torch.device('cuda'),      # or 'cpu'
    use_mixed_precision=True,         # FP16 for GPU speedup
    parallel_fbx=True,
    num_workers=7
)

output_dir = run_morphing_pipeline(config)
print(f"Results saved to: {output_dir}")
```

## Output Structure

### Default Mode

Generates 41 interpolated frames (0%, 2.5%, 5%, ..., 100%) plus heatmaps:

![Morph Progression](assets/demo_progression.png)
*Morphing progression: 0%, 25%, 50%, 75%, 100%*

```
results/YYYYMMDD_HHMMSS/face1_face2/
├── png/
│   ├── face1-000_face2-1000.png  # 0% face1, 100% face2
│   ├── face1-025_face2-975.png   # 2.5% face1, 97.5% face2
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
├── png/                           # 41 PNG frames (512×512)
├── mesh/                          # 41 FBX files
├── animation.mp4                  # 30 fps video
├── shape_displacement_components.png
├── texture_difference_components.png
├── statistics.csv                 # Summary metrics
├── vertex_displacements.csv       # Per-vertex displacement data
├── texture_differences.csv        # Per-pixel texture differences
└── session.log
```

## Heatmaps

![Heatmap Visualizations](assets/demo_heatmaps.png)
*Top: Shape displacement components. Bottom: Texture difference components.*

### Shape Displacement

Three components showing geometric changes between input meshes:

1. **Normal displacement** - Depth changes perpendicular to surface
   - Red: outward movement (fuller features)
   - Blue: inward movement (thinner features)
   - White: no depth change

2. **Tangent displacement** - Lateral sliding parallel to surface
   - Hot colors (red/yellow): high movement
   - Cool colors (blue): low movement

3. **Total displacement** - Combined 3D movement magnitude

### Texture Difference

Three components showing appearance changes:

1. **Luminance** - Brightness differences
2. **Chrominance** - Color/saturation differences
3. **Perceptual (ΔE)** - CIEDE2000 perceptual color difference

## Performance

Tested on NVIDIA GeForce RTX 3080 Laptop GPU with 18K vertex textured meshes:

| Mode | Device | Time | Notes |
|------|--------|------|-------|
| Default | CPU | ~16s | PyRender (OpenGL) |
| Default | GPU | ~18s | PyTorch3D batch rendering |
| Full | CPU | ~48s | +FBX export + video |

GPU batch rendering is active and working. For small workloads (41 frames), performance is comparable to CPU due to transfer overhead. GPU excels with larger batches or full mode processing.

**Optimizations:**
- Device transfers: 98.8% reduction (2 vs 164 unoptimized)
- Parallel heatmap generation (ThreadPoolExecutor)
- Batch rendering with chunking (10 meshes/batch)
- Optional mixed precision (FP16)
- Zero memory leaks (verified)

## CLI Reference

```
face-morph morph INPUT1 INPUT2 [OPTIONS]

Arguments:
  INPUT1, INPUT2        Input mesh files (.fbx or .obj)

Options:
  -o, --output PATH     Output directory (default: results/)
  --full                Full mode: meshes + video + CSV
  --gpu / --cpu         Device (default: cpu)
  --no-amp              Disable mixed precision (FP16)
  --log-level LEVEL     DEBUG|INFO|WARNING|ERROR
  --blender PATH        Blender executable path
  -q, --quiet           Suppress output
  --help                Show help
```

```
face-morph batch FOLDER [OPTIONS]

Process all unique pairs in FOLDER.

Discovers all .fbx/.obj files and creates unique combinations.
Example: 4 faces → 6 pairs (1+2, 1+3, 1+4, 2+3, 2+4, 3+4)
```

## Troubleshooting

**"CUDA not available"**

Check GPU and drivers:
```bash
nvidia-smi
```

Reinstall PyTorch with CUDA:
```bash
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

**"Topology mismatch"**

Meshes must have identical vertex counts. Use remeshing tools to create matching topology before morphing.

**"Out of memory" (GPU)**

Use CPU mode or disable mixed precision:
```bash
face-morph morph face1.fbx face2.fbx --cpu
face-morph morph face1.fbx face2.fbx --gpu --no-amp
```

**"Blender not found"**

Install Blender or specify path:
```bash
face-morph morph face1.fbx face2.fbx --blender /path/to/blender
```

**"No textures found"**

Ensure texture files exist alongside OBJ/MTL files, or use FBX format which embeds textures.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch3D (for GPU)
- pyrender (for CPU)
- trimesh, pillow, matplotlib, opencv-python
- Blender (external)
- FFmpeg (external, optional)

See `pyproject.toml` for complete dependency list.

## Project Structure

```
face-morph/
├── src/face_morph/
│   ├── cli/              # Command-line interface
│   ├── core/             # Mesh I/O, morphing
│   ├── rendering/        # PyTorch3D & PyRender
│   ├── visualization/    # Heatmaps, video, CSV
│   ├── pipeline/         # Orchestrator
│   └── utils/            # Logging, device management
├── data/                 # Input meshes (user-provided)
├── results/              # Output directory
├── pyproject.toml
└── README.md
```

## Roadmap

Future enhancements planned:

- **Camera and Lighting Controls** - Expose camera position and light intensity with intuitive semantics:
  - Camera position in degrees on horizontal plane (0° = facing forward, ±90° = sideways, 180° = facing away)
  - Light level from 0 (off) to 100 (maximum intensity)

## License

MIT License - see [LICENSE](LICENSE)

## Citation

If you use this software in your research, please cite appropriately.
