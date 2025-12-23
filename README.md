# face-morph

3D face morphing with GPU acceleration and quantitative analysis.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Morphing Animation](assets/demo_morph.gif)

## Features

- 🎨 Linear interpolation between 3D face meshes with texture support
- 📊 Shape and texture displacement heatmaps (normal, tangent, total components)
- ⚡ GPU-accelerated batch rendering with PyTorch3D (2.1x faster than CPU)
- 📈 CSV export for quantitative analysis
- 🎬 MP4 video generation from morph sequences
- 🚀 Optimized pipeline: GPU computation, result caching, batch processing

## Quick Start

```bash
# Clone and install
git clone https://github.com/costantinoai/face-morph.git
cd face-morph
conda env create -f environment.yml
conda activate face-morph
pip install -e .

# Run (GPU mode)
face-morph morph face1.fbx face2.fbx --gpu

# Batch process folder (recommended)
face-morph batch data/ --gpu --minimal
```

Results saved to `results/YYYYMMDD_HHMMSS/`

## Installation

### Prerequisites

- **Python 3.9-3.12** (3.10 recommended)
- **Conda** ([Miniforge](https://github.com/conda-forge/miniforge) or Anaconda)
- **Blender 3.6+** (for FBX conversion, auto-detected)
- **FFmpeg** (optional, for video generation)

### CPU-Only (Cross-Platform)

```bash
git clone https://github.com/costantinoai/face-morph.git
cd face-morph

# Create environment
conda create -n face-morph python=3.10
conda activate face-morph

# Install dependencies
conda install -c conda-forge numpy scipy pillow matplotlib scikit-image trimesh tqdm click
conda install pytorch torchvision cpuonly -c pytorch
pip install pyrender
pip install -e .
```

### GPU (Linux/Windows with CUDA)

```bash
git clone https://github.com/costantinoai/face-morph.git
cd face-morph

# Use pre-configured environment
conda env create -f environment.yml
conda activate face-morph
pip install -e .

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Note:** GPU mode requires CUDA-compatible NVIDIA GPU. macOS does not support CUDA.

### Docker (Recommended for Production)

```bash
# CPU mode
docker run -v $(pwd)/data:/data -v $(pwd)/results:/results \
  ghcr.io/costantinoai/face-morph:cpu \
  face-morph batch /data --cpu -o /results

# GPU mode (requires nvidia-docker)
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/results:/results \
  ghcr.io/costantinoai/face-morph:gpu \
  face-morph batch /data --gpu -o /results
```

See [DOCKER.md](DOCKER.md) for detailed Docker setup.

## Usage

### CLI

```bash
# Morph two faces
face-morph morph face1.fbx face2.fbx --gpu

# Minimal mode (faster, PNG + heatmaps only)
face-morph morph face1.fbx face2.fbx --gpu --minimal

# Batch process folder (recommended for multiple pairs)
face-morph batch data/ --gpu --minimal -o results/
```

**Options:**
- `--gpu` / `--cpu`: Device selection (default: CPU)
- `--minimal`: Fast mode - PNG frames + heatmaps only
- `--no-amp`: Disable mixed precision (FP16)
- `--log-level`: DEBUG|INFO|WARNING|ERROR
- `-o, --output`: Output directory (default: `results/`)

### Python API

```python
from face_morph import MorphConfig, run_morphing_pipeline
from pathlib import Path
import torch

config = MorphConfig(
    input_mesh_1=Path("face1.fbx"),
    input_mesh_2=Path("face2.fbx"),
    output_dir=Path("results"),
    output_mode="minimal",  # or "full"
    device=torch.device("cuda"),  # or "cpu"
)

output_path = run_morphing_pipeline(config)
print(f"Results: {output_path}")
```

## Output Structure

### Minimal Mode (Faster, Recommended)

```
results/YYYYMMDD_HHMMSS/face1_face2/
├── png/                                    # Morph frames (41 PNG images)
├── shape_displacement_components.png       # Normal + Tangent + Total heatmaps
├── texture_difference_components.png       # Luminance + Chroma + ΔE heatmaps
└── session.log
```

### Full Mode (Comprehensive)

Includes everything above **plus:**
```
├── mesh/                                   # 41 OBJ + FBX mesh files
├── animation.mp4                           # MP4 video (30 FPS)
├── statistics.csv                          # Summary metrics
├── vertex_displacements.csv                # Per-vertex data
└── texture_differences.csv                 # Per-pixel texture data
```

## Performance

Real-world benchmarks (NVIDIA RTX 3080 Laptop, 18K vertex meshes, 41 frames per pair):

### Batch Processing (6 Pairs)

| Configuration | Total Time | Per-Pair | Throughput | Speedup |
|---------------|------------|----------|------------|---------|
| **GPU + Minimal** ⭐ | **29.4s** | **4.8s** | 0.20 pairs/s | **2.1x** |
| GPU + Full | 247.8s | 41.1s | 0.02 pairs/s | 0.25x |
| CPU + Minimal | 61.8s | 10.2s | 0.10 pairs/s | 1.0x |
| CPU + Full | 277.3s | 46.2s | 0.02 pairs/s | 0.22x |

**Key Findings:**
- GPU is **2.1x faster** than CPU (minimal mode)
- Full mode adds **742% overhead** (FBX conversion + video encoding)
- **Recommended:** GPU + Minimal for production workflows

**When to Use Each Mode:**

| Use Case | Recommended Config | Reason |
|----------|-------------------|--------|
| Production batch processing | GPU + Minimal | Fastest (4.8s/pair) |
| Exploratory analysis | CPU + Minimal | No GPU needed |
| Publication (need meshes/videos) | GPU + Full | Comprehensive output |
| Single pair | CPU + Minimal | GPU overhead not worth it |

*Benchmarked: 2025-12-23 | Version: v1.0.0 + GPU optimizations*

## Heatmaps

### Shape Displacement

Three orthogonal components:
- **Normal:** Depth changes (expansion/contraction perpendicular to surface)
- **Tangential:** Positional changes (movement parallel to surface)
- **Total:** Complete 3D displacement magnitude

### Texture Difference

Three perceptual components:
- **Luminance:** Brightness differences
- **Chrominance:** Color/saturation differences
- **ΔE (CIEDE2000):** Perceptual color difference (industry standard)

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# With coverage
pytest --cov=face_morph tests/
```

### Profiling

```bash
# Comprehensive profiling
python scripts/profile_comprehensive.py --mode gpu --frames 41

# Real-world CLI testing
python scripts/profile_cli_batch.py
```

See profiling documentation in `PROFILING_SUMMARY.md`.

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce batch size: Add `--chunk-size 5` flag
- Use CPU mode: `--cpu` instead of `--gpu`
- Close other GPU applications

**"Blender not found"**
- Install Blender 3.6+ from [blender.org](https://www.blender.org/download/)
- Specify path: `--blender /path/to/blender`
- Auto-detection checks: `blender`, `/usr/bin/blender`, `/Applications/Blender.app/Contents/MacOS/Blender`

**"ModuleNotFoundError: pytorch3d"**
- GPU mode only: Install PyTorch3D from source (see `environment.yml`)
- Or use CPU mode: `--cpu` (uses PyRender instead)

**Slow FBX conversion**
- Expected: Blender conversion takes ~1-2s per mesh
- Use `--minimal` to skip FBX export of morphed meshes
- Pre-convert to OBJ format if possible

### Platform-Specific Notes

**Windows:**
- PyTorch3D requires Visual Studio Build Tools (GPU mode only)
- Use forward slashes in paths or raw strings: `r"C:\path\to\file.fbx"`
- Run PowerShell as Administrator for conda commands

**macOS:**
- CUDA not supported (Apple dropped NVIDIA support in 2018)
- CPU mode only
- Install Blender from DMG or via `brew install --cask blender`

**Linux:**
- Best platform for GPU acceleration
- Ensure NVIDIA drivers + CUDA toolkit installed
- Check CUDA version: `nvcc --version`

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{face_morph_2024,
  author = {Costantino, Andrea Ivan},
  title = {face-morph: 3D Face Morphing with GPU Acceleration},
  year = {2024},
  url = {https://github.com/costantinoai/face-morph},
  version = {1.0.0}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for GPU-accelerated 3D rendering
- [PyRender](https://github.com/mmatl/pyrender) for CPU rendering backend
- [Blender](https://www.blender.org/) for FBX/OBJ conversion

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed changes.

---

**Status:** Production-ready | **Version:** 1.0.0 | **Last Updated:** 2025-12-23
