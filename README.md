# face-morph

3D face morphing with GPU acceleration and quantitative analysis.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Morphing Animation](assets/demo_morph.gif)

## Features

- Linear interpolation between 3D face meshes with texture support
- Shape and texture displacement heatmaps (normal, tangent, total components)
- GPU-accelerated batch rendering with PyTorch3D
- CSV export for quantitative analysis
- MP4 video generation from morph sequences

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

### Docker (Recommended)

#### CPU Mode (No Setup Required)

```bash
# Build locally
docker build -f Dockerfile.cpu -t face-morph:cpu .

# Run
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu \
  morph /workspace/data/face1.fbx /workspace/data/face2.fbx --cpu
```

#### GPU Mode (Requires NVIDIA Setup)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (≥418.81.07)
- Docker ≥19.03

**1. Install NVIDIA Container Toolkit** (one-time setup)

**Ubuntu/Debian:**
```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Other distributions:** See [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**2. Verify Installation**
```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed.

**3. Build and Run GPU Docker**
```bash
# Build locally
docker build -f Dockerfile.gpu -t face-morph:gpu .

# Run with NVIDIA runtime
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:gpu \
  morph /workspace/data/face1.fbx /workspace/data/face2.fbx --gpu
```

**Note:** If you get `could not select device driver "" with capabilities: [[gpu]]` error, you need to install and configure the NVIDIA Container Toolkit first (step 1 above).

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
├── mesh/                                   # 41 OBJ mesh files
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

**Mesh export overhead**
- Full mode exports 41 OBJ mesh files
- Use `--minimal` to skip mesh export for faster processing
- Minimal mode includes PNG frames and heatmaps (sufficient for most use cases)

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
@software{face_morph_2025,
  author = {Costantino, Andrea Ivan},
  title = {face-morph: 3D Face Morphing},
  year = {2025},
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
