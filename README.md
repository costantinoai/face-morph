# face-morph

3D face morphing with GPU acceleration and quantitative analysis.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Morphing Animation](assets/demo_morph.gif)

![Heatmap Analysis](assets/demo_heatmaps.png)

### Shape Displacement Components (Top Row)

| Component | Description | Colormap |
|-----------|-------------|----------|
| **Normal** | Depth changes perpendicular to the surface. Red = expansion outward, Blue = contraction inward. Useful for detecting facial features that protrude or recede. | Diverging (red-blue) |
| **Tangential** | Movement parallel to the surface (sliding). Shows how vertices shift along the face without changing depth. | Sequential (purple-yellow) |
| **Total** | Combined 3D displacement magnitude (√(normal² + tangential²)). Overall measure of geometric difference. | Sequential (purple-yellow) |

### Texture Difference Components (Bottom Row)

| Component | Description | Colormap |
|-----------|-------------|----------|
| **Luminance** | Brightness differences between textures. Red = brighter in face 2, Blue = darker in face 2. | Diverging (red-blue) |
| **Chrominance** | Color/saturation differences independent of brightness. Highlights hue and saturation changes. | Sequential (purple-yellow) |
| **ΔE (Total)** | CIEDE2000 perceptual color difference (industry standard). Human-calibrated metric where ΔE > 2 is noticeable. | Sequential (purple-yellow) |

## Quick Start

### Docker CPU (Recommended)

Easiest way to get started. No CUDA setup required, works on any machine.

**Option A: Pull pre-built image (fastest)**
```bash
docker pull ghcr.io/costantinoai/face-morph:cpu
```

**Option B: Build locally**
```bash
git clone https://github.com/costantinoai/face-morph.git
cd face-morph
docker build -f Dockerfile.cpu -t face-morph:cpu .
```

**Run face morphing**

Single pair:
```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  ghcr.io/costantinoai/face-morph:cpu \
  morph /workspace/data/face1.fbx /workspace/data/face2.fbx --cpu --minimal
```

Batch processing:
```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  ghcr.io/costantinoai/face-morph:cpu \
  batch /workspace/data --cpu --minimal
```

Results saved to `results/YYYYMMDD_HHMMSS/`

**Common Options:**
- `--cpu` / `--gpu` - Device selection (use `--cpu` for Docker CPU image)
- `--minimal` - Fast mode, PNG + heatmaps only (recommended)
- `-o, --output DIR` - Output directory (default: `results/`)
- `--log-level LEVEL` - DEBUG|INFO|WARNING|ERROR (default: INFO)
- `--blender PATH` - Path to Blender executable (default: auto-detect)

### Docker GPU (Large Batches)

For processing 100+ pairs, the GPU version provides significant speedup.

> **Note:** The GPU image is ~10GB due to CUDA libraries. Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host (not just CUDA drivers).

```bash
# Pull GPU image
docker pull ghcr.io/costantinoai/face-morph:gpu

# Run with GPU
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  ghcr.io/costantinoai/face-morph:gpu \
  batch /workspace/data --gpu --minimal
```

See [DOCKER.md](DOCKER.md) for detailed GPU setup instructions.

**Bare Metal Installation:** For native installation without Docker, see [INSTALLATION.md](INSTALLATION.md).

## Features

- Linear interpolation between 3D face meshes with texture support
- Shape and texture displacement heatmaps (normal, tangent, total components)
- GPU-accelerated batch rendering with PyTorch3D
- CSV export for quantitative analysis
- MP4 video generation from morph sequences

## Usage

For bare metal installation (non-Docker), see [INSTALLATION.md](INSTALLATION.md).

### CLI (Bare Metal)

For Docker usage, see [Quick Start](#quick-start) section above.

```bash
# Morph two faces
face-morph morph face1.fbx face2.fbx --gpu

# Minimal mode (faster, PNG + heatmaps only)
face-morph morph face1.fbx face2.fbx --gpu --minimal

# Batch process folder (recommended for multiple pairs)
face-morph batch data/ --gpu --minimal
```

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

## Troubleshooting

### Docker Issues

**Permission denied writing to results:**
```bash
chmod 777 results/
# Or run as your user
docker run --user $(id -u):$(id -g) ...
```

**GPU not detected (Docker GPU mode):**
- Verify GPU on host: `nvidia-smi`
- Verify NVIDIA runtime is installed
- See [DOCKER.md](DOCKER.md) for GPU setup instructions

### Bare Metal Issues

**"CUDA out of memory"**
- Reduce batch size: Add `--chunk-size 5` flag
- Use CPU mode: `--cpu` instead of `--gpu`
- Close other GPU applications

**"Blender not found"**
- Install Blender 3.6+ from [blender.org](https://www.blender.org/download/)
- Specify path: `--blender /path/to/blender`

**"ModuleNotFoundError: pytorch3d"**
- Install PyTorch3D from source (see [INSTALLATION.md](INSTALLATION.md))
- Or use CPU mode: `--cpu` (uses PyRender instead)

**Platform-specific issues:** See [INSTALLATION.md](INSTALLATION.md) for Windows, macOS, and Linux troubleshooting.

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

## Documentation

- [INSTALLATION.md](INSTALLATION.md) - Bare metal installation on Windows, macOS, Linux
- [DOCKER.md](DOCKER.md) - Docker deployment and GPU setup

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed changes.

**Development Setup:** See [INSTALLATION.md](INSTALLATION.md) for setting up a development environment.
