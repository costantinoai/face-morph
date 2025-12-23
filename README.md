# Face Morph

**Production-ready 3D face morphing with GPU acceleration**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, modular Python package for morphing 3D face meshes with advanced heatmap visualization and quantitative analysis. Built with PyTorch3D for GPU acceleration, following modern Python packaging standards.

---

## ✨ Key Features

- 🎨 **Advanced Heatmap Visualization** - 3-component shape displacement (normal/tangent/total) and 3-component texture analysis (luminance/chroma/perceptual)
- 📊 **Quantitative Analysis** - Export CSV data for statistical analysis in Excel, R, MATLAB, or Python
- ⚡ **GPU Accelerated** - 10-20x faster morphing on GPU with mixed precision (FP16)
- 🎬 **Video Generation** - Create MP4 animations from morph sequences
- 📦 **Pip Installable** - Standard Python package with CPU and GPU support
- 🖥️ **User-Friendly CLI** - Simple commands for single pairs or batch processing
- 🔧 **Two Output Modes** - Fast mode (PNG + heatmaps) or Full mode (PNG + heatmaps + meshes + video + CSV)
- 🏭 **Production Ready** - Clean architecture, structured logging, comprehensive error handling

---

## 📋 Quick Start

### Installation

**CPU-only (recommended for most users):**
```bash
pip install -e .
```

**GPU with CUDA support:**
```bash
pip install -e .[cuda]
# Then follow prompts to install PyTorch3D with CUDA
```

See [docs/installation.md](docs/installation.md) for detailed installation instructions.

### Basic Usage

```bash
# Fast mode (PNG + heatmaps, ~5-10 min)
face-morph morph face1.fbx face2.fbx

# Full mode (PNG + heatmaps + meshes + video + CSV, ~2 hours)
face-morph morph face1.fbx face2.fbx --full --gpu

# Batch process all faces in folder
face-morph batch data/faces/ --full --gpu
```

See [docs/usage.md](docs/usage.md) for complete usage examples.

---

## 🎯 Output Modes

### Default Mode (Fast) - ~5-10 minutes

Perfect for quick visualization and iteration:

- ✅ **41 PNG images** - Rendered morph sequence (512×512)
- ✅ **6 heatmap visualizations:**
  - Shape: normal displacement, tangent displacement, total displacement
  - Texture: luminance difference, chrominance difference, perceptual difference (ΔE)
- ✅ **Session log** - Complete processing log

### Full Mode (Complete) - ~80-120 minutes

Complete research output for publication:

- ✅ **Everything from default mode**, plus:
- ✅ **41 OBJ meshes** - For external 3D analysis
- ✅ **41 FBX meshes** - Industry-standard format with embedded textures
- ✅ **MP4 animation** - 30 fps video of morph sequence
- ✅ **3 CSV files** for quantitative analysis:
  - `statistics.csv` - Summary metrics (mean, std, percentiles)
  - `vertex_displacements.csv` - Per-vertex displacement data
  - `texture_differences.csv` - Per-pixel texture metrics

---

## 📊 Heatmap Visualizations

### Shape Displacement Components

The pipeline generates **3 complementary shape heatmaps** for comprehensive analysis:

1. **Normal Displacement** (signed) - Expansion/contraction perpendicular to surface
   - Red: outward movement (fuller features)
   - Blue: inward movement (thinner features)
   - White: no depth change

2. **Tangent Displacement** (unsigned) - Sliding movement parallel to surface
   - Magnitude of lateral movement

3. **Total Displacement** (unsigned) - Overall 3D movement
   - Euclidean distance in 3D space

### Texture Difference Components

If textures are available, generates **3 texture analysis heatmaps**:

1. **Luminance Difference** (signed) - Brightness changes
2. **Chrominance Difference** (unsigned) - Color changes
3. **Perceptual Difference** (unsigned) - Industry-standard CIEDE2000 metric

See [docs/heatmaps.md](docs/heatmaps.md) for detailed methodology and interpretation.

---

## 📦 Package Structure

```
face-morph/
├── src/face_morph/          # Main package
│   ├── core/                # Core algorithms (morphing, I/O)
│   ├── rendering/           # Renderer abstraction (PyTorch3D/PyRender)
│   ├── visualization/       # Heatmaps, video, CSV export
│   ├── pipeline/            # Orchestration and configuration
│   ├── cli/                 # Command-line interface
│   └── utils/               # Logging, parallelization, context managers
├── docs/                    # Documentation
├── scripts/                 # Helper scripts
├── tests/                   # Test suite
└── pyproject.toml           # Package configuration
```

---

## 🚀 CLI Commands

### Single Pair Morphing

```bash
# Basic usage (default mode)
face-morph morph input1.fbx input2.fbx

# Full output with GPU acceleration
face-morph morph input1.fbx input2.fbx --full --gpu

# Custom output directory
face-morph morph face1.obj face2.obj -o custom_results/

# CPU mode with debug logging
face-morph morph face1.fbx face2.fbx --cpu --log-level DEBUG
```

### Batch Processing

```bash
# Process all unique pairs in folder (default mode)
face-morph batch data/faces/

# Full mode with GPU
face-morph batch data/faces/ --full --gpu

# Custom output directory
face-morph batch data/ -o batch_results/
```

### Help

```bash
# General help
face-morph --help

# Command-specific help
face-morph morph --help
face-morph batch --help
```

---

## 🐍 Python API

For programmatic access:

```python
import torch
from pathlib import Path
from face_morph.pipeline import MorphConfig, run_morphing_pipeline

# Configure morphing
config = MorphConfig(
    input_mesh_1=Path("face1.fbx"),
    input_mesh_2=Path("face2.fbx"),
    output_mode="full",  # or "default"
    device=torch.device('cuda'),  # or 'cpu'
    verbose=True
)

# Run pipeline
output_dir = run_morphing_pipeline(config)
print(f"Results saved to: {output_dir}")
```

See [docs/usage.md](docs/usage.md) for complete API examples.

---

## 🔬 Use Cases

**Research Applications:**
- Face perception studies
- Morphing stimulus generation
- Quantitative feature analysis
- Shape vs appearance comparison

**Production Applications:**
- Character animation
- Face aging simulation
- Medical visualization
- Game asset generation

---

## 📈 Performance

| Operation | CPU | GPU (CUDA) | Speedup |
|-----------|-----|------------|---------|
| Morphing (41 frames) | ~50s | ~5s | **10x** |
| Heatmap generation | ~30s | ~10s (parallel) | **3x** |
| FBX conversion (41 files) | ~60s | ~12s (parallel) | **5x** |

**GPU Benefits:**
- Mixed precision (FP16) for 2-3x faster computation
- Parallel batch operations
- Efficient memory management

---

## 🛠️ Requirements

**Minimum:**
- Python 3.9+
- 8GB RAM
- 2GB disk space

**Recommended:**
- Python 3.10+
- 16GB RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 12.4

**External Dependencies:**
- Blender (for FBX export) - `sudo snap install blender --classic`
- FFmpeg (for video creation) - `sudo apt install ffmpeg`

---

## 📚 Documentation

- [Installation Guide](docs/installation.md) - Detailed CPU and GPU installation
- [Usage Guide](docs/usage.md) - CLI and API examples
- [Heatmap Documentation](docs/heatmaps.md) - Calculation methodology and interpretation

---

## 🤝 Contributing

This is a clean, modular codebase following SOLID principles:

1. Follow existing code style and structure
2. Add type hints to all functions
3. Write comprehensive docstrings
4. Test with both CPU and GPU modes
5. Update documentation as needed

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 📧 Citation

If you use this package in your research, please cite:

```bibtex
@software{face_morph_2025,
  author = {costantinoai},
  title = {Face Morph: Production-ready 3D Face Morphing with GPU Acceleration},
  year = {2025},
  url = {https://github.com/costantinoai/face-morph-laura}
}
```

---

## 🙏 Acknowledgments

Built with:
- [PyTorch3D](https://pytorch3d.org/) - 3D deep learning framework
- [PyRender](https://pyrender.readthedocs.io/) - OpenGL-based mesh rendering
- [Click](https://click.palletsprojects.com/) - CLI framework
- [scikit-image](https://scikit-image.org/) - Perceptual color metrics

---

**Made with ❤️ for face perception research**
