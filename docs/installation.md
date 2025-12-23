# Installation Guide

This guide covers installation for both CPU-only and GPU-accelerated setups.

---

## Quick Installation (CPU-only)

For most users, CPU-only installation is recommended:

```bash
# Clone the repository
git clone https://github.com/costantinoai/face-morph-laura.git
cd face-morph-laura

# Create conda environment
conda create -n face-morph python=3.10
conda activate face-morph

# Install the package
pip install -e .
```

This installs all required dependencies for CPU-based morphing.

---

## GPU Installation (CUDA Support)

For GPU acceleration (10-20x faster):

### 1. Prerequisites

- **NVIDIA GPU** with CUDA Compute Capability 3.5+
- **CUDA Toolkit 12.4** (must match PyTorch version)
- **8GB+ GPU VRAM** (recommended)

Check your CUDA version:
```bash
nvidia-smi
nvcc --version
```

### 2. Install PyTorch with CUDA

```bash
conda create -n face-morph python=3.10
conda activate face-morph

# Install PyTorch with CUDA 12.4
conda install pytorch==2.4.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 3. Install Face Morph with CUDA extras

```bash
pip install -e .[cuda]
```

This installs additional dependencies needed for PyTorch3D compilation.

### 4. Install PyTorch3D (GPU)

**Option A: Try automated installation**
```bash
FORCE_CUDA=1 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

**Option B: Manual compilation (if Option A fails)**
```bash
# Install build dependencies
conda install -c conda-forge -c fvcore -c iopath fvcore iopath

# Install CUDA compiler (must match PyTorch CUDA version)
conda install cuda-nvcc=12.4.* cuda-toolkit -c nvidia -c conda-forge

# Set CUDA home
export CUDA_HOME=$CONDA_PREFIX

# Compile PyTorch3D with CUDA
FORCE_CUDA=1 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 5. Verify GPU Installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import pytorch3d; print('PyTorch3D with CUDA:', pytorch3d._C is not None)"
```

If both print `True`, GPU acceleration is working!

---

## External Dependencies

### Blender (for FBX export)

**Ubuntu/Debian:**
```bash
sudo snap install blender --classic
```

**macOS:**
```bash
brew install --cask blender
```

**Windows:**
Download from [blender.org](https://www.blender.org/download/)

### FFmpeg (for video creation)

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

---

## Verification

Test your installation:

```bash
# Check package imports
python -c "
from face_morph.pipeline import MorphConfig, run_morphing_pipeline
from face_morph.core import load_mesh, create_morpher
from face_morph.visualization import export_statistics_csv
print('✅ All imports successful!')
"

# Check CLI
face-morph --help
```

---

## Troubleshooting

### "CUDA not available"

**Symptoms:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia`
3. Verify CUDA toolkit version matches PyTorch

### "PyTorch3D compilation failed"

**Symptoms:** Build errors during PyTorch3D installation

**Solutions:**
1. Ensure CUDA compiler version matches PyTorch CUDA version:
   ```bash
   python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
   nvcc --version  # Should match (e.g., both 12.4)
   ```

2. Install correct CUDA compiler:
   ```bash
   conda install cuda-nvcc=12.4.* cuda-toolkit -c nvidia -c conda-forge
   ```

3. Try CPU-only mode (still fast with PyRender):
   ```bash
   pip install -e .  # Skip [cuda] extras
   ```

### "Blender not found"

**Symptoms:** FBX conversion fails

**Solutions:**
1. Install Blender (see External Dependencies above)
2. Specify path manually:
   ```bash
   face-morph morph input1.fbx input2.fbx --blender /path/to/blender
   ```

### "ffmpeg not available"

**Symptoms:** Video creation skipped

**Solutions:**
1. Install FFmpeg (see External Dependencies above)
2. Videos are optional - PNG frames are still generated

---

## Environment Management

### Activating the Environment

```bash
# Conda
conda activate face-morph

# Or use the activation script
source ./scripts/activate_env.sh
```

### Updating the Package

```bash
cd face-morph-laura
git pull
pip install -e . --upgrade
```

### Uninstalling

```bash
pip uninstall face-morph
conda deactivate
conda env remove -n face-morph
```

---

## Development Installation

For contributors:

```bash
# Install with development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

---

## Platform-Specific Notes

### Ubuntu/Debian

- Use `sudo apt install` for system packages
- OpenGL libraries required for PyRender: `sudo apt install libosmesa6-dev freeglut3-dev`

### macOS

- Use Homebrew for package management
- May need to install Xcode Command Line Tools: `xcode-select --install`

### Windows

- Use WSL2 (Windows Subsystem for Linux) for best compatibility
- Native Windows support available but less tested

---

## Next Steps

Once installed, see [docs/usage.md](usage.md) for usage examples.
