# Installation Guide

Bare metal installation instructions for face-morph across different platforms.

## Prerequisites

- **Python 3.9-3.12** (3.10 recommended)
- **Conda** ([Miniforge](https://github.com/conda-forge/miniforge) or Anaconda)
- **Blender 3.6+** (for FBX conversion, auto-detected)
- **FFmpeg** (optional, for video generation)

## CPU-Only Installation

Works on all platforms (Windows, macOS, Linux).

```bash
# Clone repository
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

# Verify installation
face-morph --help
```

## GPU Installation (Linux/Windows)

Requires CUDA-compatible NVIDIA GPU. macOS does not support CUDA.

```bash
# Clone repository
git clone https://github.com/costantinoai/face-morph.git
cd face-morph

# Use pre-configured environment
conda env create -f environment.yml
conda activate face-morph
pip install -e .

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note:** GPU mode requires CUDA-compatible NVIDIA GPU with drivers and toolkit installed.

## Platform-Specific Notes

### Windows

- **PyTorch3D:** Requires Visual Studio Build Tools (GPU mode only)
- **Paths:** Use forward slashes or raw strings: `r"C:\path\to\file.fbx"`
- **Admin Rights:** Run PowerShell as Administrator for conda commands
- **CUDA:** Install CUDA Toolkit from NVIDIA (GPU mode only)

**Install Visual Studio Build Tools:**
1. Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
2. Select "Desktop development with C++" workload
3. Restart after installation

### macOS

- **GPU:** Not supported (Apple dropped NVIDIA support in 2018)
- **CPU Only:** Use CPU-only installation above
- **Blender:** Install via DMG from [blender.org](https://www.blender.org/download/) or `brew install --cask blender`
- **FFmpeg:** Install via `brew install ffmpeg`

### Linux

- **GPU:** Best platform for GPU acceleration
- **NVIDIA Drivers:** Ensure drivers are installed (`nvidia-smi` should work)
- **CUDA Toolkit:** Install matching version for PyTorch
- **Check CUDA:** `nvcc --version` to verify installation

**Install NVIDIA drivers (Ubuntu/Debian):**
```bash
# Check recommended driver
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-535

# Reboot
sudo reboot
```

**Install CUDA Toolkit (Ubuntu/Debian):**
```bash
# Install CUDA 12.4 (example)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-4

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

## Verification

After installation, verify everything works:

```bash
# Activate environment
conda activate face-morph

# Check CLI is available
face-morph --help

# Check PyTorch (CPU)
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA (GPU only)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Check PyTorch3D (GPU only)
python -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')"

# Check Blender
blender --version

# Check FFmpeg (optional)
ffmpeg -version
```

## Troubleshooting

### "CUDA out of memory"

- Reduce batch size: Add `--chunk-size 5` flag
- Use CPU mode: `--cpu` instead of `--gpu`
- Close other GPU applications

### "Blender not found"

- Install Blender 3.6+ from [blender.org](https://www.blender.org/download/)
- Specify path: `--blender /path/to/blender`
- Auto-detection checks: `blender`, `/usr/bin/blender`, `/Applications/Blender.app/Contents/MacOS/Blender`

### "ModuleNotFoundError: pytorch3d"

- GPU mode only: Install PyTorch3D from source (see `environment.yml`)
- Or use CPU mode: `--cpu` (uses PyRender instead)

### PyTorch3D Installation Fails

**Linux/Windows:**
```bash
# Install from conda-forge
conda install pytorch3d -c pytorch3d

# Or build from source
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
```

**macOS:**
```bash
# PyTorch3D requires CUDA, so use CPU mode
# PyRender will be used automatically
```

### Environment Conflicts

```bash
# Remove and recreate environment
conda deactivate
conda env remove -n face-morph
conda env create -f environment.yml
conda activate face-morph
pip install -e .
```

## Updating

```bash
# Update to latest version
cd face-morph
git pull origin main

# Reinstall package
conda activate face-morph
pip install -e .
```

## Uninstallation

```bash
# Remove conda environment
conda deactivate
conda env remove -n face-morph

# Remove repository
cd ..
rm -rf face-morph
```

## Next Steps

- See [README.md](README.md) for usage examples
- See [DOCKER.md](DOCKER.md) for Docker deployment (recommended for production)
