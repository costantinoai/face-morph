#!/usr/bin/env bash
#
# 3D Face Morphing Pipeline - Installation Script
# =================================================
#
# Automatically installs PyTorch and PyTorch3D with correct GPU/CPU support.
# Detects NVIDIA GPU and installs appropriate packages.
#
# Usage:
#   ./install.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="face-morph"
PYTHON_VERSION="3.10"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  3D Face Morphing Pipeline - Installation${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Miniconda or Anaconda first."
    print_info "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_success "Conda found: $(conda --version)"

# Check for mamba (faster dependency solver)
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    print_success "Mamba found - will use mamba for faster installation"
else
    CONDA_CMD="conda"
    print_info "Mamba not found - using conda (consider installing mamba for faster installs)"
fi
echo ""

# Detect NVIDIA GPU
print_info "Detecting GPU..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
        print_success "NVIDIA GPU detected: ${GPU_INFO}"

        # Detect CUDA version
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_info "CUDA Version: ${CUDA_VERSION}"

        # Determine PyTorch CUDA version to install
        if [[ "$CUDA_VERSION" == 12.* ]]; then
            PYTORCH_CUDA="12.4"
        elif [[ "$CUDA_VERSION" == 11.* ]]; then
            PYTORCH_CUDA="11.8"
        else
            print_warning "CUDA version ${CUDA_VERSION} detected, defaulting to CUDA 12.4"
            PYTORCH_CUDA="12.4"
        fi

        print_info "Will install PyTorch with CUDA ${PYTORCH_CUDA}"
    else
        GPU_AVAILABLE=false
        print_warning "nvidia-smi command failed - GPU not accessible"
    fi
else
    GPU_AVAILABLE=false
    print_warning "nvidia-smi not found - no NVIDIA GPU available"
fi

if [ "$GPU_AVAILABLE" = false ]; then
    print_info "Will install CPU-only version"
fi
echo ""

# Remove old environment if it exists
print_info "Checking for existing environments..."
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '${ENV_NAME}' already exists"
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing environment '${ENV_NAME}'..."
        conda env remove -n ${ENV_NAME} -y
        print_success "Environment removed"
    else
        print_error "Installation cancelled"
        exit 1
    fi
fi

# Check for old pytorch3d environment
if conda env list | grep -q "^pytorch3d "; then
    print_warning "Old 'pytorch3d' environment found"
    read -p "Remove old environment? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing old 'pytorch3d' environment..."
        conda env remove -n pytorch3d -y
        print_success "Old environment removed"
    fi
fi
echo ""

# Create environment from environment.yml
print_info "Creating environment from environment.yml file..."
print_info "This installs all packages in one step to avoid multiple dependency resolutions"
echo ""

if [ "$GPU_AVAILABLE" = true ]; then
    print_info "Installing with GPU support (CUDA ${PYTORCH_CUDA})..."
    ${CONDA_CMD} env create -f environment.yml -y
else
    print_info "Installing CPU-only version..."
    print_warning "Modifying environment.yml for CPU-only installation..."
    # Create temporary CPU-only version
    sed 's/pytorch-cuda=12.4/# pytorch-cuda=12.4/' environment.yml | \
    sed 's/# - cpuonly/- cpuonly/' > environment-cpu.yml
    ${CONDA_CMD} env create -f environment-cpu.yml -y
    rm environment-cpu.yml
fi

print_success "Environment created with all conda packages"
echo ""

# Activate environment
print_info "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}
print_success "Environment activated"
echo ""

# Verify PyTorch installation
print_info "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ "$GPU_AVAILABLE" = true ]; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" || true
fi
echo ""

# Install PyTorch3D from source
print_info "Building PyTorch3D from source (this may take 5-10 minutes)..."

if [ "$GPU_AVAILABLE" = true ]; then
    print_info "Building with GPU support (FORCE_CUDA=1)..."

    # Set CUDA_HOME environment variable
    print_info "Detecting CUDA installation..."
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -d "$CONDA_PREFIX/pkgs/cuda-toolkit" ]; then
        export CUDA_HOME=$CONDA_PREFIX/pkgs/cuda-toolkit
    elif [ -d "$CONDA_PREFIX" ]; then
        # Use conda's CUDA
        export CUDA_HOME=$CONDA_PREFIX
    fi

    print_info "CUDA_HOME set to: $CUDA_HOME"

    FORCE_CUDA=1 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
else
    print_info "Building CPU-only version..."
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
fi
print_success "PyTorch3D built and installed"
echo ""

# Install other dependencies
print_info "Installing additional dependencies..."
pip install matplotlib pillow
print_success "Additional dependencies installed"
echo ""

# Install ffmpeg for video creation
print_info "Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    print_success "ffmpeg already installed: $(ffmpeg -version | head -n1)"
else
    print_warning "ffmpeg not found - video creation will not work"
    print_info "Install ffmpeg manually:"
    print_info "  Ubuntu/Debian: sudo apt install ffmpeg"
    print_info "  macOS: brew install ffmpeg"
    print_info "  Or: conda install -c conda-forge ffmpeg"
fi
echo ""

# Install Blender (optional, for FBX conversion)
print_info "Checking for Blender..."
if command -v blender &> /dev/null; then
    print_success "Blender already installed: $(blender --version | head -n1)"
else
    print_warning "Blender not found - FBX conversion will not work"
    print_info "Install Blender manually:"
    print_info "  Ubuntu/Debian: sudo snap install blender --classic"
    print_info "  macOS: brew install --cask blender"
    print_info "  Or download from: https://www.blender.org/download/"
fi
echo ""

# Verify installation
print_info "Verifying complete installation..."
python -c "import torch; import pytorch3d; import matplotlib; from PIL import Image; print('All imports successful!')"
echo ""

# Final verification with GPU rendering test
print_info "Testing GPU rendering support..."
if [ "$GPU_AVAILABLE" = true ]; then
    python << 'EOF'
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)

try:
    device = torch.device("cuda:0")

    # Create simple test mesh
    verts = torch.tensor([[[0.0, 0.0, 0.0]]], device=device)
    faces = torch.tensor([[[0, 0, 0]]], device=device)
    mesh = Meshes(verts=verts, faces=faces)

    # Try to create rasterizer
    R, T = look_at_view_transform(2.7, 0, 0, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(image_size=256)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # Test rasterization
    fragments = rasterizer(mesh)

    print("✓ GPU rendering support: ENABLED")
    print("  PyTorch3D rasterizer compiled with CUDA support!")
except RuntimeError as e:
    if "Not compiled with GPU support" in str(e):
        print("✗ GPU rendering support: DISABLED")
        print("  PyTorch3D rasterizer not compiled with GPU support")
        print("  Rendering will fall back to CPU (slower but functional)")
    else:
        raise
EOF
else
    print_info "Skipping GPU rendering test (no GPU available)"
    python -c "import pytorch3d; print('✓ PyTorch3D installed (CPU-only mode)')"
fi
echo ""

# Print summary
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
print_info "Environment name: ${ENV_NAME}"
print_info "Python version: $(python --version | awk '{print $2}')"
print_info "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
print_info "PyTorch3D version: $(python -c 'import pytorch3d; print(pytorch3d.__version__)')"

if [ "$GPU_AVAILABLE" = true ]; then
    print_info "GPU support: ENABLED"
    print_info "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
else
    print_info "GPU support: CPU-only mode"
fi
echo ""

print_info "To activate the environment, run:"
echo -e "  ${BLUE}conda activate ${ENV_NAME}${NC}"
echo ""

print_info "To test the installation, run:"
echo -e "  ${BLUE}./morph -i1 data/male1.fbx -i2 data/male2.fbx --gpu${NC}"
echo ""

print_success "Installation successful! Happy morphing! 🎭"
