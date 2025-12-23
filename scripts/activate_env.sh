#!/bin/bash
#
# Face Morph Environment Activation
# ==================================
#
# Usage:
#   source ./scripts/activate_env.sh
#   or: ./scripts/activate_env.sh face-morph input1.fbx input2.fbx
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}Face Morph Environment Activation${NC}"
echo "======================================"
echo ""

# Detect Python environment
if command -v conda &> /dev/null; then
    # Conda environment
    if [[ "$CONDA_DEFAULT_ENV" != "face-morph" ]]; then
        echo "Activating conda environment: face-morph"
        eval "$(conda shell.bash hook)"
        conda activate face-morph
    else
        echo -e "${GREEN}✓ Already in face-morph conda environment${NC}"
    fi
elif [[ -d "venv" ]]; then
    # Virtual environment
    echo "Activating virtual environment: venv"
    source venv/bin/activate
elif [[ -d ".venv" ]]; then
    # Alternative venv location
    echo "Activating virtual environment: .venv"
    source .venv/bin/activate
else
    echo -e "${YELLOW}Warning: No conda or venv detected. Using system Python.${NC}"
fi

# Verify installation
if ! command -v face-morph &> /dev/null; then
    echo -e "${YELLOW}Warning: face-morph command not found. Installing...${NC}"
    pip install -e .
fi

echo -e "${GREEN}✓ Environment ready!${NC}"
echo ""

# If arguments provided, run command
if [ $# -gt 0 ]; then
    echo "Running: $@"
    echo ""
    "$@"
else
    # Show usage examples
    echo "Usage examples:"
    echo "  face-morph morph input1.fbx input2.fbx"
    echo "  face-morph morph input1.fbx input2.fbx --full --gpu"
    echo "  face-morph batch data/ --full --gpu"
    echo "  face-morph --help"
    echo ""
    echo "Or run any Python command:"
    echo "  python -m face_morph.pipeline.orchestrator"
fi
