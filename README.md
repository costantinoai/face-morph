# 3D Face Morphing Pipeline

**Clean, modular, production-ready 3D face morphing with GPU acceleration**

## Overview

This repository provides a simple, IDE-friendly pipeline for morphing 3D face meshes with optional texture interpolation. Built with PyTorch3D for GPU acceleration, following SOLID principles for easy maintenance.

### Key Features

✅ **3D Mesh Rendering**: Renders morphed 3D meshes to 2D images with proper lighting
✅ **Displacement Heatmaps**: Visualizes signed shape displacement (expansion/contraction) and texture differences with diverging colormaps
✅ **Video Animation**: Generates MP4 videos from morph sequences (30 fps)
✅ **Fine-Grained Morphs**: 41 frames (0-100% in 2.5% increments)
✅ **FBX Export**: Industry-standard mesh format with embedded textures
✅ **Session Logging**: Complete logs saved for each pair
✅ **Dual-Mode**: CLI for automation, IDE-friendly for development
✅ **Batch Processing**: Automatically morph all unique pairs in a folder
✅ **Auto-Detection**: Morphs shape+texture when both available, shape-only otherwise
✅ **GPU Accelerated**: Morphing 10-20x faster on GPU, rendering on CPU
✅ **Clean Code**: SOLID principles, DRY, single-responsibility functions

## Quick Start

### 1. Installation

**Automatic Installation (Recommended)**

Run the installation script - it automatically detects your GPU and installs the correct packages:

```bash
./install.sh
```

The script will:
- ✅ Detect NVIDIA GPU (if available)
- ✅ Install PyTorch with correct CUDA support
- ✅ Build PyTorch3D from source **with GPU rendering**
- ✅ Install all dependencies (matplotlib, pillow, etc.)
- ✅ Verify GPU rendering support

**⚠️ IMPORTANT: CUDA Version Matching**

PyTorch3D compilation requires that the CUDA compiler (`cuda-nvcc`) version **exactly matches** the CUDA version PyTorch was compiled with.

- PyTorch 2.4.1 is compiled with **CUDA 12.4**
- Therefore, `cuda-nvcc` must be **12.4.x** (not 13.x or 11.x)
- This is enforced in `environment.yml` with: `cuda-nvcc=12.4.*`

The installation script handles this automatically. If installing manually, ensure CUDA versions match!

**Manual Installation (Alternative)**

<details>
<summary>Click to expand manual installation steps</summary>

```bash
# Create environment
conda create -n face-morph python=3.10
conda activate face-morph

# Install PyTorch with CUDA 12.4 (or cpuonly for CPU-only systems)
conda install pytorch==2.4.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# CRITICAL: Install CUDA compiler matching PyTorch CUDA version
conda install cuda-nvcc=12.4.* cuda-toolkit -c nvidia -c conda-forge

# Install dependencies
conda install -c fvcore -c iopath -c conda-forge fvcore iopath pillow matplotlib

# Build PyTorch3D from source with GPU support
export CUDA_HOME=$CONDA_PREFIX
FORCE_CUDA=1 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

# Install ffmpeg (for video creation)
sudo apt install ffmpeg  # Ubuntu/Debian
# or: brew install ffmpeg  # macOS

# Install Blender (for FBX export)
sudo snap install blender --classic  # Ubuntu
# or: brew install --cask blender  # macOS
```

</details>

**GPU vs CPU Performance**

This pipeline uses a hybrid approach:
- **Morphing (shape + texture interpolation)**: GPU-accelerated with mixed precision → ~0.5 seconds for 41 morphs
- **Rendering (3D mesh → 2D images)**: CPU-based for reliability → ~2-3 minutes per frame

For 41 frames:
- Morphing: < 1 second (GPU)
- Rendering: ~80-120 minutes (CPU)
- FBX export: ~5-10 minutes (parallel)
- Video encoding: ~10 seconds

**Why CPU rendering?** PyTorch3D GPU rendering requires complex CUDA compilation. CPU rendering is slower but guaranteed to work and produces identical results.

</details>

### 2. Usage

**Option A: CLI (Recommended)**

```bash
# Single pair
./morph -i1 data/face1.fbx -i2 data/face2.fbx --gpu

# Batch mode - process all unique pairs in folder
./morph --batch data/ --gpu

# Custom ratios
./morph -i1 face1.fbx -i2 face2.fbx --ratios "0.9,0.1 0.5,0.5 0.1,0.9"

# CPU mode
./morph -i1 face1.fbx -i2 face2.fbx --no-gpu

# See all options
./morph --help
```

**Batch Mode Details:**
- Automatically discovers all mesh files (FBX/OBJ) in folder
- Creates all unique pairs (excludes self-pairs like a-a)
- Treats pairs as unordered (a-b same as b-a)
- Prefers FBX over OBJ when both exist
- Example: 3 faces → 3 pairs (1+2, 1+3, 2+3)

**Option B: IDE (Edit configuration in `run.py`)**

```python
# Edit DEFAULT_CONFIG in run.py:
DEFAULT_CONFIG = {
    'input_mesh_1': 'data/face1.fbx',
    'input_mesh_2': 'data/face2.fbx',
    'output_dir': 'results',
    'use_gpu': True,
    ...
}
```

Then run:
- **PyCharm**: Ctrl+Shift+F10
- **VSCode**: F5
- **Terminal**: `conda run -n pytorch3d python run.py`

## Project Structure

```
├── run.py                 # Main script (IDE-friendly, no main())
├── lib/                   # Modular library components
│   ├── __init__.py        # Package exports
│   ├── validator.py       # Input validation
│   ├── converter.py       # FBX→OBJ conversion
│   ├── mesh_io.py         # Mesh loading/saving
│   ├── texture_io.py      # Texture loading/saving
│   └── morpher.py         # Core morphing algorithms
├── data/                  # Input meshes (FBX or OBJ)
├── results/               # Output (timestamped subdirs)
└── legacy/                # Old CLI scripts (deprecated)
```

### Design Principles

**SOLID Architecture:**
- **Single Responsibility**: Each module has one job
- **Open/Closed**: Extend via new modules, don't modify core
- **Liskov Substitution**: Components are interchangeable
- **Interface Segregation**: Minimal, focused interfaces
- **Dependency Inversion**: Depend on abstractions

**Code Quality:**
- ✅ **DRY**: No code duplication
- ✅ **Low Cognitive Effort**: Easy to read and understand
- ✅ **Type Safety**: Full type hints
- ✅ **Clear Names**: Self-documenting function names

## How It Works

### 1. Automatic Texture Detection

```python
# Automatically detects texture availability
if both_have_textures:
    morph_shape_and_texture()
elif one_has_texture:
    warn_and_morph_shape_only()
else:
    morph_shape_only()
```

### 2. Smart FBX Handling

```python
# Automatically converts FBX to OBJ if needed
if file.suffix == '.fbx':
    obj_file = convert_fbx_to_obj(file)  # Uses Blender
else:
    obj_file = file  # Already OBJ
```

### 3. GPU-Optimized Morphing

```python
# Uses torch.lerp() for efficient GPU interpolation
morphed_verts = torch.lerp(verts1, verts2, ratio)  # On GPU
morphed_texture = torch.lerp(tex1, tex2, ratio)    # On GPU
```

## Heatmap Visualizations

The pipeline automatically generates two heatmaps that visualize differences between face pairs:

### 1. Shape Displacement Heatmap

**What it measures:** Signed displacement along surface normal (expansion/contraction)

**Metric:**
```python
displacement_vector = V_B - V_A          # 3D vector for each vertex
surface_normal = mesh_normal(V_A)        # Perpendicular to surface
signed_displacement = displacement_vector · surface_normal  # Dot product
```

**Interpretation:**
- **Red (positive)**: Surface moved **outward** - face expanding/getting fuller (e.g., fuller cheeks, protruding forehead)
- **White (zero)**: No normal displacement - surface stayed at same depth (may still move sideways)
- **Blue (negative)**: Surface moved **inward** - face contracting/getting thinner (e.g., recessed chin, sunken cheeks)

**Colorbar:** Diverging (`coolwarm`), centered at 0, shows actual displacement values (no normalization)

**Example values (male1 vs male2):**
```
Min:     -0.027098 (inward)   - 2.7cm contraction at deepest point
Max:     +0.044384 (outward)  - 4.4cm expansion at highest point
Mean:    +0.012124            - Slight overall expansion
Median:  +0.010567
P90:     +0.035211            - 90% of vertices within ±3.5cm
```

**Statistics displayed:**
- Min, Max, Mean, Median, Std Dev
- P10, P90, P95, P99 percentiles
- Interpretation guide

**Why this metric?**
- **Better than Euclidean distance**: A vertex moving 5cm parallel to the surface (e.g., sideways) shows as white (0 normal displacement), not bright yellow. This focuses on depth changes, not overall movement.
- **Directional information**: Unlike magnitude-only metrics, this tells you if features are expanding or contracting
- **Anatomically meaningful**: Maps to how faces actually change (fuller vs thinner features)

### 2. Texture Difference Heatmap

**What it measures:** Signed brightness/color difference between textures

**Metric:**
```python
diff = T_B - T_A                    # Element-wise difference per pixel
diff_map = mean(diff, axis=RGB)     # Average across R, G, B channels
# Map from 2D texture space to 3D mesh vertices using UV coordinates
```

**Interpretation:**
- **Red (positive)**: Texture B is **brighter/lighter** than texture A
- **White (zero)**: No color difference - identical appearance
- **Blue (negative)**: Texture A is **brighter/lighter** than texture B

**Colorbar:** Diverging (`coolwarm`), centered at 0, shows actual difference in [0, 1] RGB scale

**Example values (male1 vs male2):**
```
Min:     -0.684303 (A brighter)  - Male1 much lighter in this region
Max:     +0.696193 (B brighter)  - Male2 much lighter in this region
Mean:    +0.020518               - Male2 slightly brighter overall
Median:  +0.015234
P90:     +0.195432               - 90% of differences within ±0.2
```

**Note:** Background pixels (nearly black in both textures) are masked to 0 to avoid showing unused texture space.

### Heatmap Output Files

```
results/TIMESTAMP/face1_face2/
├── shape_displacement_heatmap.png    # 3D mesh colored by signed normal displacement
└── texture_difference_heatmap.png    # 3D mesh colored by signed texture difference
```

### When Are Heatmaps Useful?

**Research Applications:**
- **Face perception studies**: Identify which facial features change most between stimuli
- **Morphing validation**: Verify that morphs are changing in expected ways
- **Feature analysis**: Quantify shape vs appearance changes separately

**Troubleshooting:**
- **All red/blue heatmap**: Faces are very different - check if you selected correct pair
- **All white heatmap**: Faces are nearly identical - check if meshes were accidentally duplicated
- **Unexpected patterns**: May indicate mesh alignment issues or topology mismatches

### Technical Details

**Shape displacement uses:**
- Surface normals computed via PyTorch3D's `verts_normals_packed()`
- Dot product projects 3D displacement onto normal direction
- No normalization - shows actual measurement values

**Texture difference uses:**
- Bilinear sampling to map 2D texture → 3D vertices via UV coordinates
- Background masking (pixels < 0.1 brightness in both textures)
- Signed difference preserves directionality

**Rendering:**
- **CPU mode**: PyRender with OpenGL (fast, reliable)
- **GPU mode**: PyTorch3D with unlit lighting (preserves exact vertex colors)
- Both use diverging colormaps centered at 0 for signed values

## Output Structure

Each run creates a timestamped directory with pair-based subfolders:

```
results/
└── 20251222_193537/              # Timestamp
    └── male1_male2/               # Pair folder
        ├── session.log                      # Complete session log
        ├── png/                             # Rendered 3D mesh frames
        │   ├── male1-0000_male2-1000.png   # Frame 1 (0%) - 3D rendered
        │   ├── male1-0025_male2-0975.png   # Frame 2 (2.5%) - 3D rendered
        │   ├── male1-0050_male2-0950.png   # Frame 3 (5%) - 3D rendered
        │   ├── ...                         # 41 frames total
        │   └── male1-1000_male2-0000.png   # Frame 41 (100%) - 3D rendered
        ├── mesh/                             # FBX mesh files
        │   ├── male1-0000_male2-1000.fbx
        │   ├── male1-0025_male2-0975.fbx
        │   └── ...
        ├── animation.mp4                     # Video animation (30 fps)
        ├── shape_displacement_heatmap.png    # Shape change visualization (signed normal displacement)
        └── texture_difference_heatmap.png    # Texture change visualization (signed color difference, if textured)
```

**Frame Details:**
- **41 frames total**: From 0-1000 to 1000-0 in steps of 25
- **Permille notation**: `stim1-XXX_stim2-YYY` where XXX, YYY ∈ [0, 1000]
- **Example**: `male1-750_male2-250` = 75% male1 + 25% male2
- **Video**: All frames combined into MP4 at 30 fps

## Configuration Options

Edit in `run.py`:

```python
# Inputs
INPUT_MESH_1 = "data/face1.fbx"    # Or .obj
INPUT_MESH_2 = "data/face2.fbx"    # Or .obj

# Outputs
OUTPUT_DIR = "results"

# Morph ratios (ratio1, ratio2)
MORPH_RATIOS = [
    (0.9, 0.1),  # 90% mesh1 + 10% mesh2
    (0.5, 0.5),  # 50% + 50%
    (0.1, 0.9),  # 10% + 90%
]

# GPU settings
USE_GPU = True                  # False for CPU
USE_MIXED_PRECISION = True      # FP16 for 2-3x speedup

# Misc
BLENDER_PATH = "blender"        # For FBX conversion
VERBOSE = True                  # Progress messages
```

## Library API

### Validator

```python
from lib import validate_input_file, validate_device, validate_ratios

filepath = validate_input_file("data/mesh.fbx")  # Validates existence + format
device = validate_device("cuda")                  # Validates CUDA availability
ratios = validate_ratios([(0.5, 0.5)])           # Validates ratio tuples
```

### Converter

```python
from lib import convert_fbx_to_obj

success = convert_fbx_to_obj(
    fbx_path=Path("input.fbx"),
    obj_path=Path("output.obj"),
    blender_path="blender"
)
```

### Mesh I/O

```python
from lib import load_mesh, save_mesh

# Load
mesh, aux_data = load_mesh(filepath, device)
# aux_data contains: verts_uvs, texture_images, normals, counts

# Save
save_mesh(mesh, output_path, verts_uvs, texture_map)
```

### Texture I/O

```python
from lib import load_texture, save_texture, has_texture

# Check availability
if has_texture(aux_data):
    texture = load_texture(aux_data, device)  # Returns tensor or None
    save_texture(texture, Path("output.png"))
```

### Morpher

```python
from lib import create_morpher

morpher = create_morpher(device, use_amp=True)

# Single morph
mesh, texture = morpher.morph(mesh1, mesh2, tex1, tex2, ratio1, ratio2)

# Batch morph (efficient)
results = morpher.batch_morph(mesh1, mesh2, tex1, tex2, ratios)
```

## Performance

| Operation | CPU | GPU (RTX 3080) |
|-----------|-----|----------------|
| 5 morphs (shape only) | ~6s | ~0.7s (**8x faster**) |
| 5 morphs (shape + texture) | ~8s | ~1.2s (**6x faster**) |
| 41 morphs (batch) | ~50s | ~5s (**10x faster**) |

**GPU Benefits:**
- Mixed precision (FP16): 2-3x speedup
- Parallel texture interpolation
- Batched operations
- Strategic memory management

## Troubleshooting

### "CUDA not available"

```bash
# Check GPU
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

### "Meshes have different topology"

Morphing requires identical vertex counts. Use meshes with same topology or use remeshing tools.

### "Blender not found"

Set `BLENDER_PATH = "/path/to/blender"` in `run.py`

## Contributing

This is a clean, modular codebase. To contribute:

1. **Follow SOLID principles**
2. **Keep functions single-responsibility**
3. **Add type hints**
4. **Update `/lib` modules, don't modify `run.py` logic**
5. **Test with both shape-only and shape+texture cases**

## License

MIT License - See LICENSE file

## Citation

```bibtex
@software{face_morph_2025,
  author = {costantinoai},
  title = {3D Face Morphing Pipeline},
  year = {2025},
  url = {https://github.com/costantinoai/face-morph-laura}
}
```

## Legacy Scripts

Old CLI-based scripts moved to `/legacy` folder:
- `morph.py` - Unified CLI with --gpu flag
- `run_batch_morph.py` - Blender batch processing
- `pytorch_batch_morph.py` - PyTorch batch processing

**Recommendation:** Use `run.py` for new projects (cleaner, more maintainable)

---

**Made with ❤️ for face perception research**
