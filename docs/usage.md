# Usage Guide

Complete guide to using Face Morph for 3D face morphing.

---

## CLI Usage

The command-line interface is the easiest way to use Face Morph.

### Basic Commands

```bash
# Morph two faces (default mode: PNG + heatmaps)
face-morph morph face1.fbx face2.fbx

# Full mode (PNG + heatmaps + meshes + video + CSV)
face-morph morph face1.fbx face2.fbx --full

# Batch process all faces in folder
face-morph batch data/faces/
```

### morph command

Morph two 3D face meshes with interpolation.

**Syntax:**
```bash
face-morph morph INPUT1 INPUT2 [OPTIONS]
```

**Arguments:**
- `INPUT1` - First mesh file (.fbx or .obj)
- `INPUT2` - Second mesh file (.fbx or .obj)

**Options:**
```bash
-o, --output PATH        Output directory (default: results/)
--full                   Generate complete output (meshes + video + CSV)
--gpu / --cpu            Use GPU acceleration (default: CPU)
--no-amp                 Disable mixed precision (FP16) on GPU
-v, --verbose            Verbose output (default: enabled)
-q, --quiet              Suppress output
--log-level LEVEL        Logging level: DEBUG|INFO|WARNING|ERROR
--blender PATH           Path to Blender executable
--help                   Show help message
```

**Examples:**
```bash
# Fast mode (5-10 minutes)
face-morph morph face1.fbx face2.fbx

# Full output with GPU (2 hours, includes everything)
face-morph morph face1.fbx face2.fbx --full --gpu

# Custom output directory
face-morph morph face1.fbx face2.fbx -o my_results/

# CPU mode with debug logging
face-morph morph face1.fbx face2.fbx --cpu --log-level DEBUG

# Specify custom Blender path
face-morph morph face1.fbx face2.fbx --blender /usr/local/bin/blender
```

### batch command

Process all unique pairs in a folder.

**Syntax:**
```bash
face-morph batch FOLDER [OPTIONS]
```

**Arguments:**
- `FOLDER` - Directory containing mesh files (.fbx or .obj)

**Options:**
```bash
-o, --output PATH        Output directory (default: results/)
--full                   Generate complete output for each pair
--gpu / --cpu            Use GPU acceleration
-v, --verbose            Verbose output
--log-level LEVEL        Logging level
--help                   Show help message
```

**Examples:**
```bash
# Process all pairs (default mode)
face-morph batch data/faces/

# Full mode with GPU for all pairs
face-morph batch data/faces/ --full --gpu

# Custom output directory
face-morph batch data/ -o batch_results/
```

**Batch Processing Details:**
- Discovers all .fbx and .obj files in folder
- Creates all unique combinations (excludes self-pairs)
- Treats pairs as unordered (a-b same as b-a)
- Prefers FBX over OBJ when both exist
- Example: 4 faces → 6 pairs (1+2, 1+3, 1+4, 2+3, 2+4, 3+4)

---

## Python API

For programmatic control and integration into larger pipelines.

### Basic Example

```python
import torch
from pathlib import Path
from face_morph.pipeline import MorphConfig, run_morphing_pipeline

# Configure morphing
config = MorphConfig(
    input_mesh_1=Path("data/face1.fbx"),
    input_mesh_2=Path("data/face2.fbx"),
    output_mode="default",  # or "full"
    device=torch.device('cuda'),  # or 'cpu'
    verbose=True
)

# Run pipeline
output_dir = run_morphing_pipeline(config)
print(f"Results saved to: {output_dir}")
```

### Advanced Configuration

```python
from face_morph.pipeline import MorphConfig, generate_video_ratios

config = MorphConfig(
    input_mesh_1=Path("face1.fbx"),
    input_mesh_2=Path("face2.fbx"),
    output_dir=Path("custom_results"),
    output_mode="full",

    # Custom morph ratios
    morph_ratios=generate_video_ratios(step=50),  # 21 frames instead of 41

    # Hardware
    device=torch.device('cuda'),
    use_mixed_precision=True,  # FP16 for 2-3x speedup

    # Performance
    parallel_fbx=True,  # Parallel FBX conversion
    num_workers=7,  # cpu_count - 1

    # Logging
    verbose=True,
    log_level="DEBUG",

    # Video
    video_fps=30,

    # External tools
    blender_path="blender",
    ffmpeg_path="ffmpeg"
)

output_dir = run_morphing_pipeline(config)
```

### Loading and Saving Meshes

```python
import torch
from pathlib import Path
from face_morph.core import load_mesh, save_mesh, load_texture

device = torch.device('cuda')

# Load mesh with texture
mesh, aux = load_mesh(Path("face.obj"), device)
texture = load_texture(aux, device)

print(f"Loaded mesh with {aux['num_vertices']:,} vertices")
print(f"Has texture: {texture is not None}")

# Save mesh
save_mesh(mesh, Path("output.obj"), aux['verts_uvs'], texture)
```

### Morphing Meshes

```python
from face_morph.core import create_morpher

# Create morpher
morpher = create_morpher(device, use_amp=True)

# Single morph
morphed_mesh, morphed_texture = morpher.morph(
    mesh1, mesh2,
    texture1, texture2,
    ratio1=0.7,
    ratio2=0.3
)

# Batch morph (efficient for multiple ratios)
ratios = [
    (1.0, 0.0),  # 100% mesh1
    (0.5, 0.5),  # 50-50 blend
    (0.0, 1.0),  # 100% mesh2
]

results = morpher.batch_morph(
    mesh1, mesh2,
    texture1, texture2,
    ratios
)

for (mesh, texture), (r1, r2) in zip(results, ratios):
    print(f"Morphed: {r1:.1%} mesh1 + {r2:.1%} mesh2")
```

### Computing Heatmaps

```python
from face_morph.visualization.heatmap import (
    compute_shape_displacement_components,
    compute_texture_difference_components
)

# Shape displacement
normal_disp, tangent_disp, total_disp = compute_shape_displacement_components(
    mesh1, mesh2
)

print(f"Normal displacement range: [{normal_disp.min():.4f}, {normal_disp.max():.4f}]")
print(f"Mean total displacement: {total_disp.mean():.4f}")

# Texture difference (if textures available)
if texture1 is not None and texture2 is not None:
    lum_diff, chroma_diff, delta_e = compute_texture_difference_components(
        [texture1, texture2]
    )

    print(f"Perceptual difference (ΔE) range: [{delta_e.min():.2f}, {delta_e.max():.2f}]")
```

### Exporting CSV Data

```python
from face_morph.visualization.export import (
    export_statistics_csv,
    export_vertex_data_csv,
    export_texture_data_csv
)

# Export summary statistics
export_statistics_csv(
    normal_disp, tangent_disp, total_disp,
    lum_diff, chroma_diff, delta_e,
    Path("statistics.csv")
)

# Export per-vertex data
export_vertex_data_csv(
    mesh1, mesh2,
    normal_disp, tangent_disp, total_disp,
    Path("vertex_displacements.csv")
)

# Export per-pixel texture data (downsampled)
export_texture_data_csv(
    lum_diff, chroma_diff, delta_e,
    Path("texture_differences.csv"),
    downsample_factor=4
)
```

### Parallel Processing

```python
from face_morph.utils.parallel import get_optimal_workers, create_worker_pool

# Get optimal worker count
workers = get_optimal_workers("cpu")  # cpu_count - 1
print(f"Using {workers} workers")

# Create worker pool
with create_worker_pool("cpu") as pool:
    results = pool.map(process_function, data_items)
```

---

## Output Structure

### Default Mode

```
results/
└── YYYYMMDD_HHMMSS/           # Timestamp
    └── face1_face2/            # Pair name
        ├── session.log                            # Processing log
        ├── png/                                   # Rendered images
        │   ├── face1-000_face2-1000.png          # Frame 1 (0%)
        │   ├── face1-025_face2-975.png           # Frame 2 (2.5%)
        │   └── ...                                # 41 frames total
        ├── shape_displacement_components.png      # 3-component shape heatmap
        └── texture_difference_components.png      # 3-component texture heatmap
```

### Full Mode

```
results/
└── YYYYMMDD_HHMMSS/
    └── face1_face2/
        ├── session.log
        ├── png/                                   # 41 PNG frames
        ├── mesh/                                  # Mesh files
        │   ├── face1-000_face2-1000.fbx
        │   ├── face1-025_face2-975.fbx
        │   └── ...                                # 41 FBX files
        ├── animation.mp4                          # Video (30 fps)
        ├── shape_displacement_components.png
        ├── texture_difference_components.png
        ├── statistics.csv                         # Summary metrics
        ├── vertex_displacements.csv               # Per-vertex data
        └── texture_differences.csv                # Per-pixel texture data
```

---

## File Naming Convention

Frames use **permille notation** (parts per thousand):

- `face1-000_face2-1000.png` - 0% face1, 100% face2
- `face1-250_face2-750.png` - 25% face1, 75% face2
- `face1-500_face2-500.png` - 50% face1, 50% face2
- `face1-750_face2-250.png` - 75% face1, 25% face2
- `face1-1000_face2-000.png` - 100% face1, 0% face2

---

## Tips and Best Practices

### Performance Optimization

1. **Use GPU when available** - 10-20x faster morphing
2. **Use default mode for iteration** - Fast feedback for testing
3. **Use full mode for final output** - Complete publication-ready results
4. **Batch process overnight** - Full mode can take hours for many pairs

### Quality Checks

1. **Check heatmaps** - Verify morphing is producing expected changes
2. **Review session.log** - Check for warnings or issues
3. **Inspect first/last frames** - Should match input meshes exactly
4. **Check middle frame** - 50-50 blend should look natural

### Common Workflows

**Research Workflow:**
```bash
# 1. Quick test with default mode
face-morph morph test1.fbx test2.fbx

# 2. If good, run full mode for paper
face-morph morph test1.fbx test2.fbx --full --gpu

# 3. Analyze CSV data in R/Python/Excel
# statistics.csv - summary metrics
# vertex_displacements.csv - detailed analysis
```

**Production Workflow:**
```bash
# 1. Batch process all pairs
face-morph batch data/faces/ --full --gpu

# 2. Generate videos for all pairs
# (automatically included in full mode)

# 3. Export specific frames as needed
# (all frames saved as PNG and FBX)
```

---

## Troubleshooting

### "Topology mismatch"

**Problem:** Meshes have different vertex counts

**Solution:**
- Ensure meshes have identical topology
- Use remeshing tools to create matching topology
- Check that you're using the correct pair of meshes

### "Out of memory"

**Problem:** GPU runs out of memory

**Solutions:**
- Use CPU mode: `--cpu`
- Disable mixed precision: `--no-amp`
- Reduce number of workers: Modify `num_workers` in Python API
- Process fewer meshes at a time in batch mode

### "Blender conversion failed"

**Problem:** FBX to OBJ conversion fails

**Solutions:**
- Ensure Blender is installed
- Specify Blender path: `--blender /path/to/blender`
- Check Blender version (4.0+ recommended)
- Verify FBX file is valid

### "No textures found"

**Problem:** Shape-only morphing when textures expected

**Solutions:**
- Verify OBJ files have corresponding .mtl and texture files
- Ensure texture paths in .mtl are correct
- Use FBX format which embeds textures
- Check texture file permissions

---

## Next Steps

- See [heatmaps.md](heatmaps.md) for detailed heatmap methodology
- See [installation.md](installation.md) for installation troubleshooting
- Check GitHub issues for known problems and solutions
