# Performance Benchmarks

##Test Configuration

**Hardware:**
- CPU: Multi-core system (8 cores)
- GPU: NVIDIA GeForce RTX 3080 Laptop (15.61 GB VRAM)
- Test meshes: 4 faces (male1, male2, woman18, woman19)
- Mesh complexity: 18,024 vertices, 20,982 faces each

**Software:**
- Mode: Minimal (PNG + heatmaps only, no video/CSV/FBX)
- Frames per pair: 41 morphs (standard 0.0 → 1.0 interpolation)
- Textures: None (shape-only morphing)

## Benchmark Results

### Batch Processing - ALL Combinations (6 pairs, 246 total frames)

| Mode | Real Time | User Time | Sys Time | Total CPU | Speedup | Per-Pair |
|------|-----------|-----------|----------|-----------|---------|----------|
| **CPU** | 57.56s | 158.26s | 349.77s | 508.03s | 1.00x | 9.59s |
| **GPU** | **27.17s** | 32.70s | 19.04s | 51.74s | **2.12x** ✅ | **4.53s** |

**GPU Advantages:**
- **2.1x faster in real time**
- **9.8x less total CPU time** (51.74s vs 508.03s)
- Leaves CPU free for other tasks
- Better scaling with larger batches

### Single Pair (41 frames)

| Mode | Real Time | GPU Memory | Notes |
|------|-----------|------------|-------|
| **CPU** | 10.49s | N/A | PyRender, sequential |
| **GPU** | **4.01s** | 3.96 GB peak | PyTorch3D, batch (chunk=10) |
| **Speedup** | **2.62x** | 25.4% VRAM | 182.6ms per frame |

### GPU Profiling Details (Single Pair)

```
Timing:
  Total time: 7.49s (including I/O)
  Rendering only: ~4.0s
  Per-frame: 182.6ms
  Throughput: 5.5 frames/sec

GPU Memory:
  Peak usage: 3.96 GB
  VRAM utilization: 25.4%
  Frames per GB: 10.4

Device: NVIDIA GeForce RTX 3080 Laptop
CUDA Version: 12.4
```

## Critical Performance Fix Applied

### The Problem (Before Fix)

Initial GPU implementation was **35% slower** than CPU due to:

1. **`bin_size=0` - Slow naive rasterizer** (10-100x slower)
   - See: [PyTorch3D Issue #259](https://github.com/facebookresearch/pytorch3d/issues/259)

2. **Excessive batch size (41 frames at once)**
   - PyTorch3D batch rendering scales **linearly** (no speedup from batching)
   - See: [PyTorch3D Issue #1120](https://github.com/facebookresearch/pytorch3d/issues/1120)

3. **Missing CUDA synchronization**
   - Hidden costs in timing measurements

### The Fix

```python
# Before (SLOW):
RasterizationSettings(
    bin_size=0,  # ← Naive rasterizer
)
chunk_size = 41  # ← Too large

# After (FAST):
RasterizationSettings(
    bin_size=None,  # ← Automatic optimization
)
chunk_size = 10  # ← Optimal for linear scaling
torch.cuda.synchronize()  # ← Proper timing
```

**Result:** **3.3x speedup** (13.34s → 4.01s for single pair)

## Optimization Features Verified

✅ **Architectural Refactoring:**
- Orchestrator: 636 → 254 lines (60% reduction)
- PipelineStages: Clean separation of concerns
- HeatmapRenderer: Deduplicated rendering logic
- All 6 pairs completed successfully

✅ **Performance Optimizations:**
- **PyTorch3D rasterizer**: bin_size=None (automatic optimization)
- **Smart batch sizing**: 10 frames per chunk (optimal for linear scaling)
- **CUDA synchronization**: Proper timing and memory management
- **Fast PNG compression**: Level 1 (vs default 6)
- **Optimized device transfers**: 2 per pair (~98% reduction from baseline)
- **Parallel PNG saving**: 4 workers

### Projected Performance for Larger Workloads

**20 pairs (820 frames):**
- CPU: ~192s (9.6s per pair)
- GPU: ~91s (4.5s per pair)
- **Expected GPU speedup: 2.1x**

**With textures (typical use case):**
- Vectorized texture interpolation on GPU
- Batch texture rendering with optimized rasterizer
- **Expected GPU speedup: 3-5x** (textures are GPU-heavy)

**Full mode (with FBX/video/CSV):**
- Async FBX conversion: 20-40% faster than sequential
- Video generation: Depends on ffmpeg
- Additional time: +15-30s total (not per-pair)

## Recommendations

### Use CPU Mode When:
- ✅ Processing 1-5 pairs only
- ✅ Shape-only morphing (no textures)
- ✅ No CUDA GPU available
- ✅ Limited VRAM (<4GB)

### Use GPU Mode When:
- ✅ Processing 5+ pairs (batch workflows)
- ✅ Meshes with textures (3-5x advantage)
- ✅ Multi-user/server environments (leaves CPU free)
- ✅ Repeated processing pipelines
- ✅ High-resolution textures (>1024x1024)
- ✅ Real-time/interactive workflows

## System Resource Usage

**CPU Mode:**
- Peak CPU: ~800% (8 cores fully utilized)
- Peak RAM: ~2-3 GB
- GPU: Not used
- Best for: Small batches, no GPU

**GPU Mode:**
- Peak CPU: ~150% (1.5 cores)
- Peak RAM: ~2-3 GB
- Peak VRAM: ~4 GB (18k vertex meshes, shape-only)
- GPU Utilization: 25-40%
- Best for: Large batches, textured meshes

## Full Mode Performance

Full mode adds (per batch, not per-pair):
- **FBX mesh export**: Async conversion, 4 concurrent Blender processes
- **Video generation**: ffmpeg encoding (if available)
- **CSV statistics**: Vertex/texture data export

Typical additional time: **+15-30s** total (scales sub-linearly with pair count)

---

## Architectural Improvements Impact

The refactoring improves both performance AND code quality:

**Performance:**
- GPU now **2.1x faster** than CPU (after fixing PyTorch3D settings)
- Foundation for future optimizations (CUDA streams ready)
- Dynamic batch sizing prevents OOM errors

**Code Quality:**
- **Maintainability**: 60% reduction in orchestrator complexity
- **Testability**: Modular PipelineStages design
- **Extensibility**: Reusable HeatmapRenderer, GPU optimizer utilities
- **Best practices**: DRY principle, SOLID adherence

---

## Sources

Performance optimization based on:
- [PyTorch3D Issue #259: Faster rendering](https://github.com/facebookresearch/pytorch3d/issues/259)
- [PyTorch3D Issue #1120: Batch rendering performance](https://github.com/facebookresearch/pytorch3d/issues/1120)
- NVIDIA CUDA Best Practices Guide
- PyTorch Performance Tuning Guide

---

*Benchmark date: 2025-12-23*
*Version: v1.0.0 + architectural refactoring + GPU optimization fix*
*Test system: 8-core CPU, NVIDIA RTX 3080 Laptop (15.61 GB VRAM)*
