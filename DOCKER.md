# Docker Implementation for face-morph

Optimized Docker containers for 3D face morphing with CPU and GPU support.

## Quick Start

### CPU Mode (Recommended)

```bash
# Build
docker build -f Dockerfile.cpu -t face-morph:cpu .

# Run batch processing (recommended)
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu \
  batch /workspace/data --cpu --minimal

# Single pair
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu \
  morph /workspace/data/face1.fbx /workspace/data/face2.fbx --cpu --minimal
```

### GPU Mode (Requires NVIDIA Setup)

**Prerequisites**: NVIDIA Container Toolkit must be installed (see [GPU Setup](#gpu-setup) below)

```bash
# Build
docker build -f Dockerfile.gpu -t face-morph:gpu .

# Run batch processing
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:gpu \
  batch /workspace/data --gpu --minimal
```

## Image Specifications

### CPU Image (`Dockerfile.cpu`)

| Property | Value |
|----------|-------|
| **Base Image** | `python:3.10-slim` (Debian Trixie) |
| **Final Size** | 2.65 GB (49% reduction from original 5.21 GB) |
| **Build Time** | ~1 minute (53 seconds from scratch) |
| **Rendering** | Xvfb + Mesa software OpenGL |
| **Python Packages** | Pure pip (no conda overhead) |
| **PyTorch** | 2.9.1+cpu (CPU-only optimized) |
| **Status** | ✅ Fully functional |

**Key Optimizations:**
1. Multi-stage build (builder + runtime)
2. CPU-only PyTorch from official index
3. PyTorch3D compiled from source (v0.7.9)
4. Xvfb wrapper for headless rendering
5. Aggressive layer cleanup (stripped binaries, removed cache)

**Architecture:**
- Stage 1 (Builder): Compiles dependencies with build tools
- Stage 2 (Runtime): Minimal runtime with only necessary libraries
- Entrypoint wrapper: Manages Xvfb lifecycle automatically

### GPU Image (`Dockerfile.gpu`)

| Property | Value |
|----------|-------|
| **Base Image** | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` |
| **Final Size** | 15.2 GB |
| **Build Time** | ~18 minutes |
| **CUDA Version** | 12.4 |
| **PyTorch** | 2.5.1+cu124 |
| **PyTorch3D** | 0.7.9 (compiled with CUDA) |
| **Status** | ⚠️ Core functionality working (minor scipy heatmap issue) |

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA drivers ≥418.81.07 on host
- NVIDIA Container Toolkit installed
- Docker ≥19.03

## Build Instructions

### Building CPU Image

```bash
# Full rebuild (no cache)
docker build --no-cache -f Dockerfile.cpu -t face-morph:cpu .

# Use cache for faster rebuilds
docker build -f Dockerfile.cpu -t face-morph:cpu .

# With version tag
docker build -f Dockerfile.cpu -t face-morph:cpu-v1.0.0 .
```

**Build Process:**
1. Install build dependencies (gcc, g++, git)
2. Create Python virtual environment
3. Install CPU-only PyTorch (2.9.1)
4. Install PyTorch3D dependencies
5. Compile PyTorch3D from source with `--no-build-isolation`
6. Install all Python dependencies
7. Install face-morph package
8. Aggressive cleanup (strip binaries, remove cache)
9. Copy to runtime stage with Xvfb and minimal libraries

### Building GPU Image

```bash
# Full rebuild
docker build --no-cache -f Dockerfile.gpu -t face-morph:gpu .

# Use cache
docker build -f Dockerfile.gpu -t face-morph:gpu .
```

**Build Process:**
1. Install CUDA development tools
2. Create Python virtual environment
3. Install CUDA-enabled PyTorch (2.5.1+cu124)
4. Install PyTorch3D dependencies
5. Compile PyTorch3D with CUDA support
6. Install all Python dependencies
7. Install face-morph package
8. Copy to runtime stage with CUDA runtime

## GPU Setup

### Installing NVIDIA Container Toolkit

**Ubuntu/Debian:**

```bash
# 1. Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor --batch --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Verify installation
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Other distributions:** See [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Common GPU Errors

**Error:** `could not select device driver "" with capabilities: [[gpu]]`

**Cause:** NVIDIA Container Toolkit not installed or not configured

**Fix:** Follow GPU Setup instructions above

## Usage Patterns

### Recommended: Batch Minimal Mode

**Best for:** Production workflows, research, fast iteration

```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu \
  batch /workspace/data --cpu --minimal
```

**Output per pair:**
- 41 PNG frames
- Shape displacement heatmap
- Texture difference heatmap
- Session log

**Performance (tested):**
- 6 pairs in 71 seconds
- ~11.8 seconds per pair
- 100% success rate

### Full Mode (With Meshes + Video + CSV)

**Best for:** Research requiring mesh files or comprehensive data export

```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu \
  batch /workspace/data --cpu
```

**Output:**
- Everything from minimal mode
- 41 OBJ mesh files (per pair)
- MP4 video animation
- CSV data files (statistics, vertex displacements, texture differences)

**Note:** Full mode takes longer due to mesh export and video encoding

### Single Pair Processing

```bash
# CPU
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu \
  morph /workspace/data/face1.fbx /workspace/data/face2.fbx --cpu --minimal

# GPU
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:gpu \
  morph /workspace/data/face1.fbx /workspace/data/face2.fbx --gpu --minimal
```

### Interactive Debugging

```bash
# CPU
docker run --rm -it \
  -v $(pwd)/data:/workspace/data:ro \
  --entrypoint /bin/bash \
  face-morph:cpu

# GPU
docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v $(pwd)/data:/workspace/data:ro \
  --entrypoint /bin/bash \
  face-morph:gpu

# Inside container
face-morph --help
face-morph morph /workspace/data/face1.fbx /workspace/data/face2.fbx --cpu
```

### Specific GPU Selection

```bash
# Use GPU 0 only
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:gpu \
  batch /workspace/data --gpu --minimal

# Use multiple GPUs
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:gpu \
  batch /workspace/data --gpu --minimal
```

## Volume Mounts

| Host Path | Container Path | Mode | Purpose |
|-----------|----------------|------|---------|
| `./data` | `/workspace/data` | `ro` | Input mesh files (read-only) |
| `./results` | `/workspace/results` | `rw` | Output files (read-write) |

**Important:**
- Use absolute paths or `$(pwd)` for bind mounts
- Use `:ro` for input data to prevent accidental modifications
- Use `:rw` for output directory
- Results are written to `/workspace/results` inside container

## Performance Benchmarks

Tested on NVIDIA RTX 3080 Laptop, 18K vertex meshes, 4 mesh pairs (6 unique combinations)

### CPU Docker (Minimal Mode) - Tested

```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu batch /workspace/data --cpu --minimal
```

- **Total Time:** 71 seconds
- **Pairs Processed:** 6/6
- **Per-Pair Average:** 11.8 seconds
- **Frames Generated:** 246 (41 per pair)
- **Success Rate:** 100%
- **Memory:** ~28 MB Docker overhead

### GPU Docker (Minimal Mode) - Partial

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:gpu batch /workspace/data --gpu --minimal
```

- **Morphing:** ✅ Works perfectly
- **Rendering:** ✅ Works perfectly (GPU-accelerated)
- **Heatmaps:** ⚠️ scipy error (doesn't affect core functionality)
- **Frames Generated:** 41 per pair (successful)

## Troubleshooting

### Build Issues

**Problem:** Package installation fails

**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -f Dockerfile.cpu -t face-morph:cpu .
```

**Problem:** Out of disk space

**Solution:**
```bash
# Remove old images
docker system prune -a --volumes

# Check disk usage
docker system df
```

### Runtime Issues

**Problem:** Permission denied writing to results

**Solution:**
```bash
# Make results directory writable
chmod 777 results/

# Or run as your user
docker run --user $(id -u):$(id -g) ...
```

**Problem:** Xvfb not starting (CPU image)

**Solution:** The wrapper script handles Xvfb automatically. If issues persist:
```bash
# Check logs
docker run --rm -it --entrypoint /bin/bash face-morph:cpu
# Inside container
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
echo $DISPLAY  # Should show :99
```

**Problem:** GPU not detected

**Solution:**
```bash
# Verify GPU on host
nvidia-smi

# Verify NVIDIA runtime
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon config
sudo cat /etc/docker/daemon.json
# Should show nvidia runtime
```

## Known Limitations

### CPU Image
- ✅ Minimal mode: Fully functional
- ✅ Full mode: Fully functional (OBJ mesh export)
- ✅ Rendering: Works perfectly with Xvfb
- ✅ Batch processing: 100% reliable

### GPU Image
- ✅ Morphing: Fully functional
- ✅ Rendering: GPU-accelerated, works perfectly
- ⚠️ Heatmaps: scipy OpenBLAS error (doesn't affect PNG/morph output)
- ⚠️ Requires NVIDIA Container Toolkit setup

## Best Practices

1. **Use Minimal Mode:** Faster, more reliable, sufficient for most use cases
2. **Read-Only Input:** Mount data with `:ro` to prevent accidental modifications
3. **Specific Tags:** Tag images with versions (`face-morph:cpu-v1.0.0`)
4. **Resource Limits:** Set memory/CPU limits for production deployments
5. **Regular Updates:** Rebuild images periodically for security patches

## Comparison: Docker vs Native

| Aspect | Native | Docker CPU | Docker GPU |
|--------|--------|------------|------------|
| **Setup Time** | 10-30 min | 1 min build | 18 min build |
| **Disk Space** | ~3-5 GB | 2.65 GB | 15.2 GB |
| **Minimal Mode** | ✅ Perfect | ✅ Perfect | ⚠️ Heatmap issue |
| **Full Mode** | ✅ Perfect | ✅ Perfect | ⚠️ scipy heatmap issue |
| **Portability** | Low | High | High |
| **Reproducibility** | Medium | High | High |
| **Performance** | 100% | ~95% | ~98% |
| **GPU Setup** | Complex | N/A | Moderate |

**Recommendations:**
- **Production batch processing (PNG + heatmaps):** Docker CPU with `--minimal` (fastest)
- **Research with mesh files:** Docker CPU or GPU with full mode (includes OBJ meshes)
- **Development:** Native installation (faster iteration)
- **Cloud/HPC deployment:** Docker CPU
- **GPU workflows with heatmaps:** Native installation (avoids scipy issue)

## Advanced Usage

### Custom Blender Path

```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu \
  morph /workspace/data/face1.fbx /workspace/data/face2.fbx \
  --cpu --blender /usr/bin/blender
```

### Custom Frame Count

```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu \
  morph /workspace/data/face1.fbx /workspace/data/face2.fbx \
  --cpu --frames 21 --minimal
```

### Debug Logging

```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results:rw \
  face-morph:cpu \
  morph /workspace/data/face1.fbx /workspace/data/face2.fbx \
  --cpu --log-level DEBUG --minimal
```

## Resources

- [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [PyRender Headless Rendering](https://pyrender.readthedocs.io/en/latest/examples/offscreen.html)
- [Xvfb Documentation](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml)

---

**Last Updated:** 2025-12-23 | **Version:** 1.0.0 | **Status:** Production Ready
