# Heatmap Calculations and Measurements

This document explains exactly how the shape variance and texture difference heatmaps are calculated, plotted, and what they measure.

---

## 1. Shape Variance Heatmap

### What It Measures
The **shape variance heatmap** visualizes how much each vertex on the 3D mesh **moves** (displaces) between the two input faces (Mesh A and Mesh B).

### Calculation Method

**Input:**
- `V_A`: Vertex positions of Mesh A (shape: [V, 3] where V = number of vertices)
- `V_B`: Vertex positions of Mesh B (shape: [V, 3])

**Steps:**

1. **Compute Per-Vertex Displacement** (Euclidean distance):
   ```
   displacement = ||V_B - V_A||₂ for each vertex
   ```
   This gives a scalar value for each vertex representing the L2 distance (straight-line distance in 3D space) that the vertex moved.

2. **Result:**
   ```
   shape_variance: [V,] array of displacement magnitudes
   ```

3. **Normalization for Visualization:**
   - Find `vmin = min(shape_variance)`
   - Find `vmax = 95th percentile(shape_variance)` ← Uses 95th percentile to avoid outliers
   - Normalize: `normalized = (shape_variance - vmin) / (vmax - vmin)`
   - Clip values above 95th percentile to 1.0

4. **Color Mapping:**
   - Apply colormap (default: 'hot') to normalized values
   - hot colormap: Black (0.0) → Red → Orange → Yellow → White (1.0)

### Interpretation
- **Dark/Black regions**: Minimal shape change (vertices barely moved)
- **Red/Orange regions**: Moderate shape change
- **Yellow/White regions**: Large shape change (vertices moved significantly)

**Example:** If the jaw structure differs significantly between faces, the jaw vertices will show high variance (yellow/white), while shared features like the forehead might show low variance (dark red/black).

---

## 2. Texture Difference Heatmap

### What It Measures
The **texture difference heatmap** visualizes the **signed color difference** between the texture maps of the two faces, showing which face is brighter/darker in each region.

### Calculation Method

**Input:**
- `T_A`: Texture map of Mesh A (shape: [H, W, 3] in RGB, range [0, 1])
- `T_B`: Texture map of Mesh B (shape: [H, W, 3] in RGB, range [0, 1])

**Steps:**

1. **Compute Signed Difference:**
   ```
   diff = T_B - T_A  (element-wise, per pixel, per channel)
   ```
   - Positive values: T_B is brighter than T_A
   - Negative values: T_A is brighter than T_B

2. **Average Across RGB Channels:**
   ```
   diff_map = mean(diff, axis=-1)  # [H, W]
   ```
   This gives a single scalar per pixel representing the average brightness difference.

3. **Mask Background Pixels:**
   ```
   background = (mean(T_A, axis=-1) < 0.1) AND (mean(T_B, axis=-1) < 0.1)
   diff_map[background] = 0.0
   ```
   Pixels that are nearly black in both textures (unused texture space) are set to 0.

4. **Map to Mesh Vertices:**
   Using the UV coordinates, each vertex samples the difference value from `diff_map` at its UV location.
   ```
   vertex_diff = sample(diff_map, UV_coords)  # [V,]
   ```

5. **Symmetric Normalization (Diverging Colormap):**
   - Find `vmax_abs = max(|min(vertex_diff)|, |max(vertex_diff)|)`
   - Create symmetric range: `[-vmax_abs, +vmax_abs]`
   - Normalize to [0, 1]: `normalized = (vertex_diff / vmax_abs + 1.0) / 2.0`
     - -vmax_abs → 0.0 (full blue)
     - 0.0 → 0.5 (white)
     - +vmax_abs → 1.0 (full red)

6. **Color Mapping:**
   - Apply diverging colormap (currently: 'coolwarm')
   - coolwarm: Blue (negative) → White (0) → Red (positive)

### Interpretation
- **Blue regions**: Face A is brighter than Face B (negative difference)
- **White/Light regions**: No difference or minimal difference (≈0)
- **Red regions**: Face B is brighter than Face A (positive difference)

**Example:** If Face A has lighter skin tone around the forehead and Face B has darker skin there, the forehead will show blue (A brighter). If Face B has more pronounced shadows under the eyes, those areas will show red (B darker, so difference is positive in the dark direction... wait, this depends on whether we're measuring brightness or darkness).

**Important Note:** The difference is calculated as `T_B - T_A`, so:
- Positive (Red): Face B is brighter/lighter
- Negative (Blue): Face A is brighter/lighter

---

## 3. Visualization Settings

### Shape Variance Heatmap
- **Colormap**: 'hot' (sequential)
- **Normalization**: Yes, using 95th percentile
- **Range**: [0, 1] normalized
- **Colorbar Label**: "Variance (normalized)"

### Texture Difference Heatmap
- **Colormap**: 'coolwarm' (diverging)
- **Normalization**: Symmetric around 0
- **Range**: Raw signed differences, symmetric [-max, +max]
- **Colorbar Label**: "Difference (B - A)"

### Rendering
Both heatmaps are rendered using:
- **CPU mode**: PyRender with OpenGL (fast, unlit vertex colors)
- **GPU mode**: PyTorch3D with unlit lighting (full white ambient, no diffuse/specular) to preserve exact vertex colors without lighting artifacts

---

## 4. Key Differences Between the Two Heatmaps

| Aspect | Shape Variance | Texture Difference |
|--------|---------------|-------------------|
| **Measures** | 3D geometry displacement | 2D texture color difference |
| **Input** | Vertex positions (x, y, z) | Texture RGB values |
| **Metric** | L2 distance (Euclidean norm) | Signed average RGB difference |
| **Output Type** | Magnitude (always ≥ 0) | Signed difference (can be negative) |
| **Colormap** | Sequential ('hot') | Diverging ('coolwarm') |
| **Normalization** | Percentile-based (0-1) | Symmetric around 0 |
| **Interpretation** | How much vertices moved | Which face is brighter/darker |

---

## 5. Mathematical Formulas

### Shape Variance
```
For each vertex i:
  displacement_i = sqrt((x_B[i] - x_A[i])² + (y_B[i] - y_A[i])² + (z_B[i] - z_A[i])²)

normalized_i = (displacement_i - min(displacement)) / (p95(displacement) - min(displacement))
color_i = hot_colormap(normalized_i)
```

### Texture Difference
```
For each pixel (u, v):
  diff[u,v] = mean(T_B[u,v] - T_A[u,v])  # Average over R, G, B

For each vertex i with UV coordinate (u_i, v_i):
  vertex_diff_i = sample_bilinear(diff, u_i, v_i)

vmax = max(|min(vertex_diff)|, |max(vertex_diff)|)
normalized_i = (vertex_diff_i / vmax + 1.0) / 2.0  # Map [-vmax, vmax] to [0, 1]
color_i = coolwarm_colormap(normalized_i)
```

---

## 6. Code Locations

- **Shape variance calculation**: `lib/heatmap.py:compute_shape_variance()` (lines 11-44)
- **Texture difference calculation**: `lib/heatmap.py:compute_texture_variance()` (lines 47-91)
- **Shape heatmap visualization**: `lib/heatmap.py:create_shape_variance_visualization()` (lines 171-226)
- **Texture heatmap visualization**: `lib/heatmap.py:create_texture_variance_visualization()` (lines 317-465)
- **Rendering with heatmap (PyTorch3D)**: `lib/renderer.py:render_with_heatmap()` (lines 354-477)
- **Rendering with heatmap (PyRender)**: `lib/renderer_pyrender.py:render_with_heatmap()` (lines 263-388)

---

## 7. Example Output Values

### Sample Shape Variance Statistics (male1 vs male2):
```
Min:  0.003660 (vertices barely moved)
Max:  0.057362 (vertices moved ~5.7cm)
Mean: 0.018481 (average movement ~1.8cm)
95th percentile: 0.036454 (used as vmax for normalization)
```

### Sample Texture Difference Statistics (male1 vs male2):
```
Min:  -0.684303 (male1 much brighter, in [0,1] scale)
Max:  +0.696193 (male2 much brighter, in [0,1] scale)
Mean: +0.020518 (male2 slightly brighter overall)
Symmetric range: [-0.696193, +0.696193]
```

---

## 8. Recent Changes (2025-12-23)

1. **Changed texture difference from absolute to signed**: Now calculates `T_B - T_A` instead of `|T_B - T_A|` to show directionality
2. **Switched to diverging colormap**: Changed from 'RdBu_r' to 'coolwarm' for better contrast on small differences
3. **Disabled normalization for texture difference**: Uses raw signed values with symmetric range around 0
4. **Fixed GPU rendering artifacts**: Implemented unlit rendering mode to preserve exact vertex colors without lighting modifications
