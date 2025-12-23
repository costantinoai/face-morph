# Heatmap Color Interpretation Guide

This guide explains exactly how to interpret the colors in each heatmap component.

---

## Colormap Convention

**Two colormaps used consistently:**

1. **coolwarm** (diverging) - for signed values centered at 0
   - Blue = negative values
   - White = zero (no change)
   - Red = positive values

2. **viridis** (sequential) - for magnitudes starting at 0
   - Dark purple = zero (no change)
   - Green/cyan = medium values
   - Bright yellow = maximum values

---

## Shape Displacement Components

### 1. Normal Component

**Type:** Signed (diverging colormap: coolwarm)

**What it measures:** Displacement perpendicular to the surface (depth changes)

**Color Interpretation:**
- 🔵 **Blue**: Surface moved **inward** (contraction, thinner, recessed)
  - Example: Sunken cheeks, recessed chin
- ⚪ **White**: No normal displacement (stayed at same depth)
- 🔴 **Red**: Surface moved **outward** (expansion, fuller, protruding)
  - Example: Fuller forehead, protruding nose

**Colorbar:** [-max, 0, +max] centered at white

**Key insight:** Shows if facial features became fuller or thinner

---

### 2. Tangential Component

**Type:** Magnitude (sequential colormap: viridis)

**What it measures:** Displacement parallel to the surface (positional changes)

**Color Interpretation:**
- 🟣 **Dark purple**: No tangential movement (feature stayed in place)
- 🟢 **Green/Cyan**: Medium positional change
- 🟡 **Yellow**: Large positional change (feature repositioned significantly)
  - Example: Eyebrows raised, mouth corners moved, nose shifted sideways

**Colorbar:** [0, max] starting at dark purple

**Key insight:** Shows where features moved along the surface (e.g., eyebrow height changes, facial asymmetry changes)

---

### 3. Total Component

**Type:** Magnitude (sequential colormap: viridis)

**What it measures:** Total 3D Euclidean displacement (combines normal + tangential)

**Color Interpretation:**
- 🟣 **Dark purple**: No movement at all (vertex didn't move)
- 🟢 **Green/Cyan**: Medium total displacement
- 🟡 **Yellow**: Large total displacement (vertex moved far in 3D space)

**Colorbar:** [0, max] starting at dark purple

**Key insight:** Shows overall geometric differences regardless of direction. Combines depth and positional changes.

---

## Texture Difference Components

### 1. Luminance Component

**Type:** Signed (diverging colormap: coolwarm)

**What it measures:** Brightness difference in LAB color space (L channel)

**Color Interpretation:**
- 🔵 **Blue**: Face A is **brighter/lighter** in this region
  - Example: Lighter skin tone, more illuminated, lighter makeup
- ⚪ **White**: Same brightness (no luminance difference)
- 🔴 **Red**: Face B is **brighter/lighter** in this region
  - Example: Face B has lighter skin or different lighting

**Colorbar:** [-max, 0, +max] centered at white

**Key insight:** Shows brightness differences independent of color. Useful for identifying lighting or skin tone brightness changes.

---

### 2. Chrominance Component

**Type:** Magnitude (sequential colormap: viridis)

**What it measures:** Color/hue difference (a,b channels in LAB space)

**Color Interpretation:**
- 🟣 **Dark purple**: No color difference (same hue and saturation)
  - Faces have identical color even if brightness differs
- 🟢 **Green/Cyan**: Medium color difference
- 🟡 **Yellow**: Large color difference (very different hues/saturation)
  - Example: Different skin undertones (warm vs cool), different makeup colors, sunburn vs pale

**Colorbar:** [0, max] starting at dark purple

**Key insight:** Shows pure color differences independent of brightness. Highlights skin tone, makeup, or tanning differences.

---

### 3. ΔE (Total) Component

**Type:** Magnitude (sequential colormap: viridis)

**What it measures:** Perceptual color difference using CIEDE2000 standard

**Color Interpretation:**
- 🟣 **Dark purple**: Perceptually identical color (ΔE < 1)
  - Human eye cannot distinguish the difference
- 🟢 **Green/Cyan**: Barely to moderately noticeable (ΔE ≈ 2-10)
- 🟡 **Yellow**: Very different, easily visible (ΔE > 10)

**Colorbar:** [0, max] starting at dark purple

**Perceptual Thresholds:**
- ΔE < 1.0: Imperceptible (not detectable by human eye)
- ΔE 1.0-2.0: Barely perceptible (only noticeable on close inspection)
- ΔE 2.0-10.0: Noticeable difference
- ΔE > 10.0: Very different colors

**Key insight:** Industry-standard metric for "how different do these colors look to a human?" Combines luminance and chrominance into perceptually-meaningful units.

---

## Quick Reference Table

| Component | Colormap | Dark Purple | White | Green | Yellow | Blue | Red |
|-----------|----------|-------------|-------|-------|--------|------|-----|
| **Normal** | coolwarm | - | No change | - | - | Inward | Outward |
| **Tangential** | viridis | No movement | - | Medium | Large | - | - |
| **Total** | viridis | No movement | - | Medium | Large | - | - |
| **Luminance** | coolwarm | - | No change | - | - | A brighter | B brighter |
| **Chrominance** | viridis | Same color | - | Medium diff | Large diff | - | - |
| **ΔE** | viridis | Identical | - | Noticeable | Very different | - | - |

---

## Example Interpretations

### Shape Example: Male1 vs Male2

**Normal (coolwarm):**
- Red forehead → Male2 has fuller/more prominent forehead
- Blue chin → Male2 has more recessed chin
- White nose bridge → Same depth, even if nose moved sideways

**Tangential (viridis):**
- Yellow on entire face → Features are repositioned significantly (different face proportions)
- Shows that differences aren't just "deeper/shallower" but actual repositioning

**Total (viridis):**
- Yellow everywhere → Large overall geometric differences
- Combines both depth and positional changes

### Texture Example: Male1 vs Male2

**Luminance (coolwarm):**
- Mostly white → Similar brightness/lighting
- Slight red/blue patches → Minor lighting or skin brightness differences

**Chrominance (viridis):**
- Yellow/green everywhere → Very different skin tones/colors
- Shows faces have different hue/saturation even if brightness is similar

**ΔE (viridis):**
- Yellow (high values) → Colors look very different to human perception
- Combines both brightness and color into "how different does it look?"

---

## Tips for Interpretation

1. **Compare components together:**
   - If Normal is colorful but Tangential is dark → Depth changes only (fuller/thinner)
   - If Tangential is colorful but Normal is white → Repositioning only (features moved)
   - If both are colorful → Complex changes (both depth and position)

2. **Texture components:**
   - If Luminance is colorful but Chrominance is dark → Brightness differs, color is same
   - If Chrominance is colorful but Luminance is white → Color differs, brightness is same
   - ΔE combines both into one perceptual metric

3. **Use colorbar values:**
   - Look at the colorbar to see actual magnitudes
   - Small max values = subtle differences
   - Large max values = dramatic differences

4. **Dark regions mean "no difference":**
   - In viridis: dark purple = 0 = no change
   - In coolwarm: white = 0 = no change
   - Focus on bright/saturated colors to see where changes occur
