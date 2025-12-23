"""
CSV Export Utilities
====================

Single responsibility: Export quantitative morphing data to CSV format.

This module enables quantitative analysis in external tools (Excel, R, Python, MATLAB).
Per-vertex and per-pixel data allows custom visualizations and statistical comparisons.
"""

import csv
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


def export_statistics_csv(
    normal_disp: torch.Tensor,
    tangent_disp: torch.Tensor,
    total_disp: torch.Tensor,
    luminance_diff: Optional[np.ndarray],
    chroma_diff: Optional[np.ndarray],
    delta_e: Optional[np.ndarray],
    output_path: Path
) -> None:
    """
    Export summary statistics for all displacement components to CSV.

    Computes and saves descriptive statistics (mean, std, min, max, percentiles)
    for shape and texture differences, enabling quantitative analysis and
    comparison across different morph pairs.

    Args:
        normal_disp: Normal displacement tensor (V,) - signed values
        tangent_disp: Tangent displacement tensor (V,) - unsigned magnitudes
        total_disp: Total displacement tensor (V,) - unsigned magnitudes
        luminance_diff: Luminance difference array (H, W) or None
        chroma_diff: Chrominance difference array (H, W) or None
        delta_e: Perceptual difference array (H, W) or None
        output_path: Path to save CSV file

    Output CSV Format:
        metric,component,mean,std,min,max,p25,p50,p75,p95,p99
        shape,normal_displacement,0.0234,0.0156,-0.052,0.089,0.012,0.021,0.034,0.058,0.072
        shape,tangent_displacement,0.0189,0.0123,0.000,0.067,0.009,0.017,0.026,0.045,0.059
        ...

    Example:
        >>> export_statistics_csv(normal, tangent, total, lum, chroma, de, out_path)
        # Creates statistics.csv with summary metrics
    """
    logger.info(f"Exporting statistics to {output_path}")

    def compute_stats(data: np.ndarray, name: str, category: str) -> Optional[dict]:
        """
        Compute descriptive statistics for a data array.

        Args:
            data: Numpy array to analyze
            name: Component name (e.g., 'normal_displacement')
            category: Category (e.g., 'shape' or 'texture')

        Returns:
            Dictionary with statistics or None if no valid data
        """
        data_flat = data.flatten()
        # Remove NaN/Inf values
        data_clean = data_flat[np.isfinite(data_flat)]

        if len(data_clean) == 0:
            logger.warning(f"No valid data for {category}/{name}")
            return None

        return {
            'metric': category,
            'component': name,
            'mean': float(np.mean(data_clean)),
            'std': float(np.std(data_clean)),
            'min': float(np.min(data_clean)),
            'max': float(np.max(data_clean)),
            'p25': float(np.percentile(data_clean, 25)),
            'p50': float(np.percentile(data_clean, 50)),  # median
            'p75': float(np.percentile(data_clean, 75)),
            'p95': float(np.percentile(data_clean, 95)),
            'p99': float(np.percentile(data_clean, 99)),
        }

    # Collect all statistics
    stats_rows = []

    # Shape statistics (always available)
    stats_rows.append(compute_stats(normal_disp.cpu().numpy(), 'normal_displacement', 'shape'))
    stats_rows.append(compute_stats(tangent_disp.cpu().numpy(), 'tangent_displacement', 'shape'))
    stats_rows.append(compute_stats(total_disp.cpu().numpy(), 'total_displacement', 'shape'))

    # Texture statistics (only if available)
    if luminance_diff is not None:
        stats_rows.append(compute_stats(luminance_diff, 'luminance_difference', 'texture'))
        stats_rows.append(compute_stats(chroma_diff, 'chrominance_difference', 'texture'))
        stats_rows.append(compute_stats(delta_e, 'perceptual_difference', 'texture'))

    # Remove None entries
    stats_rows = [row for row in stats_rows if row is not None]

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['metric', 'component', 'mean', 'std', 'min', 'max', 'p25', 'p50', 'p75', 'p95', 'p99']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    logger.info(f"Exported {len(stats_rows)} statistics to {output_path}")


def export_vertex_data_csv(
    mesh_a,
    mesh_b,
    normal_disp: torch.Tensor,
    tangent_disp: torch.Tensor,
    total_disp: torch.Tensor,
    output_path: Path
) -> None:
    """
    Export per-vertex displacement data to CSV for detailed analysis.

    Saves vertex positions and all displacement components, enabling:
    - Identification of high-variance regions
    - Correlation analysis between displacement types
    - Custom visualization in external tools (ParaView, Blender, etc.)

    Args:
        mesh_a: First input mesh (PyTorch3D Meshes object)
        mesh_b: Second input mesh (PyTorch3D Meshes object)
        normal_disp: Normal displacement per vertex (V,)
        tangent_disp: Tangent displacement per vertex (V,)
        total_disp: Total displacement per vertex (V,)
        output_path: Path to save CSV file

    Output CSV Format:
        vertex_id,x_a,y_a,z_a,x_b,y_b,z_b,normal_disp,tangent_disp,total_disp
        0,0.123,0.456,0.789,0.125,0.458,0.791,0.0023,0.0015,0.0028
        1,0.234,0.567,0.890,0.236,0.569,0.892,0.0025,0.0018,0.0030
        ...

    Example:
        >>> export_vertex_data_csv(mesh1, mesh2, normal, tangent, total, out_path)
        # Creates vertex_displacements.csv with 10000 rows (one per vertex)
    """
    logger.info(f"Exporting vertex data to {output_path}")

    # Get vertex positions
    verts_a = mesh_a.verts_packed().cpu().numpy()  # (V, 3)
    verts_b = mesh_b.verts_packed().cpu().numpy()  # (V, 3)

    # Convert displacement tensors to numpy
    normal_np = normal_disp.cpu().numpy()
    tangent_np = tangent_disp.cpu().numpy()
    total_np = total_disp.cpu().numpy()

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['vertex_id', 'x_a', 'y_a', 'z_a', 'x_b', 'y_b', 'z_b',
                      'normal_disp', 'tangent_disp', 'total_disp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        num_verts = len(verts_a)
        for i in range(num_verts):
            writer.writerow({
                'vertex_id': i,
                'x_a': float(verts_a[i, 0]),
                'y_a': float(verts_a[i, 1]),
                'z_a': float(verts_a[i, 2]),
                'x_b': float(verts_b[i, 0]),
                'y_b': float(verts_b[i, 1]),
                'z_b': float(verts_b[i, 2]),
                'normal_disp': float(normal_np[i]),
                'tangent_disp': float(tangent_np[i]),
                'total_disp': float(total_np[i]),
            })

    logger.info(f"Exported {num_verts} vertices to {output_path}")


def export_texture_data_csv(
    luminance_diff: np.ndarray,
    chroma_diff: np.ndarray,
    delta_e: np.ndarray,
    output_path: Path,
    downsample_factor: int = 4
) -> None:
    """
    Export per-pixel texture difference data to CSV.

    For large textures (e.g., 2048x2048 = 4M pixels), this can generate huge CSVs.
    By default, downsamples by factor of 4 (512x512 = 262k rows), and skips
    background pixels (all zeros) to further reduce file size.

    Args:
        luminance_diff: Luminance difference array (H, W)
        chroma_diff: Chrominance difference array (H, W)
        delta_e: Perceptual difference array (H, W)
        output_path: Path to save CSV file
        downsample_factor: Downsample factor (1=full resolution, 4=1/16 pixels)

    Output CSV Format:
        pixel_x,pixel_y,luminance_diff,chroma_diff,delta_e
        0,0,5.23,12.45,8.67
        0,1,4.89,11.23,7.92
        ...

    Example:
        >>> export_texture_data_csv(lum, chroma, de, out_path, downsample_factor=4)
        # Creates texture_differences.csv with downsampled texture metrics
    """
    logger.info(f"Exporting texture data to {output_path} (downsample={downsample_factor})")

    # Downsample if needed
    if downsample_factor > 1:
        luminance_diff = luminance_diff[::downsample_factor, ::downsample_factor]
        chroma_diff = chroma_diff[::downsample_factor, ::downsample_factor]
        delta_e = delta_e[::downsample_factor, ::downsample_factor]

    H, W = luminance_diff.shape

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['pixel_x', 'pixel_y', 'luminance_diff', 'chroma_diff', 'delta_e']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        rows_written = 0
        for y in range(H):
            for x in range(W):
                # Skip background pixels (all zeros) to reduce file size
                if (luminance_diff[y, x] == 0 and
                    chroma_diff[y, x] == 0 and
                    delta_e[y, x] == 0):
                    continue

                writer.writerow({
                    'pixel_x': x * downsample_factor,
                    'pixel_y': y * downsample_factor,
                    'luminance_diff': float(luminance_diff[y, x]),
                    'chroma_diff': float(chroma_diff[y, x]),
                    'delta_e': float(delta_e[y, x]),
                })
                rows_written += 1

    logger.info(f"Exported {rows_written} texture pixels to {output_path}")
