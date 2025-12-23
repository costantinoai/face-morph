"""
Heatmap Rendering Utilities
============================

Single responsibility: Provide reusable heatmap rendering utilities for both PyTorch3D and PyRender.

This module deduplicates common heatmap rendering logic (normalization, colormap application,
colorbar creation) that was previously duplicated across renderer implementations.

Design Pattern: Strategy + Template Method
- Normalization strategies encapsulated in separate methods
- Common workflow defined in template methods
- Renderer-agnostic colormap and colorbar logic
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import cm

from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


class HeatmapRenderer:
    """Reusable heatmap rendering utilities following DRY principle."""

    @staticmethod
    def normalize_values(
        values: Union[np.ndarray, torch.Tensor],
        mode: str = 'percentile',
        percentile: float = 95.0,
        symmetric: bool = False
    ) -> Tuple[Union[np.ndarray, torch.Tensor], float, float]:
        """
        Normalize values to [0, 1] range using various strategies.

        Args:
            values: Input values (numpy array or torch tensor)
            mode: Normalization mode:
                - 'percentile': Clip to percentile and normalize to [0, 1]
                - 'minmax': Simple min-max normalization to [0, 1]
                - 'none': No normalization, just scale to [0, 1] from actual range
            percentile: Percentile value (0-100) for clipping (used if mode='percentile')
            symmetric: If True, use symmetric range around 0 for diverging colormaps

        Returns:
            Tuple of (normalized_values, vmin, vmax)
        """
        is_torch = torch.is_tensor(values)

        if symmetric:
            # Symmetric diverging normalization
            if is_torch:
                vmax_abs = max(abs(values.min().item()), abs(values.max().item()))
                vmin = -vmax_abs
                vmax = vmax_abs

                if vmax_abs > 0:
                    # Map [-vmax_abs, vmax_abs] to [0, 1]
                    normalized = (values / vmax_abs + 1.0) / 2.0
                else:
                    normalized = torch.full_like(values, 0.5)  # All zeros -> center
            else:
                vmax_abs = max(abs(values.min()), abs(values.max()))
                vmin = -vmax_abs
                vmax = vmax_abs

                if vmax_abs > 0:
                    normalized = (values / vmax_abs + 1.0) / 2.0
                else:
                    normalized = np.full_like(values, 0.5)

            logger.debug(f"Symmetric diverging normalization: range=[{vmin:.6f}, {vmax:.6f}]")

        elif mode == 'percentile':
            # Percentile-based normalization (robust to outliers)
            if is_torch:
                vmin = values.min().item()
                vmax = torch.quantile(values, percentile / 100.0).item()

                if vmax > vmin:
                    clipped = torch.clamp(values, vmin, vmax)
                    normalized = (clipped - vmin) / (vmax - vmin)
                else:
                    normalized = torch.zeros_like(values)
            else:
                vmin = values.min()
                vmax = np.percentile(values, percentile)

                if vmax > vmin:
                    clipped = np.clip(values, vmin, vmax)
                    normalized = (clipped - vmin) / (vmax - vmin)
                else:
                    normalized = np.zeros_like(values)

            logger.debug(f"Percentile normalization (p{percentile}): vmin={vmin:.6f}, vmax={vmax:.6f}")

        elif mode == 'minmax' or mode == 'none':
            # Simple min-max normalization
            if is_torch:
                vmin = values.min().item()
                vmax = values.max().item()

                if vmax > vmin:
                    normalized = (values - vmin) / (vmax - vmin)
                else:
                    normalized = torch.full_like(values, 0.5)
            else:
                vmin = values.min()
                vmax = values.max()

                if vmax > vmin:
                    normalized = (values - vmin) / (vmax - vmin)
                else:
                    normalized = np.full_like(values, 0.5)

            logger.debug(f"Min-max normalization: range=[{vmin:.6f}, {vmax:.6f}]")

        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

        return normalized, vmin, vmax

    @staticmethod
    def apply_colormap(
        normalized_values: Union[np.ndarray, torch.Tensor],
        colormap_name: str = 'viridis',
        as_torch: bool = False,
        device: Optional[torch.device] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply matplotlib colormap to normalized values.

        Args:
            normalized_values: Values in [0, 1] range
            colormap_name: Matplotlib colormap name ('viridis', 'jet', 'RdBu_r', etc.)
            as_torch: If True, return torch tensor instead of numpy array
            device: Target device for torch tensor (if as_torch=True)

        Returns:
            RGB colors (N, 3) as numpy array or torch tensor
        """
        cmap = get_cmap(colormap_name)

        # Convert to numpy if needed
        if torch.is_tensor(normalized_values):
            normalized_np = normalized_values.cpu().numpy()
        else:
            normalized_np = normalized_values

        # Apply colormap
        colors = cmap(normalized_np)[:, :3]  # RGB only (drop alpha)

        # Convert back to torch if requested
        if as_torch:
            colors_torch = torch.from_numpy(colors).float()
            if device is not None:
                colors_torch = colors_torch.to(device)
            return colors_torch

        return colors

    @staticmethod
    def add_colorbar_to_image(
        image: np.ndarray,
        colormap_name: str,
        vmin: float,
        vmax: float,
        title: str = "Heatmap",
        normalize_colorbar: bool = True,
        symmetric_diverging: bool = False
    ) -> np.ndarray:
        """
        Add a colorbar overlay to a rendered image.

        Args:
            image: Rendered image (H, W, 3) as numpy array [0, 255] uint8
            colormap_name: Matplotlib colormap name
            vmin: Minimum value for colorbar
            vmax: Maximum value for colorbar
            title: Title text for the visualization
            normalize_colorbar: If True, show colorbar as 0-1 instead of actual values
            symmetric_diverging: If True, use symmetric colorbar centered at 0

        Returns:
            Image with colorbar overlay (H, W, 3) as numpy array [0, 255] uint8
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        # Create normalization and colorbar
        if symmetric_diverging:
            # Use TwoSlopeNorm for diverging colormap centered at 0
            if normalize_colorbar:
                norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
            else:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            if normalize_colorbar:
                norm = Normalize(vmin=0.0, vmax=1.0)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)

        sm = cm.ScalarMappable(cmap=colormap_name, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

        # Determine label based on title (Shape vs Texture)
        if 'Shape' in title or 'Displacement' in title:
            if symmetric_diverging:
                label = 'Displacement (B - A)'
            else:
                label = 'Displacement (normalized)' if normalize_colorbar else 'Displacement'
        else:
            if symmetric_diverging:
                label = 'Difference (B - A)'
            else:
                label = 'Difference (normalized)' if normalize_colorbar else 'Difference'

        cbar.set_label(label, rotation=270, labelpad=20, fontsize=12)

        # Convert figure to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_array = buf.reshape(h, w, 4)[:, :, :3]  # Drop alpha channel
        plt.close(fig)

        return img_array

    @staticmethod
    def render_heatmap_complete(
        values: Union[np.ndarray, torch.Tensor],
        colormap_name: str = 'viridis',
        normalize_mode: str = 'percentile',
        percentile: float = 95.0,
        symmetric_diverging: bool = False,
        as_torch: bool = False,
        device: Optional[torch.device] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], float, float]:
        """
        Complete heatmap rendering: normalize + apply colormap.

        This is a convenience method that combines normalization and colormap application.

        Args:
            values: Input values to visualize
            colormap_name: Matplotlib colormap name
            normalize_mode: Normalization mode ('percentile', 'minmax', 'none')
            percentile: Percentile for clipping (if normalize_mode='percentile')
            symmetric_diverging: Use symmetric range around 0
            as_torch: Return torch tensor instead of numpy
            device: Target device for torch tensor

        Returns:
            Tuple of (rgb_colors, vmin, vmax)
        """
        # Normalize values
        normalized, vmin, vmax = HeatmapRenderer.normalize_values(
            values,
            mode=normalize_mode,
            percentile=percentile,
            symmetric=symmetric_diverging
        )

        # Apply colormap
        colors = HeatmapRenderer.apply_colormap(
            normalized,
            colormap_name=colormap_name,
            as_torch=as_torch,
            device=device
        )

        return colors, vmin, vmax
