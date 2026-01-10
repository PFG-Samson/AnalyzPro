"""Visualization and plotting utilities."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Dict, Tuple
from ..utils import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def _quicklook_2d(arr: np.ndarray, max_dim: int = 4096, max_pixels: int = 8_000_000) -> np.ndarray:
    """Downsample large 2D arrays for plotting to avoid OOM.
    Uses stride-based decimation to keep memory small.
    """
    if arr.ndim != 2:
        return arr
    h, w = arr.shape
    if h <= 0 or w <= 0:
        return arr
    # Limit by max dimension
    step_dim = max(1, int(np.ceil(max(h, w) / max_dim)))
    # Limit by total pixels
    step_pix = max(1, int(np.ceil(np.sqrt((h * w) / max_pixels))))
    step = max(step_dim, step_pix)
    if step <= 1:
        return arr
    return arr[::step, ::step]


class Plotter:
    """Handles visualization and plotting operations."""
    
    @staticmethod
    def plot_raster(data: np.ndarray, title: str = "Raster Data",
                   cmap: str = "RdYlGn", output_path: Union[str, Path] = None,
                   vmin: float = None, vmax: float = None, colorbar_label: str = ""):
        """
        Plot raster data.
        
        Args:
            data: 2D array to plot, or 3D array with shape (1, height, width)
            title: Plot title
            cmap: Colormap
            output_path: Save path (if None, shows plot)
            vmin, vmax: Value range for colormap
            colorbar_label: Label for colorbar
        """
        # Handle 3D single-band arrays
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        elif data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]
        elif data.ndim > 2 and data.shape[0] != 3 and data.shape[2] != 3:
            raise ValueError(f"Invalid shape {data.shape} for image data. Expected 2D (H, W) or 3D (1, H, W) or (H, W, 3)")
        
        # Downsample large rasters to prevent huge RGBA allocations during rendering
        view = _quicklook_2d(data) if data.ndim == 2 else data
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(view, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {output_path}")
            plt.close()
    
    @staticmethod
    def plot_histogram(data: np.ndarray, title: str = "Data Distribution",
                      bins: int = 50, output_path: Union[str, Path] = None):
        """
        Plot data histogram.
        
        Args:
            data: Array to plot (can be 2D or 3D)
            title: Plot title
            bins: Number of bins
            output_path: Save path
        """
        # Handle 3D single-band arrays
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        elif data.ndim == 3 and data.shape[2] == 1:
            data = data[:, :, 0]
        
        # Subsample for histogram if extremely large to avoid memory/time blow-up
        if data.ndim == 2:
            h, w = data.shape
            max_samples = 2_000_000
            if h * w > max_samples:
                step = int(np.ceil(np.sqrt((h * w) / max_samples)))
                data = data[::step, ::step]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        valid_data = data[~np.isnan(data)]
        
        ax.hist(valid_data, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(valid_data)
        median_val = np.median(valid_data)
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved histogram to {output_path}")
            plt.close()
    
    @staticmethod
    def plot_comparison(data1: np.ndarray, data2: np.ndarray,
                       title1: str = "Image 1", title2: str = "Image 2",
                       cmap: str = "gray", output_path: Union[str, Path] = None):
        """
        Plot two images side by side for comparison.
        
        Args:
            data1, data2: Arrays to compare (can be 2D or 3D)
            title1, title2: Plot titles
            cmap: Colormap
            output_path: Save path
        """
        # Handle 3D single-band arrays
        if data1.ndim == 3 and data1.shape[0] == 1:
            data1 = data1[0]
        elif data1.ndim == 3 and data1.shape[2] == 1:
            data1 = data1[:, :, 0]
            
        if data2.ndim == 3 and data2.shape[0] == 1:
            data2 = data2[0]
        elif data2.ndim == 3 and data2.shape[2] == 1:
            data2 = data2[:, :, 0]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        im1 = axes[0].imshow(data1, cmap=cmap)
        axes[0].set_title(title1, fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        im2 = axes[1].imshow(data2, cmap=cmap)
        axes[1].set_title(title2, fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison to {output_path}")
            plt.close()
    
    @staticmethod
    def plot_statistics(stats: Dict, output_path: Union[str, Path] = None):
        """
        Plot statistics as bar chart.
        
        Args:
            stats: Statistics dictionary
            output_path: Save path
        """
        # Filter numeric values
        numeric_stats = {k: v for k, v in stats.items() 
                        if isinstance(v, (int, float)) and not k.endswith('_percent')}
        
        if not numeric_stats:
            logger.warning("No numeric statistics to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        keys = list(numeric_stats.keys())
        values = list(numeric_stats.values())
        
        ax.bar(keys, values, color='steelblue', alpha=0.7)
        ax.set_title('Statistical Summary', fontsize=16, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved statistics plot to {output_path}")
            plt.close()
    
    @staticmethod
    def plot_land_cover_classification(classified: np.ndarray, class_labels: Dict[int, str],
                                      class_distribution: Dict,
                                      output_path: Union[str, Path] = None):
        """
        Plot land cover classification with semantic labels and legend.
        
        Args:
            classified: Classified raster array
            class_labels: Dictionary mapping cluster ID to label name
            class_distribution: Distribution stats for each class
            output_path: Save path
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Define colors for common land cover types
        color_map = {
            'Water': '#0000ff',
            'Dense Vegetation': '#006400',
            'Vegetation': '#228b22',
            'Sparse Vegetation': '#90ee90',
            'Built-up/Urban': '#ff0000',
            'Bare Soil/Rock': '#d2691e',
            'Barren Land': '#daa520',
            'Shadow/Dark Surface': '#2f4f4f'
        }
        
        # Create custom colormap
        from matplotlib.colors import ListedColormap
        n_classes = len(class_labels)
        colors = []
        legend_labels = []
        
        for i in range(n_classes):
            label = class_labels[i]
            colors.append(color_map.get(label, f'C{i}'))
            
            # Get percentage from distribution
            pct = class_distribution.get(label, {}).get('percent', 0)
            legend_labels.append(f"{label} ({pct:.1f}%)")
        
        cmap = ListedColormap(colors)
        
        # Plot classified map
        im = ax1.imshow(classified, cmap=cmap, vmin=0, vmax=n_classes-1)
        ax1.set_title('Land Cover Classification', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=legend_labels[i]) 
                          for i in range(n_classes)]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.05),
                  ncol=2, framealpha=0.9, fontsize=10)
        
        # Plot distribution chart
        class_names = [class_labels[i] for i in range(n_classes)]
        percentages = [class_distribution.get(class_labels[i], {}).get('percent', 0) 
                      for i in range(n_classes)]
        
        bars = ax2.barh(class_names, percentages, color=colors)
        ax2.set_xlabel('Area Coverage (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Land Cover Distribution', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{pct:.1f}%', ha='left', va='center', fontsize=10, 
                    fontweight='bold', color='black')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved land cover classification to {output_path}")
            plt.close()
    
    @staticmethod
    def plot_multifigure(data_list: list, titles: list, cmaps: list = None,
                        output_path: Union[str, Path] = None,
                        cols: int = 2):
        """
        Plot multiple figures in a grid.
        
        Args:
            data_list: List of arrays to plot
            titles: List of titles
            cmaps: List of colormaps (or single cmap for all)
            output_path: Save path
            cols: Number of columns
        """
        n_plots = len(data_list)
        rows = (n_plots + cols - 1) // cols
        
        if cmaps is None:
            cmaps = ['viridis'] * n_plots
        elif isinstance(cmaps, str):
            cmaps = [cmaps] * n_plots
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, (data, title, cmap) in enumerate(zip(data_list, titles, cmaps)):
            im = axes[idx].imshow(data, cmap=cmap)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved multi-figure plot to {output_path}")
            plt.close()
    
    @staticmethod
    def plot_band_stack_overview(band_stack: np.ndarray, band_names: list,
                                output_path: Union[str, Path] = None):
        """
        Plot overview of all bands in a multi-band stack.
        
        Args:
            band_stack: Array with shape (n_bands, height, width)
            band_names: List of band names
            output_path: Save path
        """
        n_bands = band_stack.shape[0]
        
        # Calculate grid dimensions
        cols = 3
        rows = (n_bands + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten() if n_bands > 1 else [axes]
        
        for i in range(n_bands):
            band_data = band_stack[i]
            band_name = band_names[i] if i < len(band_names) else f"Band {i+1}"
            
            # Use percentile stretching for better visualization
            vmin, vmax = np.percentile(band_data[~np.isnan(band_data)], [2, 98])
            
            im = axes[i].imshow(band_data, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i].set_title(f"{band_name.upper()}", fontsize=12, fontweight='bold')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for i in range(n_bands, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Multi-Band Stack Overview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved band stack overview to {output_path}")
            plt.close()
    
    @staticmethod
    def plot_rgb_composite(rgb_data: np.ndarray, title: str = "True Color Composite (RGB)",
                          output_path: Union[str, Path] = None, 
                          show_histograms: bool = False):
        """
        Plot RGB true color composite.
        
        Args:
            rgb_data: RGB array with shape (height, width, 3)
            title: Plot title
            output_path: Save path
            show_histograms: Whether to show RGB histograms alongside the composite
        """
        if rgb_data.ndim != 3 or rgb_data.shape[2] != 3:
            raise ValueError(f"Expected RGB data with shape (height, width, 3), got {rgb_data.shape}")
        
        # Ensure values are in [0, 1] range
        rgb_clipped = np.clip(rgb_data, 0, 1)
        
        if show_histograms:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # RGB composite
            axes[0].imshow(rgb_clipped)
            axes[0].set_title(title, fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # RGB histograms
            colors = ['red', 'green', 'blue']
            band_names = ['Red', 'Green', 'Blue']
            
            for i, (color, name) in enumerate(zip(colors, band_names)):
                band_data = rgb_clipped[:, :, i].flatten()
                axes[1].hist(band_data, bins=100, color=color, alpha=0.5, label=name)
            
            axes[1].set_title('RGB Band Histograms', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Normalized Value', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(rgb_clipped)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved RGB composite to {output_path}")
            plt.close()
    
    @staticmethod
    def plot_ndvi_classification(ndvi: np.ndarray, 
                                 output_path: Union[str, Path] = None):
        """
        Plot NDVI with standard classification.
        
        Args:
            ndvi: NDVI array
            output_path: Save path
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original NDVI
        im1 = axes[0].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0].set_title('NDVI', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label('NDVI Value', fontsize=10)
        
        # Classified NDVI
        classified = np.zeros_like(ndvi)
        classified[ndvi < 0] = 0  # Water
        classified[(ndvi >= 0) & (ndvi < 0.2)] = 1  # Bare soil
        classified[(ndvi >= 0.2) & (ndvi < 0.4)] = 2  # Sparse vegetation
        classified[(ndvi >= 0.4) & (ndvi < 0.6)] = 3  # Moderate vegetation
        classified[ndvi >= 0.6] = 4  # Dense vegetation
        
        colors = ['blue', 'brown', 'yellow', 'lightgreen', 'darkgreen']
        labels = ['Water', 'Bare Soil', 'Sparse Veg.', 'Moderate Veg.', 'Dense Veg.']
        
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        im2 = axes[1].imshow(classified, cmap=cmap, vmin=0, vmax=4)
        axes[1].set_title('NDVI Classification', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        cbar2 = plt.colorbar(im2, ax=axes[1], ticks=[0, 1, 2, 3, 4], 
                            fraction=0.046, pad=0.04)
        cbar2.ax.set_yticklabels(labels, fontsize=9)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved NDVI classification to {output_path}")
            plt.close()
