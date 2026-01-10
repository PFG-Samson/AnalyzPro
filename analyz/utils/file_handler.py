"""File handling utilities for Analyz application."""

import os
from pathlib import Path
import rasterio
from typing import Union, Tuple
import numpy as np
import geopandas as gpd
from .logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Handles file I/O operations for image and vector data."""
    
    @staticmethod
    def validate_file(file_path: Union[str, Path]) -> Path:
        """
        Validate that a file exists and return Path object.
        
        Args:
            file_path: Path to file
            
        Returns:
            Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Validated file: {file_path}")
        return path
    
    @staticmethod
    def read_raster(file_path: Union[str, Path]) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
        """
        Read a raster file.
        
        Args:
            file_path: Path to raster file
            
        Returns:
            Tuple of (data array, rasterio profile)
        """
        path = FileHandler.validate_file(file_path)
        
        with rasterio.open(path) as src:
            data = src.read()
            profile = src.profile
            
        logger.info(f"Read raster: {file_path} - Shape: {data.shape}, Dtype: {data.dtype}")
        return data, profile
    
    @staticmethod
    def write_raster(file_path: Union[str, Path], data: np.ndarray, 
                     profile: rasterio.profiles.Profile, compress: str = "LZW"):
        """
        Write data to a raster file.
        
        Args:
            file_path: Output file path
            data: Array to write. Accepts (bands, H, W), (H, W), or (H, W, 3/4).
            profile: Rasterio profile
            compress: Compression type
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update profile
        profile = profile.copy()
        
        # Normalize data shape to (bands, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 3 and data.shape[0] not in (3, 4) and data.shape[-1] in (3, 4):
            # Likely channels-last (H, W, 3/4) -> transpose to (3/4, H, W)
            data = np.transpose(data, (2, 0, 1))
        
        # Set common metadata
        profile.update({
            'count': data.shape[0],
            'height': data.shape[1],
            'width': data.shape[2],
            'dtype': data.dtype,
            'compress': compress
        })
        
        # Photometric hints (memory-safe checks)
        if data.shape[0] == 3:
            profile.setdefault('photometric', 'RGB')
        elif data.shape[0] == 1:
            profile.setdefault('photometric', 'MINISBLACK')
            # Heuristic: if appears to be a small-class mask, set nodata=0
            try:
                band0 = data[0]
                h, w = band0.shape
                # Sample to ~1M pixels max
                step = max(1, int(np.sqrt((h * w) / 1_000_000)))
                sample = band0[::step, ::step]
                # If integer typed and small range within 0..5, treat as class mask
                if (np.issubdtype(sample.dtype, np.integer)):
                    vmin = int(sample.min())
                    vmax = int(sample.max())
                    if 0 <= vmin <= vmax <= 5:
                        profile.setdefault('nodata', 0)
            except Exception:
                pass
        
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data)
            
        logger.info(f"Wrote raster: {file_path}")
    
    @staticmethod
    def read_vector(file_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Read a vector file (GeoJSON, Shapefile, etc.).
        
        Args:
            file_path: Path to vector file
            
        Returns:
            GeoDataFrame
        """
        path = FileHandler.validate_file(file_path)
        gdf = gpd.read_file(path)
        logger.info(f"Read vector: {file_path} - Features: {len(gdf)}, CRS: {gdf.crs}")
        return gdf
    
    @staticmethod
    def write_vector(file_path: Union[str, Path], gdf: gpd.GeoDataFrame, 
                     driver: str = None):
        """
        Write a GeoDataFrame to file.
        
        Args:
            file_path: Output file path
            gdf: GeoDataFrame to write
            driver: GDAL driver (auto-detected if None)
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect driver from extension
        if driver is None:
            ext = path.suffix.lower()
            driver_map = {
                '.geojson': 'GeoJSON',
                '.shp': 'ESRI Shapefile',
                '.gpkg': 'GPKG',
                '.kml': 'KML'
            }
            driver = driver_map.get(ext, 'GeoJSON')
        
        gdf.to_file(path, driver=driver)
        logger.info(f"Wrote vector: {file_path}")
    
    @staticmethod
    def create_output_path(base_dir: Union[str, Path], filename: str) -> Path:
        """
        Create an output file path.
        
        Args:
            base_dir: Base output directory
            filename: Output filename
            
        Returns:
            Full output path
        """
        path = Path(base_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
