"""Boundary handling for study area clipping."""

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import mapping
from typing import Union, Tuple
from pathlib import Path
from ..utils import get_logger, FileHandler

logger = get_logger(__name__)


class BoundaryHandler:
    """Handles study area boundary operations for raster clipping."""
    
    def __init__(self, boundary_path: Union[str, Path] = None):
        """
        Initialize BoundaryHandler.
        
        Args:
            boundary_path: Path to boundary file (GeoJSON, Shapefile, etc.)
        """
        self.boundary_gdf = None
        self.boundary_path = None
        
        if boundary_path:
            self.load_boundary(boundary_path)
    
    def load_boundary(self, boundary_path: Union[str, Path]):
        """
        Load boundary from file.
        
        Args:
            boundary_path: Path to boundary file
        """
        self.boundary_path = boundary_path
        self.boundary_gdf = FileHandler.read_vector(boundary_path)
        logger.info(f"Loaded boundary with {len(self.boundary_gdf)} feature(s)")
    
    def clip_raster(self, raster_path: Union[str, Path], 
                    output_path: Union[str, Path] = None,
                    crop: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Clip raster to boundary.
        
        Args:
            raster_path: Path to raster file
            output_path: Optional output path for clipped raster
            crop: Whether to crop the extent
            
        Returns:
            Tuple of (clipped data, updated profile)
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary loaded. Use load_boundary() first.")
        
        with rasterio.open(raster_path) as src:
            # Reproject boundary to match raster CRS if needed
            boundary_gdf = self.boundary_gdf.copy()
            if boundary_gdf.crs != src.crs:
                logger.info(f"Reprojecting boundary from {boundary_gdf.crs} to {src.crs}")
                boundary_gdf = boundary_gdf.to_crs(src.crs)
            
            # Get geometries
            geoms = [mapping(geom) for geom in boundary_gdf.geometry]
            
            # Clip raster
            clipped_data, clipped_transform = mask(src, geoms, crop=crop, nodata=src.nodata)
            
            # Update profile
            profile = src.profile.copy()
            profile.update({
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'transform': clipped_transform
            })
        
        logger.info(f"Clipped raster to boundary - New shape: {clipped_data.shape}")
        
        # Write to file if output path specified
        if output_path:
            FileHandler.write_raster(output_path, clipped_data, profile)
        
        return clipped_data, profile
    
    def clip_array(self, data: np.ndarray, profile: dict,
                   crop: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Clip array using boundary without reading from file.
        
        Args:
            data: Raster data array
            profile: Rasterio profile
            crop: Whether to crop the extent
            
        Returns:
            Tuple of (clipped data, updated profile)
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary loaded. Use load_boundary() first.")
        
        # Create temporary in-memory raster
        from rasterio.io import MemoryFile
        
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(data)
            
            with memfile.open() as dataset:
                # Reproject boundary if needed
                boundary_gdf = self.boundary_gdf.copy()
                if boundary_gdf.crs != dataset.crs:
                    boundary_gdf = boundary_gdf.to_crs(dataset.crs)
                
                # Get geometries
                geoms = [mapping(geom) for geom in boundary_gdf.geometry]
                
                # Clip
                clipped_data, clipped_transform = mask(dataset, geoms, crop=crop, 
                                                       nodata=profile.get('nodata'))
        
        # Update profile
        new_profile = profile.copy()
        new_profile.update({
            'height': clipped_data.shape[1],
            'width': clipped_data.shape[2],
            'transform': clipped_transform
        })
        
        logger.info(f"Clipped array to boundary - New shape: {clipped_data.shape}")
        return clipped_data, new_profile
    
    def get_boundary_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get boundary bounding box.
        
        Returns:
            Tuple of (minx, miny, maxx, maxy)
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary loaded.")
        
        return self.boundary_gdf.total_bounds
    
    def get_boundary_area(self, crs: str = None) -> float:
        """
        Calculate boundary area.
        
        Args:
            crs: CRS for area calculation (uses current CRS if None)
            
        Returns:
            Area in square units of the CRS
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary loaded.")
        
        gdf = self.boundary_gdf.copy()
        if crs:
            gdf = gdf.to_crs(crs)
        
        return gdf.geometry.area.sum()
    
    def buffer_boundary(self, distance: float, crs: str = None):
        """
        Buffer the boundary geometry.
        
        Args:
            distance: Buffer distance in units of CRS
            crs: CRS for buffering (uses current if None)
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary loaded.")
        
        gdf = self.boundary_gdf.copy()
        if crs:
            gdf = gdf.to_crs(crs)
        
        gdf['geometry'] = gdf.geometry.buffer(distance)
        self.boundary_gdf = gdf
        logger.info(f"Buffered boundary by {distance} units")
