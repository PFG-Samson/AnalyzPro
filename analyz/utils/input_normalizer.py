"""
Unified input normalization for imagery from local uploads and online downloads.

This module normalizes imagery inputs regardless of origin (local or STAC downloads)
into a standardized format that can be processed by analysis modules.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import logging

from .file_handler import FileHandler
from .logger import get_logger

logger = get_logger(__name__)


class ImageryInput:
    """Represents a normalized imagery input from any source."""
    
    def __init__(self, source_type: str, source_path: Union[str, Path],
                 imagery_type: str = 'optical', metadata: Dict = None):
        """
        Initialize imagery input.
        
        Args:
            source_type: 'local_upload' or 'stac_download'
            source_path: Path to imagery file or folder
            imagery_type: 'optical' or 'sar'
            metadata: Optional metadata dict
        """
        self.source_type = source_type  # 'local_upload' or 'stac_download'
        self.source_path = Path(source_path)
        self.imagery_type = imagery_type
        self.metadata = metadata or {}
        
        # Discover imagery details
        self._discover_imagery()
    
    def _discover_imagery(self):
        """Discover imagery properties from source."""
        if self.source_type == 'stac_download':
            self._discover_stac_imagery()
        else:  # local_upload
            self._discover_local_imagery()
    
    def _discover_stac_imagery(self):
        """Discover properties from STAC-downloaded imagery."""
        # STAC downloads have structured folder layout
        if self.source_path.is_dir():
            # Find raster files
            raster_files = list(self.source_path.rglob('*.tif')) + \
                          list(self.source_path.rglob('*.tiff'))
            
            if raster_files:
                self.primary_file = raster_files[0]
            else:
                raise FileNotFoundError(f"No raster files found in {self.source_path}")
            
            # Load metadata if available
            metadata_file = self.source_path / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        self.metadata.update(json.load(f))
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
        else:
            self.primary_file = self.source_path
    
    def _discover_local_imagery(self):
        """Discover properties from locally uploaded imagery."""
        if self.source_path.is_file():
            self.primary_file = self.source_path
        elif self.source_path.is_dir():
            # Look for raster files in directory
            raster_files = list(self.source_path.glob('*.tif')) + \
                          list(self.source_path.glob('*.tiff')) + \
                          list(self.source_path.glob('*.jp2'))
            
            if raster_files:
                self.primary_file = raster_files[0]
            else:
                raise FileNotFoundError(f"No raster files found in {self.source_path}")
        else:
            raise FileNotFoundError(f"Source path does not exist: {self.source_path}")
    
    def read_data(self) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
        """
        Read imagery data.
        
        Returns:
            Tuple of (data array, rasterio profile)
        """
        logger.info(f"Reading imagery from {self.source_type}: {self.primary_file}")
        data, profile = FileHandler.read_raster(self.primary_file)
        return data, profile
    
    def get_metadata(self) -> Dict:
        """Get associated metadata."""
        return self.metadata.copy()
    
    def __repr__(self):
        return f"ImageryInput(type={self.source_type}, path={self.source_path})"


class InputNormalizer:
    """
    Normalizes imagery inputs from different sources into a standard format.
    
    This normalizer ensures that:
    1. Imagery from local uploads and STAC downloads are handled uniformly
    2. All analysis modules receive data in expected format
    3. Metadata is preserved throughout processing
    4. File paths and temporary files are properly managed
    """
    
    def __init__(self):
        """Initialize normalizer."""
        self.file_handler = FileHandler()
        self.processed_imagery = {}  # Cache of processed imagery
    
    def normalize_optical_imagery(self, imagery_input: ImageryInput,
                                 sensor: str = 'sentinel2') -> Tuple[np.ndarray, Dict]:
        """
        Normalize optical imagery for analysis.
        
        Args:
            imagery_input: ImageryInput instance
            sensor: Sensor type ('sentinel2', 'landsat8', 'landsat9')
            
        Returns:
            Tuple of (normalized data array, properties dict)
        """
        logger.info(f"Normalizing optical imagery from {imagery_input.source_type}")
        
        # Read imagery data
        data, profile = imagery_input.read_data()
        
        # Ensure 3D array (bands, height, width)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 3 and data.shape[-1] in (3, 4) and data.shape[0] not in (3, 4):
            # Likely channels-last format, transpose to channels-first
            data = np.transpose(data, (2, 0, 1))
        
        # Normalize data type if needed
        if data.dtype == np.uint8:
            # Keep as is (0-255)
            pass
        elif data.dtype == np.uint16:
            # Keep as is (0-65535)
            pass
        elif np.issubdtype(data.dtype, np.floating):
            # Already normalized or already float
            pass
        
        # Create properties dict
        properties = {
            'source_type': imagery_input.source_type,
            'sensor': sensor,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'bands': data.shape[0],
            'height': data.shape[1],
            'width': data.shape[2],
            'profile': profile,
            'metadata': imagery_input.get_metadata()
        }
        
        logger.info(f"Normalized optical imagery: {properties['shape']}, {data.dtype}")
        return data, properties
    
    def normalize_sar_imagery(self, imagery_input: ImageryInput,
                            sensor: str = 'sentinel1') -> Tuple[np.ndarray, Dict]:
        """
        Normalize SAR imagery for analysis.
        
        Args:
            imagery_input: ImageryInput instance
            sensor: Sensor type ('sentinel1', 'radarsat2', etc.)
            
        Returns:
            Tuple of (normalized data array, properties dict)
        """
        logger.info(f"Normalizing SAR imagery from {imagery_input.source_type}")
        
        # Read imagery data
        data, profile = imagery_input.read_data()
        
        # Ensure 3D array (bands, height, width)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim == 3 and data.shape[-1] in (1, 2, 3, 4) and data.shape[0] not in (1, 2, 3, 4):
            # Likely channels-last format, transpose to channels-first
            data = np.transpose(data, (2, 0, 1))
        
        # Convert to float if needed
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        
        # Create properties dict
        properties = {
            'source_type': imagery_input.source_type,
            'sensor': sensor,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'bands': data.shape[0],
            'height': data.shape[1],
            'width': data.shape[2],
            'profile': profile,
            'metadata': imagery_input.get_metadata()
        }
        
        logger.info(f"Normalized SAR imagery: {properties['shape']}, {data.dtype}")
        return data, properties
    
    def normalize_boundary(self, boundary_input: Union[str, Path],
                          target_crs: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """
        Normalize boundary/mask input.
        
        Args:
            boundary_input: Path to boundary file (GeoJSON, Shapefile, etc.)
            target_crs: Optional target CRS
            
        Returns:
            Tuple of (boundary geometry, properties dict)
        """
        import geopandas as gpd
        
        boundary_path = Path(boundary_input)
        logger.info(f"Normalizing boundary from {boundary_path}")
        
        # Read boundary file
        if boundary_path.suffix.lower() == '.geojson':
            gdf = gpd.read_file(boundary_path)
        elif boundary_path.suffix.lower() == '.json':
            gdf = gpd.read_file(boundary_path)
        elif boundary_path.suffix.lower() == '.shp':
            gdf = gpd.read_file(boundary_path)
        else:
            raise ValueError(f"Unsupported boundary format: {boundary_path.suffix}")
        
        # Reproject if needed
        if target_crs and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        
        properties = {
            'crs': str(gdf.crs),
            'features': len(gdf),
            'bounds': gdf.total_bounds.tolist(),
            'geometry_type': gdf.geometry.type.unique().tolist()
        }
        
        logger.info(f"Normalized boundary: {len(gdf)} features, CRS: {gdf.crs}")
        return gdf.geometry, properties
    
    def normalize_for_optical_analysis(self, imagery_input: ImageryInput,
                                       sensor: str = 'sentinel2',
                                       boundary_input: Optional[Union[str, Path]] = None
                                       ) -> Dict:
        """
        Complete normalization pipeline for optical analysis.
        
        Args:
            imagery_input: ImageryInput instance
            sensor: Sensor type
            boundary_input: Optional boundary for masking
            
        Returns:
            Normalized input dict ready for OpticalAnalyzer
        """
        # Normalize imagery
        data, img_props = self.normalize_optical_imagery(imagery_input, sensor)
        
        # Normalize boundary if provided
        boundary_geom = None
        boundary_props = None
        if boundary_input:
            boundary_geom, boundary_props = self.normalize_boundary(boundary_input)
        
        return {
            'data': data,
            'profile': img_props['profile'],
            'sensor': sensor,
            'boundary_geometry': boundary_geom,
            'imagery_properties': img_props,
            'boundary_properties': boundary_props,
            'source_type': imagery_input.source_type,
            'metadata': imagery_input.get_metadata()
        }
    
    def normalize_for_sar_analysis(self, imagery_input: ImageryInput,
                                   sensor: str = 'sentinel1',
                                   boundary_input: Optional[Union[str, Path]] = None
                                   ) -> Dict:
        """
        Complete normalization pipeline for SAR analysis.
        
        Args:
            imagery_input: ImageryInput instance
            sensor: Sensor type
            boundary_input: Optional boundary for masking
            
        Returns:
            Normalized input dict ready for SARAnalyzer
        """
        # Normalize imagery
        data, img_props = self.normalize_sar_imagery(imagery_input, sensor)
        
        # Normalize boundary if provided
        boundary_geom = None
        boundary_props = None
        if boundary_input:
            boundary_geom, boundary_props = self.normalize_boundary(boundary_input)
        
        return {
            'data': data,
            'profile': img_props['profile'],
            'sensor': sensor,
            'boundary_geometry': boundary_geom,
            'imagery_properties': img_props,
            'boundary_properties': boundary_props,
            'source_type': imagery_input.source_type,
            'metadata': imagery_input.get_metadata()
        }
    
    def copy_to_results(self, source_path: Union[str, Path],
                       destination_folder: Union[str, Path],
                       copy_recursive: bool = True) -> Path:
        """
        Copy imagery to results folder for processing.
        
        Args:
            source_path: Source path (file or folder)
            destination_folder: Destination folder
            copy_recursive: Copy entire folder structure if source is folder
            
        Returns:
            Path to copied imagery
        """
        source = Path(source_path)
        dest = Path(destination_folder)
        dest.mkdir(parents=True, exist_ok=True)
        
        if source.is_file():
            dest_file = dest / source.name
            shutil.copy2(source, dest_file)
            logger.info(f"Copied file to results: {dest_file}")
            return dest_file
        elif source.is_dir():
            if copy_recursive:
                dest_dir = dest / source.name
                shutil.copytree(source, dest_dir, dirs_exist_ok=True)
                logger.info(f"Copied directory to results: {dest_dir}")
                return dest_dir
            else:
                # Copy only raster files
                raster_files = list(source.glob('*.tif')) + list(source.glob('*.tiff'))
                for raster_file in raster_files:
                    shutil.copy2(raster_file, dest / raster_file.name)
                logger.info(f"Copied {len(raster_files)} raster files to results")
                return dest
        else:
            raise FileNotFoundError(f"Source not found: {source}")
    
    def validate_imagery_for_analysis(self, data: np.ndarray, 
                                     imagery_type: str = 'optical') -> bool:
        """
        Validate that imagery is suitable for analysis.
        
        Args:
            data: Imagery data array
            imagery_type: 'optical' or 'sar'
            
        Returns:
            True if valid, raises exception otherwise
        """
        if data.ndim not in (2, 3):
            raise ValueError(f"Invalid data dimensions: {data.ndim}. Expected 2 or 3.")
        
        if data.ndim == 3:
            bands = data.shape[0]
            
            if imagery_type == 'optical':
                if bands < 1:
                    raise ValueError(f"Optical imagery needs at least 1 band, got {bands}")
            elif imagery_type == 'sar':
                if bands < 1:
                    raise ValueError(f"SAR imagery needs at least 1 band, got {bands}")
        
        # Check for NaN/Inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("Imagery contains NaN or Inf values")
        
        logger.info(f"Imagery validation passed: {data.shape}, {data.dtype}")
        return True


def create_imagery_input(imagery_path: Union[str, Path], 
                        source_type: str,
                        imagery_type: str = 'optical',
                        metadata: Dict = None) -> ImageryInput:
    """
    Factory function to create ImageryInput from path.
    
    Args:
        imagery_path: Path to imagery
        source_type: 'local_upload' or 'stac_download'
        imagery_type: 'optical' or 'sar'
        metadata: Optional metadata dict
        
    Returns:
        ImageryInput instance
    """
    return ImageryInput(source_type, imagery_path, imagery_type, metadata)
