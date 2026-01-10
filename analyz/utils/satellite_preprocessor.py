"""Satellite data preprocessing utilities for Landsat and Sentinel imagery."""

import os
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional
import numpy as np
import rasterio
from rasterio.merge import merge
from .logger import get_logger

logger = get_logger(__name__)


class SatellitePreprocessor:
    """Handles preprocessing of raw satellite data from various sources."""
    
    # Landsat band mapping for different sensors
    LANDSAT_BANDS = {
        'landsat8': {
            'coastal': 'B1',
            'blue': 'B2',
            'green': 'B3',
            'red': 'B4',
            'nir': 'B5',
            'swir1': 'B6',
            'swir2': 'B7',
            'pan': 'B8',
            'cirrus': 'B9',
            'tir1': 'B10',
            'tir2': 'B11'
        },
        'landsat9': {
            'coastal': 'B1',
            'blue': 'B2',
            'green': 'B3',
            'red': 'B4',
            'nir': 'B5',
            'swir1': 'B6',
            'swir2': 'B7',
            'pan': 'B8',
            'cirrus': 'B9',
            'tir1': 'B10',
            'tir2': 'B11'
        }
    }
    
    # Sentinel-2 band mapping
    SENTINEL2_BANDS = {
        'coastal': 'B01',
        'blue': 'B02',
        'green': 'B03',
        'red': 'B04',
        'rededge1': 'B05',
        'rededge2': 'B06',
        'rededge3': 'B07',
        'nir': 'B08',
        'nir_narrow': 'B8A',
        'water_vapor': 'B09',
        'swir1': 'B11',
        'swir2': 'B12'
    }
    
    @staticmethod
    def extract_archive(archive_path: Union[str, Path], extract_dir: Union[str, Path]) -> Path:
        """
        Extract compressed satellite data archive.
        
        Args:
            archive_path: Path to .tar, .tar.gz, or .zip file
            extract_dir: Directory to extract to
            
        Returns:
            Path to extracted directory
        """
        archive_path = Path(archive_path)
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix in ['.tar', '.gz'] or '.tar.' in archive_path.name:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        
        logger.info(f"Extracted to {extract_dir}")
        return extract_dir
    
    @staticmethod
    def detect_satellite_type(data_path: Path) -> str:
        """
        Detect satellite type from extracted data.
        
        Args:
            data_path: Path to extracted data
            
        Returns:
            Satellite type ('landsat8', 'landsat9', 'sentinel1', 'sentinel2')
        """
        # Check for Sentinel SAFE format
        if data_path.suffix == '.SAFE' or any(data_path.glob('*.SAFE')):
            # Distinguish between Sentinel-1 (SAR) and Sentinel-2 (Optical)
            safe_dir = data_path if data_path.suffix == '.SAFE' else list(data_path.glob('*.SAFE'))[0]
            if 'S1A' in safe_dir.name or 'S1B' in safe_dir.name:
                return 'sentinel1'
            elif 'S2A' in safe_dir.name or 'S2B' in safe_dir.name:
                return 'sentinel2'
        
        # Check for Landsat files
        files = list(data_path.glob('*B*.TIF')) + list(data_path.glob('*B*.tif'))
        if files:
            filename = files[0].name.upper()
            if 'LC08' in filename or 'LO08' in filename:
                return 'landsat8'
            elif 'LC09' in filename or 'LO09' in filename:
                return 'landsat9'
        
        raise ValueError(f"Could not detect satellite type from: {data_path}")
    
    @staticmethod
    def find_landsat_bands(data_path: Path, bands: List[str] = None) -> Dict[str, Path]:
        """
        Find Landsat band files.
        
        Args:
            data_path: Path to extracted Landsat data
            bands: List of bands to find (e.g., ['B2', 'B3', 'B4'])
            
        Returns:
            Dictionary mapping band names to file paths
        """
        if bands is None:
            bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        
        band_files = {}
        for band in bands:
            # Look for band files (e.g., LC08_*_B4.TIF)
            matches = list(data_path.glob(f'*_{band}.TIF')) + list(data_path.glob(f'*_{band}.tif'))
            if matches:
                band_files[band] = matches[0]
                logger.info(f"Found {band}: {matches[0].name}")
        
        return band_files
    
    @staticmethod
    def find_sentinel_bands(data_path: Path, resolution: str = '10m', 
                           bands: List[str] = None) -> Dict[str, Path]:
        """
        Find Sentinel-2 band files.
        
        Args:
            data_path: Path to extracted Sentinel-2 .SAFE directory
            resolution: Resolution folder ('10m', '20m', '60m')
            bands: List of bands to find (e.g., ['B02', 'B03', 'B04', 'B08'])
            
        Returns:
            Dictionary mapping band names to file paths
        """
        if bands is None:
            bands = ['B02', 'B03', 'B04', 'B08']  # Default: Blue, Green, Red, NIR
        
        # Find the GRANULE folder
        safe_dir = data_path if data_path.suffix == '.SAFE' else list(data_path.glob('*.SAFE'))[0]
        granule_dir = safe_dir / 'GRANULE'
        
        if not granule_dir.exists():
            raise FileNotFoundError(f"GRANULE directory not found in {safe_dir}")
        
        # Get the first (usually only) granule
        granule = list(granule_dir.iterdir())[0]
        img_data_dir = granule / 'IMG_DATA'
        
        # Sentinel-2 L2A has resolution subfolders, L1C doesn't
        res_dir = img_data_dir / f'R{resolution}'
        if not res_dir.exists():
            res_dir = img_data_dir
        
        band_files = {}
        for band in bands:
            matches = list(res_dir.glob(f'*_{band}_*.jp2')) + list(res_dir.glob(f'*_{band}.jp2'))
            if matches:
                band_files[band] = matches[0]
                logger.info(f"Found {band}: {matches[0].name}")
        
        return band_files
    
    @staticmethod
    def stack_bands(band_files: Dict[str, Path], output_path: Path, 
                   band_order: List[str] = None) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
        """
        Stack multiple band files into a single multi-band raster.
        
        Args:
            band_files: Dictionary of band names to file paths
            output_path: Path to save stacked raster
            band_order: Order of bands in output (uses dict order if None)
            
        Returns:
            Tuple of (stacked array, profile)
        """
        if band_order is None:
            band_order = list(band_files.keys())
        
        logger.info(f"Stacking {len(band_order)} bands...")
        
        # Read first band to get metadata
        with rasterio.open(band_files[band_order[0]]) as src:
            profile = src.profile.copy()
            height, width = src.shape
        
        # Create array to hold all bands
        stacked = np.zeros((len(band_order), height, width), dtype=profile['dtype'])
        
        # Read each band
        for i, band in enumerate(band_order):
            if band not in band_files:
                logger.warning(f"Band {band} not found, filling with zeros")
                continue
            
            with rasterio.open(band_files[band]) as src:
                stacked[i] = src.read(1)
        
        # Update profile
        profile.update({
            'count': len(band_order),
            'driver': 'GTiff',
            'compress': 'LZW'
        })
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(stacked)
        
        logger.info(f"Stacked raster saved to {output_path}")
        return stacked, profile
    
    @staticmethod
    def process_landsat(archive_path: Union[str, Path], output_path: Union[str, Path],
                       bands: List[str] = None, extract_dir: Optional[Path] = None) -> Path:
        """
        Process Landsat archive to stacked GeoTIFF.
        
        Args:
            archive_path: Path to Landsat .tar or .zip file
            output_path: Path for output GeoTIFF
            bands: List of bands to include (default: B1-B7)
            extract_dir: Directory to extract to (temporary if None)
            
        Returns:
            Path to processed GeoTIFF
        """
        archive_path = Path(archive_path)
        output_path = Path(output_path)
        
        # Extract archive
        if extract_dir is None:
            extract_dir = archive_path.parent / f"{archive_path.stem}_extracted"
        extracted = SatellitePreprocessor.extract_archive(archive_path, extract_dir)
        
        try:
            # Find band files
            band_files = SatellitePreprocessor.find_landsat_bands(extracted, bands)
            
            if not band_files:
                raise ValueError(f"No Landsat bands found in {extracted}")
            
            # Stack bands
            SatellitePreprocessor.stack_bands(band_files, output_path)
            
            logger.info(f"Landsat processing complete: {output_path}")
            return output_path
            
        finally:
            # Clean up extracted files if using temporary directory
            if extract_dir.name.endswith('_extracted'):
                shutil.rmtree(extract_dir, ignore_errors=True)
    
    @staticmethod
    def process_sentinel2(archive_or_safe: Union[str, Path], output_path: Union[str, Path],
                         bands: List[str] = None, resolution: str = '10m',
                         extract_dir: Optional[Path] = None) -> Path:
        """
        Process Sentinel-2 .SAFE or archive to stacked GeoTIFF.
        
        Args:
            archive_or_safe: Path to .SAFE directory or .zip archive
            output_path: Path for output GeoTIFF
            bands: List of bands to include (default: B02, B03, B04, B08)
            resolution: Resolution ('10m', '20m', '60m')
            extract_dir: Directory to extract to (temporary if None)
            
        Returns:
            Path to processed GeoTIFF
        """
        data_path = Path(archive_or_safe)
        output_path = Path(output_path)
        
        # Extract if archive
        if data_path.suffix == '.zip':
            if extract_dir is None:
                extract_dir = data_path.parent / f"{data_path.stem}_extracted"
            data_path = SatellitePreprocessor.extract_archive(data_path, extract_dir)
        
        try:
            # Find band files
            band_files = SatellitePreprocessor.find_sentinel_bands(data_path, resolution, bands)
            
            if not band_files:
                raise ValueError(f"No Sentinel-2 bands found in {data_path}")
            
            # Stack bands
            SatellitePreprocessor.stack_bands(band_files, output_path)
            
            logger.info(f"Sentinel-2 processing complete: {output_path}")
            return output_path
            
        finally:
            # Clean up extracted files if using temporary directory
            if extract_dir and extract_dir.name.endswith('_extracted'):
                shutil.rmtree(extract_dir, ignore_errors=True)
    
    @staticmethod
    def process_sentinel1(archive_or_safe: Union[str, Path], output_path: Union[str, Path],
                         polarization: str = 'VV', extract_dir: Optional[Path] = None) -> Path:
        """
        Process Sentinel-1 SAR .SAFE or archive to GeoTIFF.
        
        Args:
            archive_or_safe: Path to .SAFE directory or .zip archive
            output_path: Path for output GeoTIFF
            polarization: Polarization to extract ('VV', 'VH', or 'both')
            extract_dir: Directory to extract to (temporary if None)
            
        Returns:
            Path to processed GeoTIFF
        """
        data_path = Path(archive_or_safe)
        output_path = Path(output_path)
        
        # Extract if archive
        if data_path.suffix == '.zip':
            if extract_dir is None:
                extract_dir = data_path.parent / f"{data_path.stem}_extracted"
            data_path = SatellitePreprocessor.extract_archive(data_path, extract_dir)
        
        try:
            # Find the SAFE directory
            safe_dir = data_path if data_path.suffix == '.SAFE' else list(data_path.glob('*.SAFE'))[0]
            
            # Find measurement files
            measurement_dir = safe_dir / 'measurement'
            if not measurement_dir.exists():
                raise FileNotFoundError(f"Measurement directory not found in {safe_dir}")
            
            # Find TIFF files (Sentinel-1 GRD products)
            tiff_files = list(measurement_dir.glob('*.tiff')) + list(measurement_dir.glob('*.tif'))
            
            if not tiff_files:
                raise ValueError(f"No measurement files found in {measurement_dir}")
            
            # Filter by polarization if specified
            if polarization.lower() == 'both':
                selected_files = tiff_files
            else:
                selected_files = [f for f in tiff_files if polarization.lower() in f.name.lower()]
                if not selected_files:
                    # Fall back to first available file
                    selected_files = [tiff_files[0]]
                    logger.warning(f"Polarization {polarization} not found, using {tiff_files[0].name}")
            
            # If single file, just copy it
            if len(selected_files) == 1:
                with rasterio.open(selected_files[0]) as src:
                    profile = src.profile.copy()
                    data = src.read()
                    profile.update({
                        'driver': 'GTiff',
                        'compress': 'LZW'
                    })
                    
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(data)
            else:
                # Stack multiple polarizations
                band_files = {f.stem.split('-')[-1]: f for f in selected_files}
                SatellitePreprocessor.stack_bands(band_files, output_path)
            
            logger.info(f"Sentinel-1 processing complete: {output_path}")
            return output_path
            
        finally:
            # Clean up extracted files if using temporary directory
            if extract_dir and extract_dir.name.endswith('_extracted'):
                shutil.rmtree(extract_dir, ignore_errors=True)
    
    @staticmethod
    def process_auto(input_path: Union[str, Path], output_path: Union[str, Path],
                    extract_dir: Optional[Path] = None, **kwargs) -> Path:
        """
        Automatically detect and process satellite data.
        
        Args:
            input_path: Path to satellite data (archive or .SAFE)
            output_path: Path for output GeoTIFF
            extract_dir: Optional custom extraction directory (for shorter paths on Windows)
            **kwargs: Additional arguments for specific processors
            
        Returns:
            Path to processed GeoTIFF
        """
        input_path = Path(input_path)
        
        # Extract if needed
        if input_path.suffix in ['.zip', '.tar', '.gz'] or '.tar.' in input_path.name:
            if extract_dir is None:
                extract_dir = input_path.parent / f"{input_path.stem}_temp"
            extracted = SatellitePreprocessor.extract_archive(input_path, extract_dir)
        else:
            extracted = input_path
            extract_dir = None
        
        try:
            # Detect satellite type
            sat_type = SatellitePreprocessor.detect_satellite_type(extracted)
            logger.info(f"Detected satellite type: {sat_type}")
            
            # Process based on type
            if 'landsat' in sat_type:
                return SatellitePreprocessor.process_landsat(
                    input_path if input_path.suffix in ['.zip', '.tar', '.gz'] else extracted,
                    output_path,
                    extract_dir=extract_dir,
                    **kwargs
                )
            elif sat_type == 'sentinel1':
                return SatellitePreprocessor.process_sentinel1(
                    extracted,
                    output_path,
                    extract_dir=extract_dir,
                    **kwargs
                )
            elif sat_type == 'sentinel2':
                return SatellitePreprocessor.process_sentinel2(
                    extracted,
                    output_path,
                    extract_dir=extract_dir,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported satellite type: {sat_type}")
                
        except Exception as e:
            if extract_dir and extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            raise e
