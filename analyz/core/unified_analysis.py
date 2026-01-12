"""
Unified analysis wrapper that integrates input normalization with existing analysis modules.

This module provides a unified interface for analyzing imagery from any source
(local uploads or STAC downloads) while maintaining compatibility with existing
OpticalAnalyzer and SARAnalyzer classes.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np
import logging

from analyz.utils import InputNormalizer, ImageryInput, FileHandler
from .optical_analysis import OpticalAnalyzer
from .sar_analysis import SARAnalyzer
from ..processing import BoundaryHandler
from ..visualization import Plotter, InsightsGenerator

logger = logging.getLogger(__name__)


class UnifiedAnalysisWrapper:
    """
    Wrapper that provides unified analysis interface for imagery from any source.
    
    This wrapper:
    1. Accepts imagery from local uploads or STAC downloads
    2. Normalizes input using InputNormalizer
    3. Feeds normalized data to existing analysis modules
    4. Returns results in standard format
    5. Maintains full compatibility with existing code
    """
    
    def __init__(self):
        """Initialize the unified analysis wrapper."""
        self.normalizer = InputNormalizer()
        self.file_handler = FileHandler()
    
    def analyze_optical_imagery(self, imagery_path: Union[str, Path],
                               source_type: str = 'local_upload',
                               analysis_type: str = 'ndvi',
                               sensor: str = 'sentinel2',
                               band_indices: Dict = None,
                               boundary_path: Optional[Union[str, Path]] = None,
                               output_dir: Optional[Union[str, Path]] = None,
                               params: Dict = None) -> Tuple[np.ndarray, Dict, str]:
        """
        Unified optical image analysis accepting imagery from any source.
        
        Args:
            imagery_path: Path to imagery file or STAC download folder
            source_type: 'local_upload' or 'stac_download'
            analysis_type: Type of analysis ('ndvi', 'ndwi', 'evi', etc.)
            sensor: Sensor type ('sentinel2', 'landsat8', etc.)
            band_indices: Dict with band assignments {red, nir, green, blue, swir1, swir2}
            boundary_path: Optional path to boundary file for clipping
            output_dir: Optional output directory for results
            params: Optional parameters dict
            
        Returns:
            Tuple of (result array, statistics dict, insights string)
        """
        logger.info(f"Starting unified optical analysis from {source_type}")
        
        params = params or {}
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        
        try:
            # Step 1: Create and normalize imagery input
            imagery_input = ImageryInput(source_type, imagery_path, 'optical', 
                                        metadata=params.get('metadata'))
            
            normalized_input = self.normalizer.normalize_for_optical_analysis(
                imagery_input, sensor, boundary_path
            )
            
            data = normalized_input['data']
            profile = normalized_input['profile']
            
            logger.info(f"Normalized imagery shape: {data.shape}, dtype: {data.dtype}")
            
            # Step 2: Apply boundary if clipping hasn't been done yet
            if boundary_path and normalized_input['boundary_geometry'] is not None:
                logger.info("Clipping to boundary...")
                boundary_handler = BoundaryHandler(boundary_path)
                data, profile = boundary_handler.clip_array(data, profile)
            
            # Step 3: Set up band indices if not provided
            if band_indices is None:
                band_indices = {
                    'red': params.get('red', 2),
                    'nir': params.get('nir', 3),
                    'green': params.get('green', 1),
                    'blue': params.get('blue', 0),
                    'swir1': params.get('swir1', 4),
                    'swir2': params.get('swir2', 5)
                }
            
            # Step 4: Run analysis using existing OpticalAnalyzer
            analyzer = OpticalAnalyzer(data, band_indices, sensor=sensor)
            
            if analysis_type == 'ndvi':
                result, stats = analyzer.ndvi()
                insights = InsightsGenerator.generate_ndvi_insights(result, stats)
            elif analysis_type == 'ndwi':
                result, stats = analyzer.ndwi()
                insights = InsightsGenerator.generate_ndwi_insights(result, stats)
            elif analysis_type == 'evi':
                result, stats = analyzer.evi()
                insights = InsightsGenerator.generate_optical_insights('EVI', result, stats)
            elif analysis_type in analyzer.available_indices().keys():
                result, stats = analyzer.compute_index(analysis_type)
                insights = InsightsGenerator.generate_optical_insights(
                    analysis_type.upper(), result, stats
                )
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Step 5: Save results if output directory specified
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save result raster
                result_file = output_dir / f'{analysis_type}_result.tif'
                self.file_handler.write_raster(result_file, result, profile)
                logger.info(f"Saved result to {result_file}")
            
            logger.info(f"Analysis complete: {analysis_type} from {source_type}")
            return result, stats, insights
            
        except Exception as e:
            logger.error(f"Error in unified optical analysis: {str(e)}")
            raise
    
    def analyze_sar_imagery(self, imagery_path: Union[str, Path],
                           source_type: str = 'local_upload',
                           analysis_type: str = 'intensity',
                           sensor: str = 'sentinel1',
                           boundary_path: Optional[Union[str, Path]] = None,
                           output_dir: Optional[Union[str, Path]] = None,
                           params: Dict = None) -> Tuple[np.ndarray, Dict, str]:
        """
        Unified SAR image analysis accepting imagery from any source.
        
        Args:
            imagery_path: Path to imagery file or STAC download folder
            source_type: 'local_upload' or 'stac_download'
            analysis_type: Type of analysis ('intensity', 'texture', 'decomposition', etc.)
            sensor: Sensor type ('sentinel1', 'radarsat2', etc.)
            boundary_path: Optional path to boundary file for clipping
            output_dir: Optional output directory for results
            params: Optional parameters dict
            
        Returns:
            Tuple of (result array, statistics dict, insights string)
        """
        logger.info(f"Starting unified SAR analysis from {source_type}")
        
        params = params or {}
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        
        try:
            # Step 1: Create and normalize imagery input
            imagery_input = ImageryInput(source_type, imagery_path, 'sar',
                                        metadata=params.get('metadata'))
            
            normalized_input = self.normalizer.normalize_for_sar_analysis(
                imagery_input, sensor, boundary_path
            )
            
            data = normalized_input['data']
            profile = normalized_input['profile']
            
            logger.info(f"Normalized SAR imagery shape: {data.shape}, dtype: {data.dtype}")
            
            # Step 2: Apply boundary if clipping hasn't been done yet
            if boundary_path and normalized_input['boundary_geometry'] is not None:
                logger.info("Clipping to boundary...")
                boundary_handler = BoundaryHandler(boundary_path)
                data, profile = boundary_handler.clip_array(data, profile)
            
            # Step 3: Run analysis using existing SARAnalyzer
            analyzer = SARAnalyzer(
                data,
                profile=profile,
                sensor=sensor,
                band_names=params.get('band_names'),
                nodata_values=params.get('nodata_values'),
                calibrated=params.get('calibrated', False),
                radiometry=params.get('radiometry', 'amplitude')
            )
            
            if analysis_type == 'intensity':
                result, stats = analyzer.intensity()
                insights = InsightsGenerator.generate_sar_insights(result, stats)
            elif analysis_type == 'texture':
                window_size = params.get('window_size', 3)
                result, stats = analyzer.texture_analysis(window_size=window_size)
                insights = InsightsGenerator.generate_sar_insights(result, stats)
            elif analysis_type == 'speckle':
                result, stats = analyzer.speckle_filtering()
                insights = InsightsGenerator.generate_sar_insights(result, stats)
            else:
                raise ValueError(f"Unknown SAR analysis type: {analysis_type}")
            
            # Step 4: Save results if output directory specified
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save result raster
                result_file = output_dir / f'{analysis_type}_result.tif'
                self.file_handler.write_raster(result_file, result, profile)
                logger.info(f"Saved result to {result_file}")
            
            logger.info(f"Analysis complete: {analysis_type} from {source_type}")
            return result, stats, insights
            
        except Exception as e:
            logger.error(f"Error in unified SAR analysis: {str(e)}")
            raise
    
    def batch_analyze(self, imagery_folder: Union[str, Path],
                     source_type: str = 'local_upload',
                     image_type: str = 'optical',
                     analysis_configs: list = None,
                     output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Batch analyze multiple imagery files.
        
        Args:
            imagery_folder: Folder containing imagery files
            source_type: 'local_upload' or 'stac_download'
            image_type: 'optical' or 'sar'
            analysis_configs: List of analysis config dicts
            output_dir: Output directory for results
            
        Returns:
            Dict with results for each imagery file
        """
        imagery_folder = Path(imagery_folder)
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        
        results = {}
        
        # Find all raster files
        raster_files = list(imagery_folder.glob('*.tif')) + \
                      list(imagery_folder.glob('*.tiff')) + \
                      list(imagery_folder.glob('*.jp2'))
        
        logger.info(f"Found {len(raster_files)} raster files for batch analysis")
        
        for raster_file in raster_files:
            try:
                file_output_dir = output_dir / raster_file.stem
                
                for config in analysis_configs or []:
                    if image_type == 'optical':
                        result, stats, insights = self.analyze_optical_imagery(
                            raster_file,
                            source_type=source_type,
                            output_dir=file_output_dir,
                            **config
                        )
                    else:  # sar
                        result, stats, insights = self.analyze_sar_imagery(
                            raster_file,
                            source_type=source_type,
                            output_dir=file_output_dir,
                            **config
                        )
                    
                    results[f"{raster_file.stem}_{config.get('analysis_type', 'unknown')}"] = {
                        'status': 'success',
                        'stats': stats,
                        'insights': insights
                    }
                    
            except Exception as e:
                logger.error(f"Error analyzing {raster_file}: {str(e)}")
                results[raster_file.stem] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def validate_imagery_source(self, imagery_path: Union[str, Path],
                               source_type: str,
                               image_type: str = 'optical') -> bool:
        """
        Validate that imagery source is valid and accessible.
        
        Args:
            imagery_path: Path to imagery
            source_type: 'local_upload' or 'stac_download'
            image_type: 'optical' or 'sar'
            
        Returns:
            True if valid, raises exception otherwise
        """
        try:
            imagery_input = ImageryInput(source_type, imagery_path, image_type)
            data, profile = imagery_input.read_data()
            self.normalizer.validate_imagery_for_analysis(data, image_type)
            logger.info(f"Imagery source validated: {imagery_path}")
            return True
        except Exception as e:
            logger.error(f"Imagery validation failed: {str(e)}")
            raise
