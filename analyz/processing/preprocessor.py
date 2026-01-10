"""Preprocessing utilities for image analysis."""

import numpy as np
from scipy import ndimage
from skimage import exposure
from typing import Tuple, Union
from ..utils import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Image preprocessing operations."""
    
    @staticmethod
    def normalize(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize data.
        
        Args:
            data: Input array
            method: Normalization method ('minmax', 'zscore', 'percentile')
            
        Returns:
            Normalized array
        """
        if method == 'minmax':
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)
            normalized = (data - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            mean = np.nanmean(data)
            std = np.nanstd(data)
            normalized = (data - mean) / std
            
        elif method == 'percentile':
            p2, p98 = np.nanpercentile(data, [2, 98])
            normalized = np.clip((data - p2) / (p98 - p2), 0, 1)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Normalized data using {method} method")
        return normalized
    
    @staticmethod
    def remove_outliers(data: np.ndarray, n_std: float = 3.0) -> np.ndarray:
        """
        Remove outliers using standard deviation threshold.
        
        Args:
            data: Input array
            n_std: Number of standard deviations for threshold
            
        Returns:
            Data with outliers set to NaN
        """
        mean = np.nanmean(data)
        std = np.nanstd(data)
        threshold = n_std * std
        
        mask = np.abs(data - mean) > threshold
        cleaned = data.copy()
        cleaned[mask] = np.nan
        
        logger.info(f"Removed {np.sum(mask)} outliers ({np.sum(mask)/data.size*100:.2f}%)")
        return cleaned
    
    @staticmethod
    def fill_nodata(data: np.ndarray, method: str = 'interpolate') -> np.ndarray:
        """
        Fill nodata values.
        
        Args:
            data: Input array with NaN values
            method: Fill method ('interpolate', 'mean', 'median', 'zero')
            
        Returns:
            Filled array
        """
        filled = data.copy()
        mask = np.isnan(filled)
        
        if not np.any(mask):
            return filled
        
        if method == 'interpolate':
            # Distance-weighted interpolation
            indices = ndimage.distance_transform_edt(
                mask, return_distances=False, return_indices=True
            )
            filled = filled[tuple(indices)]
            
        elif method == 'mean':
            filled[mask] = np.nanmean(data)
            
        elif method == 'median':
            filled[mask] = np.nanmedian(data)
            
        elif method == 'zero':
            filled[mask] = 0
            
        else:
            raise ValueError(f"Unknown fill method: {method}")
        
        logger.info(f"Filled {np.sum(mask)} nodata values using {method} method")
        return filled
    
    @staticmethod
    def enhance_contrast(data: np.ndarray, method: str = 'equalize') -> np.ndarray:
        """
        Enhance image contrast.
        
        Args:
            data: Input array
            method: Enhancement method ('equalize', 'adaptive', 'stretch')
            
        Returns:
            Contrast-enhanced array
        """
        # Normalize to 0-1 range for processing
        data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        
        if method == 'equalize':
            enhanced = exposure.equalize_hist(data_norm)
            
        elif method == 'adaptive':
            enhanced = exposure.equalize_adapthist(data_norm)
            
        elif method == 'stretch':
            p2, p98 = np.nanpercentile(data_norm, [2, 98])
            enhanced = exposure.rescale_intensity(data_norm, in_range=(p2, p98))
            
        else:
            raise ValueError(f"Unknown enhancement method: {method}")
        
        # Scale back to original range
        enhanced = enhanced * (np.nanmax(data) - np.nanmin(data)) + np.nanmin(data)
        
        logger.info(f"Enhanced contrast using {method} method")
        return enhanced
    
    @staticmethod
    def apply_mask(data: np.ndarray, mask: np.ndarray, 
                   mask_value: Union[float, None] = np.nan) -> np.ndarray:
        """
        Apply a mask to data.
        
        Args:
            data: Input array
            mask: Boolean mask (True = keep, False = mask)
            mask_value: Value to use for masked pixels
            
        Returns:
            Masked array
        """
        masked = data.copy()
        masked[~mask] = mask_value
        
        logger.info(f"Applied mask - Masked {np.sum(~mask)} pixels")
        return masked
    
    @staticmethod
    def resample_bands(data: np.ndarray, target_shape: Tuple[int, int],
                       method: str = 'bilinear') -> np.ndarray:
        """
        Resample multi-band data to target shape.
        
        Args:
            data: Input array (bands, height, width)
            target_shape: Target (height, width)
            method: Resampling method
            
        Returns:
            Resampled array
        """
        from scipy.ndimage import zoom
        
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        
        zoom_factors = (1, target_shape[0]/data.shape[1], target_shape[1]/data.shape[2])
        
        order_map = {
            'nearest': 0,
            'bilinear': 1,
            'cubic': 3
        }
        order = order_map.get(method, 1)
        
        resampled = zoom(data, zoom_factors, order=order)
        
        logger.info(f"Resampled from {data.shape} to {resampled.shape}")
        return resampled
