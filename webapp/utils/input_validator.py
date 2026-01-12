"""
Input validation module for STAC search interface.

Validates user inputs (ROI, dates, cloud coverage) and provides
helpful error messages and guidance.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analyz.utils.logger import setup_logger

logger = setup_logger("INFO")


class InputValidator:
    """Validate user inputs for STAC imagery search."""
    
    # Validation constraints
    MAX_CLOUD_COVERAGE = 100
    MIN_CLOUD_COVERAGE = 0
    MAX_ROI_AREA_KM2 = 1_000_000  # 1 million km²
    MIN_ROI_AREA_KM2 = 0.01  # 0.01 km²
    MAX_DATE_RANGE_DAYS = 3650  # 10 years
    MIN_DATE_RANGE_DAYS = 1
    
    # Satellite specifications
    SATELLITE_SPECS = {
        'sentinel2': {
            'name': 'Sentinel-2',
            'type': 'optical',
            'min_cloud': 0,
            'max_cloud': 100,
            'min_area': 0.01,  # km²
            'availability': 'global',
            'start_date': datetime(2015, 6, 23),
            'bands': ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12']
        },
        'sentinel1': {
            'name': 'Sentinel-1',
            'type': 'sar',
            'min_cloud': None,  # N/A for SAR
            'max_cloud': None,
            'min_area': 0.01,
            'availability': 'global',
            'start_date': datetime(2014, 4, 3),
            'polarizations': ['VV', 'VH', 'HH', 'HV']
        },
        'landsat8': {
            'name': 'Landsat 8',
            'type': 'optical',
            'min_cloud': 0,
            'max_cloud': 100,
            'min_area': 185 * 185 / 1_000_000,  # 185x185 km scene
            'availability': 'global',
            'start_date': datetime(2013, 4, 11),
            'bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        },
        'landsat9': {
            'name': 'Landsat 9',
            'type': 'optical',
            'min_cloud': 0,
            'max_cloud': 100,
            'min_area': 185 * 185 / 1_000_000,
            'availability': 'global',
            'start_date': datetime(2021, 9, 27),
            'bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        }
    }
    
    @staticmethod
    def validate_roi(geometry: Dict) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Validate ROI geometry.
        
        Args:
            geometry: GeoJSON geometry object
        
        Returns:
            Tuple of (is_valid, error_message, area_km2)
        """
        try:
            if not geometry:
                return False, "ROI is empty. Please draw an area on the map.", None
            
            geo_type = geometry.get('type', '').lower()
            
            if geo_type not in ['polygon', 'multipolygon', 'rectangle']:
                return False, f"Invalid geometry type: {geo_type}. Use Polygon or Rectangle.", None
            
            # Check coordinates exist
            coords = geometry.get('coordinates')
            if not coords:
                return False, "Geometry has no coordinates.", None
            
            # Basic bounds check
            if geo_type == 'polygon':
                bounds = InputValidator._get_bounds(coords[0])
            elif geo_type == 'rectangle':
                bounds = InputValidator._get_bounds(coords)
            else:
                bounds = InputValidator._get_bounds_multipolygon(coords)
            
            if not bounds:
                return False, "Could not extract bounds from geometry.", None
            
            # Calculate area (rough approximation)
            area_km2 = InputValidator._calculate_area_km2(bounds)
            
            # Validate area
            if area_km2 < InputValidator.MIN_ROI_AREA_KM2:
                return False, f"ROI too small ({area_km2:.4f} km²). Minimum: {InputValidator.MIN_ROI_AREA_KM2} km².", area_km2
            
            if area_km2 > InputValidator.MAX_ROI_AREA_KM2:
                return False, f"ROI too large ({area_km2:.0f} km²). Maximum: {InputValidator.MAX_ROI_AREA_KM2:,.0f} km².", area_km2
            
            return True, None, area_km2
        
        except Exception as e:
            logger.error(f"ROI validation error: {e}")
            return False, f"Error validating ROI: {str(e)}", None
    
    @staticmethod
    def validate_date_range(
        start_date: str, 
        end_date: str,
        satellite: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate date range.
        
        Args:
            start_date: Start date as ISO string (YYYY-MM-DD)
            end_date: End date as ISO string (YYYY-MM-DD)
            satellite: Satellite/mission name
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            now = datetime.now()
            
            # Check start/end relationship
            if start > end:
                return False, f"Start date ({start_date}) cannot be after end date ({end_date})"
            
            if start == end:
                return False, "Start and end dates must be different"
            
            # Check future dates
            if end > now:
                return False, f"End date ({end_date}) cannot be in the future"
            
            if start > now:
                return False, f"Start date ({start_date}) cannot be in the future"
            
            # Check satellite availability
            if satellite in InputValidator.SATELLITE_SPECS:
                spec = InputValidator.SATELLITE_SPECS[satellite]
                if start < spec['start_date']:
                    min_date = spec['start_date'].strftime('%Y-%m-%d')
                    return False, f"{spec['name']} not available before {min_date}"
            
            # Check date range span
            delta = (end - start).days
            if delta < InputValidator.MIN_DATE_RANGE_DAYS:
                return False, f"Date range too short. Minimum: {InputValidator.MIN_DATE_RANGE_DAYS} day(s)"
            
            if delta > InputValidator.MAX_DATE_RANGE_DAYS:
                max_years = InputValidator.MAX_DATE_RANGE_DAYS // 365
                return False, f"Date range too long. Maximum: {max_years} years"
            
            return True, None
        
        except ValueError as e:
            return False, f"Invalid date format. Use YYYY-MM-DD: {str(e)}"
        except Exception as e:
            logger.error(f"Date validation error: {e}")
            return False, f"Error validating dates: {str(e)}"
    
    @staticmethod
    def validate_cloud_coverage(
        cloud_coverage: float,
        satellite: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate cloud coverage parameter.
        
        Args:
            cloud_coverage: Maximum cloud coverage percentage (0-100)
            satellite: Satellite/mission name
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check range
            if not isinstance(cloud_coverage, (int, float)):
                return False, "Cloud coverage must be a number"
            
            if cloud_coverage < InputValidator.MIN_CLOUD_COVERAGE:
                return False, f"Cloud coverage cannot be less than {InputValidator.MIN_CLOUD_COVERAGE}%"
            
            if cloud_coverage > InputValidator.MAX_CLOUD_COVERAGE:
                return False, f"Cloud coverage cannot exceed {InputValidator.MAX_CLOUD_COVERAGE}%"
            
            # Check satellite type
            if satellite in InputValidator.SATELLITE_SPECS:
                spec = InputValidator.SATELLITE_SPECS[satellite]
                if spec['type'] == 'sar':
                    return False, "Cloud coverage filter not applicable for SAR data"
            
            return True, None
        
        except Exception as e:
            logger.error(f"Cloud coverage validation error: {e}")
            return False, f"Error validating cloud coverage: {str(e)}"
    
    @staticmethod
    def validate_satellite_selection(satellite: str) -> Tuple[bool, Optional[str]]:
        """
        Validate satellite selection.
        
        Args:
            satellite: Satellite identifier
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not satellite:
            return False, "Please select a satellite/mission"
        
        if satellite not in InputValidator.SATELLITE_SPECS:
            available = ', '.join(InputValidator.SATELLITE_SPECS.keys())
            return False, f"Unknown satellite: {satellite}. Available: {available}"
        
        return True, None
    
    @staticmethod
    def validate_complete_search(
        geometry: Dict,
        start_date: str,
        end_date: str,
        satellite: str,
        cloud_coverage: float = None
    ) -> Tuple[bool, Dict]:
        """
        Validate all inputs for search operation.
        
        Args:
            geometry: GeoJSON geometry
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            satellite: Satellite name
            cloud_coverage: Cloud coverage percentage (for optical only)
        
        Returns:
            Tuple of (is_valid, validation_results_dict)
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Validate satellite
        sat_valid, sat_error = InputValidator.validate_satellite_selection(satellite)
        if not sat_valid:
            results['errors'].append(sat_error)
            results['valid'] = False
        else:
            results['info']['satellite'] = InputValidator.SATELLITE_SPECS[satellite]['name']
        
        # Validate ROI
        roi_valid, roi_error, area = InputValidator.validate_roi(geometry)
        if not roi_valid:
            results['errors'].append(roi_error)
            results['valid'] = False
        else:
            results['info']['roi_area_km2'] = area
        
        # Validate dates
        date_valid, date_error = InputValidator.validate_date_range(start_date, end_date, satellite)
        if not date_valid:
            results['errors'].append(date_error)
            results['valid'] = False
        else:
            results['info']['date_range_days'] = (
                (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days
            )
        
        # Validate cloud coverage (if optical)
        if satellite in InputValidator.SATELLITE_SPECS:
            spec = InputValidator.SATELLITE_SPECS[satellite]
            if spec['type'] == 'optical' and cloud_coverage is not None:
                cloud_valid, cloud_error = InputValidator.validate_cloud_coverage(cloud_coverage, satellite)
                if not cloud_valid:
                    results['errors'].append(cloud_error)
                    results['valid'] = False
                else:
                    results['info']['max_cloud_coverage'] = cloud_coverage
        
        return results['valid'], results
    
    @staticmethod
    def _get_bounds(coordinates: List) -> Optional[Tuple[float, float, float, float]]:
        """Extract bounds from coordinate list. Returns (minlon, minlat, maxlon, maxlat)."""
        try:
            if not coordinates:
                return None
            
            lons = [c[0] for c in coordinates]
            lats = [c[1] for c in coordinates]
            
            return (min(lons), min(lats), max(lons), max(lats))
        except:
            return None
    
    @staticmethod
    def _get_bounds_multipolygon(coordinates: List) -> Optional[Tuple]:
        """Extract bounds from multipolygon coordinates."""
        try:
            all_lons = []
            all_lats = []
            
            for polygon in coordinates:
                for ring in polygon:
                    all_lons.extend([c[0] for c in ring])
                    all_lats.extend([c[1] for c in ring])
            
            if all_lons and all_lats:
                return (min(all_lons), min(all_lats), max(all_lons), max(all_lats))
        except:
            pass
        return None
    
    @staticmethod
    def _calculate_area_km2(bounds: Tuple[float, float, float, float]) -> float:
        """
        Rough calculation of area from bounds.
        Uses simple approximation, not precise.
        """
        minlon, minlat, maxlon, maxlat = bounds
        
        # Rough approximation: 1 degree lat ≈ 111 km, 1 degree lon varies by latitude
        lat_km = (maxlat - minlat) * 111
        lon_km = (maxlon - minlon) * 111 * __import__('math').cos(__import__('math').radians((maxlat + minlat) / 2))
        
        return abs(lat_km * lon_km)


class ValidationErrorFormatter:
    """Format validation errors for user-friendly display."""
    
    @staticmethod
    def format_error(error_message: str) -> Dict:
        """
        Format validation error with helpful guidance.
        
        Returns:
            Dict with error info and suggestions
        """
        return {
            'message': error_message,
            'severity': 'error',
            'icon': '❌',
            'dismissible': False
        }
    
    @staticmethod
    def format_warning(warning_message: str) -> Dict:
        """Format warning message."""
        return {
            'message': warning_message,
            'severity': 'warning',
            'icon': '⚠️',
            'dismissible': True
        }
    
    @staticmethod
    def format_info(info_message: str) -> Dict:
        """Format informational message."""
        return {
            'message': info_message,
            'severity': 'info',
            'icon': 'ℹ️',
            'dismissible': True
        }
    
    @staticmethod
    def format_validation_results(results: Dict) -> Dict:
        """
        Format complete validation results for API response.
        
        Args:
            results: Validation results dict from InputValidator
        
        Returns:
            Formatted dict ready for JSON response
        """
        return {
            'valid': results['valid'],
            'errors': [
                ValidationErrorFormatter.format_error(e) 
                for e in results.get('errors', [])
            ],
            'warnings': [
                ValidationErrorFormatter.format_warning(w) 
                for w in results.get('warnings', [])
            ],
            'info': results.get('info', {}),
            'summary': (
                f"✓ All inputs valid" if results['valid']
                else f"❌ {len(results.get('errors', []))} error(s) found"
            )
        }
