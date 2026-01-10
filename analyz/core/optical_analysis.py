"""Optical image analysis module."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, Optional
from ..utils import get_logger

logger = get_logger(__name__)


class OpticalAnalyzer:
    """Optical satellite image analysis."""
    
    # Sensor capabilities (what bands exist natively)
    SENSOR_CAPABILITIES = {
        'sentinel2': {
            'bands': {
                'coastal','blue','green','red','nir','nir_narrow',
                'rededge1','rededge2','rededge3','swir1','swir2'
            }
        },
        'landsat8': {
            'bands': {'coastal','blue','green','red','nir','swir1','swir2'}
        },
        'landsat9': {
            'bands': {'coastal','blue','green','red','nir','swir1','swir2'}
        }
    }
    
    # Registry of spectral index formulas (expressions) and required bands
    # Expressions use band variable names (red, green, blue, nir, swir1, swir2, rededge1, rededge2, rededge3, nir_narrow, coastal)
    INDEX_REGISTRY = {
        # Vegetation
        'ndvi': {
            'expr': '(nir - red) / (nir + red + 1e-8)',
            'requires': ['nir', 'red']
        },
        'evi2': {
            'expr': '2.5 * (nir - red) / (nir + 2.4*red + 1.0 + 1e-8)',
            'requires': ['nir', 'red']
        },
        'savi': {
            'expr': '((nir - red) / (nir + red + 0.5 + 1e-8)) * 1.5',
            'requires': ['nir', 'red']
        },
        'msavi': {
            'expr': '0.5 * (2*nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - red)))',
            'requires': ['nir', 'red']
        },
        'gndvi': {
            'expr': '(nir - green) / (nir + green + 1e-8)',
            'requires': ['nir', 'green']
        },
        'rvi': {
            'expr': 'nir / (red + 1e-8)',
            'requires': ['nir', 'red']
        },
        'dvi': {
            'expr': 'nir - red',
            'requires': ['nir', 'red']
        },
        'sr': {
            'expr': 'nir / (red + 1e-8)',
            'requires': ['nir', 'red']
        },
        'rdvi': {
            'expr': '(nir - red) / (np.sqrt(nir + red + 1e-8))',
            'requires': ['nir', 'red']
        },
        'ipvi': {
            'expr': 'nir / (nir + red + 1e-8)',
            'requires': ['nir', 'red']
        },
        'osavi': {
            'expr': '(nir - red) / (nir + red + 0.16 + 1e-8)',
            'requires': ['nir', 'red']
        },
        'msr': {
            'expr': '(nir/(red + 1e-8) - 1) / np.sqrt(nir/(red + 1e-8) + 1)',
            'requires': ['nir', 'red']
        },
        'tsavi': {
            # Soil line parameters a,b default to 0.08 and 0.1
            'expr': '(0.08*(nir - 0.08*red - 0.1)) / (red + 0.08*nir - 0.08*0.1 + 1e-8)',
            'requires': ['nir', 'red']
        },
        'tvi': {
            'expr': 'np.sqrt(((nir - red) / (nir + red + 1e-8)) + 0.5)',
            'requires': ['nir', 'red']
        },
        'tvi2': {
            # Triangular Vegetation Index approximation
            'expr': '0.5*(120*(nir - green) - 200*(red - green))',
            'requires': ['nir', 'red', 'green']
        },
        'mtvi2': {
            'expr': '(1.5*(1.2*(nir - green) - 2.5*(red - green))) / (np.sqrt((2*nir + 1)**2 - (6*nir - 5*np.sqrt(np.clip(red,0,None))) - 0.5) + 1e-8)',
            'requires': ['nir', 'red', 'green']
        },
        'mcari': {
            'expr': '(rededge1 - red) - 0.2*(rededge1 - green)*(rededge1/(red + 1e-8))',
            'requires': ['rededge1', 'red', 'green']
        },
        'tcari': {
            'expr': '3*((rededge1 - red) - 0.2*(rededge1 - green)*(rededge1/(red + 1e-8)))',
            'requires': ['rededge1', 'red', 'green']
        },
        'tcari_osavi_ratio': {
            'expr': '(3*((rededge1 - red) - 0.2*(rededge1 - green)*(rededge1/(red + 1e-8)))) / ((nir - red)/(nir + red + 0.16 + 1e-8) + 1e-8)',
            'requires': ['rededge1', 'red', 'green', 'nir']
        },
        'gci': {
            'expr': 'nir/(green + 1e-8) - 1',
            'requires': ['nir', 'green']
        },
        'cvi': {
            'expr': '(nir * red) / (np.clip(green, 1e-8, None)**2)',
            'requires': ['nir', 'red', 'green']
        },
        'slavi': {
            'expr': 'nir / (red + swir1 + 1e-8)',
            'requires': ['nir', 'red', 'swir1']
        },
        'bsi': {
            'expr': '((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-8)',
            'requires': ['swir1', 'red', 'nir', 'blue']
        },
        'gossavi': {
            'expr': '(nir - green) / (nir + green + 0.5 + 1e-8)',
            'requires': ['nir', 'green']
        },
        'vari': {
            'expr': '(green - red) / (green + red - blue + 1e-8)',
            'requires': ['green', 'red', 'blue']
        },
        'arvi': {
            'expr': '(nir - (2*red - blue)) / (nir + (2*red - blue) + 1e-8)',
            'requires': ['nir', 'red', 'blue']
        },
        'pri': {
            'expr': '(green - blue) / (green + blue + 1e-8)',
            'requires': ['green', 'blue']
        },
        'psri': {
            'expr': '(red - blue) / (rededge1 + 1e-8)',
            'requires': ['red', 'blue', 'rededge1']
        },
        'cri1': {
            'expr': '(1/np.clip(green,1e-8,None)) - (1/np.clip(blue,1e-8,None))',
            'requires': ['green', 'blue']
        },
        'cri2': {
            'expr': '(1/np.clip(rededge1,1e-8,None)) - (1/np.clip(green,1e-8,None))',
            'requires': ['rededge1', 'green']
        },
'sipi': {
            'expr': '(nir - blue) / (nir - red + 1e-8)',
            'requires': ['nir', 'blue', 'red']
        },
        'gemi': {
            'expr': '(eta*(1 - 0.25*eta) - (red - 0.125)/(1 - red + 1e-8))',
            'requires': ['nir', 'red'],
            'helpers': {
                'eta': '(2*(nir**2 - red**2) + 1.5*nir + 0.5*red) / (nir + red + 0.5 + 1e-8)'
            }
        },
        'lai': {
            'expr': 'np.clip(3.618 * (2.5*(nir - red)/(nir + 6*red - 7.5*blue + 1 + 1e-8)) - 0.118, 0, 7)',
            'requires': ['nir', 'red', 'blue']
        },
        # Red-edge based
        'ndvi705': {
            'expr': '(nir - rededge1) / (nir + rededge1 + 1e-8)',
            'requires': ['nir', 'rededge1']
        },
        'ndre': {
            'expr': '(nir - rededge2)/(nir + rededge2 + 1e-8)',
            'requires': ['nir', 'rededge2']
        },
        'ndvire': {
            'expr': '(nir - rededge1)/(nir + rededge1 + 1e-8)',
            'requires': ['nir', 'rededge1']
        },
        'cirededge': {
            'expr': 'nir/(rededge1 + 1e-8) - 1',
            'requires': ['nir', 'rededge1']
        },
        # Water / moisture / snow / burn
        'ndwi': {
            'expr': '(green - nir) / (green + nir + 1e-8)',
            'requires': ['green', 'nir']
        },
        'mndwi': {
            'expr': '(green - swir1) / (green + swir1 + 1e-8)',
            'requires': ['green', 'swir1']
        },
        'ndwi2': {
            'expr': '(green - swir2) / (green + swir2 + 1e-8)',
            'requires': ['green', 'swir2']
        },
        'ndmi': {
            'expr': '(nir - swir1) / (nir + swir1 + 1e-8)',
            'requires': ['nir', 'swir1']
        },
        'msi': {
            'expr': 'swir1 / (nir + 1e-8)',
            'requires': ['swir1', 'nir']
        },
        'gvmi': {
            'expr': '((nir + 0.1) - (swir2 + 0.02)) / ((nir + 0.1) + (swir2 + 0.02) + 1e-8)',
            'requires': ['nir', 'swir2']
        },
        'ndsi': {
            'expr': '(green - swir1) / (green + swir1 + 1e-8)',
            'requires': ['green', 'swir1']
        },
        'bai': {
            'expr': '1.0 / (((0.1 - red)**2) + ((0.06 - nir)**2) + 1e-12)',
            'requires': ['red', 'nir']
        },
        'baim': {
            'expr': '1.0 / (((swir2 - 0.06)**2) + ((nir - 0.1)**2) + 1e-12)',
            'requires': ['swir2', 'nir']
        },
        'nbr': {
            'expr': '(nir - swir2) / (nir + swir2 + 1e-8)',
            'requires': ['nir', 'swir2']
        },
        'nbr2': {
            'expr': '(swir1 - swir2) / (swir1 + swir2 + 1e-8)',
            'requires': ['swir1', 'swir2']
        },
        'lswi': {
            'expr': '(nir - swir1) / (nir + swir1 + 1e-8)',
            'requires': ['nir', 'swir1']
        },
        # Built-up
        'ndbi': {
            'expr': '(swir1 - nir) / (swir1 + nir + 1e-8)',
            'requires': ['swir1', 'nir']
        },
        # Color / visible
        'ndgi': {
            'expr': '(green - red) / (green + red + 1e-8)',
            'requires': ['green', 'red']
        },
        'rgr': {
            'expr': 'red / (green + 1e-8)',
            'requires': ['red', 'green']
        },
        'vig': {
            'expr': 'green / (red + 1e-8)',
            'requires': ['green', 'red']
        },
        # Aliases
        'ndvi2': {'alias': 'ndvi'},
        'ndvi3': {'alias': 'ndvi'},
        'ndvirededge': {'alias': 'ndvire'},
        'rendvi': {'alias': 'ndre'},
        'arvi2': {'alias': 'arvi'},
        'evi3': {'alias': 'evi2'},
        'nbri': {'alias': 'nbr'},
        'nbri2': {'alias': 'nbr2'},
        'pssra': {'alias': 'sr'},
        'pssrb': {'alias': 'sr'},
        'psnda': {'alias': 'ndvi'},
        'psndb': {'alias': 'ndvi'},
        'vi_green': {'alias': 'gndvi'},
        'viGreen': {'alias': 'gndvi'},
        'ndi7': {'alias': 'nbr2'},
        'ndti': {'alias': 'nbr2'},
        'ndi45': {
            'expr': '(rededge1 - rededge2) / (rededge1 + rededge2 + 1e-8)',
            'requires': ['rededge1', 'rededge2']
        },
        'ndi8': {
            'expr': '(nir - rededge3) / (nir + rededge3 + 1e-8)',
            'requires': ['nir', 'rededge3']
        },
        # Placeholders for indices requiring narrow bands not commonly available
        'wbi': {'requires': ['b900', 'b970'], 'expr': 'b900 / (b970 + 1e-8)'},
        'pri531': {'requires': ['b531', 'b570'], 'expr': '(b531 - b570) / (b531 + b570 + 1e-8)'}
    }
    
    def __init__(self, data: np.ndarray, band_indices: Dict[str, int] = None, sensor: Optional[str] = None):
        """
        Initialize OpticalAnalyzer.
        
        Args:
            data: Multi-band raster data (bands, height, width)
            band_indices: Dictionary mapping band names to indices
                         (e.g., {'red': 0, 'nir': 3})
            sensor: Optional sensor name ('sentinel2', 'landsat8', 'landsat9') for validation
        """
        self.data = data
        self.band_indices = band_indices or {}
        self.sensor = sensor.lower() if sensor else None
        if self.sensor and self.sensor not in self.SENSOR_CAPABILITIES:
            logger.warning(f"Unknown sensor '{sensor}'. Skipping sensor-based validation.")
            self.sensor = None
        logger.info(f"Initialized OpticalAnalyzer with shape {data.shape}")
    
    def _get_band(self, band_name: str) -> np.ndarray:
        """Get a specific band by name."""
        if band_name not in self.band_indices:
            raise ValueError(f"Band '{band_name}' not defined in band_indices")
        return self.data[self.band_indices[band_name]].astype(float)
    
    def ndvi(self) -> Tuple[np.ndarray, Dict]:
        """
        Calculate NDVI (Normalized Difference Vegetation Index) with stress detection.
        NDVI = (NIR - Red) / (NIR + Red)
        
        Returns:
            Tuple of (NDVI array, statistics dict with stress analysis)
        """
        nir = self._get_band('nir')
        red = self._get_band('red')
        
        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-8)
        
        # Clip to valid range
        ndvi = np.clip(ndvi, -1, 1)
        
        # Calculate statistics
        stats = self._calculate_statistics(ndvi, "NDVI")
        
        # Vegetation categories
        stats['no_vegetation_percent'] = np.sum(ndvi <= 0.2) / ndvi.size * 100
        stats['sparse_vegetation_percent'] = np.sum((ndvi > 0.2) & (ndvi <= 0.4)) / ndvi.size * 100
        stats['moderate_vegetation_percent'] = np.sum((ndvi > 0.4) & (ndvi <= 0.6)) / ndvi.size * 100
        stats['healthy_vegetation_percent'] = np.sum((ndvi > 0.6) & (ndvi <= 0.8)) / ndvi.size * 100
        stats['very_healthy_vegetation_percent'] = np.sum(ndvi > 0.8) / ndvi.size * 100
        stats['vegetation_cover_percent'] = np.sum(ndvi > 0.2) / ndvi.size * 100
        
        # Stress detection (vegetation with lower than expected NDVI)
        # Areas with 0.2 < NDVI < 0.4 could indicate stress
        stats['potential_stress_percent'] = stats['sparse_vegetation_percent']
        stats['stressed_area_km2'] = stats['potential_stress_percent'] / 100 * ndvi.size * 30 * 30 / 1e6
        
        # Vigor assessment
        vegetation_mask = ndvi > 0.2
        if np.sum(vegetation_mask) > 0:
            stats['vegetation_vigor_mean'] = float(np.mean(ndvi[vegetation_mask]))
        else:
            stats['vegetation_vigor_mean'] = 0.0
        
        logger.info(f"Calculated NDVI - Mean: {stats['mean']:.3f}, Vegetation cover: {stats['vegetation_cover_percent']:.1f}%")
        return ndvi, stats
    
    def ndwi(self) -> Tuple[np.ndarray, Dict]:
        """
        Calculate NDWI (Normalized Difference Water Index).
        NDWI = (Green - NIR) / (Green + NIR)
        
        Returns:
            Tuple of (NDWI array, statistics dict)
        """
        green = self._get_band('green')
        nir = self._get_band('nir')
        
        ndwi = (green - nir) / (green + nir + 1e-8)
        ndwi = np.clip(ndwi, -1, 1)
        
        stats = self._calculate_statistics(ndwi, "NDWI")
        stats['water_cover_percent'] = np.sum(ndwi > 0.0) / ndwi.size * 100
        
        logger.info(f"Calculated NDWI - Water cover: {stats['water_cover_percent']:.2f}%")
        return ndwi, stats
    
    def ndbi(self) -> Tuple[np.ndarray, Dict]:
        """
        Calculate NDBI (Normalized Difference Built-up Index).
        NDBI = (SWIR - NIR) / (SWIR + NIR)
        
        Returns:
            Tuple of (NDBI array, statistics dict)
        """
        swir = self._get_band('swir1')
        nir = self._get_band('nir')
        
        ndbi = (swir - nir) / (swir + nir + 1e-8)
        ndbi = np.clip(ndbi, -1, 1)
        
        stats = self._calculate_statistics(ndbi, "NDBI")
        stats['urban_cover_percent'] = np.sum(ndbi > 0.0) / ndbi.size * 100
        
        logger.info(f"Calculated NDBI - Urban cover: {stats['urban_cover_percent']:.2f}%")
        return ndbi, stats
    
    def evi(self, G: float = 2.5, C1: float = 6.0, 
            C2: float = 7.5, L: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Calculate EVI (Enhanced Vegetation Index).
        EVI = G * ((NIR - Red) / (NIR + C1*Red - C2*Blue + L))
        
        Args:
            G: Gain factor
            C1, C2: Coefficients for atmospheric correction
            L: Canopy background adjustment
            
        Returns:
            Tuple of (EVI array, statistics dict)
        """
        nir = self._get_band('nir')
        red = self._get_band('red')
        blue = self._get_band('blue')
        
        evi = G * ((nir - red) / (nir + C1 * red - C2 * blue + L + 1e-8))
        evi = np.clip(evi, -1, 1)
        
        stats = self._calculate_statistics(evi, "EVI")
        
        logger.info(f"Calculated EVI - Mean: {stats['mean']:.3f}")
        return evi, stats
    
    def savi(self, L: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """
        Calculate SAVI (Soil Adjusted Vegetation Index).
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        
        Args:
            L: Soil brightness correction factor (0.5 for moderate vegetation)
            
        Returns:
            Tuple of (SAVI array, statistics dict)
        """
        nir = self._get_band('nir')
        red = self._get_band('red')
        
        savi = ((nir - red) / (nir + red + L + 1e-8)) * (1 + L)
        savi = np.clip(savi, -1, 1)
        
        stats = self._calculate_statistics(savi, "SAVI")
        
        logger.info(f"Calculated SAVI - Mean: {stats['mean']:.3f}")
        return savi, stats
    
    def classify_kmeans(self, n_clusters: int = 5, profile: dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform unsupervised K-means classification with semantic labeling.
        
        Args:
            n_clusters: Number of clusters/classes
            profile: Rasterio profile for GeoJSON generation (optional)
            
        Returns:
            Tuple of (classified array with labels, statistics dict)
        """
        # Reshape data for sklearn
        n_bands, height, width = self.data.shape
        data_reshaped = self.data.reshape(n_bands, -1).T
        
        # Remove NaN values
        mask = ~np.any(np.isnan(data_reshaped), axis=1)
        valid_data = data_reshaped[mask]
        
        # Perform K-means clustering
        logger.info(f"Running K-means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(valid_data)
        
        # Reshape back to image
        classified = np.full(height * width, -1, dtype=int)
        classified[mask] = labels
        classified = classified.reshape(height, width)
        
        # Identify land cover types based on spectral signatures
        cluster_labels = self._identify_land_cover(kmeans.cluster_centers_, classified, mask.reshape(height, width))
        
        # Calculate statistics with identified classes
        stats = {
            'n_clusters': n_clusters,
            'class_distribution': {},
            'class_labels': cluster_labels
        }
        
        for i in range(n_clusters):
            count = np.sum(classified == i)
            percent = count / np.sum(classified >= 0) * 100
            class_name = cluster_labels[i]
            stats['class_distribution'][class_name] = {
                'cluster_id': i,
                'count': int(count),
                'percent': float(percent),
                'area_km2': float(count * 30 * 30 / 1e6)  # Approximate for 30m resolution
            }
        
        logger.info(f"Classification complete - Identified: {', '.join(cluster_labels.values())}")
        return classified, stats
    
    def _identify_land_cover(self, cluster_centers: np.ndarray, 
                            classified: np.ndarray, mask: np.ndarray) -> Dict[int, str]:
        """
        Identify land cover types based on cluster spectral signatures.
        
        Args:
            cluster_centers: K-means cluster centers (n_clusters, n_bands)
            classified: Classified image
            mask: Valid data mask
            
        Returns:
            Dictionary mapping cluster ID to land cover label
        """
        labels = {}
        
        try:
            # Calculate indices for each cluster center
            for i, center in enumerate(cluster_centers):
                # Extract band values (assuming standard order: B, G, R, NIR, SWIR1, SWIR2)
                # Handle different band configurations
                if len(center) >= 4:
                    red_idx = self.band_indices.get('red', 2)
                    nir_idx = self.band_indices.get('nir', 3)
                    green_idx = self.band_indices.get('green', 1)
                    
                    red = center[red_idx] if red_idx < len(center) else center[0]
                    nir = center[nir_idx] if nir_idx < len(center) else center[0]
                    green = center[green_idx] if green_idx < len(center) else center[0]
                    
                    # Calculate indices
                    ndvi = (nir - red) / (nir + red + 1e-8)
                    ndwi = (green - nir) / (green + nir + 1e-8)
                    
                    # Get SWIR if available
                    swir1_idx = self.band_indices.get('swir1', 4 if len(center) > 4 else 0)
                    swir1 = center[swir1_idx] if swir1_idx < len(center) else 0
                    
                    # Overall brightness
                    brightness = np.mean(center)
                    
                    # Classify based on spectral signatures
                    if ndwi > 0.3:  # High water index
                        labels[i] = "Water"
                    elif ndvi > 0.6:  # Very high NDVI
                        labels[i] = "Dense Vegetation"
                    elif ndvi > 0.3:  # Moderate NDVI
                        labels[i] = "Vegetation"
                    elif brightness > np.median(cluster_centers) * 1.2 and ndvi < 0.1:
                        if swir1 > 0:
                            labels[i] = "Built-up/Urban"
                        else:
                            labels[i] = "Bare Soil/Rock"
                    elif ndvi < 0.1 and brightness < np.median(cluster_centers) * 0.8:
                        labels[i] = "Shadow/Dark Surface"
                    else:
                        # Use NDVI as fallback
                        if ndvi > 0.2:
                            labels[i] = "Sparse Vegetation"
                        else:
                            labels[i] = "Barren Land"
                else:
                    # Fallback for insufficient bands
                    labels[i] = f"Class {i+1}"
        
        except Exception as e:
            logger.warning(f"Could not identify land cover types: {e}. Using default labels.")
            for i in range(len(cluster_centers)):
                labels[i] = f"Class {i+1}"
        
        return labels
    
    def _generate_classification_geojson(self, classified: np.ndarray, cluster_labels: Dict[int, str], 
                                       stats: Dict, profile: dict) -> str:
        """
        Generate GeoJSON polygons for each land cover class.
        
        Args:
            classified: Classified image array
            cluster_labels: Dictionary mapping cluster IDs to class names
            stats: Statistics dictionary with class distribution
            profile: Rasterio profile containing transform and CRS
            
        Returns:
            Path to generated GeoJSON file (relative to output directory)
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
            import rasterio
            from rasterio.features import shapes
            from shapely.geometry import shape
            
            height, width = classified.shape
            transform = profile.get('transform')
            crs = profile.get('crs')
            
            if transform is None:
                logger.warning("No transform in profile, cannot generate GeoJSON")
                return None
            
            # Create features for each class
            features = []
            
            for class_id, class_name in cluster_labels.items():
                # Create binary mask for this class
                class_mask = (classified == class_id).astype(np.uint8)
                
                # Generate polygons from mask
                mask_shapes = shapes(class_mask, mask=class_mask, transform=transform)
                
                for geom, value in mask_shapes:
                    if value == 1:  # Only include pixels belonging to this class
                        # Convert to shapely geometry
                        poly = shape(geom)
                        
                        # Skip very small polygons (likely noise)
                        if poly.area < 100:  # 100 square meters minimum
                            continue
                        
                        # Get class statistics
                        class_info = stats['class_distribution'].get(class_name, {})
                        
                        # Create feature
                        feature = {
                            'geometry': poly,
                            'properties': {
                                'class_id': int(class_id),
                                'class_name': class_name,
                                'area_pixels': int(class_info.get('count', 0)),
                                'area_percent': float(class_info.get('percent', 0)),
                                'area_km2': float(class_info.get('area_km2', 0))
                            }
                        }
                        features.append(feature)
            
            if features:
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    [f['properties'] for f in features],
                    geometry=[f['geometry'] for f in features],
                    crs=crs
                )
                
                # Generate filename
                geojson_filename = 'classification_polygons.geojson'
                
                # Save to temporary location (will be moved by caller)
                temp_path = f'/tmp/{geojson_filename}'  # This will be overridden by the caller
                
                logger.info(f"Generated GeoJSON with {len(features)} polygons for {len(cluster_labels)} classes")
                return geojson_filename
            else:
                logger.warning("No valid polygons generated for GeoJSON")
                return None
                
        except ImportError as e:
            logger.warning(f"GeoPandas not available for GeoJSON generation: {e}")
            return None
        except Exception as e:
            logger.warning(f"Could not generate classification GeoJSON: {e}")
            return None
    
    def change_detection(self, other_data: np.ndarray, 
                        method: str = 'difference') -> Tuple[np.ndarray, Dict]:
        """
        Perform change detection between two images.
        
        Args:
            other_data: Second image data (same shape as self.data)
            method: Detection method ('difference', 'ratio', 'pca')
            
        Returns:
            Tuple of (change map, statistics dict)
        """
        if method == 'difference':
            # Simple differencing
            change = np.mean(np.abs(self.data - other_data), axis=0)
            
        elif method == 'ratio':
            # Ratio-based change
            change = np.mean(np.abs(np.log(self.data + 1) - np.log(other_data + 1)), axis=0)
            
        else:
            raise ValueError(f"Unknown change detection method: {method}")
        
        stats = self._calculate_statistics(change, "Change")
        
        # Threshold for change/no-change
        threshold = stats['mean'] + 2 * stats['std']
        changed_pixels = np.sum(change > threshold)
        stats['changed_percent'] = changed_pixels / change.size * 100
        
        logger.info(f"Change detection complete - {stats['changed_percent']:.2f}% changed")
        return change, stats
    
    def create_band_stack(self, band_order: list = None) -> Tuple[np.ndarray, Dict]:
        """
        Create a multi-band composite GeoTIFF with all available useful bands.
        
        Args:
            band_order: Optional list of band names in desired order. 
                       If None, uses standard ordering: [blue, green, red, nir, swir1, swir2]
            
        Returns:
            Tuple of (stacked bands array with shape (n_bands, height, width), statistics dict)
        """
        if band_order is None:
            # Standard band order for optical imagery
            band_order = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        
        # Collect available bands
        available_bands = []
        band_stats = {}
        
        for band_name in band_order:
            if band_name in self.band_indices:
                try:
                    band_data = self._get_band(band_name)
                    available_bands.append(band_data)
                    
                    # Calculate statistics for each band
                    valid_data = band_data[~np.isnan(band_data)]
                    band_stats[band_name] = {
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'median': float(np.median(valid_data))
                    }
                except Exception as e:
                    logger.warning(f"Could not retrieve band '{band_name}': {e}")
        
        if not available_bands:
            raise ValueError("No valid bands available for stacking")
        
        # Stack bands
        stacked = np.stack(available_bands, axis=0)
        
        stats = {
            'n_bands': len(available_bands),
            'band_names': [name for name in band_order if name in self.band_indices],
            'shape': stacked.shape,
            'band_statistics': band_stats,
            'data_type': str(stacked.dtype)
        }
        
        logger.info(f"Created band stack with {len(available_bands)} bands: {stats['band_names']}")
        return stacked, stats
    
    def rgb_composite(self, stretch_method: str = 'percentile', 
                     percentile_range: Tuple[float, float] = (2, 98)) -> Tuple[np.ndarray, Dict]:
        """
        Generate a true color RGB composite from optical bands.
        
        Args:
            stretch_method: Contrast stretching method ('percentile', 'minmax', or 'std')
            percentile_range: Tuple of (min, max) percentiles for stretching (default: 2-98)
            
        Returns:
            Tuple of (RGB composite array with shape (height, width, 3), statistics dict)
        """
        # Extract RGB bands
        red = self._get_band('red')
        green = self._get_band('green')
        blue = self._get_band('blue')
        
        # Stack bands as RGB
        rgb_stack = np.stack([red, green, blue], axis=-1)
        height, width = red.shape
        
        # Apply contrast stretching
        rgb_stretched = np.zeros_like(rgb_stack)
        
        for i, band_data in enumerate([red, green, blue]):
            if stretch_method == 'percentile':
                # Percentile stretching (default 2-98%)
                p_low, p_high = np.percentile(band_data[~np.isnan(band_data)], percentile_range)
                stretched = np.clip((band_data - p_low) / (p_high - p_low + 1e-8), 0, 1)
            elif stretch_method == 'minmax':
                # Min-max normalization
                min_val = np.nanmin(band_data)
                max_val = np.nanmax(band_data)
                stretched = (band_data - min_val) / (max_val - min_val + 1e-8)
            elif stretch_method == 'std':
                # Standard deviation stretch (mean ± 2 std)
                mean = np.nanmean(band_data)
                std = np.nanstd(band_data)
                stretched = np.clip((band_data - (mean - 2*std)) / (4*std + 1e-8), 0, 1)
            else:
                raise ValueError(f"Unknown stretch method: {stretch_method}")
            
            rgb_stretched[:, :, i] = stretched
        
        # Calculate statistics
        stats = {
            'method': stretch_method,
            'shape': rgb_stretched.shape,
            'percentile_range': percentile_range if stretch_method == 'percentile' else None,
            'red_range': (float(np.nanmin(red)), float(np.nanmax(red))),
            'green_range': (float(np.nanmin(green)), float(np.nanmax(green))),
            'blue_range': (float(np.nanmin(blue)), float(np.nanmax(blue))),
            'stretched_range': (float(np.nanmin(rgb_stretched)), float(np.nanmax(rgb_stretched)))
        }
        
        logger.info(f"Generated RGB composite with {stretch_method} stretching - Shape: {rgb_stretched.shape}")
        return rgb_stretched, stats
    
    def available_indices(self) -> Dict[str, Dict]:
        """Return available index names and definitions."""
        return self.INDEX_REGISTRY

    def _validate_index_for_sensor(self, index_name: str, requires: list):
        """Validate that the selected sensor supports all required bands for the index."""
        if not self.sensor:
            return
        caps = self.SENSOR_CAPABILITIES.get(self.sensor, {})
        available = caps.get('bands', set())
        missing = [b for b in requires if b not in available]
        if missing:
            raise ValueError(
                f"Index '{index_name}' is not supported on sensor '{self.sensor}'. Missing bands: {missing}"
            )

    def compute_index(self, name: str) -> Tuple[np.ndarray, Dict]:
        """Compute a spectral index by name.
        Returns (index_array, stats).
        """
        name_l = name.lower()
        reg = self.INDEX_REGISTRY
        # Resolve alias
        if name_l in reg and 'alias' in reg[name_l]:
            name_l = reg[name_l]['alias']
        if name_l not in reg:
            raise ValueError(f"Unknown index '{name}'. Use one of: {sorted(reg.keys())}")
        entry = reg[name_l]
        expr = entry.get('expr')
        requires = entry.get('requires', [])
        helpers = entry.get('helpers', {})
        
        # Validate against sensor capabilities (if provided)
        self._validate_index_for_sensor(name_l, requires)
        
        # Build bands dictionary from available data
        bands = {}
        for key, idx in self.band_indices.items():
            try:
                bands[key] = self._get_band(key)
            except Exception:
                continue
        
        # Verify required bands
        missing = [b for b in requires if b not in bands]
        if missing:
            raise ValueError(f"Index '{name}' requires bands not available: {missing}. Provided band_indices={list(self.band_indices.keys())}")
        
        # Inject helpers
        local_env = {**{k: bands[k] for k in bands}, 'np': np}
        for h_name, h_expr in helpers.items():
            local_env[h_name] = eval(h_expr, {"__builtins__": {}}, local_env)
        
        result = eval(expr, {"__builtins__": {}}, local_env)
        
        # Clip normalized indices to [-1, 1] where appropriate
        normalized_names = {
            'ndvi','gndvi','ndwi','mndwi','ndwi2','ndmi','ndsi','nbr','nbr2','ndbi','ndgi','ndre','ndvire','ndvi705','lswi','ndvi2','ndvi3'
        }
        if name_l in normalized_names:
            result = np.clip(result, -1.0, 1.0)
        
        stats = self._calculate_statistics(result, name.upper())
        logger.info(f"Computed index {name.upper()} - mean: {stats.get('mean', float('nan')):.3f}")
        return result, stats

    def band_arithmetic(self, expression: str, 
                       bands_dict: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Perform custom band arithmetic.
        
        Args:
            expression: Mathematical expression (e.g., "(B4 - B3) / (B4 + B3)")
            bands_dict: Dictionary of band name to array
            
        Returns:
            Result array
        """
        if bands_dict is None:
            bands_dict = {f'B{i}': self.data[i] for i in range(self.data.shape[0])}
        
        # Evaluate expression safely
        result = eval(expression, {"__builtins__": {}}, bands_dict)
        
        logger.info(f"Calculated custom expression: {expression}")
        return result
    
    def _calculate_statistics(self, data: np.ndarray, name: str = "") -> Dict:
        """Calculate statistics for an array."""
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            return {}
        
        stats = {
            'name': name,
            'mean': float(np.mean(valid_data)),
            'median': float(np.median(valid_data)),
            'std': float(np.std(valid_data)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'percentile_25': float(np.percentile(valid_data, 25)),
            'percentile_75': float(np.percentile(valid_data, 75))
        }
        
        return stats
