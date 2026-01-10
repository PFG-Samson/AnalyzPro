"""SAR (Synthetic Aperture Radar) image analysis module."""

import numpy as np
from scipy import ndimage
from scipy.ndimage import generic_filter
from skimage.feature import graycomatrix, graycoprops
from typing import Dict, Tuple
from ..utils import get_logger

logger = get_logger(__name__)


class SARAnalyzer:
    """SAR image analysis with speckle filtering and texture analysis."""
    
    def __init__(self, data: np.ndarray, profile: dict = None, mask: np.ndarray = None,
                 sensor: str = None, band_names: list = None, nodata_values: list = None,
                 calibrated: bool = None, radiometry: str = None):
        """
        Initialize SARAnalyzer.
        
        Args:
            data: SAR image data (can be single or multi-band)
            profile: Optional rasterio profile (to read pixel size, nodata, etc.)
            mask: Optional boolean mask (H, W) where True indicates NoData/invalid pixels
        """
        self.data = data
        if data.ndim == 2:
            # If 2D, expand to 3D with a single band for consistency
            self.data = data[np.newaxis, :, :]
        elif data.ndim == 3:
            # Data is already in the expected (bands, height, width) format
            pass
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}. Expected 2 or 3.")
        
        # Derive pixel area from profile if available (km^2)
        self.pixel_area_km2 = 0.0001  # default 10m x 10m = 100 m^2 = 1e-4 km^2
        if profile is not None:
            try:
                transform = profile.get('transform')
                if transform is not None:
                    # Affine (a, b, c, d, e, f) where a=px width, e= -px height typically
                    if hasattr(transform, 'a') and hasattr(transform, 'e'):
                        px_w = float(transform.a)
                        px_h = float(abs(transform.e))
                    else:
                        px_w = float(transform[0])
                        px_h = float(abs(transform[4]))
                    self.pixel_area_km2 = (px_w * px_h) / 1e6
            except Exception:
                pass
        
        # Infer calibration/radiometry and band names
        self.sensor = sensor
        self.band_names = band_names
        self.radiometry = radiometry
        if calibrated is None:
            dtype_str = None
            if profile is not None:
                dtype_str = profile.get('dtype')
            if dtype_str is not None:
                self.calibrated = not ("int" in str(dtype_str).lower() and "float" not in str(dtype_str).lower())
            else:
                self.calibrated = np.issubdtype(self.data.dtype, np.floating)
        else:
            self.calibrated = bool(calibrated)
        
        # Build and apply NoData mask if provided or derivable from profile or params
        self.mask = None
        if mask is not None and mask.shape == self.data.shape[-2:]:
            self.mask = mask.astype(bool)
        else:
            nodata_list = []
            if profile is not None and profile.get('nodata') is not None:
                nodata_list.append(profile.get('nodata'))
            if nodata_values is not None:
                try:
                    nodata_list.extend(list(nodata_values))
                except Exception:
                    nodata_list.append(nodata_values)
            # Common for preview/byte SAR: zero-valued background
            if not self.calibrated and np.issubdtype(self.data.dtype, np.integer):
                nodata_list.append(0)
            if len(nodata_list) > 0:
                m = np.zeros(self.data.shape[-2:], dtype=bool)
                for v in nodata_list:
                    try:
                        m |= np.any(self.data == v, axis=0)
                    except Exception:
                        continue
                self.mask = m if np.any(m) else None
        
        if self.mask is not None:
            # Set masked pixels to NaN across all bands to exclude from stats and filters
            self.data = self.data.astype(float)
            self.data[:, self.mask] = np.nan
        
        logger.info(f"Initialized SARAnalyzer with shape {self.data.shape}, pixel_area_km2={self.pixel_area_km2:.8f}")
    
    def _normalize01(self, img: np.ndarray) -> np.ndarray:
        """Normalize an array to 0-1 using robust percentiles (1-99%)."""
        img = img.astype(float)
        if np.isnan(img).all():
            return img
        lo = np.nanpercentile(img, 1)
        hi = np.nanpercentile(img, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-10:
            lo = np.nanmin(img)
            hi = np.nanmax(img)
        return np.clip((img - lo) / (hi - lo + 1e-10), 0, 1)
    
    def _select_band(self, band=None) -> np.ndarray:
        """Select a band by name ('VV','VH','HH','HV','R','G','B') or index; default first band."""
        if band is None:
            return self.data[0]
        # Index selection
        if isinstance(band, (int, np.integer)):
            idx = int(band)
            idx = max(0, min(self.data.shape[0] - 1, idx))
            return self.data[idx]
        # Name selection
        name = str(band).upper()
        # If band_names provided, try exact match
        if self.band_names:
            try:
                idx = [bn.upper() for bn in self.band_names].index(name)
                return self.data[idx]
            except Exception:
                pass
        default_map = {'VV': 0, 'VH': 1, 'HH': 0, 'HV': 1, 'R': 0, 'G': 1, 'B': 2}
        idx = default_map.get(name, 0)
        idx = max(0, min(self.data.shape[0] - 1, idx))
        return self.data[idx]
    
    def _area_to_pixels(self, min_area_km2: float = None, fallback_pixels: int = None) -> int:
        """Convert area (km^2) to pixel count using image pixel area; fallback to given pixels."""
        if min_area_km2 is not None and min_area_km2 > 0:
            px_area = max(self.pixel_area_km2, 1e-12)
            return int(np.ceil(float(min_area_km2) / px_area))
        return int(fallback_pixels or 0)
    
    def _apply_speckle_filter(self, method: str = 'lee', window_size: int = 5, looks: int = 1) -> np.ndarray:
        """
        Internal preprocessing: Apply speckle filter.
        
        Args:
            method: Filter method ('lee', 'median')
            window_size: Filter window size
            looks: Number of looks (affects Lee filter weighting)
            
        Returns:
            Filtered array
        """
        # Prepare input by filling NaNs with local median to avoid NaN propagation
        def _fill_nan(img):
            if np.isnan(img).any():
                med = np.nanmedian(img)
                return np.nan_to_num(img, nan=float(med))
            return img
        
        if method == 'lee':
            def _lee_filter_single(img, win_size, n_looks):
                img = _fill_nan(img)
                cu = 1.0 / np.sqrt(max(n_looks, 1))
                cu2 = cu * cu
                mean = ndimage.uniform_filter(img, win_size)
                sqr_mean = ndimage.uniform_filter(img ** 2, win_size)
                variance = np.maximum(sqr_mean - mean ** 2, 0.0)
                weights = variance / (variance + mean ** 2 * cu2 + 1e-10)
                return mean + weights * (img - mean)
            
            filtered_data = np.zeros_like(self.data, dtype=float)
            for i in range(self.data.shape[0]):
                filtered_data[i] = _lee_filter_single(self.data[i], window_size, looks)
        else:  # median
            filtered_data = np.zeros_like(self.data, dtype=float)
            for i in range(self.data.shape[0]):
                filtered_data[i] = ndimage.median_filter(_fill_nan(self.data[i]), size=window_size)
        
        # Re-apply mask after filtering (keep NaNs at masked pixels)
        if self.mask is not None:
            filtered_data[:, self.mask] = np.nan
        
        return filtered_data
    
    def _to_db(self, data: np.ndarray) -> np.ndarray:
        """Convert intensity to dB scale."""
        return 10 * np.log10(data + 1e-10)
    
    def _calculate_texture_features(self, data: np.ndarray) -> Dict:
        """Internal: Calculate GLCM texture features for classification."""
        img_norm = ((data - data.min()) / (data.max() - data.min() + 1e-10) * 255).astype(np.uint8)
        glcm = graycomatrix(img_norm, distances=[1], angles=[0], 
                           levels=256, symmetric=True, normed=True)
        
        return {
            'contrast': float(graycoprops(glcm, 'contrast').mean()),
            'dissimilarity': float(graycoprops(glcm, 'dissimilarity').mean()),
            'homogeneity': float(graycoprops(glcm, 'homogeneity').mean()),
            'energy': float(graycoprops(glcm, 'energy').mean()),
            'correlation': float(graycoprops(glcm, 'correlation').mean())
        }
    
    def oil_spill_detection(self, window_size: int = 51, k_threshold: float = 1.5,
                           min_area_pixels: int = 100, looks: int = 1,
                           band=None, min_area_km2: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Detect potential oil spills using adaptive local threshold on dark patches.
        Product: GRD, Polarization: VV preferred
        
        Args:
            window_size: Local threshold window size
            k_threshold: Standard deviation multiplier for threshold
            min_area_pixels: Minimum area in pixels to filter small artifacts (used if min_area_km2 is None)
            looks: Number of looks for Lee filter
            band: Band selector (e.g., 'VV','VH','HH', or index); defaults to first band
            min_area_km2: Minimum slick area in km^2 (overrides min_area_pixels if provided)
            
        Returns:
            Tuple of (oil spill mask, statistics dict)
        """
        from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
        from skimage.measure import label
        from scipy.ndimage import uniform_filter
        
        # Select requested band (defaults to first)
        vv = self._select_band(band).astype(float)
        
        # Apply speckle filter as preprocessing
        filtered = self._apply_speckle_filter(method='lee', window_size=5, looks=looks)
        vv_filtered = filtered[0]
        
        # Adaptive local threshold for dark patches
        local_mean = uniform_filter(vv_filtered, size=window_size)
        local_std = np.sqrt(uniform_filter((vv_filtered - local_mean)**2, size=window_size))
        threshold = local_mean - k_threshold * local_std
        
        # Detect dark patches (potential oil slicks)
        dark_mask = vv_filtered < threshold
        
        # Clean small speckle and objects; avoid memory-heavy labeling on huge images
        min_size = self._area_to_pixels(min_area_km2, min_area_pixels)
        huge = dark_mask.size > 200_000_000  # ~200M pixels threshold
        if huge:
            # Morphological cleanup only (approximate) to avoid full connected-component labeling
            dark_mask_cleaned = binary_opening(dark_mask, selem=disk(1))
            dark_mask_cleaned = binary_closing(dark_mask_cleaned, selem=disk(1))
            # Approximate slick count and bboxes using downsampled labeling
            ds_factor = int(np.ceil(np.sqrt(dark_mask.size / 50_000_000)))  # target ~50M pixels
            ds_factor = max(2, ds_factor)
            ds_mask = dark_mask_cleaned[::ds_factor, ::ds_factor]
            labeled_small = label(ds_mask)
            num_slicks = int(labeled_small.max())
            # Build per-slick metadata scaled back to full-res pixel coords
            slick_list = []
            if num_slicks > 0:
                from skimage.measure import regionprops
                for r in regionprops(labeled_small):
                    x0, y0, x1, y1 = r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]
                    slick_list.append({
                        'label': int(r.label),
                        'area_pixels': int(r.area) * (ds_factor * ds_factor),
                        'area_km2': float(r.area * (ds_factor * ds_factor) * self.pixel_area_km2),
                        'bbox_pixels': (int(x0 * ds_factor), int(y0 * ds_factor), int(x1 * ds_factor), int(y1 * ds_factor))
                    })
        else:
            # Remove small objects precisely
            dark_mask_cleaned = remove_small_objects(dark_mask, min_size=min_size)
            # Label connected regions
            labeled_mask = label(dark_mask_cleaned)
            num_slicks = labeled_mask.max()
        
        # Calculate statistics
        slick_pixels = int(np.sum(dark_mask_cleaned))
        slick_area_km2 = slick_pixels * self.pixel_area_km2
        
        # Backscatter/intensity analysis for stats
        vv_metric = self._to_db(vv_filtered) if self.calibrated else self._normalize01(vv_filtered)
        
        stats = {
            'num_detected_slicks': int(num_slicks),
            'total_slick_pixels': int(slick_pixels),
            'total_slick_area_km2': float(slick_area_km2),
            'coverage_percent': float(slick_pixels / dark_mask.size * 100),
'mean_backscatter_slick_db': float(np.mean(vv_metric[dark_mask_cleaned])) if slick_pixels > 0 else 0,
            'mean_backscatter_background_db': float(np.mean(vv_metric[~dark_mask_cleaned])),
            'k_threshold': k_threshold,
            'window_size': window_size
        }
        
        # Individual slick sizes and per-slick metadata
        if num_slicks > 0:
            from skimage.measure import regionprops
            slick_sizes = []
            slick_list = slick_list if 'slick_list' in locals() else []
            regions = [] if 'labeled_mask' not in locals() else regionprops(labeled_mask)
            for r in regions:
                size_km2 = r.area * self.pixel_area_km2
                slick_sizes.append(size_km2)
                x0, y0, x1, y1 = r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]
                slick_list.append({
                    'label': int(r.label),
                    'area_km2': float(size_km2),
                    'area_pixels': int(r.area),
                    'bbox_pixels': (int(x0), int(y0), int(x1), int(y1))
                })
            # If we used downsampled labeling, accumulate sizes from that list as well
            if 'slick_list' in locals() and regions == []:
                slick_sizes.extend([s['area_km2'] for s in slick_list])
            stats['largest_slick_km2'] = float(max(slick_sizes))
            stats['smallest_slick_km2'] = float(min(slick_sizes))
            stats['mean_slick_size_km2'] = float(np.mean(slick_sizes))
            stats['slicks'] = slick_list
        
        logger.info(f"Oil spill detection - Found {num_slicks} potential slicks covering {slick_area_km2:.2f} km²")
        return dark_mask_cleaned.astype(int), stats
    
    def ship_detection(self, cfar_window: int = 50, guard_cells: int = 5,
                      false_alarm_rate: float = 1e-5, min_ship_pixels: int = 5,
                      band=None, min_ship_area_km2: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Detect ships using CFAR (Constant False Alarm Rate) on bright targets.
        Product: GRD, Polarization: VV primary
        
        Args:
            cfar_window: Background estimation window size
            guard_cells: Guard cells around target
            false_alarm_rate: False alarm rate for threshold
            min_ship_pixels: Minimum pixels to classify as ship
            
        Returns:
            Tuple of (ship detection mask, statistics dict)
        """
        from skimage.morphology import remove_small_objects, binary_dilation, binary_opening, disk, binary_closing
        from skimage.measure import regionprops, label
        
        # Select requested band (defaults to first)
        vv = self._select_band(band).astype(float)
        
        # CFAR-like detection with calibration-aware stats
        # Fill NaNs to avoid bias in local stats
        if np.isnan(vv).any():
            vv = np.nan_to_num(vv, nan=float(np.nanmedian(vv)))
        
        if self.calibrated:
            # Use Gaussian CFAR on dB-equivalent signal proxy
            from scipy.stats import norm
            k = norm.ppf(1 - false_alarm_rate)
            local_mean = ndimage.uniform_filter(vv, size=cfar_window)
            local_std = np.sqrt(np.maximum(ndimage.uniform_filter((vv - local_mean)**2, size=cfar_window), 0.0))
            threshold = local_mean + k * local_std
        else:
            # Use robust median + MAD for byte/preview products
            med = ndimage.median_filter(vv, size=cfar_window)
            abs_dev = np.abs(vv - med)
            mad = ndimage.median_filter(abs_dev, size=cfar_window)
            sigma = 1.4826 * mad + 1e-6
            threshold = med + 6.0 * sigma  # aggressive to suppress clutter
        
        ship_mask = vv > threshold
        
        # Morphological cleanup: open then slight dilate to connect ship pixels
        ship_mask = binary_opening(ship_mask, selem=disk(1))
        ship_mask = binary_dilation(ship_mask, selem=disk(1))
        
        # Remove small objects (noise) with size-aware mode
        min_size = self._area_to_pixels(min_ship_area_km2, min_ship_pixels)
        huge = ship_mask.size > 200_000_000
        if huge:
            ship_mask_cleaned = binary_opening(ship_mask, selem=disk(1))
            ship_mask_cleaned = binary_dilation(ship_mask_cleaned, selem=disk(1))
            # Approximate properties via downsampled labeling
            ds_factor = int(np.ceil(np.sqrt(ship_mask.size / 50_000_000)))
            ds_factor = max(2, ds_factor)
            labeled_small = label(ship_mask_cleaned[::ds_factor, ::ds_factor])
            num_ships = int(labeled_small.max())
            regions = regionprops(labeled_small)
            ship_properties = []
            filtered_mask = ship_mask_cleaned  # keep cleaned mask at full res
            for reg in regions:
                x0, y0, x1, y1 = reg.bbox[1], reg.bbox[0], reg.bbox[3], reg.bbox[2]
                area_pixels = int(reg.area) * (ds_factor * ds_factor)
                area_m2 = area_pixels * (self.pixel_area_km2 * 1e6)
                ship_properties.append({
                    'label': int(reg.label),
                    'area_m2': float(area_m2),
                    'centroid_rc': (float(reg.centroid[0] * ds_factor), float(reg.centroid[1] * ds_factor)),
                    'bbox_pixels': (int(x0 * ds_factor), int(y0 * ds_factor), int(x1 * ds_factor), int(y1 * ds_factor)),
                    'eccentricity': float(reg.eccentricity)
                })
        else:
            ship_mask_cleaned = remove_small_objects(ship_mask, min_size=min_size)
            
            # Label ships
            labeled_ships = label(ship_mask_cleaned)
            num_ships = labeled_ships.max()
            
            # Extract ship properties
            regions = regionprops(labeled_ships)
            
            ship_properties = []
            filtered_mask = np.zeros_like(ship_mask_cleaned, dtype=bool)
            min_size = self._area_to_pixels(min_ship_area_km2, min_ship_pixels)
            for region in regions:
                if region.area < max(min_size, 8):
                    continue
                # Filter out overly elongated noise and huge regions
                if not (0.4 <= region.eccentricity <= 0.99):
                    continue
                if region.area > ship_mask_cleaned.size * 0.05:
                    continue
                rr0, cc0, rr1, cc1 = region.bbox
                filtered_mask[region.coords[:, 0], region.coords[:, 1]] = True
                area_m2 = region.area * (self.pixel_area_km2 * 1e6)
                ship_properties.append({
                    'label': int(region.label),
                    'area_m2': float(area_m2),
                    'centroid_rc': (float(region.centroid[0]), float(region.centroid[1])),
                    'bbox_pixels': (int(cc0), int(rr0), int(cc1), int(rr1)),  # x0,y0,x1,y1
                    'eccentricity': float(region.eccentricity)
                })
        
        ship_mask_cleaned = filtered_mask if 'filtered_mask' in locals() else ship_mask_cleaned
        
        stats = {
            'num_detected_ships': int(num_ships),
            'total_ship_pixels': int(np.sum(ship_mask_cleaned)),
            'false_alarm_rate': false_alarm_rate,
            'cfar_k_value': float(k),
            'ships': ship_properties
        }
        
        if num_ships > 0:
            areas = [s['area_m2'] for s in ship_properties]
            stats['largest_ship_m2'] = float(max(areas))
            stats['smallest_ship_m2'] = float(min(areas))
            stats['mean_ship_size_m2'] = float(np.mean(areas))
        
        logger.info(f"Ship detection - Found {num_ships} potential ships")
        return ship_mask_cleaned.astype(int), stats
    
    def crop_monitoring(self, vv_data: np.ndarray = None, vh_data: np.ndarray = None,
                       temporal_stack: list = None) -> Tuple[np.ndarray, Dict]:
        """
        Agriculture & crop monitoring using dual-pol time-series or single scene.
        Product: GRD time-series, Polarization: VV + VH
        
        Args:
            vv_data: VV polarization (if None, uses self.data[0])
            vh_data: VH polarization (if None, uses self.data[1] if available)
            temporal_stack: List of additional temporal scenes for time-series (optional)
            
        Returns:
            Tuple of (crop vigor index, statistics dict)
        """
        # Get VV and VH polarizations
        if vv_data is None:
            vv_data = self.data[0]
        if vh_data is None:
            if self.data.shape[0] > 1:
                vh_data = self.data[1]
            else:
                raise ValueError("VH polarization required for crop monitoring")
        
        # Apply preprocessing
        filtered_data = self._apply_speckle_filter(method='lee', window_size=7)
        vv_filtered = filtered_data[0]
        vh_filtered = filtered_data[1] if filtered_data.shape[0] > 1 else vh_data
        
        # Use metric (dB if calibrated, else normalized intensity)
        vv_db = self._to_db(vv_filtered) if self.calibrated else self._normalize01(vv_filtered)
        vh_db = self._to_db(vh_filtered) if self.calibrated else self._normalize01(vh_filtered)
        
        # Crop Vigor Index: ratio-based proxy for vegetation volume
        # Higher VH relative to VV indicates more vegetation volume scattering
        rvi = (4 * vh_filtered) / (vv_filtered + vh_filtered + 1e-10)  # Radar Vegetation Index
        rvi = np.clip(rvi, 0, 1)
        
        # Cross-pol ratio
        cross_pol_ratio = vh_filtered / (vv_filtered + 1e-10)
        
        # Classify vegetation conditions based on backscatter
        crop_vigor_class = np.zeros_like(vv_db, dtype=int)
        # 0: No vegetation/bare soil, 1: Low vigor, 2: Moderate, 3: High vigor
        crop_vigor_class[rvi < 0.2] = 0  # Bare soil/no veg
        crop_vigor_class[(rvi >= 0.2) & (rvi < 0.4)] = 1  # Low vigor
        crop_vigor_class[(rvi >= 0.4) & (rvi < 0.6)] = 2  # Moderate
        crop_vigor_class[rvi >= 0.6] = 3  # High vigor
        
        stats = self._calculate_statistics(rvi, "Crop Vigor (RVI)")
        stats['vv_mean_db'] = float(np.mean(vv_db))
        stats['vh_mean_db'] = float(np.mean(vh_db))
        stats['cross_pol_ratio_mean'] = float(np.mean(cross_pol_ratio))
        
        # Vegetation cover percentages
        pixel_area_km2 = self.pixel_area_km2
        stats['bare_soil_percent'] = float(np.sum(crop_vigor_class == 0) / crop_vigor_class.size * 100)
        stats['low_vigor_percent'] = float(np.sum(crop_vigor_class == 1) / crop_vigor_class.size * 100)
        stats['moderate_vigor_percent'] = float(np.sum(crop_vigor_class == 2) / crop_vigor_class.size * 100)
        stats['high_vigor_percent'] = float(np.sum(crop_vigor_class == 3) / crop_vigor_class.size * 100)
        stats['vegetated_area_km2'] = float(np.sum(crop_vigor_class > 0) * pixel_area_km2)
        
        logger.info(f"Crop monitoring - Mean RVI: {stats['mean']:.3f}, {stats['high_vigor_percent']:.1f}% high vigor")
        return rvi, stats
    
    def land_cover_classification(self, num_classes: int = 4) -> Tuple[np.ndarray, Dict]:
        """
        Land cover classification using SAR features and unsupervised clustering.
        Product: GRD (dual-pol preferred), Polarization: VV + VH
        
        Args:
            num_classes: Number of land cover classes
            
        Returns:
            Tuple of (classified map, statistics dict)
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Collect features
        vv_data = self.data[0]
        
        # Convert to metric (dB if calibrated, else normalized intensity)
        vv_db = self._to_db(vv_data) if self.calibrated else self._normalize01(vv_data)
        
        features_list = [vv_db.flatten()]
        feature_names = ['VV_metric']
        
        # Add VH if available
        if self.data.shape[0] > 1:
            vh_data = self.data[1]
            vh_db = self._to_db(vh_data) if self.calibrated else self._normalize01(vh_data)
            features_list.append(vh_db.flatten())
            feature_names.append('VH_metric')
            
            # Add ratio
            ratio = vv_db - vh_db
            features_list.append(ratio.flatten())
            feature_names.append('VV_VH_Ratio')
        
        # Add texture features (simplified)
        texture = self._calculate_texture_features(vv_data)
        contrast_map = ndimage.generic_filter(vv_data, np.std, size=5)
        features_list.append(contrast_map.flatten())
        feature_names.append('Texture_Std')
        
        # Stack features
        feature_matrix = np.column_stack(features_list)
        
        # Remove NaN/Inf
        valid_mask = np.all(np.isfinite(feature_matrix), axis=1)
        feature_matrix_clean = feature_matrix[valid_mask]
        
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix_clean)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
        labels_clean = kmeans.fit_predict(features_scaled)
        
        # Map back to image space
        classified_map = np.full(vv_data.size, -1, dtype=int)
        classified_map[valid_mask] = labels_clean
        classified_map = classified_map.reshape(vv_data.shape)
        
        # Calculate class statistics
        pixel_area_km2 = self.pixel_area_km2
        class_stats = {}
        for i in range(num_classes):
            class_pixels = np.sum(classified_map == i)
            class_stats[f'class_{i}_percent'] = float(class_pixels / np.sum(classified_map >= 0) * 100)
            class_stats[f'class_{i}_area_km2'] = float(class_pixels * pixel_area_km2)
            class_stats[f'class_{i}_mean_vv_db'] = float(np.mean(vv_db[classified_map == i]))
        
        stats = {
            'num_classes': num_classes,
            'feature_names': feature_names,
            **class_stats
        }
        
        logger.info(f"Land cover classification - {num_classes} classes identified")
        return classified_map, stats
    
    def biomass_estimation(self, reference_biomass: float = None) -> Tuple[np.ndarray, Dict]:
        """
        Forest biomass estimation using SAR backscatter and texture.
        Product: GRD, Polarization: VV + VH (dual-pol preferred)
        Note: Requires calibration with field data for absolute biomass.
        
        Args:
            reference_biomass: Reference biomass value for calibration (tons/ha)
            
        Returns:
            Tuple of (biomass index map, statistics dict)
        """
        # Get VV and VH
        vv_data = self.data[0]
        
        # Use metric (dB if calibrated, else normalized intensity)
        vv_db = self._to_db(vv_data) if self.calibrated else self._normalize01(vv_data)
        
        # Calculate texture (roughness proxy for structure)
        texture_features = self._calculate_texture_features(vv_data)
        texture_std = ndimage.generic_filter(vv_data, np.std, size=7)
        
        # Biomass index: combination of backscatter and texture
        # Higher backscatter + higher texture = more biomass
        # Normalize components
        vv_norm = (vv_db - np.percentile(vv_db, 5)) / (np.percentile(vv_db, 95) - np.percentile(vv_db, 5) + 1e-10)
        vv_norm = np.clip(vv_norm, 0, 1)
        
        texture_norm = (texture_std - texture_std.min()) / (texture_std.max() - texture_std.min() + 1e-10)
        
        # Biomass index (0-1 scale)
        biomass_index = 0.7 * vv_norm + 0.3 * texture_norm
        
        # If dual-pol available, include VH
        if self.data.shape[0] > 1:
            vh_data = self.data[1]
            vh_db = self._to_db(vh_data) if self.calibrated else self._normalize01(vh_data)
            vh_norm = (vh_db - np.percentile(vh_db, 5)) / (np.percentile(vh_db, 95) - np.percentile(vh_db, 5) + 1e-10)
            vh_norm = np.clip(vh_norm, 0, 1)
            biomass_index = 0.5 * vv_norm + 0.3 * vh_norm + 0.2 * texture_norm
        
        # Classify biomass levels
        biomass_class = np.zeros_like(biomass_index, dtype=int)
        biomass_class[biomass_index < 0.2] = 0  # Very low/no forest
        biomass_class[(biomass_index >= 0.2) & (biomass_index < 0.4)] = 1  # Low
        biomass_class[(biomass_index >= 0.4) & (biomass_index < 0.6)] = 2  # Medium
        biomass_class[(biomass_index >= 0.6) & (biomass_index < 0.8)] = 3  # High
        biomass_class[biomass_index >= 0.8] = 4  # Very high
        
        stats = self._calculate_statistics(biomass_index, "Biomass Index")
        stats['vv_mean_db'] = float(np.mean(vv_db))
        stats['texture_contrast'] = texture_features['contrast']
        stats['texture_homogeneity'] = texture_features['homogeneity']
        
        # Area statistics
        pixel_area_km2 = self.pixel_area_km2
        stats['very_low_biomass_percent'] = float(np.sum(biomass_class == 0) / biomass_class.size * 100)
        stats['low_biomass_percent'] = float(np.sum(biomass_class == 1) / biomass_class.size * 100)
        stats['medium_biomass_percent'] = float(np.sum(biomass_class == 2) / biomass_class.size * 100)
        stats['high_biomass_percent'] = float(np.sum(biomass_class == 3) / biomass_class.size * 100)
        stats['very_high_biomass_percent'] = float(np.sum(biomass_class == 4) / biomass_class.size * 100)
        stats['forest_area_km2'] = float(np.sum(biomass_class > 0) * pixel_area_km2)
        
        logger.info(f"Biomass estimation - Mean index: {stats['mean']:.3f}, Forest area: {stats['forest_area_km2']:.2f} km²")
        return biomass_index, stats
    
    def wildfire_burn_mapping(self, pre_fire_data: np.ndarray = None,
                             post_fire_data: np.ndarray = None,
                             use_dual_pol: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Wildfire burn area mapping using pre/post-fire SAR change detection.
        Product: GRD time-series, Polarization: VV & VH (dual-pol preferred)
        
        Args:
            pre_fire_data: Pre-fire SAR data (if None, uses self.data as pre-fire)
            post_fire_data: Post-fire SAR data (required for change detection)
            use_dual_pol: Use VH for better vegetation loss detection
            
        Returns:
            Tuple of (burn severity map, statistics dict)
        """
        if post_fire_data is None:
            raise ValueError("post_fire_data is required for burn mapping")
        
        # Ensure consistent dimensions
        if post_fire_data.ndim == 2:
            post_fire_data = post_fire_data[np.newaxis, :, :]
        
        if pre_fire_data is None:
            pre_fire_data = self.data
        elif pre_fire_data.ndim == 2:
            pre_fire_data = pre_fire_data[np.newaxis, :, :]
        
        # Apply speckle filtering
        pre_filtered = self._apply_speckle_filter(method='lee', window_size=7)
        # Create temp analyzer for post-fire
        post_analyzer = SARAnalyzer(post_fire_data)
        post_filtered = post_analyzer._apply_speckle_filter(method='lee', window_size=7)
        
        # Use VH if available (better for vegetation volume loss)
        if use_dual_pol and pre_filtered.shape[0] > 1 and post_filtered.shape[0] > 1:
            pre_band = pre_filtered[1]  # VH
            post_band = post_filtered[1]  # VH
            band_name = "VH"
        else:
            pre_band = pre_filtered[0]  # VV
            post_band = post_filtered[0]  # VV
            band_name = "VV"
        
        # Use metric (dB if calibrated, else normalized intensity)
        pre_db = self._to_db(pre_band) if self.calibrated else self._normalize01(pre_band)
        post_db = self._to_db(post_band) if self.calibrated else self._normalize01(post_band)
        
        # Delta metric (burned areas show decrease in VH)
        delta_db = pre_db - post_db  # Positive = decrease (burn)
        
        # Burn severity classification
        # Based on magnitude of backscatter decrease
        burn_severity = np.zeros_like(delta_db, dtype=int)
        burn_severity[delta_db < 1] = 0  # No burn
        burn_severity[(delta_db >= 1) & (delta_db < 2)] = 1  # Low severity
        burn_severity[(delta_db >= 2) & (delta_db < 3)] = 2  # Moderate severity
        burn_severity[(delta_db >= 3) & (delta_db < 4)] = 3  # High severity
        burn_severity[delta_db >= 4] = 4  # Very high severity
        
        # Calculate burn statistics
        pixel_area_km2 = self.pixel_area_km2
        burned_pixels = int(np.sum(burn_severity > 0))
        burned_area_km2 = burned_pixels * pixel_area_km2
        
        stats = self._calculate_statistics(delta_db, "Delta Backscatter (dB)")
        stats['band_used'] = band_name
        stats['burned_area_km2'] = float(burned_area_km2)
        stats['burned_percent'] = float(burned_pixels / burn_severity.size * 100)
        stats['unburned_percent'] = float(np.sum(burn_severity == 0) / burn_severity.size * 100)
        stats['low_severity_percent'] = float(np.sum(burn_severity == 1) / burn_severity.size * 100)
        stats['moderate_severity_percent'] = float(np.sum(burn_severity == 2) / burn_severity.size * 100)
        stats['high_severity_percent'] = float(np.sum(burn_severity == 3) / burn_severity.size * 100)
        stats['very_high_severity_percent'] = float(np.sum(burn_severity == 4) / burn_severity.size * 100)
        stats['mean_backscatter_decrease_db'] = float(np.mean(delta_db[burn_severity > 0])) if burned_pixels > 0 else 0
        
        logger.info(f"Wildfire burn mapping - {burned_area_km2:.2f} km² burned ({stats['burned_percent']:.2f}%)")
        return burn_severity, stats
    
    def geology_terrain_analysis(self, dem_data: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Geology and terrain analysis using SAR texture and backscatter.
        Product: GRD, Polarization: VV or VV+VH
        
        Args:
            dem_data: Optional DEM data for slope/aspect analysis
            
        Returns:
            Tuple of (roughness map, statistics dict)
        """
        from skimage.filters import sobel
        from skimage.morphology import binary_closing, disk
        
        vv_data = self.data[0]
        
        # Use metric (dB if calibrated, else normalized intensity)
        vv_metric = self._to_db(vv_data) if self.calibrated else self._normalize01(vv_data)
        
        # Calculate texture features for roughness
        texture_features = self._calculate_texture_features(vv_data)
        
        # Local standard deviation as roughness proxy
        roughness = ndimage.generic_filter(vv_data, np.std, size=7)
        
        # Normalize roughness to 0-1
        roughness_norm = (roughness - roughness.min()) / (roughness.max() - roughness.min() + 1e-10)
        
        # Edge detection for lineaments (geological structures)
        edges = sobel(vv_metric)
        edges_norm = (edges - edges.min()) / (edges.max() - edges.min() + 1e-10)
        
        # Combine texture metrics for terrain classification
        terrain_index = 0.5 * roughness_norm + 0.3 * edges_norm + 0.2 * (texture_features['contrast'] / 100)
        terrain_index = np.clip(terrain_index, 0, 1)
        
        # Classify terrain types based on roughness
        terrain_class = np.zeros_like(terrain_index, dtype=int)
        terrain_class[terrain_index < 0.2] = 0  # Very smooth (water, flat)
        terrain_class[(terrain_index >= 0.2) & (terrain_index < 0.4)] = 1  # Smooth (plains)
        terrain_class[(terrain_index >= 0.4) & (terrain_index < 0.6)] = 2  # Moderate (hills)
        terrain_class[(terrain_index >= 0.6) & (terrain_index < 0.8)] = 3  # Rough (mountains)
        terrain_class[terrain_index >= 0.8] = 4  # Very rough (steep terrain)
        
        # Lineament detection (strong edges)
        lineament_threshold = np.percentile(edges_norm, 95)
        lineaments = edges_norm > lineament_threshold
        lineaments = binary_closing(lineaments, disk(2))  # Clean up
        
        stats = self._calculate_statistics(terrain_index, "Terrain Roughness Index")
        stats['texture_contrast'] = texture_features['contrast']
        stats['texture_dissimilarity'] = texture_features['dissimilarity']
        stats['texture_homogeneity'] = texture_features['homogeneity']
        stats['texture_energy'] = texture_features['energy']
        stats['lineament_density_percent'] = float(np.sum(lineaments) / lineaments.size * 100)
        
        # Area statistics
        pixel_area_km2 = self.pixel_area_km2
        stats['very_smooth_percent'] = float(np.sum(terrain_class == 0) / terrain_class.size * 100)
        stats['smooth_percent'] = float(np.sum(terrain_class == 1) / terrain_class.size * 100)
        stats['moderate_percent'] = float(np.sum(terrain_class == 2) / terrain_class.size * 100)
        stats['rough_percent'] = float(np.sum(terrain_class == 3) / terrain_class.size * 100)
        stats['very_rough_percent'] = float(np.sum(terrain_class == 4) / terrain_class.size * 100)
        stats['mountainous_area_km2'] = float(np.sum(terrain_class >= 3) * pixel_area_km2)
        
        # If DEM provided, add slope analysis
        if dem_data is not None:
            from scipy.ndimage import sobel as sobel_filter
            sx = sobel_filter(dem_data, axis=0)
            sy = sobel_filter(dem_data, axis=1)
            slope = np.sqrt(sx**2 + sy**2)
            stats['mean_slope'] = float(np.mean(slope))
            stats['max_slope'] = float(np.max(slope))
        
        logger.info(f"Geology terrain analysis - Mean roughness: {stats['mean']:.3f}, Lineament density: {stats['lineament_density_percent']:.2f}%")
        return terrain_index, stats
    
    
    def flood_mapping(self, threshold_method: str = 'otsu',
                     manual_threshold: float = None, band=None) -> Tuple[np.ndarray, Dict]:
        """
        Detect water/flood areas in SAR image with enhanced analysis.
        
        Args:
            threshold_method: Thresholding method ('otsu', 'manual')
            manual_threshold: Manual threshold value (if method='manual')
            
        Returns:
            Tuple of (flood mask, statistics dict)
        """
        img = self._select_band(band)
        
        # Use dB if calibrated, else raw intensity
        db = 10 * np.log10(img + 1e-10) if self.calibrated else img.astype(float)
        
        if threshold_method == 'otsu':
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(db[~np.isnan(db)])
        elif threshold_method == 'manual':
            threshold = manual_threshold if manual_threshold else -15
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
        
        # Water has low backscatter
        flood_mask = db < threshold
        
        # Calculate area using pixel size from profile if available
        pixel_area_km2 = self.pixel_area_km2
        water_area_km2 = int(np.sum(flood_mask)) * pixel_area_km2
        
        stats = {
            'threshold': float(threshold),
            'water_percent': float(np.sum(flood_mask) / flood_mask.size * 100),
            'water_pixels': int(np.sum(flood_mask)),
            'water_area_km2': float(water_area_km2),
            'backscatter_mean_water': float(np.mean(db[flood_mask])),
            'backscatter_mean_land': float(np.mean(db[~flood_mask]))
        }
        
        logger.info(f"Flood mapping complete - {stats['water_percent']:.2f}% water ({water_area_km2:.2f} km²)")
        return flood_mask.astype(int), stats
    
    def polarimetric_decomposition(self, vv_data: np.ndarray = None, 
                                   vh_data: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Perform polarimetric analysis using VV and VH polarizations.
        Dual-pol ratio (VV/VH) is useful for land cover discrimination.
        
        Args:
            vv_data: VV polarization data (if None, uses self.data[0])
            vh_data: VH polarization data (if None, uses self.data[1] if available)
            
        Returns:
            Tuple of (polarimetric features, statistics dict)
        """
        if vv_data is None:
            vv_data = self._select_band('VV')
        
        if vh_data is None:
            if self.data.shape[0] > 1:
                # Try VH band selection; falls back to second band
                try:
                    vh_data = self._select_band('VH')
                except Exception:
                    vh_data = self.data[1]
            else:
                raise ValueError("VH polarization data not available. Please provide vh_data.")
        
        # Calculate dual-pol ratio (use dB if calibrated, else normalized intensity)
        vv_db = self._to_db(vv_data) if self.calibrated else self._normalize01(vv_data)
        vh_db = self._to_db(vh_data) if self.calibrated else self._normalize01(vh_data)
        ratio = vv_db - vh_db  # VV/VH ratio metric
        
        # Calculate cross-polarization index
        cross_pol_ratio = vh_data / (vv_data + 1e-10)
        
        stats = self._calculate_statistics(ratio, "VV/VH Ratio (dB)")
        stats['vv_mean_db'] = float(np.mean(vv_db[~np.isnan(vv_db)]))
        stats['vh_mean_db'] = float(np.mean(vh_db[~np.isnan(vh_db)]))
        stats['cross_pol_mean'] = float(np.mean(cross_pol_ratio[~np.isnan(cross_pol_ratio)]))
        
        # Land cover discrimination based on ratio
        stats['forest_percent'] = float(np.sum(ratio < 10) / ratio.size * 100)  # Low ratio
        stats['urban_percent'] = float(np.sum(ratio > 15) / ratio.size * 100)    # High ratio
        stats['agricultural_percent'] = float(np.sum((ratio >= 10) & (ratio <= 15)) / ratio.size * 100)
        
        logger.info(f"Polarimetric analysis complete - VV/VH mean ratio: {stats['mean']:.2f} dB")
        return ratio, stats
    
    def soil_moisture_estimation(self, incidence_angle: float = 39.0) -> Tuple[np.ndarray, Dict]:
        """
        Estimate relative soil moisture from SAR backscatter.
        Uses simplified water cloud model approach.
        
        Args:
            incidence_angle: Radar incidence angle in degrees (default: 39° for Sentinel-1)
            
        Returns:
            Tuple of (soil moisture index, statistics dict)
        """
        img = self.data[0]
        
        # Use metric (dB if calibrated, else normalized intensity)
        metric = 10 * np.log10(img + 1e-10) if self.calibrated else self._normalize01(img)
        
        # Normalize considering incidence angle
        angle_factor = np.cos(np.radians(incidence_angle))
        db_normalized = metric * angle_factor
        
        # Soil moisture index (simplified empirical relationship)
        # Higher backscatter generally indicates higher soil moisture
        # Normalize to 0-1 range
        db_min = np.percentile(db_normalized[~np.isnan(db_normalized)], 5)
        db_max = np.percentile(db_normalized[~np.isnan(db_normalized)], 95)
        
        soil_moisture_index = (db_normalized - db_min) / (db_max - db_min + 1e-10)
        soil_moisture_index = np.clip(soil_moisture_index, 0, 1)
        
        stats = self._calculate_statistics(soil_moisture_index, "Soil Moisture Index")
        stats['incidence_angle'] = incidence_angle
        stats['very_dry_percent'] = float(np.sum(soil_moisture_index < 0.2) / soil_moisture_index.size * 100)
        stats['dry_percent'] = float(np.sum((soil_moisture_index >= 0.2) & (soil_moisture_index < 0.4)) / soil_moisture_index.size * 100)
        stats['moderate_percent'] = float(np.sum((soil_moisture_index >= 0.4) & (soil_moisture_index < 0.6)) / soil_moisture_index.size * 100)
        stats['moist_percent'] = float(np.sum((soil_moisture_index >= 0.6) & (soil_moisture_index < 0.8)) / soil_moisture_index.size * 100)
        stats['very_moist_percent'] = float(np.sum(soil_moisture_index >= 0.8) / soil_moisture_index.size * 100)
        
        logger.info(f"Soil moisture estimation complete - Mean index: {stats['mean']:.3f}")
        return soil_moisture_index, stats
    
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
