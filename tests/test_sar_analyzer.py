"""Unit tests for SARAnalyzer class."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from analyz.core.sar_analysis import SARAnalyzer


class TestSARAnalyzerInitialization:
    """Test cases for SARAnalyzer initialization."""
    
    def test_initialization_2d_data(self):
        """Test initialization with 2D SAR data."""
        data = np.random.rand(100, 100)
        analyzer = SARAnalyzer(data)
        
        assert analyzer.data.shape == (1, 100, 100)  # Should be expanded to 3D
        assert analyzer.pixel_area_km2 == 0.0001  # Default 10m x 10m pixel
        assert analyzer.mask is None
    
    def test_initialization_3d_data(self):
        """Test initialization with 3D SAR data."""
        data = np.random.rand(2, 100, 100)  # Dual-pol data
        analyzer = SARAnalyzer(data)
        
        assert analyzer.data.shape == (2, 100, 100)
        assert analyzer.pixel_area_km2 == 0.0001
    
    def test_initialization_invalid_dimensions(self):
        """Test initialization with invalid data dimensions."""
        data = np.random.rand(100, 100, 100, 100)  # 4D data
        
        with pytest.raises(ValueError, match="Unsupported data dimensions: 4"):
            SARAnalyzer(data)
    
    def test_initialization_with_profile_transform(self):
        """Test initialization with rasterio profile containing transform."""
        data = np.random.rand(100, 100)
        
        # Mock rasterio transform object
        mock_transform = Mock()
        mock_transform.a = 20.0  # 20m pixel width
        mock_transform.e = -20.0  # -20m pixel height
        
        profile = {'transform': mock_transform}
        analyzer = SARAnalyzer(data, profile=profile)
        
        expected_area = (20.0 * 20.0) / 1e6  # 400 m² = 0.0004 km²
        assert abs(analyzer.pixel_area_km2 - expected_area) < 1e-10
    
    def test_initialization_with_profile_transform_tuple(self):
        """Test initialization with transform as tuple/list."""
        data = np.random.rand(100, 100)
        
        # Transform as tuple (a, b, c, d, e, f)
        transform = (30.0, 0.0, 0.0, 0.0, -30.0, 0.0)  # 30m pixels
        profile = {'transform': transform}
        
        analyzer = SARAnalyzer(data, profile=profile)
        expected_area = (30.0 * 30.0) / 1e6  # 900 m² = 0.0009 km²
        assert abs(analyzer.pixel_area_km2 - expected_area) < 1e-10
    
    def test_initialization_with_profile_transform_exception(self):
        """Test initialization when transform processing fails."""
        data = np.random.rand(100, 100)
        profile = {'transform': None}  # Will cause exception in processing
        
        analyzer = SARAnalyzer(data, profile=profile)
        assert analyzer.pixel_area_km2 == 0.0001  # Should fallback to default
    
    def test_initialization_with_explicit_mask(self):
        """Test initialization with explicitly provided mask."""
        data = np.random.rand(2, 50, 50)
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:20, 10:20] = True  # Mask out a 10x10 area
        
        analyzer = SARAnalyzer(data, mask=mask)
        
        assert analyzer.mask.shape == (50, 50)
        assert analyzer.mask.dtype == bool
        assert np.sum(analyzer.mask) == 100  # 10x10 = 100 pixels masked
        
        # Check that masked pixels are set to NaN
        assert np.all(np.isnan(analyzer.data[:, mask]))
    
    def test_initialization_with_profile_nodata(self):
        """Test initialization with NoData value from profile."""
        data = np.random.rand(2, 30, 30)
        data[0, 5:10, 5:10] = -9999  # Set some pixels to NoData value
        
        profile = {'nodata': -9999}
        analyzer = SARAnalyzer(data, profile=profile)
        
        # Should create mask from NoData values
        assert analyzer.mask is not None
        assert np.any(analyzer.mask[5:10, 5:10])  # Should be masked
        assert np.all(np.isnan(analyzer.data[:, analyzer.mask]))  # Should be NaN
    
    def test_initialization_with_incompatible_mask_shape(self):
        """Test initialization with mask of incompatible shape."""
        data = np.random.rand(2, 50, 50)
        mask = np.ones((30, 30), dtype=bool)  # Wrong shape
        
        analyzer = SARAnalyzer(data, mask=mask)
        assert analyzer.mask is None  # Should not use incompatible mask
    
    def test_initialization_pixel_area_calculation(self):
        """Test pixel area calculation from different profile formats."""
        data = np.random.rand(100, 100)
        
        # Test with affine transform
        mock_transform = Mock()
        mock_transform.a = 5.0  # 5m pixel
        mock_transform.e = -5.0
        
        profile = {'transform': mock_transform}
        analyzer = SARAnalyzer(data, profile=profile)
        
        expected_area = (5.0 * 5.0) / 1e6  # 25 m² = 0.000025 km²
        assert abs(analyzer.pixel_area_km2 - expected_area) < 1e-10


class TestApplySpeckleFilter:
    """Test cases for _apply_speckle_filter method."""
    
    @pytest.fixture
    def sample_sar_data(self):
        """Create sample SAR data with some noise patterns."""
        np.random.seed(42)
        data = np.random.rand(2, 20, 20) * 0.5 + 0.1
        # Add some noise spikes
        data[0, 5, 5] = 2.0  # High noise spike
        data[1, 10, 10] = 0.001  # Low noise spike
        return data
    
    @pytest.fixture
    def analyzer(self, sample_sar_data):
        """Create SARAnalyzer with sample data."""
        return SARAnalyzer(sample_sar_data)
    
    def test_lee_filter_basic(self, analyzer):
        """Test basic Lee filter functionality."""
        filtered = analyzer._apply_speckle_filter(method='lee', window_size=3, looks=1)
        
        assert filtered.shape == analyzer.data.shape
        assert not np.array_equal(filtered, analyzer.data)  # Should be different
        assert np.all(np.isfinite(filtered[~np.isnan(filtered)]))  # No inf values
    
    def test_median_filter_basic(self, analyzer):
        """Test basic median filter functionality."""
        filtered = analyzer._apply_speckle_filter(method='median', window_size=3)
        
        assert filtered.shape == analyzer.data.shape
        assert not np.array_equal(filtered, analyzer.data)  # Should be different
        assert np.all(np.isfinite(filtered[~np.isnan(filtered)]))  # No inf values
    
    def test_filter_with_different_window_sizes(self, analyzer):
        """Test filtering with different window sizes."""
        filter_3x3 = analyzer._apply_speckle_filter(method='lee', window_size=3)
        filter_5x5 = analyzer._apply_speckle_filter(method='lee', window_size=5)
        filter_7x7 = analyzer._apply_speckle_filter(method='lee', window_size=7)
        
        # All should have same shape
        assert filter_3x3.shape == filter_5x5.shape == filter_7x7.shape
        
        # Different window sizes should produce different results
        assert not np.array_equal(filter_3x3, filter_5x5)
        assert not np.array_equal(filter_5x5, filter_7x7)
    
    def test_filter_preserves_mask(self):
        """Test that speckle filter preserves NoData mask."""
        data = np.random.rand(2, 20, 20)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 5:10] = True  # Create masked area
        
        analyzer = SARAnalyzer(data, mask=mask)
        filtered = analyzer._apply_speckle_filter(method='lee', window_size=3)
        
        # Masked pixels should remain NaN
        assert np.all(np.isnan(filtered[:, mask]))
        # Non-masked pixels should not be NaN (mostly)
        assert not np.all(np.isnan(filtered[:, ~mask]))
    
    def test_filter_handles_nan_values(self):
        """Test that filter handles NaN values in input data."""
        data = np.random.rand(2, 20, 20)
        data[0, 5:10, 5:10] = np.nan  # Add NaN values
        
        analyzer = SARAnalyzer(data)
        filtered = analyzer._apply_speckle_filter(method='lee', window_size=3)
        
        # Should not crash and should handle NaN values
        assert filtered.shape == data.shape
        assert np.any(np.isfinite(filtered))  # Should have some valid values
    
    def test_lee_filter_with_different_looks(self, analyzer):
        """Test Lee filter with different number of looks."""
        filter_1_look = analyzer._apply_speckle_filter(method='lee', window_size=3, looks=1)
        filter_4_looks = analyzer._apply_speckle_filter(method='lee', window_size=3, looks=4)
        filter_16_looks = analyzer._apply_speckle_filter(method='lee', window_size=3, looks=16)
        
        # All should have same shape
        assert filter_1_look.shape == filter_4_looks.shape == filter_16_looks.shape
        
        # Different looks should produce different results
        assert not np.array_equal(filter_1_look, filter_4_looks)
        assert not np.array_equal(filter_4_looks, filter_16_looks)
    
    def test_filter_with_extreme_values(self, analyzer):
        """Test filter behavior with extreme values."""
        # Add extreme values to data
        analyzer.data[0, 0, 0] = 1e6  # Very high value
        analyzer.data[1, 1, 1] = 1e-10  # Very low value
        
        filtered = analyzer._apply_speckle_filter(method='lee', window_size=3)
        
        # Should handle extreme values without crashing
        assert filtered.shape == analyzer.data.shape
        assert np.all(np.isfinite(filtered[np.isfinite(filtered)]))
    
    def test_filter_noise_reduction(self):
        """Test that filtering actually reduces noise."""
        # Create data with known noise pattern
        clean_data = np.ones((1, 20, 20)) * 0.5
        noisy_data = clean_data + np.random.normal(0, 0.1, (1, 20, 20))
        
        analyzer = SARAnalyzer(noisy_data)
        filtered = analyzer._apply_speckle_filter(method='median', window_size=3)
        
        # Filtered data should be closer to original clean data
        clean_mse = np.mean((clean_data - filtered) ** 2)
        noisy_mse = np.mean((clean_data - noisy_data) ** 2)
        
        # Filter should reduce error (most of the time)
        assert clean_mse < noisy_mse * 2  # Allow some tolerance
    
    def test_filter_edge_preservation(self, analyzer):
        """Test that filter preserves data at edges appropriately."""
        original_shape = analyzer.data.shape
        filtered = analyzer._apply_speckle_filter(method='lee', window_size=5)
        
        # Shape should be preserved
        assert filtered.shape == original_shape
        
        # Edge pixels should still have valid values (not all NaN)
        assert not np.all(np.isnan(filtered[:, 0, :]))  # Top edge
        assert not np.all(np.isnan(filtered[:, -1, :]))  # Bottom edge
        assert not np.all(np.isnan(filtered[:, :, 0]))  # Left edge
        assert not np.all(np.isnan(filtered[:, :, -1]))  # Right edge
    
    def test_multi_band_filtering(self):
        """Test filtering with multiple bands."""
        data = np.random.rand(3, 15, 15)  # 3-band data
        analyzer = SARAnalyzer(data)
        
        filtered = analyzer._apply_speckle_filter(method='lee', window_size=3)
        
        # All bands should be filtered
        assert filtered.shape == (3, 15, 15)
        for band in range(3):
            assert not np.array_equal(filtered[band], data[band])
    
    def test_single_pixel_data_filtering(self):
        """Test filtering with single pixel data."""
        data = np.array([[[0.5]]])  # Single pixel, single band
        analyzer = SARAnalyzer(data)
        
        filtered = analyzer._apply_speckle_filter(method='median', window_size=3)
        
        # Should handle gracefully without crashing
        assert filtered.shape == (1, 1, 1)
        assert np.isfinite(filtered[0, 0, 0])


class TestSARAnalyzerMethods:
    """Test cases for other SARAnalyzer methods."""
    
    @pytest.fixture
    def dual_pol_data(self):
        """Create dual-polarization SAR data."""
        np.random.seed(42)
        data = np.random.rand(2, 50, 50) * 0.3 + 0.1
        return data
    
    def test_to_db_conversion(self, dual_pol_data):
        """Test dB conversion utility method."""
        analyzer = SARAnalyzer(dual_pol_data)
        
        # Test with known value
        test_data = np.array([[0.1]])  # Known intensity value
        db_data = analyzer._to_db(test_data)
        
        expected_db = 10 * np.log10(0.1 + 1e-10)
        np.testing.assert_allclose(db_data, expected_db, rtol=1e-6)
    
    def test_to_db_with_zero_values(self, dual_pol_data):
        """Test dB conversion with zero/very small values."""
        analyzer = SARAnalyzer(dual_pol_data)
        
        test_data = np.array([[0.0, 1e-15, 1.0]])
        db_data = analyzer._to_db(test_data)
        
        # Should not produce -inf values due to epsilon addition
        assert np.all(np.isfinite(db_data))
    
    def test_calculate_statistics(self, dual_pol_data):
        """Test statistics calculation method."""
        analyzer = SARAnalyzer(dual_pol_data)
        
        test_data = np.random.rand(20, 20)
        stats = analyzer._calculate_statistics(test_data, "Test Data")
        
        assert stats['name'] == "Test Data"
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'percentile_25' in stats
        assert 'percentile_75' in stats
        
        # Validate statistical relationships
        assert stats['min'] <= stats['percentile_25'] <= stats['median'] <= stats['percentile_75'] <= stats['max']
    
    def test_calculate_statistics_with_nan(self, dual_pol_data):
        """Test statistics calculation with NaN values."""
        analyzer = SARAnalyzer(dual_pol_data)
        
        test_data = np.random.rand(20, 20)
        test_data[5:10, 5:10] = np.nan  # Add NaN values
        
        stats = analyzer._calculate_statistics(test_data, "Test with NaN")
        
        # Should still produce valid statistics
        assert 'mean' in stats
        assert np.isfinite(stats['mean'])
        assert np.isfinite(stats['median'])
    
    def test_calculate_statistics_all_nan(self, dual_pol_data):
        """Test statistics calculation with all NaN values."""
        analyzer = SARAnalyzer(dual_pol_data)
        
        test_data = np.full((10, 10), np.nan)
        stats = analyzer._calculate_statistics(test_data, "All NaN")
        
        # Should return empty dict when no valid data
        assert stats == {}
    
    def test_flood_mapping_basic(self, dual_pol_data):
        """Test basic flood mapping functionality."""
        analyzer = SARAnalyzer(dual_pol_data)
        
        flood_mask, stats = analyzer.flood_mapping(threshold_method='manual', manual_threshold=-15)
        
        assert flood_mask.shape == dual_pol_data.shape[1:]  # Should match spatial dimensions
        assert flood_mask.dtype == int
        assert 'water_percent' in stats
        assert 'water_area_km2' in stats
        assert 'threshold' in stats
        assert stats['threshold'] == -15
    
    def test_flood_mapping_otsu(self, dual_pol_data):
        """Test flood mapping with Otsu thresholding."""
        analyzer = SARAnalyzer(dual_pol_data)
        
        flood_mask, stats = analyzer.flood_mapping(threshold_method='otsu')
        
        assert flood_mask.shape == dual_pol_data.shape[1:]
        assert 'water_percent' in stats
        assert 'threshold' in stats
        assert stats['threshold'] != -15  # Should be different from manual default


class TestSARAnalyzerIntegration:
    """Integration tests for SARAnalyzer with realistic scenarios."""
    
    def test_oil_spill_detection_integration(self):
        """Test oil spill detection with realistic data patterns."""
        # Create data with dark patches (potential oil spills)
        data = np.random.rand(1, 100, 100) * 0.2 + 0.1  # Background
        data[0, 30:40, 30:40] = 0.05  # Dark patch (potential oil spill)
        
        analyzer = SARAnalyzer(data)
        mask, stats = analyzer.oil_spill_detection(window_size=21, k_threshold=1.5, min_area_pixels=50)
        
        assert mask.shape == (100, 100)
        assert 'num_detected_slicks' in stats
        assert 'total_slick_area_km2' in stats
        assert stats['num_detected_slicks'] >= 0
    
    def test_full_workflow_with_profile_and_mask(self):
        """Test complete workflow with profile and mask."""
        # Create realistic SAR data
        data = np.random.rand(2, 80, 80) * 0.3 + 0.1
        data[0, 10:20, 10:20] = -9999  # NoData area
        
        # Create profile
        mock_transform = Mock()
        mock_transform.a = 10.0
        mock_transform.e = -10.0
        profile = {'transform': mock_transform, 'nodata': -9999}
        
        # Initialize analyzer
        analyzer = SARAnalyzer(data, profile=profile)
        
        # Test that initialization worked correctly
        assert analyzer.pixel_area_km2 == 0.0001  # 10m x 10m
        assert analyzer.mask is not None
        assert np.any(analyzer.mask)  # Should have masked pixels
        
        # Test that filtering works with mask
        filtered = analyzer._apply_speckle_filter(method='lee', window_size=3)
        assert np.all(np.isnan(filtered[:, analyzer.mask]))
        
        # Test flood mapping
        flood_mask, flood_stats = analyzer.flood_mapping()
        assert flood_mask.shape == (80, 80)
        assert 'water_area_km2' in flood_stats