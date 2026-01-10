"""Unit tests for OpticalAnalyzer class."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from analyz.core.optical_analysis import OpticalAnalyzer


class TestOpticalAnalyzer:
    """Test cases for OpticalAnalyzer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample multi-band optical data for testing."""
        # Create synthetic data (6 bands, 100x100 pixels)
        np.random.seed(42)  # For reproducible tests
        data = np.random.rand(6, 100, 100) * 0.5 + 0.1
        return data
    
    @pytest.fixture
    def band_indices(self):
        """Standard band indices mapping."""
        return {
            'blue': 0,
            'green': 1, 
            'red': 2,
            'nir': 3,
            'swir1': 4,
            'swir2': 5
        }
    
    @pytest.fixture
    def analyzer(self, sample_data, band_indices):
        """Create OpticalAnalyzer instance for testing."""
        return OpticalAnalyzer(sample_data, band_indices, sensor='sentinel2')
    
    def test_initialization(self, sample_data, band_indices):
        """Test OpticalAnalyzer initialization."""
        analyzer = OpticalAnalyzer(sample_data, band_indices, sensor='sentinel2')
        
        assert analyzer.data.shape == sample_data.shape
        assert analyzer.band_indices == band_indices
        assert analyzer.sensor == 'sentinel2'
        
    def test_initialization_unknown_sensor(self, sample_data, band_indices):
        """Test initialization with unknown sensor."""
        with patch('analyz.core.optical_analysis.logger') as mock_logger:
            analyzer = OpticalAnalyzer(sample_data, band_indices, sensor='unknown_sensor')
            assert analyzer.sensor is None
            mock_logger.warning.assert_called_once()
    
    def test_get_band_valid(self, analyzer):
        """Test _get_band with valid band name."""
        red_band = analyzer._get_band('red')
        assert red_band.shape == (100, 100)
        assert red_band.dtype == float
        
    def test_get_band_invalid(self, analyzer):
        """Test _get_band with invalid band name."""
        with pytest.raises(ValueError, match="Band 'invalid' not defined in band_indices"):
            analyzer._get_band('invalid')


class TestComputeIndex:
    """Test cases for compute_index method."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with known values for index calculation."""
        # Create predictable data for testing index calculations
        data = np.zeros((6, 10, 10))
        data[0] = 0.1  # blue
        data[1] = 0.2  # green
        data[2] = 0.3  # red
        data[3] = 0.5  # nir
        data[4] = 0.4  # swir1
        data[5] = 0.35  # swir2
        return data
    
    @pytest.fixture
    def band_indices(self):
        """Standard band indices mapping."""
        return {
            'blue': 0,
            'green': 1,
            'red': 2,
            'nir': 3,
            'swir1': 4,
            'swir2': 5
        }
    
    @pytest.fixture
    def analyzer(self, sample_data, band_indices):
        """Create OpticalAnalyzer instance for testing."""
        return OpticalAnalyzer(sample_data, band_indices, sensor='sentinel2')
    
    def test_compute_ndvi(self, analyzer):
        """Test NDVI calculation."""
        ndvi, stats = analyzer.compute_index('ndvi')
        
        # Expected NDVI = (0.5 - 0.3) / (0.5 + 0.3) = 0.25
        expected_value = 0.25
        np.testing.assert_allclose(ndvi, expected_value, rtol=1e-6)
        
        assert stats['name'] == 'NDVI'
        assert abs(stats['mean'] - expected_value) < 1e-6
        assert -1 <= stats['min'] <= stats['max'] <= 1  # NDVI should be clipped to [-1, 1]
    
    def test_compute_ndwi(self, analyzer):
        """Test NDWI calculation."""
        ndwi, stats = analyzer.compute_index('ndwi')
        
        # Expected NDWI = (0.2 - 0.5) / (0.2 + 0.5) = -0.428571...
        expected_value = (0.2 - 0.5) / (0.2 + 0.5 + 1e-8)
        np.testing.assert_allclose(ndwi, expected_value, rtol=1e-4)
        
        assert stats['name'] == 'NDWI'
        assert -1 <= stats['min'] <= stats['max'] <= 1
    
    def test_compute_evi2(self, analyzer):
        """Test EVI2 calculation."""
        evi2, stats = analyzer.compute_index('evi2')
        
        # Expected EVI2 = 2.5 * (0.5 - 0.3) / (0.5 + 2.4*0.3 + 1.0)
        expected_value = 2.5 * (0.5 - 0.3) / (0.5 + 2.4*0.3 + 1.0 + 1e-8)
        np.testing.assert_allclose(evi2, expected_value, rtol=1e-6)
        
        assert stats['name'] == 'EVI2'
    
    def test_compute_savi(self, analyzer):
        """Test SAVI calculation."""
        savi, stats = analyzer.compute_index('savi')
        
        # Expected SAVI = ((0.5 - 0.3) / (0.5 + 0.3 + 0.5)) * 1.5
        expected_value = ((0.5 - 0.3) / (0.5 + 0.3 + 0.5 + 1e-8)) * 1.5
        np.testing.assert_allclose(savi, expected_value, rtol=1e-6)
        
        assert stats['name'] == 'SAVI'
    
    def test_compute_ndbi(self, analyzer):
        """Test NDBI calculation."""
        ndbi, stats = analyzer.compute_index('ndbi')
        
        # Expected NDBI = (0.4 - 0.5) / (0.4 + 0.5)
        expected_value = (0.4 - 0.5) / (0.4 + 0.5 + 1e-8)
        np.testing.assert_allclose(ndbi, expected_value, rtol=1e-6)
        
        assert stats['name'] == 'NDBI'
        assert -1 <= stats['min'] <= stats['max'] <= 1
    
    def test_compute_bsi(self, analyzer):
        """Test BSI (Bare Soil Index) calculation."""
        bsi, stats = analyzer.compute_index('bsi')
        
        # BSI = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
        # Expected BSI = ((0.4 + 0.3) - (0.5 + 0.1)) / ((0.4 + 0.3) + (0.5 + 0.1))
        expected_value = ((0.4 + 0.3) - (0.5 + 0.1)) / ((0.4 + 0.3) + (0.5 + 0.1) + 1e-8)
        np.testing.assert_allclose(bsi, expected_value, rtol=1e-6)
        
        assert stats['name'] == 'BSI'
    
    def test_compute_index_with_helpers(self, analyzer):
        """Test index computation with helper expressions (GEMI)."""
        gemi, stats = analyzer.compute_index('gemi')
        
        # GEMI has a helper 'eta' that should be computed first
        assert stats['name'] == 'GEMI'
        assert isinstance(gemi, np.ndarray)
        assert gemi.shape == (10, 10)
    
    def test_compute_complex_indices(self, analyzer):
        """Test computation of complex indices requiring multiple bands."""
        # Test TVI2 which requires nir, red, green
        tvi2, stats = analyzer.compute_index('tvi2')
        assert stats['name'] == 'TVI2'
        assert isinstance(tvi2, np.ndarray)
        
        # Test MTVI2 which has complex formula
        mtvi2, stats = analyzer.compute_index('mtvi2')
        assert stats['name'] == 'MTVI2'
        assert isinstance(mtvi2, np.ndarray)
    
    def test_compute_index_alias_resolution(self, analyzer):
        """Test that index aliases are properly resolved."""
        # Test alias resolution
        ndvi1, _ = analyzer.compute_index('ndvi')
        ndvi2, _ = analyzer.compute_index('ndvi2')  # This is an alias for ndvi
        
        np.testing.assert_array_equal(ndvi1, ndvi2)
    
    def test_compute_unknown_index(self, analyzer):
        """Test compute_index with unknown index name."""
        with pytest.raises(ValueError, match="Unknown index 'unknown_index'"):
            analyzer.compute_index('unknown_index')
    
    def test_compute_index_missing_bands(self, sample_data):
        """Test compute_index when required bands are missing."""
        # Create analyzer with limited band indices
        limited_bands = {'red': 2, 'nir': 3}  # Missing blue, green, swir bands
        analyzer = OpticalAnalyzer(sample_data, limited_bands)
        
        # NDVI should work (only needs red, nir)
        ndvi, _ = analyzer.compute_index('ndvi')
        assert isinstance(ndvi, np.ndarray)
        
        # BSI should fail (needs swir1, red, nir, blue)
        with pytest.raises(ValueError, match="requires bands not available"):
            analyzer.compute_index('bsi')


class TestSensorValidation:
    """Test cases for sensor capability validation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return np.random.rand(6, 50, 50)
    
    def test_validate_index_for_sentinel2(self, sample_data):
        """Test validation for Sentinel-2 sensor capabilities."""
        band_indices = {'red': 0, 'nir': 1, 'rededge1': 2}
        analyzer = OpticalAnalyzer(sample_data, band_indices, sensor='sentinel2')
        
        # NDVI should be valid (red, nir available in Sentinel-2)
        analyzer._validate_index_for_sensor('ndvi', ['red', 'nir'])
        
        # Index requiring rededge1 should be valid for Sentinel-2
        analyzer._validate_index_for_sensor('ndvi705', ['nir', 'rededge1'])
    
    def test_validate_index_for_landsat8(self, sample_data):
        """Test validation for Landsat 8 sensor capabilities."""
        band_indices = {'red': 0, 'nir': 1, 'rededge1': 2}
        analyzer = OpticalAnalyzer(sample_data, band_indices, sensor='landsat8')
        
        # NDVI should be valid (red, nir available in Landsat 8)
        analyzer._validate_index_for_sensor('ndvi', ['red', 'nir'])
        
        # Index requiring rededge1 should fail for Landsat 8 (no red edge bands)
        with pytest.raises(ValueError, match="not supported on sensor 'landsat8'"):
            analyzer._validate_index_for_sensor('ndvi705', ['nir', 'rededge1'])
    
    def test_validate_index_missing_bands_for_sensor(self, sample_data):
        """Test validation when sensor doesn't support required bands."""
        band_indices = {'red': 0, 'nir': 1}
        analyzer = OpticalAnalyzer(sample_data, band_indices, sensor='landsat8')
        
        # Index requiring rededge1 band should fail for Landsat 8
        with pytest.raises(ValueError, match="Missing bands: \\['rededge1'\\]"):
            analyzer._validate_index_for_sensor('test_index', ['red', 'nir', 'rededge1'])
    
    def test_no_sensor_validation(self, sample_data):
        """Test that validation is skipped when no sensor is specified."""
        band_indices = {'red': 0, 'nir': 1, 'rededge1': 2}
        analyzer = OpticalAnalyzer(sample_data, band_indices, sensor=None)
        
        # Should not raise error even with invalid band for non-existent sensor
        analyzer._validate_index_for_sensor('test_index', ['red', 'nir', 'rededge1'])
    
    def test_sensor_validation_integration(self, sample_data):
        """Test sensor validation integration with compute_index."""
        # Test with Landsat 8 - should work with basic indices
        basic_bands = {'red': 0, 'nir': 1, 'blue': 2, 'green': 3}
        analyzer = OpticalAnalyzer(sample_data, basic_bands, sensor='landsat8')
        
        ndvi, _ = analyzer.compute_index('ndvi')
        assert isinstance(ndvi, np.ndarray)
        
        # Should fail with red-edge indices on Landsat 8
        with pytest.raises(ValueError, match="not supported on sensor 'landsat8'"):
            analyzer.compute_index('ndvi705')
    
    def test_available_indices_method(self, sample_data):
        """Test available_indices method returns the full registry."""
        band_indices = {'red': 0, 'nir': 1}
        analyzer = OpticalAnalyzer(sample_data, band_indices)
        
        available = analyzer.available_indices()
        assert isinstance(available, dict)
        assert 'ndvi' in available
        assert 'ndwi' in available
        assert 'bsi' in available
        
        # Check structure of registry entries
        ndvi_entry = available['ndvi']
        assert 'expr' in ndvi_entry
        assert 'requires' in ndvi_entry
        assert ndvi_entry['requires'] == ['nir', 'red']


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_data_with_nan_values(self):
        """Test handling of NaN values in input data."""
        data = np.random.rand(4, 20, 20)
        data[0, 5:10, 5:10] = np.nan  # Add some NaN values
        
        band_indices = {'red': 0, 'nir': 1, 'green': 2, 'blue': 3}
        analyzer = OpticalAnalyzer(data, band_indices)
        
        # Should handle NaN values gracefully
        ndvi, stats = analyzer.compute_index('ndvi')
        assert isinstance(ndvi, np.ndarray)
        assert 'mean' in stats
    
    def test_data_with_extreme_values(self):
        """Test handling of extreme values."""
        data = np.ones((4, 10, 10))
        data[0] = 10000  # Very high values
        data[1] = 0.00001  # Very low values
        
        band_indices = {'red': 0, 'nir': 1, 'green': 2, 'blue': 3}
        analyzer = OpticalAnalyzer(data, band_indices)
        
        # Should handle extreme values and clipping
        ndvi, stats = analyzer.compute_index('ndvi')
        assert np.all(ndvi >= -1) and np.all(ndvi <= 1)  # Should be clipped
    
    def test_single_pixel_data(self):
        """Test with single pixel data."""
        data = np.random.rand(4, 1, 1)
        band_indices = {'red': 0, 'nir': 1, 'green': 2, 'blue': 3}
        analyzer = OpticalAnalyzer(data, band_indices)
        
        ndvi, stats = analyzer.compute_index('ndvi')
        assert ndvi.shape == (1, 1)
        assert 'mean' in stats