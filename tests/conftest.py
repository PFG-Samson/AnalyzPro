"""Pytest configuration and shared fixtures for test suite."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import MagicMock


@pytest.fixture(scope="session")
def sample_optical_data():
    """Create sample optical satellite data for testing."""
    np.random.seed(42)  # For reproducible tests
    # Create synthetic 6-band optical data (Blue, Green, Red, NIR, SWIR1, SWIR2)
    data = np.random.rand(6, 100, 100) * 0.6 + 0.1
    return data


@pytest.fixture(scope="session") 
def sample_sar_data():
    """Create sample SAR data for testing."""
    np.random.seed(43)
    # Create dual-pol SAR data (VV, VH)
    data = np.random.rand(2, 80, 80) * 0.4 + 0.05
    return data


@pytest.fixture
def standard_band_indices():
    """Standard optical band indices mapping."""
    return {
        'blue': 0,
        'green': 1,
        'red': 2,
        'nir': 3,
        'swir1': 4,
        'swir2': 5
    }


@pytest.fixture
def sample_raster_profile():
    """Create a standard rasterio profile for testing."""
    return {
        'driver': 'GTiff',
        'count': 1,
        'height': 100,
        'width': 100,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': None,  # Identity transform
        'nodata': None,
        'compress': 'LZW'
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture  
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
        tmp_path = tmp.name
    
    yield Path(tmp_path)
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def mock_rasterio_dataset():
    """Create a mock rasterio dataset for testing."""
    mock_dataset = MagicMock()
    mock_dataset.read.return_value = np.random.rand(3, 50, 50)
    mock_dataset.profile = {
        'driver': 'GTiff',
        'count': 3,
        'height': 50,
        'width': 50,
        'dtype': 'float64',
        'crs': 'EPSG:4326'
    }
    return mock_dataset


# Test data constants
TEST_SPECTRAL_VALUES = {
    'blue': 0.1,
    'green': 0.15,
    'red': 0.2,
    'nir': 0.4,
    'swir1': 0.25,
    'swir2': 0.22
}


def create_test_data_with_known_values(shape=(100, 100), values=None):
    """Create test data with known spectral values for predictable index calculations."""
    if values is None:
        values = TEST_SPECTRAL_VALUES
    
    data = np.zeros((len(values), *shape))
    for i, (band, value) in enumerate(values.items()):
        data[i] = value
    
    return data