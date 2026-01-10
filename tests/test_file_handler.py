"""Unit tests for FileHandler class."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import rasterio
from analyz.utils.file_handler import FileHandler


class TestFileHandlerValidation:
    """Test cases for file validation methods."""
    
    def test_validate_existing_file(self):
        """Test validation of an existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            result = FileHandler.validate_file(tmp_path)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            os.unlink(tmp_path)
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file raises error."""
        nonexistent_path = "/path/to/nonexistent/file.tif"
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            FileHandler.validate_file(nonexistent_path)
    
    def test_validate_file_with_pathlib_path(self):
        """Test validation with pathlib.Path input."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = Path(tmp.name)
        
        try:
            result = FileHandler.validate_file(tmp_path)
            assert isinstance(result, Path)
            assert result == tmp_path
        finally:
            tmp_path.unlink()


class TestWriteRaster:
    """Test cases for write_raster method with different array shapes."""
    
    @pytest.fixture
    def sample_profile(self):
        """Create a sample rasterio profile."""
        return {
            'driver': 'GTiff',
            'count': 1,
            'height': 100,
            'width': 100,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': rasterio.Affine.identity(),
            'nodata': None
        }
    
    def test_write_raster_2d_array(self, sample_profile):
        """Test writing 2D array (single band)."""
        data = np.random.rand(100, 100).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Mock rasterio.open to avoid actual file I/O
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                mock_dst = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dst
                
                FileHandler.write_raster(tmp_path, data, sample_profile)
                
                # Verify the data was reshaped to (1, H, W)
                written_data = mock_dst.write.call_args[0][0]
                assert written_data.shape == (1, 100, 100)
                
                # Verify profile was updated correctly
                write_args = mock_open.call_args[1]
                assert write_args['count'] == 1
                assert write_args['height'] == 100
                assert write_args['width'] == 100
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_write_raster_3d_chw_array(self, sample_profile):
        """Test writing 3D array in CHW format (channels first)."""
        data = np.random.rand(3, 100, 100).astype(np.float32)  # RGB data
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                mock_dst = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dst
                
                FileHandler.write_raster(tmp_path, data, sample_profile)
                
                # Data should remain as (3, H, W)
                written_data = mock_dst.write.call_args[0][0]
                assert written_data.shape == (3, 100, 100)
                
                # Profile should be updated for RGB
                write_args = mock_open.call_args[1]
                assert write_args['count'] == 3
                assert write_args['photometric'] == 'RGB'
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_write_raster_3d_hwc_array(self, sample_profile):
        """Test writing 3D array in HWC format (channels last)."""
        data = np.random.rand(100, 100, 3).astype(np.float32)  # HWC format
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                mock_dst = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dst
                
                FileHandler.write_raster(tmp_path, data, sample_profile)
                
                # Data should be transposed to (3, H, W)
                written_data = mock_dst.write.call_args[0][0]
                assert written_data.shape == (3, 100, 100)
                
                # Verify the transpose happened correctly
                np.testing.assert_array_equal(written_data[0], data[:, :, 0])
                np.testing.assert_array_equal(written_data[1], data[:, :, 1])
                np.testing.assert_array_equal(written_data[2], data[:, :, 2])
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_write_raster_4_channel_hwc_array(self, sample_profile):
        """Test writing 4-channel array in HWC format (e.g., RGBA)."""
        data = np.random.rand(50, 50, 4).astype(np.float32)  # RGBA data
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                mock_dst = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dst
                
                FileHandler.write_raster(tmp_path, data, sample_profile)
                
                # Data should be transposed to (4, H, W)
                written_data = mock_dst.write.call_args[0][0]
                assert written_data.shape == (4, 50, 50)
                
                # Profile should be updated for 4 bands
                write_args = mock_open.call_args[1]
                assert write_args['count'] == 4
                # Should not set RGB photometric for 4 channels
                assert 'photometric' not in write_args or write_args.get('photometric') != 'RGB'
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_write_raster_profile_update(self, sample_profile):
        """Test that profile is properly updated with data properties."""
        data = np.random.rand(2, 80, 120).astype(np.uint16)  # Different dtype and size
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                mock_dst = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dst
                
                FileHandler.write_raster(tmp_path, data, sample_profile, compress="DEFLATE")
                
                # Verify profile updates
                write_args = mock_open.call_args[1]
                assert write_args['count'] == 2
                assert write_args['height'] == 80
                assert write_args['width'] == 120
                assert write_args['dtype'] == np.uint16
                assert write_args['compress'] == 'DEFLATE'
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_write_raster_directory_creation(self, sample_profile):
        """Test that output directories are created if they don't exist."""
        data = np.random.rand(100, 100).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a nested path that doesn't exist
            output_path = Path(tmp_dir) / "nested" / "directory" / "output.tif"
            
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                mock_dst = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dst
                
                FileHandler.write_raster(str(output_path), data, sample_profile)
                
                # Verify the directory was created
                assert output_path.parent.exists()
                assert output_path.parent.is_dir()
    
    def test_write_raster_ambiguous_3d_array(self, sample_profile):
        """Test handling of ambiguous 3D array shapes."""
        # Create data where first dimension could be channels or spatial
        data = np.random.rand(5, 100, 100).astype(np.float32)  # 5 could be bands
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                mock_dst = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dst
                
                FileHandler.write_raster(tmp_path, data, sample_profile)
                
                # Should be treated as CHW (5 bands)
                written_data = mock_dst.write.call_args[0][0]
                assert written_data.shape == (5, 100, 100)
                
                write_args = mock_open.call_args[1]
                assert write_args['count'] == 5
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_write_raster_preserves_original_profile(self, sample_profile):
        """Test that original profile is not modified."""
        data = np.random.rand(100, 100).astype(np.float32)
        original_profile = sample_profile.copy()
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                mock_dst = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dst
                
                FileHandler.write_raster(tmp_path, data, sample_profile)
                
                # Original profile should be unchanged
                assert sample_profile == original_profile
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestReadRaster:
    """Test cases for read_raster method."""
    
    def test_read_raster_success(self):
        """Test successful raster reading."""
        mock_data = np.random.rand(3, 100, 100).astype(np.float32)
        mock_profile = {'count': 3, 'height': 100, 'width': 100, 'dtype': 'float32'}
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                mock_src = MagicMock()
                mock_src.read.return_value = mock_data
                mock_src.profile = mock_profile
                mock_open.return_value.__enter__.return_value = mock_src
                
                # Create the file so validation passes
                Path(tmp_path).touch()
                
                data, profile = FileHandler.read_raster(tmp_path)
                
                assert np.array_equal(data, mock_data)
                assert profile == mock_profile
                mock_src.read.assert_called_once()
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_read_raster_nonexistent_file(self):
        """Test reading non-existent raster file."""
        with pytest.raises(FileNotFoundError):
            FileHandler.read_raster("/path/to/nonexistent.tif")


class TestVectorOperations:
    """Test cases for vector file operations."""
    
    @patch('analyz.utils.file_handler.gpd.read_file')
    @patch.object(FileHandler, 'validate_file')
    def test_read_vector_success(self, mock_validate, mock_read_file):
        """Test successful vector reading."""
        # Setup mocks
        mock_path = Path("/path/to/vector.geojson")
        mock_validate.return_value = mock_path
        
        mock_gdf = MagicMock()
        mock_read_file.return_value = mock_gdf
        
        result = FileHandler.read_vector("/path/to/vector.geojson")
        
        assert result == mock_gdf
        mock_validate.assert_called_once_with("/path/to/vector.geojson")
        mock_read_file.assert_called_once_with(mock_path)
    
    @patch('analyz.utils.file_handler.gpd.read_file')
    def test_read_vector_validation_failure(self, mock_read_file):
        """Test vector reading with validation failure."""
        with pytest.raises(FileNotFoundError):
            FileHandler.read_vector("/path/to/nonexistent.geojson")
        
        # read_file should not be called if validation fails
        mock_read_file.assert_not_called()
    
    def test_write_vector_with_auto_driver_detection(self):
        """Test vector writing with automatic driver detection."""
        mock_gdf = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test different extensions
            test_cases = [
                ("test.geojson", "GeoJSON"),
                ("test.shp", "ESRI Shapefile"),
                ("test.gpkg", "GPKG"),
                ("test.kml", "KML"),
                ("test.xyz", "GeoJSON")  # Unknown extension defaults to GeoJSON
            ]
            
            for filename, expected_driver in test_cases:
                output_path = Path(tmp_dir) / filename
                
                FileHandler.write_vector(str(output_path), mock_gdf)
                
                # Verify to_file was called with correct driver
                mock_gdf.to_file.assert_called_with(output_path, driver=expected_driver)
    
    def test_write_vector_with_explicit_driver(self):
        """Test vector writing with explicit driver specification."""
        mock_gdf = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test.geojson"
            
            FileHandler.write_vector(str(output_path), mock_gdf, driver="GPKG")
            
            mock_gdf.to_file.assert_called_with(output_path, driver="GPKG")
    
    def test_write_vector_creates_directory(self):
        """Test that vector writing creates output directory."""
        mock_gdf = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "directory" / "output.geojson"
            
            FileHandler.write_vector(str(output_path), mock_gdf)
            
            # Verify directory was created
            assert output_path.parent.exists()
            assert output_path.parent.is_dir()


class TestUtilityMethods:
    """Test cases for utility methods."""
    
    def test_create_output_path(self):
        """Test output path creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            filename = "output.tif"
            
            result = FileHandler.create_output_path(base_dir, filename)
            
            expected_path = base_dir / filename
            assert result == expected_path
            assert result.parent.exists()  # Directory should be created
    
    def test_create_output_path_with_nested_structure(self):
        """Test output path creation with nested directory structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir) / "level1" / "level2"  # Doesn't exist yet
            filename = "output.tif"
            
            result = FileHandler.create_output_path(base_dir, filename)
            
            expected_path = base_dir / filename
            assert result == expected_path
            assert result.parent.exists()  # Nested directories should be created
            assert result.parent.is_dir()
    
    def test_create_output_path_string_input(self):
        """Test output path creation with string inputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = FileHandler.create_output_path(tmp_dir, "output.tif")
            
            expected_path = Path(tmp_dir) / "output.tif"
            assert result == expected_path


class TestIntegrationScenarios:
    """Integration test scenarios for FileHandler."""
    
    def test_roundtrip_raster_operations(self):
        """Test reading and writing raster data roundtrip."""
        # Create test data
        original_data = np.random.rand(3, 50, 50).astype(np.float32)
        test_profile = {
            'driver': 'GTiff',
            'count': 3,
            'height': 50,
            'width': 50,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': rasterio.Affine.identity(),
        }
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Mock the rasterio operations for both read and write
            with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                # Setup write mock
                mock_dst = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_dst
                
                # Write the data
                FileHandler.write_raster(tmp_path, original_data, test_profile)
                
                # Verify write was called correctly
                written_data = mock_dst.write.call_args[0][0]
                assert written_data.shape == original_data.shape
                np.testing.assert_array_equal(written_data, original_data)
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_different_data_types_handling(self):
        """Test handling of different numpy data types."""
        test_cases = [
            np.uint8,
            np.uint16,
            np.int16,
            np.float32,
            np.float64
        ]
        
        sample_profile = {
            'driver': 'GTiff',
            'count': 1,
            'height': 10,
            'width': 10,
            'crs': 'EPSG:4326',
            'transform': rasterio.Affine.identity(),
        }
        
        for dtype in test_cases:
            data = np.random.rand(10, 10).astype(dtype)
            
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                with patch('analyz.utils.file_handler.rasterio.open') as mock_open:
                    mock_dst = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_dst
                    
                    FileHandler.write_raster(tmp_path, data, sample_profile)
                    
                    # Verify dtype was preserved in profile
                    write_args = mock_open.call_args[1]
                    assert write_args['dtype'] == dtype
                    
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)