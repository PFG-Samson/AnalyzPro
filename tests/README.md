# Analyz Test Suite

This directory contains comprehensive unit tests for the Analyz satellite image analysis application.

## Test Coverage

### OpticalAnalyzer Tests (`test_optical_analyzer.py`)
- ✅ **Spectral Index Calculations**: Tests for various spectral indices (NDVI, NDWI, EVI2, SAVI, NDBI, BSI, etc.)
- ✅ **Sensor Validation**: Tests for Sentinel-2 and Landsat sensor capability validation
- ✅ **Band Requirements**: Tests for proper validation of required bands for each index
- ✅ **Edge Cases**: Tests for NaN values, extreme values, and error handling

### SARAnalyzer Tests (`test_sar_analyzer.py`)
- ✅ **Initialization with Profile**: Tests for proper initialization with rasterio profiles
- ✅ **Pixel Area Calculation**: Tests for correct pixel area calculation from transform data
- ✅ **NoData Mask Handling**: Tests for proper mask creation and application
- ✅ **Speckle Filter Implementation**: Tests for Lee and median filters with NaN handling
- ✅ **Filter Parameter Variations**: Tests for different window sizes, looks parameters

### FileHandler Tests (`test_file_handler.py`)
- ✅ **Array Shape Handling**: Tests for 2D, CHW (channels-first), and HWC (channels-last) formats
- ✅ **Profile Updates**: Tests for proper rasterio profile updates with array properties
- ✅ **Directory Creation**: Tests for automatic output directory creation
- ✅ **Data Type Preservation**: Tests for preservation of numpy data types
- ✅ **Vector Operations**: Tests for GeoDataFrame reading/writing with driver detection

## Test Fixtures

The test suite uses several shared fixtures defined in `conftest.py`:

- `sample_optical_data`: Synthetic 6-band optical data (100x100 pixels)
- `sample_sar_data`: Synthetic dual-pol SAR data (80x80 pixels) 
- `standard_band_indices`: Standard optical band mapping
- `sample_raster_profile`: Template rasterio profile for testing
- `temp_directory` / `temp_file`: Temporary file system resources

## Running Tests

### Option 1: Using pytest directly
```bash
pytest tests/ -v
```

### Option 2: Using the test runner script
```bash
python run_tests.py
```

### Option 3: Run specific test files
```bash
pytest tests/test_optical_analyzer.py -v
pytest tests/test_sar_analyzer.py -v  
pytest tests/test_file_handler.py -v
```

### Option 4: Run with coverage
```bash
pytest tests/ --cov=analyz --cov-report=html
```

## Test Requirements

The following packages are required to run the tests (already included in `requirements.txt`):

- `pytest>=7.4.0`: Main testing framework
- `pytest-cov>=4.1.0`: Coverage reporting
- `numpy`: Array operations and testing utilities
- `unittest.mock`: Mocking for external dependencies

## Test Design Principles

1. **Isolation**: Each test is independent and doesn't rely on external files or network resources
2. **Mocking**: External dependencies (rasterio, file I/O) are mocked to ensure fast, reliable tests
3. **Coverage**: Tests cover both normal operation and edge cases (NaN values, extreme values, errors)
4. **Reproducibility**: Random seeds are set for consistent test results
5. **Real-world Scenarios**: Tests simulate realistic satellite data processing workflows

## Adding New Tests

When adding new functionality to the codebase:

1. Add corresponding test methods to the appropriate test class
2. Use descriptive test names that clearly indicate what is being tested
3. Include both positive (success) and negative (error) test cases
4. Mock external dependencies to keep tests fast and independent
5. Add any new test fixtures to `conftest.py` if they will be reused

## Test Organization

Tests are organized into logical classes:
- `TestClassInitialization`: Constructor and setup tests
- `TestMethodName`: Tests for specific methods
- `TestEdgeCases`: Error handling and boundary condition tests
- `TestIntegration`: End-to-end workflow tests