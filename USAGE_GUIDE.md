# Analyz Usage Guide

Complete guide for using the Analyz optical and SAR image analysis application.

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Optical Analysis](#optical-analysis)
4. [SAR Analysis](#sar-analysis)
5. [Study Area Boundaries](#study-area-boundaries)
6. [Configuration](#configuration)
7. [Command Line Interface](#command-line-interface)
8. [Python API](#python-api)

## Installation

```powershell
# Navigate to project directory
cd automatedAnalysis

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line - NDVI Analysis
```powershell
python app.py --image "C:\path\to\optical.tif" `
              --image-type optical `
              --analysis ndvi `
              --boundary "C:\path\to\boundary.geojson" `
              --output "outputs\ndvi_analysis"
```

### Command Line - SAR Speckle Filtering
```powershell
python app.py --image "C:\path\to\sar.tif" `
              --image-type sar `
              --analysis lee_filter `
              --window-size 5 `
              --output "outputs\sar_filtered"
```

## Optical Analysis

### Available Analyses

1. **NDVI** - Vegetation health and coverage
2. **NDWI** - Water body detection
3. **NDBI** - Built-up/urban area mapping
4. **EVI** - Enhanced vegetation monitoring
5. **SAVI** - Soil-adjusted vegetation index
6. **Classification** - Land cover classification

### Example: NDVI Analysis

```python
from analyz import OpticalAnalyzer, FileHandler, Plotter, InsightsGenerator

# Load image
data, profile = FileHandler.read_raster("optical_image.tif")

# Define band indices (adjust based on your sensor)
band_indices = {
    'red': 2,    # Band 3 for Sentinel-2
    'nir': 3,    # Band 4 for Sentinel-2
    'green': 1,  # Band 2 for Sentinel-2
    'blue': 0    # Band 1 for Sentinel-2
}

# Initialize analyzer
analyzer = OpticalAnalyzer(data, band_indices)

# Calculate NDVI
ndvi, stats = analyzer.ndvi()

# Visualize
Plotter.plot_ndvi_classification(ndvi, output_path="ndvi_result.png")

# Generate insights
insights = InsightsGenerator.generate_ndvi_insights(ndvi, stats)
InsightsGenerator.format_insights_report(insights, "ndvi_report.txt")
```

### Band Configurations for Common Sensors

#### Sentinel-2 (L2A)
```python
band_indices = {
    'blue': 0,    # B2 (490 nm)
    'green': 1,   # B3 (560 nm)
    'red': 2,     # B4 (665 nm)
    'nir': 3,     # B8 (842 nm)
    'swir1': 4,   # B11 (1610 nm)
    'swir2': 5    # B12 (2190 nm)
}
```

#### Landsat 8/9
```python
band_indices = {
    'blue': 1,    # Band 2 (452-512 nm)
    'green': 2,   # Band 3 (533-590 nm)
    'red': 3,     # Band 4 (636-673 nm)
    'nir': 4,     # Band 5 (851-879 nm)
    'swir1': 5,   # Band 6 (1566-1651 nm)
    'swir2': 6    # Band 7 (2107-2294 nm)
}
```

## SAR Analysis

### Available Analyses

1. **Lee Filter** - Adaptive speckle filtering
2. **Frost Filter** - Edge-preserving speckle reduction
3. **Median Filter** - Simple speckle reduction
4. **Backscatter Analysis** - Intensity analysis in dB
5. **Texture Analysis** - GLCM texture features
6. **Coherence Estimation** - Temporal coherence
7. **Flood Mapping** - Water detection
8. **Change Detection** - Multi-temporal analysis

### Example: Speckle Filtering

```python
from analyz import SARAnalyzer, FileHandler, Plotter

# Load SAR image
data, profile = FileHandler.read_raster("sar_image.tif")

# Initialize analyzer
analyzer = SARAnalyzer(data)

# Apply Lee filter
filtered, stats = analyzer.lee_filter(window_size=5, num_looks=1)

# Visualize comparison
Plotter.plot_comparison(data[0], filtered[0], 
                       "Original", "Lee Filtered",
                       output_path="lee_comparison.png")
```

### Example: Flood Mapping

```python
# Detect flood/water areas
flood_mask, stats = analyzer.flood_mapping(threshold_method='otsu')

print(f"Water coverage: {stats['water_percent']:.2f}%")

# Visualize
Plotter.plot_raster(flood_mask, "Flood Map", cmap='Blues',
                   output_path="flood_map.png")

# Generate insights
insights = InsightsGenerator.generate_sar_insights("Flood Mapping", 
                                                   flood_mask, stats)
```

## Study Area Boundaries

Reduce analysis to specific areas using boundary files.

### Supported Formats
- GeoJSON (`.geojson`)
- Shapefile (`.shp`)
- GeoPackage (`.gpkg`)
- KML (`.kml`)

### Using Boundaries

```python
from analyz import BoundaryHandler

# Load boundary
boundary = BoundaryHandler("study_area.geojson")

# Clip raster
clipped_data, clipped_profile = boundary.clip_raster(
    "image.tif",
    output_path="clipped_image.tif"
)

# Get boundary info
area = boundary.get_boundary_area(crs="EPSG:32633")  # Area in m²
bounds = boundary.get_boundary_bounds()  # (minx, miny, maxx, maxy)
```

### CLI with Boundary

```powershell
python app.py --image optical.tif `
              --image-type optical `
              --analysis ndvi `
              --boundary study_area.geojson `
              --output outputs\ndvi_clipped
```

## Configuration

Edit `config.yaml` to customize defaults:

```yaml
# Processing settings
processing:
  max_memory_mb: 4096
  num_threads: 4
  resampling_method: "bilinear"

# Optical settings
optical:
  default_bands:
    red: 3
    nir: 4
    green: 2
    blue: 1
  ndvi:
    threshold_low: 0.2
    threshold_high: 0.8

# Visualization settings
visualization:
  dpi: 300
  figure_size: [12, 8]
  colormap: "RdYlGn"
```

## Command Line Interface

### General Syntax

```powershell
python app.py --image <path> `
              --image-type <optical|sar> `
              --analysis <analysis_type> `
              [--boundary <path>] `
              [--output <dir>] `
              [options]
```

### Optical Options

```powershell
--band-indices "red:2,nir:3,green:1,blue:0"  # Custom band mapping
--n-clusters 5                                # For classification
```

### SAR Options

```powershell
--window-size 5     # Filter window size
--num-looks 1       # Number of looks
```

### Examples

**NDVI with custom bands:**
```powershell
python app.py --image sentinel2.tif `
              --image-type optical `
              --analysis ndvi `
              --band-indices "red:2,nir:7,green:1,blue:0" `
              --output outputs\ndvi
```

**Land cover classification:**
```powershell
python app.py --image optical.tif `
              --image-type optical `
              --analysis classification `
              --n-clusters 7 `
              --boundary aoi.geojson `
              --output outputs\classification
```

**SAR flood mapping:**
```powershell
python app.py --image sar.tif `
              --image-type sar `
              --analysis flood_mapping `
              --boundary flood_area.geojson `
              --output outputs\flood
```

## Python API

### Complete Workflow Example

```python
from pathlib import Path
from analyz import *

# Setup
setup_logger("INFO", "analysis.log")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Load and clip image
data, profile = FileHandler.read_raster("image.tif")
boundary = BoundaryHandler("boundary.geojson")
data, profile = boundary.clip_array(data, profile)

# Optical analysis
band_indices = {'red': 2, 'nir': 3, 'green': 1, 'blue': 0}
optical = OpticalAnalyzer(data, band_indices)

# Multiple indices
ndvi, ndvi_stats = optical.ndvi()
ndwi, ndwi_stats = optical.ndwi()
ndbi, ndbi_stats = optical.ndbi()

# Visualize
Plotter.plot_multifigure(
    [ndvi, ndwi, ndbi],
    ['NDVI', 'NDWI', 'NDBI'],
    ['RdYlGn', 'Blues', 'Reds'],
    output_path=output_dir / "indices.png"
)

# Generate insights
for idx_name, idx_data, idx_stats in [
    ('ndvi', ndvi, ndvi_stats),
    ('ndwi', ndwi, ndwi_stats)
]:
    if idx_name == 'ndvi':
        insights = InsightsGenerator.generate_ndvi_insights(idx_data, idx_stats)
    else:
        insights = InsightsGenerator.generate_ndwi_insights(idx_data, idx_stats)
    
    InsightsGenerator.format_insights_report(
        insights, 
        output_dir / f"{idx_name}_insights.txt"
    )

# Save results
FileHandler.write_raster(output_dir / "ndvi.tif", ndvi, profile)
FileHandler.write_raster(output_dir / "ndwi.tif", ndwi, profile)
FileHandler.write_raster(output_dir / "ndbi.tif", ndbi, profile)
```

## Tips and Best Practices

1. **Memory Management**: For large images, process in chunks or use the `max_memory_mb` config
2. **Band Order**: Always verify band order in your imagery before analysis
3. **Coordinate Systems**: Ensure boundary CRS matches image CRS (auto-reprojection is supported)
4. **Speckle Filtering**: Start with Lee filter for SAR; adjust window size based on image resolution
5. **Thresholds**: Use Otsu thresholding first, then adjust manually if needed
6. **Insights**: Always review generated insights and statistics for quality control

## Troubleshooting

**Issue**: "Band 'nir' not defined in band_indices"
- **Solution**: Verify band_indices match your image band order

**Issue**: CRS mismatch warnings
- **Solution**: Boundary will auto-reproject, but verify results

**Issue**: Memory errors with large images
- **Solution**: Reduce `max_memory_mb` or clip to smaller area first

**Issue**: Poor classification results
- **Solution**: Try different n_clusters values or preprocess with contrast enhancement

For more examples, see the `examples/` directory.
