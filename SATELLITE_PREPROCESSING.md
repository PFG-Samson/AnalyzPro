# Satellite Data Preprocessing Guide

The app now supports **automatic preprocessing** of raw satellite data archives from Landsat and Sentinel-2!

## Supported Formats

### Landsat (8 & 9)
- **Archives**: `.tar`, `.tar.gz`, `.zip`
- **Bands**: B1-B7 (coastal, blue, green, red, NIR, SWIR1, SWIR2)
- **Sources**: USGS EarthExplorer, Google Earth Engine

### Sentinel-2
- **Archives**: `.zip` containing `.SAFE` directory
- **Bands**: B02-B08, B11-B12 (blue, green, red, NIR, SWIR)
- **Resolutions**: 10m, 20m, 60m
- **Sources**: Copernicus Open Access Hub, Google Earth Engine

## How It Works

When you upload a satellite archive (`.tar`, `.zip`) to the web app:

1. **Auto-detection**: System detects whether it's Landsat or Sentinel-2
2. **Extraction**: Archive is extracted temporarily
3. **Band Finding**: Locates individual band files (e.g., `B4.TIF`, `B08.jp2`)
4. **Stacking**: Combines bands into a single multi-band GeoTIFF
5. **Analysis**: Runs your chosen analysis (NDVI, NDWI, etc.)

## Using the Web App

### Upload Archive Directly
1. Start the web app: `python webapp/app_web.py`
2. Go to Upload page
3. Select your Landsat `.tar` or Sentinel `.zip` file
4. Choose analysis type (NDVI, NDWI, etc.)
5. Click "Analyze" - preprocessing happens automatically!

The app will show status:
- **Preprocessing**: Extracting and stacking bands
- **Processing**: Running analysis
- **Completed**: Results ready!

## Using the CLI Tool

For more control, preprocess first then upload:

### Basic Usage
```bash
# Auto-detect satellite type
python preprocess_satellite.py landsat_data.tar landsat_processed.tif

# Process Sentinel-2
python preprocess_satellite.py sentinel_data.zip sentinel_processed.tif
```

### Advanced Options

**Select specific bands (Landsat):**
```bash
python preprocess_satellite.py LC08_data.tar output.tif --bands B2 B3 B4 B5
# Only includes: Blue, Green, Red, NIR
```

**Select specific bands (Sentinel-2):**
```bash
python preprocess_satellite.py S2_data.zip output.tif --bands B02 B03 B04 B08
# Only includes: Blue, Green, Red, NIR
```

**Choose resolution (Sentinel-2):**
```bash
python preprocess_satellite.py S2_data.zip output.tif --resolution 20m
# Options: 10m (default), 20m, 60m
```

**Verbose mode:**
```bash
python preprocess_satellite.py data.tar output.tif --verbose
```

## Band Mapping Reference

### Landsat 8/9
| Band | Name | Wavelength | Use |
|------|------|------------|-----|
| B1 | Coastal/Aerosol | 0.43-0.45 µm | Coastal/aerosol |
| B2 | Blue | 0.45-0.51 µm | Blue |
| B3 | Green | 0.53-0.59 µm | Green |
| B4 | Red | 0.64-0.67 µm | Red |
| B5 | NIR | 0.85-0.88 µm | NIR (vegetation) |
| B6 | SWIR1 | 1.57-1.65 µm | SWIR1 |
| B7 | SWIR2 | 2.11-2.29 µm | SWIR2 |

### Sentinel-2
| Band | Name | Resolution | Use |
|------|------|------------|-----|
| B02 | Blue | 10m | Blue |
| B03 | Green | 10m | Green |
| B04 | Red | 10m | Red |
| B05 | Red Edge 1 | 20m | Vegetation |
| B06 | Red Edge 2 | 20m | Vegetation |
| B07 | Red Edge 3 | 20m | Vegetation |
| B08 | NIR | 10m | NIR (vegetation) |
| B8A | NIR Narrow | 20m | NIR narrow |
| B11 | SWIR1 | 20m | SWIR1 |
| B12 | SWIR2 | 20m | SWIR2 |

## Default Band Selection

When processing automatically, these bands are included:

**Landsat**: B1-B7 (all optical bands)
**Sentinel-2 (10m)**: B02, B03, B04, B08 (Blue, Green, Red, NIR)

This provides all bands needed for:
- NDVI (Normalized Difference Vegetation Index)
- NDWI (Normalized Difference Water Index)
- NDBI (Normalized Difference Built-up Index)
- EVI (Enhanced Vegetation Index)
- SAVI (Soil-Adjusted Vegetation Index)

## Tips

1. **Large files**: Preprocessing can take time for large archives (especially Sentinel-2). Be patient!

2. **Disk space**: Ensure you have enough space. Archives are extracted temporarily:
   - Landsat: ~1-2 GB per scene
   - Sentinel-2: ~5-8 GB per scene

3. **Band selection**: If you only need NDVI, you can preprocess with just the required bands:
   ```bash
   # Landsat NDVI only needs Red and NIR
   python preprocess_satellite.py landsat.tar output.tif --bands B4 B5
   
   # Sentinel-2 NDVI
   python preprocess_satellite.py sentinel.zip output.tif --bands B04 B08
   ```

4. **Cloud masking**: This tool does NOT perform cloud masking. Use pre-processed or cloud-free scenes.

## Where to Get Satellite Data

### Landsat 8/9
- **USGS EarthExplorer**: https://earthexplorer.usgs.gov/
  - Register (free)
  - Search by location/date
  - Download Level-1 or Level-2 products
  - Choose "LandsatLook Quality L1" for Level-1

### Sentinel-2
- **Copernicus Open Access Hub**: https://scihub.copernicus.eu/
  - Register (free)
  - Search by location/date
  - Download L1C or L2A products
  
- **Google Earth Engine** (requires coding):
  - Can export preprocessed data
  - Includes cloud masking options

## Troubleshooting

**"Could not detect satellite type"**
- Archive may be corrupted
- Not a Landsat/Sentinel-2 archive
- Try extracting manually first

**"No bands found"**
- Archive structure unexpected
- May need to extract to different location
- Check file naming conventions

**"Out of memory"**
- Scene too large for available RAM
- Try processing smaller area
- Use lower resolution (Sentinel-2)

## Python API

You can also use the preprocessor in your own scripts:

```python
from analyz.utils import SatellitePreprocessor

# Auto-detect and process
SatellitePreprocessor.process_auto(
    'landsat.tar', 
    'output.tif'
)

# Process Landsat with specific bands
SatellitePreprocessor.process_landsat(
    'LC08_data.tar',
    'landsat.tif',
    bands=['B2', 'B3', 'B4', 'B5']
)

# Process Sentinel-2
SatellitePreprocessor.process_sentinel2(
    'S2A_data.zip',
    'sentinel.tif',
    bands=['B02', 'B03', 'B04', 'B08'],
    resolution='10m'
)
```

## Need Help?

Check the logs for detailed information about what's happening during preprocessing. Use `--verbose` flag for maximum detail.
