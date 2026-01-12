# Analyz - Optical and SAR Image Analysis Tool

A comprehensive Python application for automated analysis of optical and SAR (Synthetic Aperture Radar) images with study area boundary support and automated insights generation.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 **NEW in Version 2.0**

### ✨ Automatic Land Cover Classification with Semantic Labels
- **Smart Class Identification**: Automatically labels classes as Water, Vegetation, Urban, Bare Soil, etc.
- **No more "Class 0, Class 1"**: Get meaningful land cover names based on spectral signatures
- **Area Calculations**: See coverage in km² for each land cover type
- **Beautiful Visualizations**: Dual-panel plots with legends and distribution charts

### 🌿 Enhanced NDVI with Vegetation Stress Detection
- **Vegetation Vigor Index**: Overall health metric (0-1 scale)
- **Stress Detection**: Automatically identifies stressed vegetation areas
- **5-Level Categorization**: No Veg, Sparse/Stressed, Moderate, Healthy, Very Healthy
- **Smart Alerts**: Warns when >15% of area shows stress
- **Area-based Metrics**: Stressed area calculations in km²

### 🛰️ Three New SAR Analyses
- **Polarimetric Analysis**: VV/VH ratio for forest/urban/agriculture discrimination
- **Soil Moisture Estimation**: 5-level classification from SAR backscatter
- **Coherence Analysis**: Surface change detection between acquisitions

### 🐛 Bug Fixes
- Fixed server restart issue during large file uploads
- Improved stability and performance

---

## Features

### Optical Image Analysis
- **NDVI** - Vegetation health with **stress detection** ⚡NEW
- **NDWI** - Water body detection
- **NDBI** - Urban area mapping
- **EVI** - Enhanced vegetation monitoring
- **SAVI** - Soil adjusted vegetation index
- **Land Cover Classification** - With **semantic labels** ⚡NEW
- **Change Detection** - Multi-temporal analysis
- **Band Arithmetic** - Custom index calculations

### SAR Image Analysis (Application-Focused - 10 Analyses)
- **Oil Spill Detection** - Dark patch detection with CFAR/adaptive threshold ⚡NEW
- **Ship Detection** - Maritime surveillance using CFAR on bright targets ⚡NEW
- **Crop Monitoring** - Agriculture monitoring with Radar Vegetation Index (RVI) ⚡NEW
- **Land Cover Classification** - ML-based classification using SAR features ⚡NEW
- **Biomass Estimation** - Forest biomass index from backscatter + texture ⚡NEW
- **Wildfire Burn Mapping** - Pre/post-fire change detection with burn severity ⚡NEW
- **Geology & Terrain Analysis** - Roughness, lineament detection, terrain classification ⚡NEW
- **Flood Mapping** - Water detection with area calculations
- **Polarimetric Decomposition** - VV/VH ratio for land cover discrimination
- **Soil Moisture Estimation** - Relative soil moisture from SAR backscatter

### Core Features
- **Study Area Boundary** - Clip analysis to specific AOI (GeoJSON, Shapefile)
- **Automated Insights** - Statistical summaries and interpretation
- **Visualization** - Interactive plots and maps
- **Batch Processing** - Process multiple images
- **Export Options** - GeoTIFF, PNG, CSV, PDF reports

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 🌐 Web Application (No Coding Required!)

**Easiest way to get started:**

1. Install dependencies: `pip install -r requirements.txt`
2. Launch web app: **Double-click `start_webapp.bat`**
3. Open browser: **http://localhost:5000**
4. Upload imagery, configure, and analyze!

See **`WEB_APP_QUICKSTART.md`** for complete web app guide.

### 🐍 Python API

```python
from analyz import Analyz

# Initialize the app
app = Analyz()

# Load optical image
app.load_optical_image("path/to/image.tif")

# Set study area boundary
app.set_study_area("path/to/boundary.geojson")

# Run NDVI analysis
results = app.run_analysis("ndvi")

# Generate insights and visualizations
app.generate_insights(results)
app.visualize(results, output_path="ndvi_analysis.png")
```

## Usage

### Command Line Interface

```bash
# Optical NDVI analysis
python app.py --image optical.tif --analysis ndvi --boundary aoi.geojson --output results/

# SAR speckle filtering
python app.py --image sar.tif --analysis speckle_filter --filter-type lee --boundary aoi.shp --output results/

# Change detection
python app.py --image1 before.tif --image2 after.tif --analysis change_detection --boundary aoi.geojson --output results/
```

### Python API

See `examples/` directory for detailed usage examples.

## Project Structure

```
automatedAnalysis/
├── analyz/
│   ├── __init__.py
│   ├── core/
│   │   ├── optical_analysis.py
│   │   ├── sar_analysis.py
│   │   └── base_analyzer.py
│   ├── processing/
│   │   ├── boundary_handler.py
│   │   ├── preprocessor.py
│   │   └── postprocessor.py
│   ├── visualization/
│   │   ├── plotter.py
│   │   └── insights_generator.py
│   └── utils/
│       ├── file_handler.py
│       └── logger.py
├── examples/
├── tests/
├── app.py
├── config.yaml
├── requirements.txt
└── README.md
```

## 📚 Documentation

- **[Manager.md](Manager.md)** - Executive overview for managers (architecture, tech stack, deployment)
- **[docs/API_USAGE_GUIDE.md](docs/API_USAGE_GUIDE.md)** - Complete Python API reference and CLI guide
- **[docs/WEB_APP_GUIDE.md](docs/WEB_APP_GUIDE.md)** - Web application user guide and configuration
- **[webapp/README_WEB.md](webapp/README_WEB.md)** - Web app implementation details

## 🚀 Data Sources

### Optical
- **Landsat 8/9**: Free, 30m resolution - [USGS EarthExplorer](https://earthexplorer.usgs.gov/)
- **Sentinel-2**: Free, 10m resolution - [Copernicus Hub](https://scihub.copernicus.eu/)

### SAR
- **Sentinel-1**: Free, 10m resolution - [Copernicus Hub](https://scihub.copernicus.eu/)
- Format: GRD (Ground Range Detected), IW mode, VV+VH polarization

## 🎯 Applications

### SAR Applications ⚡NEW
- **Maritime Surveillance**: Oil spill detection, ship detection, illegal fishing monitoring
- **Agriculture**: Crop vigor monitoring (RVI), phenology tracking, soil moisture
- **Forestry**: Biomass estimation, forest structure analysis, deforestation detection
- **Disaster Response**: Flood mapping (all-weather), wildfire burn mapping, damage assessment
- **Land Management**: Land cover classification, urban mapping, change detection
- **Geology & Mining**: Terrain roughness analysis, lineament/fault detection, geological mapping

### Optical Applications
- **Agriculture**: Crop health monitoring (NDVI), stress detection, yield prediction
- **Water Resources**: Water body detection (NDWI), wetland monitoring, reservoir tracking
- **Urban Planning**: Built-up area mapping (NDBI), settlement growth analysis
- **Environmental**: Vegetation monitoring (EVI), ecosystem health assessment

## License

MIT License

## Contributors

Samson Adeyomoye

## ⭐ Acknowledgments

- Landsat missions (NASA/USGS)
- Sentinel missions (ESA/Copernicus)
- Open-source geospatial community
