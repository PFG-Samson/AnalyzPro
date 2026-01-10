# Analyz - Complete User Training Course
## Satellite Imagery Analysis Made Simple

---

# Course Overview

**Welcome to the Analyz Training Course!**

This comprehensive course will teach you everything you need to know to use Analyz effectively—from basic setup to advanced analysis workflows. Whether you're a GIS professional, environmental researcher, agricultural specialist, or someone new to remote sensing, this course is designed for you.

**What You'll Learn:**
- How to install and configure Analyz
- Navigate both the Web Interface and Command Line Interface
- Perform Optical and SAR satellite imagery analyses
- Generate professional reports with automated insights
- Troubleshoot common issues and optimize your workflow

**Course Duration:** Approximately 4-6 hours (self-paced)

**Prerequisites:** 
- Basic computer skills
- No programming experience required (though Python knowledge is a plus!)

---

# Chapter 1: Introduction to Analyz

## 1.1 What is Analyz?

**Analyz** is a powerful, open-source Python application for automated analysis of satellite imagery. It processes two primary types of satellite data:

1. **Optical Imagery** (from satellites like Sentinel-2 and Landsat 8/9)
2. **SAR (Synthetic Aperture Radar) Imagery** (from satellites like Sentinel-1)

<!-- [IMAGE PLACEHOLDER: Analyz Logo and Interface Overview - Show the main web interface dashboard] -->

### Key Capabilities at a Glance

| Feature | Description |
|---------|-------------|
| **NDVI Analysis** | Assess vegetation health and detect stressed crops |
| **Land Cover Classification** | Automatically identify water, vegetation, urban areas, and bare soil |
| **Flood Mapping** | Detect water bodies using all-weather SAR data |
| **Soil Moisture Estimation** | Monitor irrigation needs from SAR backscatter |
| **Change Detection** | Track land use changes over time |
| **Automated Insights** | Get professional reports with statistical summaries |

---

## 1.2 Who is Analyz For?

Analyz is designed for diverse users across multiple industries:

### Environmental Scientists
- Monitor vegetation health and ecosystem changes
- Track deforestation and land degradation
- Assess wetland and water body extent

### Agricultural Professionals
- Monitor crop health with NDVI stress detection
- Plan irrigation with soil moisture analysis
- Classify crop types and track growth cycles

### Urban Planners
- Map built-up areas with NDBI analysis
- Track urban expansion and settlement growth
- Monitor infrastructure development

### Disaster Response Teams
- Rapid flood mapping with SAR data
- Damage assessment and recovery planning
- All-weather monitoring capabilities

### Research Institutions
- Batch processing for large-scale studies
- Reproducible analysis with command-line workflows
- Integration with existing GIS pipelines

<!-- [IMAGE PLACEHOLDER: Industry Applications Infographic - Show icons representing different industries] -->

---

## 1.3 What Problems Does Analyz Solve?

### Challenge 1: Complex GIS Software
**Problem:** Traditional remote sensing software (ENVI, ERDAS) requires extensive training and expensive licenses.

**Solution:** Analyz provides a user-friendly web interface and is completely free and open-source.

### Challenge 2: Technical Knowledge Barrier
**Problem:** Many analysis tools require programming expertise.

**Solution:** Analyz's web interface requires zero coding—just upload, configure, and analyze!

### Challenge 3: Time-Consuming Processing
**Problem:** Manually calculating indices and generating reports takes hours.

**Solution:** Analyz automates the entire workflow: load → analyze → visualize → generate insights.

### Challenge 4: SAR Data Complexity
**Problem:** SAR analysis tools are rare in free software.

**Solution:** Analyz provides 10+ SAR analysis types, including flood mapping, soil moisture, and polarimetric analysis.

---

## 1.4 Key Features Overview

### Optical Image Analysis Suite
- **NDVI** - Vegetation health with stress detection and 5-level categorization
- **NDWI** - Water body detection and wetland monitoring
- **NDBI** - Urban and built-up area mapping
- **EVI** - Enhanced vegetation monitoring for dense canopies
- **SAVI** - Soil-adjusted vegetation index for arid regions
- **Land Cover Classification** - Automatic semantic labeling (Water, Vegetation, Urban, Bare Soil)
- **Change Detection** - Multi-temporal analysis

### SAR Image Analysis Suite
- **Speckle Filtering** - Lee, Frost, and Median filters for noise reduction
- **Flood Mapping** - All-weather water detection
- **Polarimetric Analysis** - VV/VH ratio for land cover discrimination
- **Soil Moisture** - 5-level moisture classification
- **Coherence Analysis** - Surface change detection
- **Ship Detection** - Maritime surveillance
- **Oil Spill Detection** - Environmental monitoring
- **Biomass Estimation** - Forest structure analysis

### Core Features
- **Study Area Boundaries** - Clip analysis to your area of interest
- **Automated Insights** - AI-generated statistical summaries
- **Professional Visualizations** - Publication-quality maps and charts
- **Flexible Export** - GeoTIFF, PNG, CSV, PDF reports

---

## 1.5 How Analyz Works: The Big Picture

Analyz follows a simple four-step workflow:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   1. LOAD    │ ──► │  2. PROCESS  │ ──► │  3. ANALYZE  │ ──► │  4. EXPORT   │
│              │     │              │     │              │     │              │
│ Upload your  │     │ Optional:    │     │ Calculate    │     │ Download     │
│ satellite    │     │ Clip to      │     │ indices and  │     │ results as   │
│ imagery      │     │ study area   │     │ classify     │     │ GeoTIFF +    │
│              │     │              │     │              │     │ reports      │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

<!-- [IMAGE PLACEHOLDER: Workflow Diagram - Visual representation of the 4-step process] -->

---

# Chapter 2: Setup and Installation

## 2.1 System Requirements

### Minimum Requirements
| Component | Requirement |
|-----------|-------------|
| **Operating System** | Windows 10+, macOS 10.15+, or Ubuntu 18.04+ |
| **Processor** | Intel/AMD x64 processor (2 GHz or faster) |
| **RAM** | 4 GB (8 GB recommended) |
| **Storage** | 2 GB free space + space for data |
| **Python** | Version 3.8 or higher |

### Recommended for Large Images
| Component | Recommendation |
|-----------|----------------|
| **Processor** | Quad-core CPU or better |
| **RAM** | 16 GB or more |
| **Storage** | SSD with 50+ GB free space |

<!-- [IMAGE PLACEHOLDER: System Requirements Check - Screenshot of system properties window] -->

---

## 2.2 Installing Python

### Windows Installation

1. **Download Python**
   - Visit [python.org/downloads](https://python.org/downloads)
   - Download the latest Python 3.8+ installer
   
2. **Run the Installer**
   - ✅ **IMPORTANT:** Check "Add Python to PATH" at the bottom
   - Click "Install Now"
   
3. **Verify Installation**
   - Open PowerShell (right-click Start → Windows PowerShell)
   - Type: `python --version`
   - You should see: `Python 3.x.x`

<!-- [IMAGE PLACEHOLDER: Python Installer - Screenshot showing "Add to PATH" checkbox] -->

### macOS Installation

```bash
# Using Homebrew (recommended)
brew install python

# Verify installation
python3 --version
```

### Linux (Ubuntu/Debian) Installation

```bash
sudo apt update
sudo apt install python3 python3-pip

# Verify installation
python3 --version
```

---

## 2.3 Installing GDAL (Required Dependency)

GDAL is a core geospatial library that Analyz depends on.

### Windows (Using Conda - Recommended)

```powershell
# Install Miniconda first from: https://docs.conda.io/en/latest/miniconda.html
# Then run:
conda install -c conda-forge gdal
```

### macOS

```bash
brew install gdal
```

### Linux

```bash
sudo apt install gdal-bin libgdal-dev
```

---

## 2.4 Installing Analyz

### Step 1: Download or Clone the Repository

If you received Analyz as a folder, simply extract it to your desired location (e.g., `C:\Analyz` or `~/Documents/Analyz`).

### Step 2: Install Python Dependencies

Open a terminal/PowerShell window and navigate to the Analyz folder:

```powershell
# Windows
cd C:\path\to\Analyz

# macOS/Linux
cd ~/path/to/Analyz
```

Install all required packages:

```powershell
pip install -r requirements.txt
```

This installs:
- **Flask** - Web framework
- **Rasterio** - Geospatial raster I/O
- **NumPy & SciPy** - Scientific computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Visualization
- And more...

<!-- [IMAGE PLACEHOLDER: Terminal Installation - Screenshot of successful pip install] -->

### Step 3: Verify Installation

```powershell
python -c "import analyz; print('✅ Analyz installed successfully!')"
```

If you see the success message, you're ready to go!

---

## 2.5 Launching Analyz

### Option 1: Web Application (Recommended for Beginners)

**Easiest method - Double-click to launch:**

1. Navigate to the Analyz folder
2. **Double-click** `start_webapp.bat` (Windows) or run `./launch_webapp.ps1`
3. Your browser will open automatically to `http://localhost:5000`

**Manual launch via command line:**

```powershell
cd webapp
python app_web.py
```

Then open your browser to: **http://localhost:5000**

<!-- [IMAGE PLACEHOLDER: Web App Launch - Screenshot of browser with Analyz homepage] -->

### Option 2: Command Line Interface (For Advanced Users)

```powershell
python app.py --help
```

This displays all available options and analysis types.

---

## 2.6 Configuration (config.yaml)

Analyz uses a YAML configuration file to customize default settings. The file is located at `config.yaml` in the root folder.

### Key Settings You Can Customize

```yaml
# Processing settings
processing:
  max_memory_mb: 4096    # Increase for larger images
  num_threads: 4         # Match your CPU cores
  resampling_method: "bilinear"

# Optical analysis defaults
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
  dpi: 300              # Image resolution
  figure_size: [12, 8]
  colormap: "RdYlGn"
```

> **💡 Tip:** For most users, the default settings work perfectly. Only modify these if you have specific requirements.

---

## 2.7 Getting Satellite Data

### Free Optical Data Sources

| Source | Satellite | Resolution | How to Access |
|--------|-----------|------------|---------------|
| **USGS EarthExplorer** | Landsat 8/9 | 30m | [earthexplorer.usgs.gov](https://earthexplorer.usgs.gov) |
| **Copernicus Open Hub** | Sentinel-2 | 10m | [scihub.copernicus.eu](https://scihub.copernicus.eu) |

### Free SAR Data Sources

| Source | Satellite | Resolution | How to Access |
|--------|-----------|------------|---------------|
| **Copernicus Open Hub** | Sentinel-1 | 10m | [scihub.copernicus.eu](https://scihub.copernicus.eu) |
| **Alaska SAR Facility** | Various | Various | [asf.alaska.edu](https://asf.alaska.edu) |

### Recommended SAR Format
- **Product Type:** GRD (Ground Range Detected)
- **Mode:** IW (Interferometric Wide)
- **Polarization:** VV + VH (dual-polarization)

<!-- [IMAGE PLACEHOLDER: Copernicus Hub Interface - Screenshot of data download portal] -->

---

# Chapter 3: Interface Orientation

## 3.1 The Web Application Interface

The web interface is designed for users who prefer a visual, no-code experience.

### Home Page

**What you'll see:**
- Welcome message and quick start guide
- Feature overview cards
- "Start Analysis" button to begin

<!-- [IMAGE PLACEHOLDER: Home Page - Full screenshot of the web app homepage] -->

---

### Upload Page

This is where you configure and submit your analysis.

**Section 1: Image Type Selection**
- **Optical** - For Sentinel-2, Landsat 8/9 imagery
- **SAR** - For Sentinel-1 radar data

<!-- [IMAGE PLACEHOLDER: Upload Page - Image Type Selection section] -->

**Section 2: File Upload**
- **Image File** - Upload your GeoTIFF (.tif, .tiff) - up to 500MB
- **Boundary File** (Optional) - Upload GeoJSON, Shapefile, or KML to clip analysis

<!-- [IMAGE PLACEHOLDER: Upload Page - File upload area with drag-and-drop] -->

**Section 3: Analysis Configuration**

For **Optical** analyses:
- Choose analysis type (NDVI, NDWI, NDBI, EVI, SAVI, Classification)
- Configure band indices (auto-configured for common sensors)
- Set number of classes for classification

For **SAR** analyses:
- Choose analysis type (Lee Filter, Flood Mapping, Soil Moisture, etc.)
- Configure window size and parameters

<!-- [IMAGE PLACEHOLDER: Upload Page - Analysis Configuration dropdown] -->

**Section 4: Start Analysis**
- Click the "Start Analysis" button
- Monitor progress in real-time

---

### Results Page

After analysis completes, you'll see:

**Progress Section:**
- Real-time progress bar (0-100%)
- Status updates (Queued → Processing → Completed)

**Results Section:**
- 📊 **Main Visualization** - Color-coded analysis result
- 📈 **Histogram** - Data distribution chart
- 📋 **Statistics** - Mean, median, min, max values
- 💡 **Insights** - AI-generated interpretation
- 📥 **Download Links** - All result files

<!-- [IMAGE PLACEHOLDER: Results Page - Completed analysis with all sections visible] -->

---

## 3.2 Band Indices Configuration

When working with optical imagery, you need to tell Analyz which bands contain which colors/wavelengths.

### Sentinel-2 Configuration
```
Blue: 0, Green: 1, Red: 2, NIR: 3, SWIR1: 4, SWIR2: 5
```

### Landsat 8/9 Configuration
```
Blue: 1, Green: 2, Red: 3, NIR: 4, SWIR1: 5, SWIR2: 6
```

> **💡 The web interface provides presets for common sensors, so you often don't need to configure this manually!**

<!-- [IMAGE PLACEHOLDER: Band Configuration - Screenshot of band index input fields] -->

---

## 3.3 The Command Line Interface (CLI)

The CLI provides more control and is ideal for:
- Batch processing multiple images
- Automated workflows and scripts
- Integration with other tools

### Basic CLI Syntax

```powershell
python app.py --image <path> --image-type <optical|sar> --analysis <type> [options]
```

### Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--image` | Path to input image | `--image C:\data\image.tif` |
| `--image-type` | Type of imagery | `--image-type optical` |
| `--analysis` | Analysis to perform | `--analysis ndvi` |
| `--boundary` | Study area boundary | `--boundary C:\data\aoi.geojson` |
| `--output` | Output directory | `--output C:\results` |
| `--band-indices` | Band mapping | `--band-indices "red:2,nir:3"` |
| `--window-size` | SAR filter size | `--window-size 5` |

### Example Commands

**NDVI Analysis:**
```powershell
python app.py --image sentinel2.tif --image-type optical --analysis ndvi --output results/ndvi
```

**SAR Flood Mapping:**
```powershell
python app.py --image sentinel1.tif --image-type sar --analysis flood_mapping --output results/flood
```

<!-- [IMAGE PLACEHOLDER: CLI Help Output - Screenshot of python app.py --help] -->

---

## 3.4 Python API (For Developers)

For programmers who want to integrate Analyz into their workflows:

```python
from analyz import OpticalAnalyzer, FileHandler, Plotter, InsightsGenerator

# Load image
data, profile = FileHandler.read_raster("optical_image.tif")

# Define band indices
band_indices = {'red': 2, 'nir': 3, 'green': 1, 'blue': 0}

# Analyze
analyzer = OpticalAnalyzer(data, band_indices)
ndvi, stats = analyzer.ndvi()

# Visualize
Plotter.plot_ndvi_classification(ndvi, output_path="ndvi_result.png")

# Generate insights
insights = InsightsGenerator.generate_ndvi_insights(ndvi, stats)
InsightsGenerator.format_insights_report(insights, "ndvi_report.txt")
```

---

# Chapter 4: Core Workflows

## 4.1 Workflow 1: NDVI Vegetation Analysis (Optical)

**Objective:** Assess vegetation health and detect stressed areas

**When to Use:**
- Monitoring crop health
- Assessing forest conditions
- Tracking seasonal vegetation changes

### Step-by-Step (Web Interface)

**Step 1:** Launch the web application
- Double-click `start_webapp.bat` or navigate to `http://localhost:5000`

**Step 2:** Click "Start Analysis" on the homepage

**Step 3:** Configure your analysis:
- Image Type: **Optical**
- Upload: Your Sentinel-2 or Landsat GeoTIFF
- Boundary (optional): Your study area GeoJSON
- Analysis: **NDVI - Vegetation Index**
- Verify band indices are correct

**Step 4:** Click "Start Analysis"

**Step 5:** Wait for processing (progress bar shows status)

**Step 6:** Download results:
- `result.tif` - Georeferenced NDVI values
- `ndvi_plot.png` - Visualization
- `statistics.json` - Numerical stats
- `insights.txt` - Interpretation

<!-- [IMAGE PLACEHOLDER: NDVI Result - Sample NDVI color-coded map] -->

### Understanding NDVI Results

| NDVI Value | Color | Interpretation |
|------------|-------|----------------|
| ≤ 0.2 | Brown/Red | No vegetation or water |
| 0.2 - 0.4 | Yellow | Sparse/Stressed vegetation ⚠️ |
| 0.4 - 0.6 | Light Green | Moderate vegetation |
| 0.6 - 0.8 | Green | Healthy vegetation |
| > 0.8 | Dark Green | Very healthy, dense vegetation |

### Stress Detection Feature

Analyz automatically detects stressed vegetation and provides:
- **Vegetation Vigor Index** (0-1 scale)
- **Stressed Area** in km²
- **Alert** when >15% of vegetation shows stress

---

## 4.2 Workflow 2: Land Cover Classification (Optical)

**Objective:** Automatically identify and map land cover types

**When to Use:**
- Creating land use maps
- Urban planning and zoning
- Environmental assessments

### Step-by-Step (Web Interface)

**Step 1:** Launch web application

**Step 2:** Configure analysis:
- Image Type: **Optical**
- Upload: Multi-band satellite image
- Analysis: **Land Cover Classification**
- Number of Classes: **4-7** (recommended)

**Step 3:** Run analysis and wait for results

### Understanding Classification Results

Analyz automatically assigns semantic labels:

| Class | Description | Typical Color |
|-------|-------------|---------------|
| 🌊 Water | Lakes, rivers, reservoirs | Blue |
| 🌳 Dense Vegetation | Forests, dense crops | Dark Green |
| 🌿 Vegetation | Moderate plant cover | Light Green |
| 🏙️ Built-up/Urban | Cities, roads, buildings | Red/Gray |
| 🪨 Bare Soil/Rock | Exposed earth, rocks | Brown/Tan |
| 🏜️ Barren Land | Desert, degraded areas | Beige |

<!-- [IMAGE PLACEHOLDER: Classification Result - Land cover map with legend] -->

### Output Files

- **result.tif** - Classified raster (each pixel has a class value)
- **classification_plot.png** - Dual-panel visualization with legend
- **statistics.json** - Area (km²) per class

---

## 4.3 Workflow 3: Flood Mapping (SAR)

**Objective:** Detect and map flooded areas using radar imagery

**When to Use:**
- Emergency flood response
- Wetland monitoring
- Reservoir tracking
- All-weather water detection (clouds don't affect SAR!)

### Why SAR for Floods?

Optical satellites can't see through clouds. During flooding events, clouds often block the view. SAR (radar) works day and night, rain or shine!

<!-- [IMAGE PLACEHOLDER: SAR vs Optical Comparison - Side-by-side showing cloud blocking optical but SAR sees through] -->

### Step-by-Step (Web Interface)

**Step 1:** Obtain Sentinel-1 GRD data for your flood date

**Step 2:** Configure analysis:
- Image Type: **SAR**
- Upload: Sentinel-1 GeoTIFF (VV or VH band)
- Boundary (optional): Flood-prone region
- Analysis: **Flood & Water Mapping**

**Step 3:** Run analysis

### Understanding Flood Map Results

- **Blue areas** = Detected water (low backscatter)
- **Other areas** = Land (higher backscatter)

Output includes:
- Water area in **km²**
- Percentage of flooded area
- Mean backscatter values for water vs. land

---

## 4.4 Workflow 4: Soil Moisture Estimation (SAR)

**Objective:** Estimate relative soil moisture levels from SAR backscatter

**When to Use:**
- Irrigation planning
- Drought monitoring
- Agricultural management

### Step-by-Step

**Step 1:** Obtain Sentinel-1 VV-polarization data

**Step 2:** Configure analysis:
- Image Type: **SAR**
- Analysis: **Soil Moisture Estimation**
- Incidence Angle: **39°** (default for Sentinel-1)

**Step 3:** Run and review results

### Understanding Moisture Levels

| Level | Moisture | Interpretation |
|-------|----------|----------------|
| 1 | Very Dry (<0.2) | Needs irrigation |
| 2 | Dry (0.2-0.4) | Low moisture |
| 3 | Moderate (0.4-0.6) | Acceptable moisture |
| 4 | Moist (0.6-0.8) | Good moisture |
| 5 | Very Moist (>0.8) | Saturated/wet |

> **Note:** This is a relative moisture index, not absolute volumetric percentage.

<!-- [IMAGE PLACEHOLDER: Soil Moisture Map - Color-coded moisture levels] -->

---

## 4.5 Workflow 5: Complete Analysis Pipeline

For comprehensive analysis, combine multiple indices in one session.

### Using CLI for Multi-Index Analysis

```powershell
# Calculate NDVI
python app.py --image sentinel2.tif --image-type optical --analysis ndvi --boundary aoi.geojson --output results/ndvi

# Calculate NDWI
python app.py --image sentinel2.tif --image-type optical --analysis ndwi --boundary aoi.geojson --output results/ndwi

# Calculate NDBI
python app.py --image sentinel2.tif --image-type optical --analysis ndbi --boundary aoi.geojson --output results/ndbi
```

### Using Python API

```python
from analyz import *

# Load once
data, profile = FileHandler.read_raster("sentinel2.tif")
boundary = BoundaryHandler("aoi.geojson")
data, profile = boundary.clip_array(data, profile)

# Multiple analyses
band_indices = {'red': 2, 'nir': 3, 'green': 1, 'blue': 0, 'swir1': 4}
optical = OpticalAnalyzer(data, band_indices)

ndvi, ndvi_stats = optical.ndvi()
ndwi, ndwi_stats = optical.ndwi()
ndbi, ndbi_stats = optical.ndbi()

# Combined visualization
Plotter.plot_multifigure(
    [ndvi, ndwi, ndbi],
    ['NDVI', 'NDWI', 'NDBI'],
    ['RdYlGn', 'Blues', 'Reds'],
    output_path="combined_indices.png"
)
```

<!-- [IMAGE PLACEHOLDER: Multi-Index Result - Combined visualization with 3 panels] -->

---

# Chapter 5: Intermediate and Advanced Features

## 5.1 Study Area Boundaries

Clip your analysis to a specific area of interest (AOI) to:
- Speed up processing
- Focus results on your study area
- Reduce file sizes

### Supported Formats
- **GeoJSON** (.geojson) - Recommended
- **Shapefile** (.shp)
- **GeoPackage** (.gpkg)
- **KML** (.kml)

### Using Boundaries in Web Interface

1. Upload your boundary file in the "Boundary File" section
2. Analyz will automatically clip the image to your boundary
3. Area calculations will be based on the clipped region

<!-- [IMAGE PLACEHOLDER: Boundary Clipping - Before/after showing full image vs clipped] -->

### Using Boundaries in CLI

```powershell
python app.py --image optical.tif --image-type optical --analysis ndvi --boundary study_area.geojson --output results/
```

### Using Boundaries in Python

```python
from analyz import BoundaryHandler

boundary = BoundaryHandler("study_area.geojson")
clipped_data, clipped_profile = boundary.clip_raster(
    "image.tif",
    output_path="clipped_image.tif"
)

# Get boundary info
area_m2 = boundary.get_boundary_area(crs="EPSG:32633")
print(f"Study area: {area_m2 / 1e6:.2f} km²")
```

---

## 5.2 SAR Speckle Filtering

SAR images contain inherent "speckle" noise. Always filter before analysis!

### Available Filters

| Filter | Best For | Characteristics |
|--------|----------|-----------------|
| **Lee Filter** | General use | Good balance of noise reduction and edge preservation |
| **Frost Filter** | Urban areas | Better edge preservation |
| **Median Filter** | Quick preview | Simple but effective |

### Recommended Settings

| Parameter | Recommendation |
|-----------|----------------|
| Window Size | **5** for 10m resolution, **7** for 20m resolution |
| Number of Looks | **1** for GRD products |

<!-- [IMAGE PLACEHOLDER: Speckle Filter Comparison - Original vs Lee-filtered SAR image] -->

### CLI Example

```powershell
python app.py --image sentinel1.tif --image-type sar --analysis lee_filter --window-size 5 --num-looks 1 --output filtered/
```

---

## 5.3 Polarimetric Analysis (Advanced SAR)

Use dual-polarization (VV + VH) SAR data for advanced land cover discrimination.

### VV/VH Ratio Interpretation

| Ratio (dB) | Surface Type |
|------------|--------------|
| < 10 dB | Volume scattering (forests, dense vegetation) |
| 10-15 dB | Mixed/Agricultural areas |
| > 15 dB | Surface scattering (urban, bare surfaces) |

### Requirements
- Sentinel-1 IW mode with VV + VH polarization
- Both bands in the same GeoTIFF file

<!-- [IMAGE PLACEHOLDER: Polarimetric Analysis - VV/VH ratio map showing forest/urban discrimination] -->

---

## 5.4 Batch Processing

Process multiple images efficiently using scripts.

### Windows Batch Script

Create a file named `batch_ndvi.bat`:

```batch
@echo off
echo Starting batch NDVI analysis...

for %%f in (input\*.tif) do (
    echo Processing: %%f
    python app.py --image "%%f" --analysis ndvi --output "output\%%~nf_ndvi"
)

echo Batch processing complete!
pause
```

### PowerShell Script

```powershell
# batch_process.ps1
$inputDir = "input"
$outputDir = "output"

Get-ChildItem "$inputDir\*.tif" | ForEach-Object {
    $name = $_.BaseName
    Write-Host "Processing: $name"
    python app.py --image $_.FullName --image-type optical --analysis ndvi --output "$outputDir\$name"
}

Write-Host "Batch processing complete!"
```

---

## 5.5 Custom Band Arithmetic

Create custom indices for specialized applications.

### Example: Custom Index via Python

```python
from analyz import OpticalAnalyzer

# Load data
analyzer = OpticalAnalyzer(data, band_indices)

# Access raw bands
red = data[band_indices['red']]
nir = data[band_indices['nir']]
green = data[band_indices['green']]

# Custom calculation
my_custom_index = (nir - red + green) / (nir + red + green + 0.5)
```

---

## 5.6 Network Access (Team Sharing)

Share Analyz with your team on the local network.

### Step 1: Find Your IP Address

```powershell
ipconfig
# Look for "IPv4 Address" (e.g., 192.168.1.100)
```

### Step 2: Launch with Network Access

Edit `webapp/app_web.py` or use:

```powershell
python app_web.py
# Default binding: 0.0.0.0:5000 (accessible on network)
```

### Step 3: Team Members Access

Other computers on the same network can access:
```
http://192.168.1.100:5000
```
(Replace with your actual IP address)

---

# Chapter 6: Troubleshooting and FAQs

## 6.1 Installation Issues

### Problem: "GDAL Import Error"
```
ImportError: No module named 'osgeo'
```

**Solution (Windows):**
```powershell
conda install -c conda-forge gdal
```

**Solution (macOS):**
```bash
brew install gdal
pip install GDAL==$(gdal-config --version)
```

---

### Problem: "Memory Error" During Analysis
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Use a boundary file** to process a smaller area
2. **Increase `max_memory_mb`** in `config.yaml`
3. **Close other applications** to free RAM
4. **Process in chunks** (automatic for large files)

---

### Problem: "Port 5000 Already in Use"
```
OSError: Address already in use
```

**Solution:**
1. Find and kill the process using port 5000:
   ```powershell
   netstat -ano | findstr :5000
   taskkill /PID <PID> /F
   ```
2. Or change the port in `webapp/app_web.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=8080)
   ```

---

## 6.2 Analysis Issues

### Problem: "Band 'nir' not defined"
```
ValueError: Band 'nir' not defined in band_indices
```

**Solution:**
Verify your band configuration matches your satellite:

```python
# Check your image bands
from analyz import FileHandler
data, profile = FileHandler.read_raster("image.tif")
print(f"Number of bands: {data.shape[0]}")
```

Then configure band indices accordingly.

---

### Problem: Poor Classification Results

**Solutions:**
1. Try different numbers of classes (5-7 usually works best)
2. Ensure all required bands are present (Red, NIR, Green, Blue)
3. Use cloud-free imagery
4. Apply preprocessing (contrast enhancement)

---

### Problem: CRS Mismatch Warning
```
WARNING: Boundary CRS does not match image CRS
```

**Explanation:** This is usually OK—Analyz automatically reprojects boundaries to match the image. However, always verify results for accuracy.

---

## 6.3 Web Application Issues

### Problem: "File Too Large"

**Solutions:**
1. Increase upload limit in `webapp/app_web.py`:
   ```python
   app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB
   ```
2. Clip your image to a smaller area before uploading

---

### Problem: "Analysis Stuck at X%"

**Solutions:**
1. Check browser console (F12) for JavaScript errors
2. Check the terminal/command prompt for Python errors
3. For very large images, processing may take 10-15 minutes
4. Restart the server if needed

---

### Problem: Results Not Displaying

**Solutions:**
1. Clear browser cache (Ctrl + Shift + Delete)
2. Refresh the page
3. Check if results were saved to `webapp/results/{session_id}/`

---

## 6.4 Frequently Asked Questions

### Q: What satellite data can I use?
**A:** Analyz works with:
- **Optical:** Sentinel-2, Landsat 8/9, PlanetScope, MODIS
- **SAR:** Sentinel-1, RADARSAT, TerraSAR-X, ALOS PALSAR
- Any multi-band GeoTIFF file

### Q: Is my data sent to the cloud?
**A:** No! All processing happens locally on your computer. Your data never leaves your machine.

### Q: How long are results stored?
**A:** Results are stored temporarily in the `webapp/results/` folder. Download what you need—temporary files are cleaned up automatically.

### Q: Can I run Analyz without internet?
**A:** Yes! Once installed, Analyz runs completely offline.

### Q: What's the largest image I can process?
**A:** Depends on your RAM:
- 8 GB RAM → ~5,000 x 5,000 pixels comfortably
- 16 GB RAM → ~10,000 x 10,000 pixels
- For larger images, use boundary clipping

### Q: How do I cite Analyz in research?
**A:** Include a reference to Analyz version and the analysis date. Contact the development team for formal citation guidelines.

---

# Chapter 7: Real-World Case Studies

## 7.1 Case Study: Agricultural Crop Monitoring

### Scenario
A farm manager needs to identify stressed crops across 500 hectares of farmland to prioritize irrigation.

### Approach

**Step 1:** Downloaded cloud-free Sentinel-2 imagery from Copernicus Hub

**Step 2:** Created a boundary file (GeoJSON) outlining the farm using QGIS

**Step 3:** Ran NDVI analysis with stress detection:
```powershell
python app.py --image sentinel2_farm.tif --image-type optical --analysis ndvi --boundary farm_boundary.geojson --output crop_health/
```

<!-- [IMAGE PLACEHOLDER: Farm NDVI Map - Showing healthy (green) and stressed (yellow) crop areas] -->

### Results
- **82%** of vegetation classified as healthy (NDVI > 0.6)
- **12%** showing moderate stress (NDVI 0.4-0.6)
- **6%** with severe stress (NDVI 0.2-0.4)

### Action Taken
The farmer prioritized irrigation in the yellow-flagged areas, saving water while addressing crop stress early.

---

## 7.2 Case Study: Flood Emergency Response

### Scenario
Following heavy rainfall, emergency responders need to quickly map flooded areas to plan rescue operations.

### Approach

**Step 1:** Obtained Sentinel-1 SAR imagery (unaffected by clouds!)

**Step 2:** Ran flood mapping analysis using the web interface:
- Image Type: **SAR**
- Analysis: **Flood & Water Mapping**

<!-- [IMAGE PLACEHOLDER: Flood Map - Showing pre-flood vs post-flood water extent] -->

### Results
- **Flooded area:** 15.7 km²
- **Affected settlements:** 3 villages identified
- **Processing time:** 4 minutes

### Action Taken
Emergency teams used the flood map to identify accessible routes and prioritize evacuation of the most affected areas.

---

## 7.3 Case Study: Urban Expansion Monitoring

### Scenario
A city planning department needs to document urban growth over 5 years.

### Approach

**Step 1:** Obtained Landsat 8 imagery from 2019 and 2024

**Step 2:** Ran NDBI (built-up index) analysis on both images

**Step 3:** Compared results to quantify changes

<!-- [IMAGE PLACEHOLDER: Urban Change Detection - 2019 vs 2024 NDBI comparison] -->

### Results
- **2019 built-up area:** 42.3 km²
- **2024 built-up area:** 58.7 km²
- **Urban expansion:** +38.8% over 5 years

### Application
The data informed the city's master plan revision and identified priority areas for infrastructure development.

---

## 7.4 Case Study: Forest Health Assessment

### Scenario
A conservation organization monitors forest health across a 100 km² protected area.

### Approach

**Step 1:** Quarterly Sentinel-2 imagery acquisition

**Step 2:** NDVI analysis with vegetation stress detection

**Step 3:** Land cover classification to identify deforested areas

<!-- [IMAGE PLACEHOLDER: Forest Health Dashboard - NDVI time series and classification] -->

### Results
- **Healthy forest:** 78%
- **Stressed vegetation:** 15% (primarily edge areas)
- **Deforestation detected:** 3.2 km² in northwest sector

### Action Taken
Rangers investigated the stressed areas, finding illegal logging activity. Enforcement actions were initiated.

---

# Chapter 8: Closing Chapter

## 8.1 Congratulations!

🎉 **You've completed the Analyz Training Course!**

You now know how to:
- ✅ Install and configure Analyz
- ✅ Use both Web and Command Line interfaces
- ✅ Perform Optical analyses (NDVI, NDWI, NDBI, EVI, SAVI, Classification)
- ✅ Perform SAR analyses (Flood Mapping, Soil Moisture, Speckle Filtering)
- ✅ Use study area boundaries
- ✅ Troubleshoot common issues
- ✅ Apply these skills to real-world scenarios

---

## 8.2 Getting Support

### Documentation
- **README.md** - Quick start and feature overview
- **USAGE_GUIDE.md** - Detailed usage instructions
- **WEB_APP_QUICKSTART.md** - Web interface guide
- **ENHANCEMENTS.md** - Latest feature updates

### Getting Help
- Check the documentation in the `docs/` folder
- Review example scripts in `examples/`
- Contact the development team for technical support

---

## 8.3 Staying Updated

### Version Information
- **Current Version:** 2.0
- **Author:** Samson Adeyomoye
- **License:** MIT (Free and Open Source)

### What's Coming Next
- 🔜 InSAR for ground deformation monitoring
- 🔜 Machine learning classification with training data
- 🔜 Time series analysis
- 🔜 Enhanced report generation (PDF export)

---

## 8.4 Continue Learning

### Recommended Resources

**Remote Sensing Fundamentals:**
- NASA ARSET Training Program: [arset.gsfc.nasa.gov](https://arset.gsfc.nasa.gov)
- ESA SAR Educational Toolkit

**Satellite Data Portals:**
- USGS EarthExplorer: [earthexplorer.usgs.gov](https://earthexplorer.usgs.gov)
- Copernicus Open Access Hub: [scihub.copernicus.eu](https://scihub.copernicus.eu)

**GIS Software (for advanced analysis):**
- QGIS (Free): [qgis.org](https://qgis.org)
- Google Earth Engine: [earthengine.google.com](https://earthengine.google.com)

### Practice Exercises

1. **Beginner:** Download a Sentinel-2 image of your local area and run NDVI analysis. Identify parks and green spaces.

2. **Intermediate:** Download pre- and post-monsoon Sentinel-1 images and create flood extent maps for comparison.

3. **Advanced:** Create a batch processing script that analyzes multiple images and generates a summary report.

---

## 8.5 Quick Reference Card

Keep this summary handy for quick reference:

### Web Application
```
Launch: Double-click start_webapp.bat
URL: http://localhost:5000
```

### CLI Cheat Sheet
```powershell
# NDVI
python app.py --image img.tif --image-type optical --analysis ndvi --output results/

# Land Cover
python app.py --image img.tif --image-type optical --analysis classification --n-clusters 5 --output results/

# Flood Mapping
python app.py --image sar.tif --image-type sar --analysis flood_mapping --output results/

# Speckle Filtering
python app.py --image sar.tif --image-type sar --analysis lee_filter --window-size 5 --output results/
```

### Band Indices
```
Sentinel-2: Blue:0, Green:1, Red:2, NIR:3, SWIR1:4
Landsat 8:  Blue:1, Green:2, Red:3, NIR:4, SWIR1:5
```

---

## 8.6 Thank You!

Thank you for choosing Analyz for your satellite imagery analysis needs. We hope this course has empowered you to make the most of this powerful tool.

**Happy Analyzing! 🛰️📊🌍**

---

*Course Version: 1.0*  
*Last Updated: December 2024*  
*Developed for: Analyz v2.0*

---

# Appendix A: Complete Analysis Reference

## Optical Analyses

| Analysis | CLI Flag | Description |
|----------|----------|-------------|
| NDVI | `ndvi` | Vegetation health index |
| NDWI | `ndwi` | Water body detection |
| NDBI | `ndbi` | Built-up area index |
| EVI | `evi` | Enhanced vegetation |
| SAVI | `savi` | Soil-adjusted vegetation |
| Classification | `classification` | Land cover mapping |

## SAR Analyses

| Analysis | CLI Flag | Description |
|----------|----------|-------------|
| Lee Filter | `lee_filter` | Speckle reduction |
| Frost Filter | `frost_filter` | Edge-preserving filter |
| Median Filter | `median_filter` | Simple speckle reduction |
| Flood Mapping | `flood_mapping` | Water detection |
| Soil Moisture | `soil_moisture` | Moisture estimation |
| Polarimetric | `polarimetric` | VV/VH ratio |
| Backscatter | `backscatter` | Intensity analysis |
| Coherence | `coherence` | Change detection |

---

# Appendix B: Output Files Reference

| File | Format | Description |
|------|--------|-------------|
| `result.tif` | GeoTIFF | Georeferenced analysis result |
| `{analysis}_plot.png` | PNG (300 DPI) | Main visualization |
| `{analysis}_histogram.png` | PNG | Data distribution |
| `statistics.json` | JSON | Numerical statistics |
| `insights.txt` | Text | Automated interpretation |

---

# Appendix C: Keyboard Shortcuts (Web Interface)

| Shortcut | Action |
|----------|--------|
| Tab | Navigate between fields |
| Enter | Submit form |
| F5 | Refresh page |
| F12 | Open developer tools |

---

*End of Course*
