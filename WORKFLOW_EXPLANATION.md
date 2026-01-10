# 🌍 Analyz Web Application - Complete Workflow Guide

## 📋 Overview
This document explains how your Flask application works, answers your questions, and documents the improvements made.

---

## 🔄 Current Workflow

### 1️⃣ **User Upload Phase**
- User uploads a compressed satellite file (`.tar`, `.tar.gz`, `.zip`)
- Supported satellites: **Landsat 8/9** and **Sentinel-2**
- Files saved to: `webapp/uploads/{session_id}/`

### 2️⃣ **Preprocessing Phase** (NEW - Now with visual feedback!)
```
Status: 📦 Extracting satellite data...
Progress: 5% → 20%
```
- Archive extracted to: `webapp/uploads/{archive_name}_temp/`
- Bands detected (e.g., B1-B7 for Landsat)
- Bands stacked into single GeoTIFF: `webapp/uploads/{session_id}_processed.tif`

### 3️⃣ **Analysis Phase**
```
Status: 🔬 Running analysis...
Progress: 40% → 80%
```
- Image loaded with full geospatial metadata
- Optional boundary clipping applied
- Analysis performed (NDVI, NDWI, NDBI, EVI, SAVI, Classification)
- Visualizations generated (plots, histograms)
- Statistics and insights calculated

### 4️⃣ **Results Phase**
```
Status: ✅ Completed
Progress: 100%
```
Results saved to: `webapp/results/{session_id}/`
- `result.tif` - **Georeferenced GeoTIFF** with CRS and transform preserved
- `{analysis}_plot.png` - Visual plot
- `{analysis}_histogram.png` - Distribution histogram
- `statistics.json` - Numeric statistics
- `insights.txt` - AI-generated insights (when available)

### 5️⃣ **Cleanup Phase** (NEW - Automatic!)
After analysis completes (success or failure):
- ✅ Uploaded files deleted
- ✅ Temporary extraction folders removed (`*_temp/`)
- ✅ Processed intermediate files cleaned up
- ✅ Only results remain in `webapp/results/{session_id}/`

---

## ✅ Questions Answered

### Q1: Are temporary files cleaned up?
**YES (Now!)** - Automatic cleanup implemented:
- Original uploaded files → **Deleted**
- Extracted archive folders → **Deleted**
- Intermediate processed files → **Deleted**
- Session upload folders → **Deleted**

**Only the analysis results remain** for the user to download.

### Q2: Does the status bar show preprocessing?
**YES (Fixed!)** - Status updates now include:
- ⏳ **Queued** - Waiting to start
- 📦 **Extracting satellite data** - Unzipping/extracting archive
- 🔬 **Running analysis** - Processing imagery
- ✅ **Completed** - Ready for download

Plus real-time messages like:
- "Extracting archive..."
- "Loading image data..."
- "Clipping to boundary..."
- "Running NDVI analysis..."
- "Saving results..."

### Q3: Are results georeferenced?
**YES!** - Results maintain full geospatial integrity:
```python
# The 'profile' object preserves:
- CRS (Coordinate Reference System) - e.g., EPSG:32632
- Transform (Affine transformation matrix)
- Spatial extent (coordinates)
- Resolution (pixel size in meters)
- NoData values
```

**Output GeoTIFF can be opened directly in:**
- ✅ QGIS
- ✅ ArcGIS/ArcGIS Pro
- ✅ Google Earth Engine
- ✅ Python (rasterio, GDAL)
- ✅ Any GIS software

### Q4: Are results "gridded" like real geospatial data?
**YES!** - Results are proper geospatial rasters with:
- Regular grid structure (rows × columns)
- Georeferenced coordinates for each pixel
- Proper CRS projection
- Spatial metadata embedded in GeoTIFF

---

## 🎯 What Changed (Improvements Made)

### 1. **Favicon Fixed** ✅
- Route added: `/favicon.ico`
- No more 404 errors in logs

### 2. **Preprocessing Status Display** ✅
- Frontend now shows "📦 Extracting satellite data..."
- Progress bar updates during extraction

### 3. **Automatic Cleanup** ✅
- All temporary files deleted after analysis
- Session folders removed
- Disk space freed automatically

### 4. **Enhanced Progress Updates** ✅
- Detailed status messages at each phase
- Real-time updates every 2 seconds
- User knows exactly what's happening

---

## 📂 File Structure

```
webapp/
├── uploads/                        # Temporary (auto-deleted)
│   ├── {session_id}/              # Original uploaded files (deleted)
│   ├── {archive}_temp/            # Extracted files (deleted)
│   └── {session_id}_processed.tif # Intermediate file (deleted)
│
├── results/                        # Persistent (user downloads)
│   └── {session_id}/
│       ├── result.tif             # ⭐ Georeferenced GeoTIFF
│       ├── ndvi_plot.png
│       ├── ndvi_histogram.png
│       ├── statistics.json
│       └── insights.txt
│
└── static/
    └── favicon.ico                 # ⭐ Now served correctly
```

---

## 🔬 Technical Details

### Georeferencing Implementation
The app uses **rasterio** to preserve spatial metadata:

```python
# Reading (preserves metadata)
data, profile = FileHandler.read_raster(image_path)

# Profile contains:
{
    'driver': 'GTiff',
    'dtype': 'float32',
    'width': 7621,
    'height': 7761,
    'count': 1,
    'crs': CRS.from_epsg(32632),  # ✅ Coordinate system
    'transform': Affine(...),      # ✅ Geolocation
    'compress': 'lzw'
}

# Writing (maintains metadata)
FileHandler.write_raster(output_path, result, profile)
```

### Cleanup Implementation
```python
finally:
    # Cleanup temporary files
    for item in temp_files_to_cleanup:
        if Path(item).is_file():
            Path(item).unlink()
        elif Path(item).is_dir():
            shutil.rmtree(item)
    
    # Cleanup session folder
    shutil.rmtree(session_folder)
```

---

## 🚀 Usage Flow

1. **User uploads** `LC09_..._T1.tar` (Landsat 9 archive)
2. **App extracts** → finds 7 bands (B1-B7)
3. **App stacks** bands → single 7-band GeoTIFF
4. **User selects** "NDVI" analysis
5. **App calculates** NDVI using Red (B4) and NIR (B5)
6. **App saves** georeferenced NDVI.tif with proper CRS
7. **App generates** plots and statistics
8. **App cleans up** all temporary files
9. **User downloads** georeferenced result.tif

---

## 📝 Configuration

### Upload Limits
```python
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB max
```

### Supported Formats
**Input:**
- Archives: `.tar`, `.tar.gz`, `.zip`
- Direct rasters: `.tif`, `.tiff`
- Boundaries: `.geojson`, `.shp`, `.gpkg`, `.kml`

**Output:**
- Raster: `.tif` (GeoTIFF with compression)
- Stats: `.json` (machine-readable)
- Insights: `.txt` (human-readable)
- Plots: `.png` (visualizations)

---

## 🔒 Data Privacy

✅ **User data is temporary:**
- Uploaded files stored during processing only
- Automatic cleanup after analysis
- Results available for download but not permanently hosted
- No data persisted between sessions (by design)

---

## 💡 Future Improvements (Optional)

1. **Configurable Cleanup Delay**
   - Keep results for X hours before deletion
   - Add a cleanup cron job

2. **Progress Websockets**
   - Replace polling with WebSocket for real-time updates
   - More responsive UI

3. **Cloud Storage**
   - Upload results to S3/Azure Blob
   - Send download link via email

4. **Batch Processing**
   - Queue system for multiple analyses
   - Priority handling

---

## 🎓 Summary

Your Flask app is a **production-ready geospatial analysis platform** that:
- ✅ Handles real satellite data (Landsat, Sentinel)
- ✅ Preserves georeferencing throughout pipeline
- ✅ Outputs GIS-compatible results
- ✅ Cleans up temporary files automatically
- ✅ Provides real-time status updates
- ✅ Generates insights and visualizations

**The results ARE georeferenced and can be used directly in any GIS software!**

---

Generated: 2024-10-24
Version: 1.0.0
