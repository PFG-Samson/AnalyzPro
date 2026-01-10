# Satellite Data Preprocessing - Quick Start

## What's New? 🎉

Your app now **automatically handles raw satellite archives**! No need to manually extract and process Landsat/Sentinel-2 data anymore.

## TL;DR

### Web App
```bash
python webapp/app_web.py
```
Just upload your `.tar` or `.zip` satellite archive directly - preprocessing happens automatically!

### Command Line
```bash
# Preprocess first
python preprocess_satellite.py landsat.tar landsat.tif

# Then analyze
python app.py --image landsat.tif --analysis ndvi
```

## What Files Can I Upload Now?

### Before (only worked with):
- ✓ Pre-processed GeoTIFF (`.tif`)

### Now (also works with):
- ✓ Landsat archives (`.tar`, `.tar.gz`, `.zip`)
- ✓ Sentinel-2 archives (`.zip` with `.SAFE`)
- ✓ Pre-processed GeoTIFF (`.tif`) - still works!

## Workflow Examples

### Example 1: Landsat NDVI Analysis
```bash
# Download from USGS EarthExplorer
# File: LC08_L1TP_123045_20231015_20231016_02_T1.tar

# Option A: Web App (easiest)
python webapp/app_web.py
# Upload the .tar file, select NDVI, click Analyze

# Option B: CLI with preprocessing
python preprocess_satellite.py LC08_L1TP_123045_20231015_20231016_02_T1.tar landsat.tif
python app.py --image landsat.tif --analysis ndvi --output results/
```

### Example 2: Sentinel-2 Water Detection
```bash
# Download from Copernicus
# File: S2A_MSIL2A_20231015T103031_N0509_R108_T32TQM_20231015T145855.zip

# Preprocess with specific bands
python preprocess_satellite.py S2A_MSIL2A*.zip sentinel.tif --bands B03 B08

# Analyze for water (NDWI)
python app.py --image sentinel.tif --analysis ndwi --output results/
```

### Example 3: Custom Band Selection
```bash
# Only need Red and NIR for NDVI? Extract only those bands:
python preprocess_satellite.py landsat.tar landsat_ndvi.tif --bands B4 B5

# Much faster and smaller file!
python webapp/app_web.py
# Upload landsat_ndvi.tif and run NDVI
```

## Default Bands

When you don't specify bands, these are included:

**Landsat**: B1, B2, B3, B4, B5, B6, B7
- Covers all visible, NIR, and SWIR bands
- Works with all analyses (NDVI, NDWI, NDBI, EVI, SAVI)

**Sentinel-2**: B02, B03, B04, B08 (at 10m resolution)
- Blue, Green, Red, NIR
- Works with NDVI, NDWI
- For NDBI add B11: `--bands B02 B03 B04 B08 B11`

## Where to Download Satellite Data

### Landsat 8/9
🌐 **USGS EarthExplorer**: https://earthexplorer.usgs.gov/
1. Register (free account)
2. Search by location and date
3. Select "Landsat" → "Landsat Collection 2 Level-1"
4. Download `.tar` file

### Sentinel-2
🌐 **Copernicus Open Access Hub**: https://scihub.copernicus.eu/
1. Register (free account)
2. Search by location and date
3. Select "Sentinel-2" → "L1C" or "L2A"
4. Download `.zip` file

## Tips

✓ **Start with pre-processed TIFF** if you're new - easier to understand
✓ **Archives are large** - Landsat ~1GB, Sentinel-2 ~5-8GB
✓ **Preprocessing takes time** - be patient, especially for Sentinel-2
✓ **Select fewer bands** if you only need specific analyses (saves time/space)
✓ **Check cloud cover** before downloading - choose scenes with <10% clouds

## Need More Info?

📖 **Full Guide**: See `SATELLITE_PREPROCESSING.md` for:
- Detailed band information
- Troubleshooting
- Python API usage
- Advanced options

🚀 **Quick Start Video Guides**: Check project README for links

## Questions?

**Q: Can I use Google Earth Engine exports?**
A: Yes! Export as GeoTIFF or as band files that can be zipped and uploaded.

**Q: Do you support cloud masking?**
A: Not yet. Download cloud-free scenes or pre-masked data.

**Q: What about other satellites (MODIS, Planet, etc.)?**
A: Currently only Landsat 8/9 and Sentinel-2. More coming!

**Q: Why is my archive not being detected?**
A: Check the file naming and structure. Should match standard USGS/ESA formats.

**Q: Can I batch process multiple scenes?**
A: Use the CLI tool in a loop. Web app processes one at a time.

---

🎯 **Ready to go!** Download a Landsat or Sentinel scene and try it out!
