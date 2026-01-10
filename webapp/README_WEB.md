# Analyz Web Application

Browser-based interface for optical and SAR image analysis.

## Features

- 🌐 **Web Interface** - No coding required, use through browser
- 📤 **Easy Upload** - Drag-and-drop file uploads
- ⚙️ **Interactive Configuration** - Configure analysis parameters visually
- 📊 **Real-time Progress** - Live progress updates during processing
- 🖼️ **Visual Results** - View maps, charts, and statistics in browser
- 💾 **Download Results** - Get GeoTIFF, images, and reports
- 📱 **Responsive** - Works on desktop, tablet, and mobile

## Quick Start

### 1. Install Dependencies

```powershell
cd automatedAnalysis
pip install -r requirements.txt
```

### 2. Start the Server

```powershell
cd webapp
python app_web.py
```

### 3. Open in Browser

Navigate to: **http://localhost:5000**

## Using the Web App

### Step 1: Home Page
- View available analysis types
- Learn about features
- Click "Start Analysis"

### Step 2: Upload & Configure
1. **Select Image Type**: Choose Optical or SAR
2. **Upload Image**: Select your .tif/.tiff file (max 500MB)
3. **Upload Boundary** (Optional): Add study area boundary
4. **Choose Analysis**: Pick from available analysis types
5. **Configure Parameters**: 
   - **Optical**: Band indices, classification clusters
   - **SAR**: Window size, number of looks
6. **Start Analysis**: Click the button to begin

### Step 3: View Results
- Watch real-time progress bar
- View generated visualizations
- Review statistics and insights
- Download results:
  - Result GeoTIFF
  - Visualization images
  - Statistics JSON
  - Insights report

## Analysis Types

### Optical (Sentinel-2, Landsat, etc.)
- **NDVI** - Vegetation health index
- **NDWI** - Water detection
- **NDBI** - Urban/built-up areas
- **EVI** - Enhanced vegetation
- **SAVI** - Soil-adjusted vegetation
- **Classification** - Land cover classes

### SAR (Sentinel-1, RADARSAT, etc.)
- **Lee Filter** - Speckle reduction
- **Frost Filter** - Edge-preserving filter
- **Median Filter** - Simple smoothing
- **Backscatter** - Intensity analysis
- **Flood Mapping** - Water detection

## Configuration

### Band Indices (Optical)
Adjust to match your imagery:
- **Sentinel-2**: Red=2, NIR=3, Green=1, Blue=0
- **Landsat 8/9**: Red=3, NIR=4, Green=2, Blue=1

### SAR Parameters
- **Window Size**: Filter kernel size (3, 5, 7, 9, 11)
- **Number of Looks**: Multi-look processing parameter

## File Requirements

### Image Files
- **Format**: GeoTIFF (.tif, .tiff)
- **Size**: Up to 500MB
- **Bands**: 
  - Optical: Multi-band (RGB + NIR + SWIR recommended)
  - SAR: Single or multi-band

### Boundary Files (Optional)
- **Formats**: GeoJSON (.geojson), Shapefile (.shp), GeoPackage (.gpkg), KML (.kml)
- **Purpose**: Clip analysis to specific area of interest

## Output Files

For each analysis session, you get:
- `result.tif` - Processed GeoTIFF
- `{analysis}_plot.png` - Main visualization (300 DPI)
- `{analysis}_histogram.png` - Data distribution
- `statistics.json` - Numerical statistics
- `insights.txt` - Automated interpretation (when available)

## Server Configuration

Edit `app_web.py` to customize:

```python
# File size limit
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Server port
app.run(debug=True, host='0.0.0.0', port=5000)

# Secret key (change for production!)
app.config['SECRET_KEY'] = 'your-secret-key-here'
```

## Deployment

### Local Network Access

To allow access from other devices on your network:

```python
# In app_web.py, change:
app.run(debug=False, host='0.0.0.0', port=5000)
```

Then access from other devices using your computer's IP:
`http://192.168.x.x:5000`

### Production Deployment

For production use, deploy with WSGI server:

```powershell
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app_web:app
```

Or use Waitress on Windows:

```powershell
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app_web:app
```

## Troubleshooting

### Port Already in Use
```powershell
# Change port in app_web.py:
app.run(debug=True, host='0.0.0.0', port=8080)
```

### Large File Upload Fails
- Increase `MAX_CONTENT_LENGTH` in `app_web.py`
- Check available disk space in `uploads/` and `results/`

### Analysis Takes Too Long
- Use boundary file to clip to smaller area
- Reduce image resolution before upload
- Check server resources (CPU, RAM)

### Can't Access from Browser
- Check firewall settings
- Ensure Flask server is running
- Verify correct URL and port

## File Structure

```
webapp/
├── app_web.py              # Main Flask application
├── templates/              # HTML templates
│   ├── base.html          # Base template
│   ├── index.html         # Home page
│   ├── upload.html        # Upload form
│   └── results.html       # Results display
├── static/                 # Static assets
│   ├── css/
│   │   └── style.css      # Styles
│   └── js/
│       └── main.js        # JavaScript utilities
├── uploads/                # Temporary uploaded files
├── results/                # Analysis results
└── README_WEB.md          # This file
```

## API Endpoints

### Web Pages
- `GET /` - Home page
- `GET /upload` - Upload form
- `POST /upload` - Submit analysis
- `GET /results/<session_id>` - View results

### API Routes
- `GET /api/status/<session_id>` - Get analysis status (JSON)
- `GET /api/view/<session_id>/<filename>` - View file
- `GET /api/download/<session_id>/<filename>` - Download file

## Tips

1. **File Names**: Use descriptive names for uploaded files
2. **Boundaries**: Pre-clip images to reduce processing time
3. **Multiple Analyses**: Each analysis gets unique session ID
4. **Results**: Results are stored temporarily - download what you need
5. **Refresh**: Results page auto-updates during processing

## Security Notes

⚠️ **For Development/Local Use Only**

Before deploying publicly:
- Change SECRET_KEY
- Add authentication
- Implement rate limiting
- Validate file contents
- Set up HTTPS
- Configure CORS properly

## Support

For issues or questions:
- Check `app_web.py` logs
- Review browser console for errors
- Ensure all dependencies are installed
- Verify file formats and sizes

## Version

Web App Version: 1.0.0  
Compatible with: Analyz Core 1.0.0

---

**Ready to analyze?** Start the server and open http://localhost:5000 in your browser!
