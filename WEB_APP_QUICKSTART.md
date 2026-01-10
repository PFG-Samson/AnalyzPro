# 🌐 Analyz Web Application - Quick Start Guide

Transform your image analysis workflow with a **browser-based interface** - no coding required!

## What You Get

✅ **Full-Featured Web Interface**  
✅ **Real-Time Progress Updates**  
✅ **Interactive Results Display**  
✅ **Easy File Upload & Download**  
✅ **Mobile-Friendly Design**

---

## 🚀 Launch in 3 Steps

### Option 1: Double-Click Launch (Easiest!)

Simply **double-click** `start_webapp.bat` and your browser will open automatically!

### Option 2: PowerShell

```powershell
.\launch_webapp.ps1
```

### Option 3: Manual

```powershell
cd webapp
python app_web.py
```

Then open: **http://localhost:5000**

---

## 📋 First-Time Setup

### 1. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

This installs:
- Flask (web framework)
- Rasterio (geospatial processing)
- All analysis libraries
- Visualization tools

### 2. Verify Installation

```powershell
python -c "import flask; print('✓ Ready to go!')"
```

---

## 🎯 Using the Web App

### **Home Page** (/)
- Overview of features
- Links to documentation
- "Start Analysis" button

### **Upload Page** (/upload)

**Step 1: Select Image Type**
- Optical (Sentinel-2, Landsat)
- SAR (Sentinel-1, RADARSAT)

**Step 2: Upload Files**
- **Image**: Your .tif/.tiff file (up to 500MB)
- **Boundary** (Optional): GeoJSON, Shapefile, etc.

**Step 3: Configure Analysis**

**For Optical:**
- Choose: NDVI, NDWI, NDBI, EVI, SAVI, or Classification
- Set band indices (auto-configured for Sentinel-2/Landsat)
- For classification: Set number of classes

**For SAR:**
- Choose: Lee/Frost/Median filter, Backscatter, or Flood mapping
- Set window size (3, 5, 7, 9, 11)
- Set number of looks

**Step 4: Start Analysis** 🚀

### **Results Page** (/results/{id})

**Real-Time Updates:**
- Progress bar (0-100%)
- Status indicators (Queued → Processing → Completed)

**When Complete:**
- 📊 **Visualizations** - Main analysis plot
- 📈 **Histogram** - Data distribution
- 📋 **Statistics** - Mean, median, min, max, percentiles
- 💡 **Insights** - Automated interpretation
- 📥 **Downloads** - All result files

---

## 📊 Analysis Examples

### NDVI Analysis (Optical)
1. Upload multi-band satellite image
2. Select "Optical" type
3. Choose "NDVI" analysis
4. Verify band indices:
   - Red: 2, NIR: 3 (for Sentinel-2)
5. Start analysis
6. Get vegetation health map + statistics

### Flood Mapping (SAR)
1. Upload SAR image
2. Select "SAR" type
3. Choose "Flood Mapping"
4. Use default settings or adjust threshold
5. Start analysis
6. Get water detection map + coverage stats

---

## 🔧 Configuration

### Band Indices Quick Reference

**Sentinel-2 (L2A):**
```
Blue: 0, Green: 1, Red: 2
NIR: 3, SWIR1: 4, SWIR2: 5
```

**Landsat 8/9:**
```
Blue: 1, Green: 2, Red: 3
NIR: 4, SWIR1: 5, SWIR2: 6
```

### Change Server Port

Edit `webapp/app_web.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Changed to 8080
```

### Increase Upload Limit

Edit `webapp/app_web.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1GB
```

---

## 🌐 Access from Other Devices

### On Same Network

1. Find your computer's IP address:
```powershell
ipconfig
# Look for IPv4 Address (e.g., 192.168.1.100)
```

2. On other device, navigate to:
```
http://192.168.1.100:5000
```

### Share with Team

Deploy on cloud server or use ngrok:
```powershell
pip install pyngrok
python
>>> from pyngrok import ngrok
>>> ngrok.connect(5000)
```

---

## 📁 File Requirements

### Images
- **Format**: GeoTIFF (.tif, .tiff)
- **Size**: Up to 500MB (configurable)
- **Projection**: Any (boundary will be reprojected)
- **Bands**:
  - Optical: Minimum 4 bands (RGB + NIR)
  - SAR: 1 or more bands

### Boundaries (Optional)
- **Formats**: GeoJSON, Shapefile, GeoPackage, KML
- **Purpose**: Clip analysis to area of interest
- **Benefit**: Faster processing, focused results

---

## 💾 Output Files

Each analysis generates:

| File | Description |
|------|-------------|
| `result.tif` | Processed GeoTIFF |
| `{analysis}_plot.png` | Main visualization (300 DPI) |
| `{analysis}_histogram.png` | Data distribution |
| `statistics.json` | All numerical stats |
| `insights.txt` | Automated interpretation |

**Location:** `webapp/results/{session-id}/`

---

## ⚡ Performance Tips

1. **Use Boundaries** - Clip to area of interest first
2. **Reduce Resolution** - Downsample large images before upload
3. **Close Other Apps** - Free up CPU and RAM
4. **Monitor Progress** - Progress bar shows real-time status
5. **Download Results** - Files are temporary

---

## 🐛 Troubleshooting

### "Port 5000 already in use"
```powershell
# Find process using port
netstat -ano | findstr :5000
# Kill process or change port in app_web.py
```

### "File too large"
- Increase `MAX_CONTENT_LENGTH` in `app_web.py`
- Or compress/crop image before upload

### "Analysis stuck at X%"
- Check browser console (F12) for errors
- Check terminal/command prompt for server errors
- Restart server if needed

### "Can't open in browser"
- Verify server started successfully
- Check firewall isn't blocking port 5000
- Try `http://127.0.0.1:5000` instead of localhost

### "Results not displaying"
- Clear browser cache
- Check browser console for JavaScript errors
- Ensure all files downloaded correctly

---

## 🔒 Security Notes

⚠️ **Important**: This web app is designed for **local/development use**

Before deploying publicly:
- [ ] Change `SECRET_KEY` in `app_web.py`
- [ ] Add user authentication
- [ ] Implement file upload validation
- [ ] Set up HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Add rate limiting
- [ ] Implement session cleanup

---

## 🎓 Learn More

- **Full Documentation**: See `README.md`
- **Python API**: Check `USAGE_GUIDE.md`
- **Web App Details**: Read `webapp/README_WEB.md`
- **Examples**: Explore `examples/` directory

---

## 🆘 Need Help?

**Common Questions:**

**Q: Can I run multiple analyses simultaneously?**  
A: Yes! Each analysis gets a unique session ID.

**Q: How long are results stored?**  
A: Temporarily. Download results you need.

**Q: Can I use this offline?**  
A: Yes! Everything runs locally on your machine.

**Q: What about my data privacy?**  
A: All processing is local. Nothing is uploaded to cloud.

**Q: Can I integrate this into my workflow?**  
A: Yes! Use the API endpoints or Python CLI.

---

## 🎉 You're Ready!

1. **Run**: `start_webapp.bat` or `python webapp/app_web.py`
2. **Open**: http://localhost:5000
3. **Upload**: Your satellite imagery
4. **Analyze**: Choose analysis type
5. **Download**: Get professional results!

---

**Version:** 1.0.0  
**Created:** 2025  
**Author:** Samson Adeyomoye

**Enjoy analyzing! 🛰️📊🌍**
