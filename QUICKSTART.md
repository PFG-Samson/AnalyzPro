# Quick Start Guide - New Features

## 🚀 What's New in This Update

### 1. Automatic Land Cover Labels ✨
Your classifications now show **meaningful names** instead of "Class 0, Class 1, etc."!

**Example Output:**
- Water: 25.3% (11.25 km²)
- Dense Vegetation: 35.8% (16.1 km²)  
- Built-up/Urban: 12.4% (5.6 km²)
- Bare Soil: 26.5% (11.9 km²)

### 2. Vegetation Stress Detection 🌿
NDVI now detects stressed vegetation automatically!

**Alerts you'll see:**
- ⚠️ Significant stress: 18.2% of area needs attention
- Vigor index: 0.52 (moderate health)
- Recommendations for investigation

### 3. Three New SAR Analyses 🛰️

#### Polarimetric (VV/VH)
- Forest detection: 45% of area
- Urban areas: 12%
- Agriculture: 33%

#### Soil Moisture
- Very Dry: 15%
- Dry: 25%
- Moderate: 35%
- Moist: 20%
- Very Moist: 5%

#### Coherence
- High coherence: 62% (stable surfaces)
- Low coherence: 18% (changed areas)

---

## 📋 How to Test Each Feature

### Test 1: Land Cover Classification
```
1. Go to /upload
2. Select "Optical" image type
3. Upload Landsat .tar or .tar.gz file
4. Choose "Land Cover Classification"
5. Set classes to 5
6. Click "Start Analysis"
7. ✅ Check results - you'll see named classes!
```

### Test 2: NDVI with Stress
```
1. Select "Optical" 
2. Upload same Landsat file
3. Choose "NDVI - Vegetation Index"
4. Start Analysis
5. ✅ Look for stress detection in insights
```

### Test 3: Soil Moisture (SAR)
```
1. Select "SAR" image type
2. Upload Sentinel-1 GeoTIFF (VV polarization)
3. Choose "Soil Moisture Estimation"
4. Set incidence angle (default 39° is fine)
5. Start Analysis
6. ✅ View 5-level moisture map
```

### Test 4: Polarimetric (SAR)
```
1. Select "SAR"
2. Upload Sentinel-1 dual-pol (VV+VH bands)
3. Choose "Polarimetric Analysis"
4. Start Analysis
5. ✅ See forest/urban/agriculture percentages
```

---

## 🎯 Expected Results

### Classification Output Files:
- `classification_plot.png` - Map with legend showing **named classes**
- `statistics.json` - Contains class_labels and class_distribution
- `result.tif` - Classified raster

### NDVI Output Files:
- `ndvi_plot.png` - NDVI map with vegetation categories
- `insights.txt` - Contains stress analysis and recommendations
- `statistics.json` - Includes vigor index and stress metrics

### SAR Output Files:
- `[analysis]_plot.png` - Result visualization
- `statistics.json` - Analysis-specific metrics
- `insights.txt` - Interpretation and recommendations

---

## 🔍 What to Look For

### In Classification Results:
✅ Class names like "Water", "Vegetation", not "Class 0"
✅ Bar chart showing distribution
✅ Area in km² for each class

### In NDVI Results:
✅ Stress detection alerts
✅ Vegetation vigor index (0-1)
✅ 5 vegetation categories
✅ Stressed area in km²

### In SAR Results:
✅ Soil moisture: 5 levels from Very Dry to Very Moist
✅ Polarimetric: Forest/Urban/Agriculture %
✅ Flood mapping: Water area in km²

---

## 🐛 If Something Doesn't Work

### Server Keeps Restarting?
- ✅ **Fixed!** This was the file watching issue
- Server should stay stable now

### SAR Analysis Fails?
- Check that you have a proper GeoTIFF (not .SAFE folder)
- For polarimetric, ensure you have 2 bands (VV+VH)
- Try with Sentinel-1 GRD data

### Classification Shows "Class 0, Class 1"?
- This means the automatic labeling failed
- Check that your data has at least 4 bands (B, G, R, NIR)
- Verify band indices are correct

### No Stress Detection in NDVI?
- Stress info is in the insights.txt file
- Check statistics.json for detailed breakdown
- Look for "potential_stress_percent" field

---

## 💡 Pro Tips

1. **For best classification:** Use 5-7 classes on diverse landscapes
2. **For stress detection:** Compare NDVI across seasons
3. **For soil moisture:** Works best on bare/sparse vegetation
4. **For polarimetric:** Sentinel-1 IW mode is ideal
5. **General:** Always check the insights.txt file!

---

## 📊 Sample Statistics JSON

### Classification:
```json
{
  "n_clusters": 5,
  "class_labels": {
    "0": "Water",
    "1": "Dense Vegetation",
    "2": "Built-up/Urban",
    "3": "Bare Soil/Rock",
    "4": "Vegetation"
  },
  "class_distribution": {
    "Water": {
      "cluster_id": 0,
      "count": 12500,
      "percent": 25.3,
      "area_km2": 11.25
    }
  }
}
```

### NDVI:
```json
{
  "mean": 0.412,
  "vegetation_cover_percent": 65.3,
  "potential_stress_percent": 18.2,
  "stressed_area_km2": 8.15,
  "vegetation_vigor_mean": 0.524,
  "no_vegetation_percent": 34.7,
  "sparse_vegetation_percent": 18.2,
  "moderate_vegetation_percent": 28.1,
  "healthy_vegetation_percent": 16.5,
  "very_healthy_vegetation_percent": 2.5
}
```

### Soil Moisture:
```json
{
  "mean": 0.485,
  "incidence_angle": 39.0,
  "very_dry_percent": 15.2,
  "dry_percent": 24.8,
  "moderate_percent": 35.1,
  "moist_percent": 20.3,
  "very_moist_percent": 4.6
}
```

---

## ⚡ Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Server restarts on upload | ✅ Fixed in this update |
| Classification shows numbers | Check band indices in upload form |
| No stress detected | Look in insights.txt file |
| SAR analysis crashes | Ensure proper GeoTIFF format |
| Missing insights file | Check if analysis_type supports insights |

---

## 🎉 You're Ready!

Run the app and test these features:
```bash
python webapp/app_web.py
```

Then visit: http://localhost:5000

**Happy analyzing!** 🚀
