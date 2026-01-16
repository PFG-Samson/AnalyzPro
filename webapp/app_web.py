"""Flask web application for Analyz."""

import os
import uuid
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import threading

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyz import (
    OpticalAnalyzer, SARAnalyzer, BoundaryHandler,
    Plotter, InsightsGenerator, FileHandler, setup_logger
)
from analyz.utils.satellite_preprocessor import SatellitePreprocessor
from analyz.utils.cog import convert_to_cog
from analyz.utils.stac_downloader import STACDownloadManager, download_imagery_async
from webapp.utils.temporary_storage import TemporaryStorageManager
from webapp.utils.thumbnail_generator import ThumbnailGenerator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'results'
app.config['TEMP_DOWNLOADS_FOLDER'] = Path(__file__).parent / 'temp_downloads'
app.config['PERMANENT_IMAGERY_FOLDER'] = Path(__file__).parent / 'permanent_imagery'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max file size

# Ensure folders exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(exist_ok=True)
app.config['TEMP_DOWNLOADS_FOLDER'].mkdir(exist_ok=True)
app.config['PERMANENT_IMAGERY_FOLDER'].mkdir(exist_ok=True)

# Initialize STAC Download Manager
download_manager = STACDownloadManager(
    temp_folder=app.config['TEMP_DOWNLOADS_FOLDER'],
    permanent_folder=app.config['PERMANENT_IMAGERY_FOLDER']
)

# Initialize Temporary Storage Manager for metadata tracking
storage_manager = TemporaryStorageManager(
    temp_folder=app.config['TEMP_DOWNLOADS_FOLDER']
)

# Initialize Thumbnail Generator
thumbnail_generator = ThumbnailGenerator(
    cache_dir=Path(app.static_folder) / 'thumbnails' if app.static_folder else Path(__file__).parent / 'static' / 'thumbnails'
)
if not (Path(__file__).parent / 'static' / 'thumbnails').exists():
    (Path(__file__).parent / 'static' / 'thumbnails').mkdir(parents=True, exist_ok=True)

# Download sessions tracking
download_sessions = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'geotiff', 'geojson', 'json', 'shp', 'gpkg', 'kml', 'zip', 'tar', 'gz'}
SATELLITE_ARCHIVES = {'zip', 'tar', 'gz'}

# Store analysis status
STATUS_FILE = Path(__file__).parent / 'analysis_status.json'

def load_analysis_status():
    """Load analysis status from disk."""
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load status file: {e}")
    return {}

def save_analysis_status():
    """Save analysis status to disk."""
    try:
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(analysis_status, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Could not save status file: {e}")

analysis_status = load_analysis_status()

logger = setup_logger("INFO")


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_analysis_async(session_id, image_path, boundary_path, image_type, 
                       analysis_type, params):
    """Run analysis in background thread."""
    session_folder = app.config['UPLOAD_FOLDER'] / session_id
    temp_files_to_cleanup = []
    
    try:
        analysis_status[session_id]['status'] = 'processing'
        analysis_status[session_id]['progress'] = 10
        save_analysis_status()
        
        # Check if input is a satellite archive that needs preprocessing
        image_path = Path(image_path)
        if image_path.suffix.lower().lstrip('.') in SATELLITE_ARCHIVES or '.tar.' in image_path.name.lower():
            logger.info(f"Detected satellite archive: {image_path.name}")
            analysis_status[session_id]['status'] = 'preprocessing'
            analysis_status[session_id]['progress'] = 5
            analysis_status[session_id]['message'] = 'Extracting archive...'
            save_analysis_status()
            
            # Process the archive with a short temp directory path to avoid Windows path limits
            # Use a temp directory at the root of uploads folder with just the session ID
            temp_extract_dir = app.config['UPLOAD_FOLDER'] / f"tmp_{session_id[:8]}"
            processed_path = app.config['UPLOAD_FOLDER'] / f"{session_id[:8]}_proc.tif"
            
            try:
                SatellitePreprocessor.process_auto(
                    image_path, 
                    processed_path,
                    extract_dir=temp_extract_dir
                )
                image_path = processed_path
                temp_files_to_cleanup.append(processed_path)
                if temp_extract_dir.exists():
                    temp_files_to_cleanup.append(temp_extract_dir)
                logger.info(f"Preprocessing complete: {processed_path}")
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                raise ValueError(f"Could not process satellite archive: {str(e)}")
        
        # If request is COG conversion only, handle here and return early
        if analysis_type == 'cog':
            try:
                analysis_status[session_id]['status'] = 'processing'
                analysis_status[session_id]['progress'] = 25
                analysis_status[session_id]['message'] = 'Converting to COG...'
                save_analysis_status()

                result_path = app.config['RESULTS_FOLDER'] / session_id
                result_path.mkdir(exist_ok=True)
                cog_out = result_path / 'result_cog.tif'

                # COG options
                reproject_epsg = None
                try:
                    if str(params.get('cog_epsg', '')).strip():
                        reproject_epsg = int(params.get('cog_epsg'))
                    elif bool(params.get('cog_web_mercator', False)):
                        reproject_epsg = 3857
                except Exception:
                    reproject_epsg = 3857 if bool(params.get('cog_web_mercator', False)) else None

                convert_to_cog(
                    image_path,
                    cog_out,
                    reproject_epsg=reproject_epsg,
                )

                # Build minimal stats for display
                stats = {
                    'operation': 'COG Conversion',
                    'source_file': str(Path(image_path).name),
                    'reprojected_epsg': reproject_epsg or 'preserved',
                }
                try:
                    import rasterio
                    with rasterio.open(cog_out) as src:
                        stats.update({
                            'output_crs': str(src.crs) if src.crs else None,
                            'width': src.width,
                            'height': src.height,
                            'count': src.count,
                            'dtype': src.dtypes[0] if src.count > 0 else None,
                            'driver': src.driver,
                        })
                except Exception:
                    pass

                # Save stats
                with open(result_path / 'statistics.json', 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2)

                # Update status/result files
                analysis_status[session_id]['progress'] = 100
                analysis_status[session_id]['status'] = 'completed'
                analysis_status[session_id]['message'] = 'COG created successfully.'
                analysis_status[session_id]['result_files'] = {
                    'cog': 'result_cog.tif',
                    'statistics': 'statistics.json'
                }
                analysis_status[session_id]['session_folder'] = str(session_folder)
                save_analysis_status()
                return
            except Exception as e:
                logger.error(f"COG conversion failed: {e}")
                raise

        analysis_status[session_id]['status'] = 'processing'
        analysis_status[session_id]['progress'] = 20
        analysis_status[session_id]['message'] = 'Loading image data...'
        save_analysis_status()
        
        # Load image
        logger.info(f"Loading image: {image_path}")
        data, profile = FileHandler.read_raster(image_path)
        analysis_status[session_id]['progress'] = 30
        
        # Apply boundary if provided
        if boundary_path and Path(boundary_path).exists():
            logger.info(f"Applying boundary: {boundary_path}")
            analysis_status[session_id]['message'] = 'Clipping to boundary...'
            boundary_handler = BoundaryHandler(boundary_path)
            data, profile = boundary_handler.clip_array(data, profile)
        analysis_status[session_id]['progress'] = 40
        analysis_status[session_id]['message'] = f'Running {analysis_type.upper()} analysis...'
        
        # Run analysis based on type
        result_path = app.config['RESULTS_FOLDER'] / session_id
        result_path.mkdir(exist_ok=True)
        
        if image_type == 'optical':
            result, stats, insights = run_optical_analysis(
                data, profile, analysis_type, params, result_path
            )
        else:  # SAR
            result, stats, insights = run_sar_analysis(
                data, profile, analysis_type, params, result_path, image_path=str(image_path)
            )
        
        analysis_status[session_id]['progress'] = 80
        analysis_status[session_id]['message'] = 'Saving results...'
        
        # Save results
        FileHandler.write_raster(result_path / "result.tif", result, profile)
        
        # Save statistics
        with open(result_path / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save insights
        if insights:
            InsightsGenerator.format_insights_report(
                insights, 
                result_path / "insights.txt"
            )
        
        analysis_status[session_id]['progress'] = 100
        analysis_status[session_id]['status'] = 'completed'
        analysis_status[session_id]['message'] = 'Analysis complete!'
        
        # Build result files list
        result_files = {
            'result': 'result.tif',
            'plot': f'{analysis_type}_plot.png',
            'histogram': f'{analysis_type}_histogram.png',
            'statistics': 'statistics.json',
            'insights': 'insights.txt'
        }
        
        # Include vector detections if provided
        try:
            if isinstance(stats, dict):
                if stats.get('vector_geojson'):
                    result_files['vectors'] = stats['vector_geojson']
                if analysis_type == 'classification' and image_type == 'optical' and stats.get('geojson_name'):
                    result_files['polygons'] = stats['geojson_name']
        except Exception:
            pass
            
        analysis_status[session_id]['result_files'] = result_files
        analysis_status[session_id]['session_folder'] = str(session_folder)
        save_analysis_status()
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        analysis_status[session_id]['status'] = 'failed'
        analysis_status[session_id]['error'] = str(e)
        analysis_status[session_id]['session_folder'] = str(session_folder)
        save_analysis_status()

    finally:
        # Robust cleanup of all temporary artifacts and session folders
        logger.info(f"Starting final cleanup for session {session_id}")
        import shutil as _shutil
        
        # 1. Clean up specific temp files tracked
        for item in list(temp_files_to_cleanup):
            try:
                p = Path(item)
                if p.is_file():
                    p.unlink(missing_ok=True)
                elif p.is_dir():
                    _shutil.rmtree(p, ignore_errors=True)
                logger.info(f"Cleaned up temp artifact: {p}")
            except Exception as _e:
                logger.warning(f"Could not clean up {item}: {_e}")
        
        # 2. Clean up the entire session upload folder
        if session_folder.exists():
            try:
                _shutil.rmtree(session_folder, ignore_errors=True)
                logger.info(f"Cleaned up session folder: {session_folder}")
            except Exception as _e:
                logger.warning(f"Could not clean up session folder {session_folder}: {_e}")


def run_optical_analysis(data, profile, analysis_type, params, output_dir):
    """Run optical image analysis."""
    # Parse band indices from params
    band_indices = {
        'red': params.get('red', 2),
        'nir': params.get('nir', 3),
        'green': params.get('green', 1),
        'blue': params.get('blue', 0),
        'swir1': params.get('swir1', 4),
        'swir2': params.get('swir2', 5)
    }
    
    analyzer = OpticalAnalyzer(data, band_indices, sensor=params.get('sensor'))
    
    # Initialize geojson_name for classification
    geojson_name = None
    
    # Run analysis
    if analysis_type == 'ndvi':
        result, stats = analyzer.ndvi()
        Plotter.plot_ndvi_classification(result, output_path=output_dir / 'ndvi_plot.png')
        insights = InsightsGenerator.generate_ndvi_insights(result, stats)
    elif analysis_type == 'ndwi':
        result, stats = analyzer.ndwi()
        Plotter.plot_raster(result, "NDWI", cmap='Blues', 
                           output_path=output_dir / 'ndwi_plot.png')
        insights = InsightsGenerator.generate_ndwi_insights(result, stats)
    elif analysis_type == 'ndbi':
        result, stats = analyzer.ndbi()
        Plotter.plot_raster(result, "NDBI", cmap='Reds',
                           output_path=output_dir / 'ndbi_plot.png')
        insights = InsightsGenerator.generate_optical_insights('NDBI', result, stats)
    elif analysis_type == 'evi':
        result, stats = analyzer.evi()
        Plotter.plot_raster(result, "EVI", cmap='RdYlGn',
                           output_path=output_dir / 'evi_plot.png')
        insights = InsightsGenerator.generate_optical_insights('EVI', result, stats)
    elif analysis_type == 'savi':
        result, stats = analyzer.savi()
        Plotter.plot_raster(result, "SAVI", cmap='RdYlGn',
                           output_path=output_dir / 'savi_plot.png')
        insights = InsightsGenerator.generate_optical_insights('SAVI', result, stats)
    elif analysis_type in analyzer.available_indices().keys():
        # Generic spectral index path
        result, stats = analyzer.compute_index(analysis_type)
        # Heuristic colormap selection
        name = analysis_type.lower()
        cmap = 'RdYlGn'
        if 'ndwi' in name or 'mndwi' in name:
            cmap = 'Blues'
        elif 'nbr' in name or 'bai' in name:
            cmap = 'inferno'
        elif 'ndbi' in name or 'bsi' in name:
            cmap = 'Greys'
        Plotter.plot_raster(result, f"{analysis_type.upper()}", cmap=cmap,
                           output_path=output_dir / f'{analysis_type}_plot.png')
        insights = InsightsGenerator.generate_optical_insights(analysis_type.upper(), result, stats)
    elif analysis_type == 'classification':
        n_clusters = params.get('n_clusters', 5)
        result, stats = analyzer.classify_kmeans(n_clusters=n_clusters)
        
        # Save GeoJSON for land cover classification
        geojson_name = 'classification_polygons.geojson'
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
            import rasterio
            from rasterio.features import shapes
            from shapely.geometry import shape
            
            print(f"DEBUG: Starting GeoJSON generation for result shape: {result.shape}")
            height, width = result.shape
            unique_result_vals = np.unique(result)
            print(f"DEBUG: Unique values in classification result: {unique_result_vals}")
            transform = profile.get('transform')
            crs = profile.get('crs')
            print(f"DEBUG: Transform: {transform}, CRS: {crs}")
            
            if transform is None:
                print("DEBUG: No transform found in profile, skipping GeoJSON")
                geojson_name = None
            else:
                # Calculate minimum polygon area based on pixel size (at least 1 pixel area)
                pixel_area = abs(transform[0] * transform[4])  # width * height
                min_area = max(pixel_area, 1.0)  # At least 1 square unit
                print(f"DEBUG: Pixel area: {pixel_area}, min polygon area: {min_area}")
                # Create features for each class
                features = []
                cluster_labels = stats.get('class_labels', {})
                print(f"DEBUG: Cluster labels: {cluster_labels}")
                
                for class_id, class_name in cluster_labels.items():
                    print(f"DEBUG: Processing class {class_id}: {class_name}")
                    # Create binary mask for this class
                    class_mask = (result == class_id).astype(np.uint8)
                    unique_vals = np.unique(class_mask)
                    print(f"DEBUG: Class mask unique values: {unique_vals}, shape: {class_mask.shape}")
                    
                    # Generate polygons from mask
                    try:
                        mask_shapes = list(shapes(class_mask, mask=class_mask, transform=transform))
                        print(f"DEBUG: Generated {len(mask_shapes)} shapes for class {class_name}")
                        
                        for geom, value in mask_shapes:
                            if value == 1:  # Only include pixels belonging to this class
                                # Convert to shapely geometry
                                poly = shape(geom)
                                print(f"DEBUG: Polygon area: {poly.area} (pixel threshold: {min_area})")
                                
                                # Get class statistics
                                class_info = stats['class_distribution'].get(class_name, {})
                                
                                # Create feature (don't filter yet)
                                feature = {
                                    'geometry': poly,
                                    'properties': {
                                        'class_id': int(class_id),
                                        'class_name': class_name,
                                        'area_pixels': int(class_info.get('count', 0)),
                                        'area_percent': float(class_info.get('percent', 0)),
                                        'area_km2': float(class_info.get('area_km2', 0)),
                                        'polygon_area': poly.area  # Store actual polygon area
                                    }
                                }
                                features.append(feature)
                                print(f"DEBUG: Added feature for {class_name}")
                    except Exception as e:
                        print(f"DEBUG: Error processing class {class_name}: {e}")
                        continue
                
                print(f"DEBUG: Total features generated: {len(features)}")
                
                if features:
                    # Filter out extremely small polygons (less than 1/10 of pixel area)
                    min_keep_area = min_area / 10.0
                    filtered_features = [f for f in features if f['properties']['polygon_area'] >= min_keep_area]
                    
                    # If filtering removed too many, keep the largest ones
                    if len(filtered_features) == 0 and len(features) > 0:
                        print(f"DEBUG: No polygons above minimum area, keeping largest polygons")
                        # Sort by area and keep top 100
                        features.sort(key=lambda f: f['properties']['polygon_area'], reverse=True)
                        filtered_features = features[:100]
                    
                    print(f"DEBUG: Filtered from {len(features)} to {len(filtered_features)} features")
                    
                    # Sort features by area (largest first) and take top N to avoid too many small polygons
                    filtered_features.sort(key=lambda f: f['properties']['polygon_area'], reverse=True)
                    # Keep at most 1000 features to avoid huge files
                    final_features = filtered_features[:1000]
                    
                    # Remove the polygon_area property before saving
                    for f in final_features:
                        del f['properties']['polygon_area']
                    
                    print(f"DEBUG: Keeping top {len(final_features)} features by area")
                    
                    # Create GeoDataFrame
                    gdf = gpd.GeoDataFrame(
                        [f['properties'] for f in final_features],
                        geometry=[f['geometry'] for f in final_features],
                        crs=crs
                    )
                    
                    # Save GeoJSON
                    geojson_path = output_dir / geojson_name
                    gdf.to_file(geojson_path, driver='GeoJSON')
                    logger.info(f"Saved GeoJSON with {len(final_features)} polygons to {geojson_path}")
                    
                else:
                    logger.warning("No valid polygons generated for GeoJSON")
                    geojson_name = None
        
        except Exception as e:
            logger.warning(f"Could not save classification GeoJSON: {e}")
            print(f"DEBUG: Exception in GeoJSON generation: {e}")
            import traceback
            traceback.print_exc()
            geojson_name = None
        
        # Use new land cover classification plot with semantic labels
        Plotter.plot_land_cover_classification(
            result, 
            stats.get('class_labels', {}),
            stats.get('class_distribution', {}),
            output_path=output_dir / 'classification_plot.png'
        )
        insights = InsightsGenerator.generate_optical_insights('Classification', result, stats)
    elif analysis_type == 'rgb':
        stretch_method = params.get('stretch_method', 'percentile')
        result, stats = analyzer.rgb_composite(stretch_method=stretch_method)
        # Plot RGB composite with histograms
        Plotter.plot_rgb_composite(
            result,
            title="True Color Composite (RGB)",
            output_path=output_dir / 'rgb_plot.png',
            show_histograms=True
        )
        # Also create a simple version without histograms
        Plotter.plot_rgb_composite(
            result,
            title="RGB Composite",
            output_path=output_dir / 'rgb_histogram.png',
            show_histograms=False
        )
        insights = InsightsGenerator.generate_optical_insights('RGB Composite', result, stats)
    elif analysis_type == 'band_stack':
        # Create multi-band stack with all available bands
        result, stats = analyzer.create_band_stack()
        
        # Create overview visualization showing all bands
        Plotter.plot_band_stack_overview(
            result,
            stats['band_names'],
            output_path=output_dir / 'band_stack_plot.png'
        )
        
        # Create a simple summary plot (just use first band for histogram placeholder)
        Plotter.plot_histogram(
            result[0], 
            f"Band 1 ({stats['band_names'][0].upper()}) Distribution",
            output_path=output_dir / 'band_stack_histogram.png'
        )
        
        insights = InsightsGenerator.generate_optical_insights('Band Stack', result, stats)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    # Generate histogram (skip for RGB and band_stack as they have their own visualizations)
    if analysis_type not in ['rgb', 'band_stack']:
        Plotter.plot_histogram(result, f"{analysis_type.upper()} Distribution",
                              output_path=output_dir / f'{analysis_type}_histogram.png')
    
    # Add geojson_name to stats for classification
    if analysis_type == 'classification':
        stats['geojson_name'] = geojson_name
    
    return result, stats, insights


def run_sar_analysis(data, profile, analysis_type, params, output_dir, image_path:str=None):
    """Run SAR image analysis with sensor-aware defaults."""
    def infer_sensor(path, profile, data):
        try:
            name = (Path(path).name if path else "").upper()
        except Exception:
            name = ""
        if "CAPELLA" in name:
            return "capella"
        if "S1" in name or "SENTINEL-1" in name or "SENTINEL1" in name:
            return "sentinel1"
        if "RADARSAT" in name or "RS2" in name:
            return "radarsat2"
        # Heuristics
        if np.issubdtype(data.dtype, np.uint8):
            return "capella"
        return "sentinel1"
    
    def default_params(sensor:str, analysis_type:str):
        d = {
            'capella': {
                'common': {'calibrated': False, 'nodata_values': [0], 'band': 'HH'},
                'oil_spill': {'window_size': 101, 'k_threshold': 2.0, 'min_area_km2': 0.01},
                'ship_detection': {'cfar_window': 100, 'min_ship_area_km2': 5e-5, 'band': 'HH'},
            },
            'sentinel1': {
                'common': {'calibrated': True, 'band': 'VV'},
                'oil_spill': {'window_size': 51, 'k_threshold': 1.5},
                'ship_detection': {'cfar_window': 50},
            },
            'radarsat2': {
                'common': {'calibrated': True, 'band': 'HH'},
                'oil_spill': {'window_size': 51, 'k_threshold': 1.8, 'min_area_km2': 0.02},
                'ship_detection': {'cfar_window': 60, 'band': 'HH'},
            }
        }
        s = d.get(sensor, d['sentinel1'])
        out = {}
        out.update(s.get('common', {}))
        out.update(s.get(analysis_type, {}))
        return out
    
    import numpy as np
    sensor = params.get('sensor') or infer_sensor(image_path, profile, data)
    dflt = default_params(sensor, analysis_type)
    # Build analyzer with sensor-aware calibration and nodata handling
    analyzer = SARAnalyzer(
        data,
        profile=profile,
        sensor=sensor,
        calibrated=dflt.get('calibrated'),
        nodata_values=dflt.get('nodata_values')
    )
    band = params.get('band') or dflt.get('band')
    
    # 1. Oil Spill Detection
    if analysis_type == 'oil_spill':
        result, stats = analyzer.oil_spill_detection(
            window_size=int(params.get('window_size', dflt.get('window_size', 51))),
            k_threshold=float(params.get('k_threshold', dflt.get('k_threshold', 1.5))),
            min_area_pixels=int(params.get('min_area_pixels', 100)),
            band=band,
            min_area_km2=float(params.get('min_area_km2', dflt.get('min_area_km2', 0) or 0)) or None
        )
        cmap = 'Reds'
        insight_type = 'OIL SPILL DETECTION'
    
    # 2. Ship Detection
    elif analysis_type == 'ship_detection':
        result, stats = analyzer.ship_detection(
            cfar_window=int(params.get('cfar_window', dflt.get('cfar_window', 50))),
            false_alarm_rate=float(params.get('false_alarm_rate', 1e-5)),
            min_ship_pixels=int(params.get('min_ship_pixels', 5)),
            band=band,
            min_ship_area_km2=float(params.get('min_ship_area_km2', dflt.get('min_ship_area_km2', 0) or 0)) or None
        )
        cmap = 'hot'
        insight_type = 'SHIP DETECTION'
    
    # 3. Crop Monitoring
    elif analysis_type == 'crop_monitoring':
        result, stats = analyzer.crop_monitoring()
        cmap = 'YlGn'
        insight_type = 'CROP MONITORING'
    
    # 4. Land Cover Classification
    elif analysis_type == 'land_cover':
        result, stats = analyzer.land_cover_classification(
            num_classes=params.get('num_classes', 4)
        )
        cmap = 'tab10'
        insight_type = 'LAND COVER CLASSIFICATION'
    
    # 5. Biomass Estimation
    elif analysis_type == 'biomass':
        result, stats = analyzer.biomass_estimation()
        cmap = 'Greens'
        insight_type = 'BIOMASS ESTIMATION'
    
    # 6. Wildfire Burn Mapping (requires pre/post images - skip for now)
    elif analysis_type == 'wildfire':
        # Would need pre-fire and post-fire images
        raise ValueError("Wildfire burn mapping requires pre-fire and post-fire images")
    
    # 7. Geology & Terrain Analysis
    elif analysis_type == 'geology':
        result, stats = analyzer.geology_terrain_analysis()
        cmap = 'terrain'
        insight_type = 'GEOLOGY TERRAIN ANALYSIS'
    
    # 8. Flood Mapping
    elif analysis_type == 'flood_mapping':
        result, stats = analyzer.flood_mapping(
            threshold_method=params.get('threshold_method', 'otsu'),
            band=band
        )
        cmap = 'Blues'
        insight_type = 'FLOOD MAPPING'
    
    # 9. Polarimetric Decomposition
    elif analysis_type == 'polarimetric':
        result, stats = analyzer.polarimetric_decomposition()
        cmap = 'RdYlBu'
        insight_type = 'POLARIMETRIC DECOMPOSITION'
    
    # 10. Soil Moisture Estimation
    elif analysis_type == 'soil_moisture':
        result, stats = analyzer.soil_moisture_estimation(
            incidence_angle=params.get('incidence_angle', 39.0)
        )
        cmap = 'Blues'
        insight_type = 'SOIL MOISTURE ESTIMATION'
    
    else:
        raise ValueError(f"Unknown SAR analysis type: {analysis_type}")
    
    # Ensure mask-like outputs are uint8 for GIS compatibility (memory-safe checks)
    try:
        if isinstance(result, np.ndarray) and result.ndim <= 3:
            arr = result
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            # If integer/bool typed, use min/max check
            if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
                vmin = int(np.nanmin(arr))
                vmax = int(np.nanmax(arr))
                if 0 <= vmin <= vmax <= 5:
                    result = result.astype(np.uint8)
            else:
                # Sample to check for small-class integer mask in float arrays
                if arr.ndim == 2:
                    h, w = arr.shape
                else:
                    h, w = arr.shape[-2], arr.shape[-1]
                step = max(1, int(np.sqrt((h * w) / 1_000_000)))  # about 1M samples
                sample = arr[::step, ::step]
                if np.all(np.isfinite(sample)) and np.allclose(sample, np.round(sample), atol=1e-6):
                    vmin = int(np.nanmin(sample))
                    vmax = int(np.nanmax(sample))
                    if 0 <= vmin <= vmax <= 5:
                        result = result.astype(np.uint8)
    except Exception:
        pass
    
    # Generate visualizations
    Plotter.plot_raster(result, f"{analysis_type.upper().replace('_', ' ')}", cmap=cmap,
                       output_path=output_dir / f'{analysis_type}_plot.png')
    
    # Generate histogram for SAR analyses
    Plotter.plot_histogram(result, f"{analysis_type.upper().replace('_', ' ')} Distribution",
                          output_path=output_dir / f'{analysis_type}_histogram.png')
    
    # Vector export: detections as GeoJSON (bbox polygons) when applicable
    geojson_name = None
    try:
        import geopandas as gpd
        from shapely.geometry import box
        import rasterio
        from rasterio.transform import xy
        if analysis_type in ['ship_detection', 'oil_spill'] and stats:
            transform = profile.get('transform')
            crs = profile.get('crs')
            feats = []
            # Ships: expect 'ships' list with bbox_pixels
            if analysis_type == 'ship_detection' and 'ships' in stats:
                for s in stats['ships']:
                    if 'bbox_pixels' not in s:
                        continue
                    x0, y0, x1, y1 = s['bbox_pixels']
                    # Convert pixel edges to geospatial bounds
                    # rasterio.xy returns (x,y) of pixel center; use transform to compute bounds
                    x_min, y_min = xy(transform, y1, x0, offset='ul')
                    x_max, y_max = xy(transform, y0, x1, offset='lr')
                    geom = box(x_min, y_min, x_max, y_max)
                    props = {k: v for k, v in s.items() if k != 'bbox_pixels'}
                    feats.append({'geometry': geom, 'properties': props})
            # Oil spills: prefer per-slick metadata from stats; fallback to regionprops
            if analysis_type == 'oil_spill':
                if 'slicks' in stats and stats['slicks']:
                    for s in stats['slicks']:
                        x0, y0, x1, y1 = s.get('bbox_pixels', (None, None, None, None))
                        if None in (x0, y0, x1, y1):
                            continue
                        x_min, y_min = xy(transform, y1, x0, offset='ul')
                        x_max, y_max = xy(transform, y0, x1, offset='lr')
                        geom = box(x_min, y_min, x_max, y_max)
                        props = {k: v for k, v in s.items() if k != 'bbox_pixels'}
                        feats.append({'geometry': geom, 'properties': props})
                else:
                    from skimage.measure import label, regionprops
                    labeled = label(result.astype(bool))
                    for r in regionprops(labeled):
                        if r.area <= 0:
                            continue
                        x0, y0, x1, y1 = r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]
                        x_min, y_min = xy(transform, y1, x0, offset='ul')
                        x_max, y_max = xy(transform, y0, x1, offset='lr')
                        geom = box(x_min, y_min, x_max, y_max)
                        props = {'area_pixels': int(r.area)}
                        feats.append({'geometry': geom, 'properties': props})
            if feats:
                gdf = gpd.GeoDataFrame([f['properties'] for f in feats], geometry=[f['geometry'] for f in feats], crs=crs)
                geojson_name = f'{analysis_type}_detections.geojson'
                gdf.to_file(output_dir / geojson_name, driver='GeoJSON')
    except Exception as _e:
        logger.warning(f"Vector export skipped: {_e}")
    
    # Generate insights
    insights = InsightsGenerator.generate_sar_insights(
        insight_type, result, stats
    )
    
    # Augment result files with vector export if present
    if geojson_name:
        # analysis_status updated by caller; here we just return path via stats
        stats['vector_geojson'] = geojson_name
    
    return result, stats, insights


@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return send_file('static/favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/download', methods=['GET'])
def download():
    """Satellite imagery download page."""
    return render_template('download.html')


@app.route('/download-progress')
def download_progress():
    """Download progress tracking page."""
    download_id = request.args.get('id')
    if not download_id:
        flash('No download ID provided', 'error')
        return redirect('/')
    
    return render_template('download_progress.html', download_id=download_id)


@app.route('/cache-management')
def cache_management():
    """Cache management and storage analytics page."""
    return render_template('cache_management.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload files and configure analysis."""
    if request.method == 'POST':
        # Validate inputs
        if 'image' not in request.files:
            flash('No image file uploaded', 'error')
            return redirect(request.url)
        
        image_file = request.files['image']
        if image_file.filename == '':
            flash('No image selected', 'error')
            return redirect(request.url)
        
        if not allowed_file(image_file.filename):
            flash('Invalid image file type', 'error')
            return redirect(request.url)
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        session_folder = app.config['UPLOAD_FOLDER'] / session_id
        session_folder.mkdir(exist_ok=True)
        
        # Save image file
        image_filename = secure_filename(image_file.filename)
        image_path = session_folder / image_filename
        image_file.save(image_path)
        
        # Save boundary file if provided
        boundary_path = None
        if 'boundary' in request.files and request.files['boundary'].filename:
            boundary_file = request.files['boundary']
            if allowed_file(boundary_file.filename):
                boundary_filename = secure_filename(boundary_file.filename)
                boundary_path = session_folder / boundary_filename
                boundary_file.save(boundary_path)
        
        # Get analysis parameters
        image_type = request.form.get('image_type')
        analysis_type = request.form.get('analysis_type')
        
        # Get additional parameters
        params = {}
        if image_type == 'optical':
            params = {
                'analysis_type': analysis_type,
                'sensor': (request.form.get('sensor') or '').strip() or None,
                'red': int(request.form.get('red', 2)),
                'nir': int(request.form.get('nir', 3)),
                'green': int(request.form.get('green', 1)),
                'blue': int(request.form.get('blue', 0)),
                'swir1': int(request.form.get('swir1', 4)),
                'swir2': int(request.form.get('swir2', 5)),
                'n_clusters': int(request.form.get('n_clusters', 5)),
                'stretch_method': request.form.get('stretch_method', 'percentile')
            }
        else:  # SAR
            params = {
                'analysis_type': analysis_type,
                'window_size': int(request.form.get('window_size', 5)),
                'num_looks': int(request.form.get('num_looks', 1)),
                'incidence_angle': float(request.form.get('incidence_angle', 39.0))
            }
        
        # Add COG options if selected
        if analysis_type == 'cog':
            params.update({
                'cog_web_mercator': True if request.form.get('cog_web_mercator') else False,
                'cog_epsg': request.form.get('cog_epsg') or None,
            })

        # Initialize analysis status
        analysis_status[session_id] = {
            'status': 'queued',
            'progress': 0,
            'created': datetime.now().isoformat(),
            'image_type': image_type,
            'analysis_type': analysis_type
        }
        save_analysis_status()
        
        # Start analysis in background thread
        thread = threading.Thread(
            target=run_analysis_async,
            args=(session_id, str(image_path), str(boundary_path) if boundary_path else None,
                  image_type, analysis_type, params)
        )
        thread.start()
        
        flash('Analysis started! You will be redirected to results.', 'success')
        return redirect(url_for('results', session_id=session_id))
    
    return render_template('upload.html')


@app.route('/results/<session_id>')
def results(session_id):
    """View analysis results."""
    if session_id not in analysis_status:
        flash('Analysis session not found', 'error')
        return redirect(url_for('index'))
    
    return render_template('results.html', session_id=session_id)


@app.route('/api/status/<session_id>')
def get_status(session_id):
    """Get analysis status (AJAX endpoint)."""
    if session_id in analysis_status:
        return jsonify(analysis_status[session_id])
    return jsonify({'status': 'not_found'}), 404


@app.route('/api/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """Download result file."""
    if session_id not in analysis_status:
        return "Session not found", 404
    
    file_path = app.config['RESULTS_FOLDER'] / session_id / filename
    if not file_path.exists():
        return "File not found", 404
    
    return send_file(file_path, as_attachment=True)


@app.route('/api/cleanup/<session_id>', methods=['POST', 'GET'])
def cleanup_session(session_id):
    """Clean up session files and data."""
    import shutil
    
    try:
        # Get temp files and folders from status
        temp_files = []
        session_folder = None
        if session_id in analysis_status:
            temp_files = analysis_status[session_id].get('temp_files', [])
            session_folder = analysis_status[session_id].get('session_folder')
        
        # Clean up temporary processing files
        for item in temp_files:
            try:
                item_path = Path(item)
                if item_path.is_file():
                    item_path.unlink(missing_ok=True)
                    logger.info(f"Cleaned up temp file: {item}")
                elif item_path.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                    logger.info(f"Cleaned up temp directory: {item}")
            except Exception as cleanup_error:
                logger.warning(f"Could not cleanup {item}: {cleanup_error}")
        
        # Clean up session folder
        if session_folder:
            try:
                session_path = Path(session_folder)
                if session_path.exists():
                    shutil.rmtree(session_path, ignore_errors=True)
                    logger.info(f"Cleaned up session folder: {session_folder}")
            except Exception as cleanup_error:
                logger.warning(f"Could not cleanup session folder: {cleanup_error}")
        
        # Remove result files
        result_folder = app.config['RESULTS_FOLDER'] / session_id
        if result_folder.exists():
            shutil.rmtree(result_folder, ignore_errors=True)
            logger.info(f"Cleaned up result folder: {result_folder}")
        
        # Remove upload files (if any remain)
        upload_folder = app.config['UPLOAD_FOLDER'] / session_id
        if upload_folder.exists():
            shutil.rmtree(upload_folder, ignore_errors=True)
            logger.info(f"Cleaned up upload folder: {upload_folder}")
        
        # Remove from analysis status
        if session_id in analysis_status:
            del analysis_status[session_id]
            save_analysis_status()
        
        return jsonify({'status': 'success', 'message': 'All session files cleaned up successfully'})
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/view/<session_id>/<filename>')
def view_file(session_id, filename):
    """View result file (for images)."""
    if session_id not in analysis_status:
        return "Session not found", 404
    
    file_path = app.config['RESULTS_FOLDER'] / session_id / filename
    if not file_path.exists():
        return "File not found", 404
    
    return send_file(file_path)


# Online Imagery Routes
@app.route('/online-imagery', methods=['GET'])
def online_imagery():
    """Online imagery search and discovery page."""
    return render_template('online_imagery.html')


@app.route('/api/search-stac', methods=['POST'])
def search_stac():
    """Search STAC catalog for imagery matching criteria."""
    try:
        data = request.get_json()
        
        # Validate inputs
        if not data.get('geometry'):
            return jsonify({'error': 'No geometry provided'}), 400
        
        geometry = data.get('geometry')
        satellite = data.get('satellite', 'sentinel2')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        max_cloud = data.get('max_cloud', 100)
        sar_options = data.get('sar_options', {})
        
        # Call STAC search function
        results = query_stac_catalog(
            geometry=geometry,
            satellite=satellite,
            start_date=start_date,
            end_date=end_date,
            max_cloud=max_cloud,
            sar_options=sar_options
        )
        
        return jsonify({'features': results})
    
    except ImportError as e:
        logger.error(f"pystac-client not installed: {str(e)}")
        return jsonify({
            'error': 'pystac-client is required for STAC queries. Install with: pip install pystac-client'
        }), 500
    except Exception as e:
        logger.error(f"STAC search error: {str(e)}")
        return jsonify({'error': f'STAC query failed: {str(e)}'}), 500


@app.route('/api/prepare-download', methods=['POST'])
def prepare_download():
    """Prepare selected scenes for download."""
    try:
        data = request.get_json()
        scene_ids = data.get('scene_ids', [])
        scenes = data.get('scenes', [])
        
        if not scene_ids or not scenes:
            return jsonify({'error': 'No scenes selected'}), 400
        
        # Create a download session
        download_id = str(uuid.uuid4())
        download_folder = app.config['UPLOAD_FOLDER'] / f"download_{download_id}"
        download_folder.mkdir(exist_ok=True)
        
        # Store download info
        download_info = {
            'download_id': download_id,
            'scene_ids': scene_ids,
            'scenes': scenes,
            'status': 'preparing',
            'created': datetime.now().isoformat(),
            'folder': str(download_folder)
        }
        
        # Save download metadata
        metadata_file = download_folder / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(download_info, f, indent=2)
        
        logger.info(f"Download session created: {download_id}")
        
        return jsonify({
            'download_id': download_id,
            'message': f'{len(scene_ids)} scene(s) prepared for download'
        })
    
    except Exception as e:
        logger.error(f"Download preparation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/start-stac-download', methods=['POST'])
def start_stac_download():
    """
    Start downloading imagery from STAC.
    
    Expects JSON with:
    - download_id: Download session ID
    - scenes: List of scene objects
    - keep_imagery: Whether to keep imagery after analysis
    - auto_analyze: Whether to auto-start analysis after download
    """
    try:
        data = request.json
        download_id = data.get('download_id')
        scenes = data.get('scenes', [])
        keep_imagery = data.get('keep_imagery', False)
        auto_analyze = data.get('auto_analyze', False)
        
        if not download_id or not scenes:
            return jsonify({'error': 'Missing download_id or scenes'}), 400
        
        # Create download session
        manager_session = download_manager.create_session(
            download_id,
            scenes,
            keep_imagery=keep_imagery
        )
        
        # Track in download_sessions
        download_sessions[download_id] = {
            'status': 'downloading',
            'download_id': download_id,
            'scenes': scenes,
            'keep_imagery': keep_imagery,
            'auto_analyze': auto_analyze,
            'progress': 0,
            'errors': [],
            'created': datetime.now().isoformat()
        }
        
        # Register downloads in temporary storage manager
        for scene in scenes:
            metadata = {
                'scene_id': scene.get('id'),
                'platform': scene.get('platform'),
                'acquired': scene.get('acquired'),
                'cloud_cover': scene.get('cloud_cover', 0),
                'bbox': scene.get('geometry', {}).get('bbox'),
                'stac_collection': scene.get('collection')
            }
            storage_manager.register_download(download_id, scene.get('id'), metadata)
        
        # Define progress callback
        def progress_callback(progress_info):
            session = download_sessions.get(download_id)
            if session:
                if progress_info['type'] == 'scene_complete':
                    session['progress'] = progress_info['percent']
                    logger.info(f"Download progress: {progress_info['percent']:.1f}%")
                elif progress_info['type'] == 'complete':
                    session['status'] = 'completed'
                    session['progress'] = 100
                    logger.info(f"Download completed: {download_id}")
                elif progress_info['type'] == 'error':
                    session['errors'].append(progress_info.get('error', 'Unknown error'))
                    logger.error(f"Download error: {progress_info.get('error')}")
        
        # Start download in background thread
        download_thread = threading.Thread(
            target=download_imagery_async,
            args=(download_manager, download_id, scenes, progress_callback),
            daemon=True
        )
        download_thread.start()
        
        logger.info(f"STAC download started: {download_id} ({len(scenes)} scenes)")
        
        return jsonify({
            'download_id': download_id,
            'status': 'downloading',
            'message': 'Download started in background'
        })
    
    except Exception as e:
        logger.error(f"STAC download start error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-progress/<download_id>')
def get_download_progress(download_id):
    """
    Get download progress as Server-Sent Events stream.
    
    Streams real-time progress updates to client.
    """
    def generate():
        last_update = 0
        while True:
            session = download_sessions.get(download_id)
            if not session:
                yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                break
            
            progress = session.get('progress', 0)
            status = session.get('status', 'unknown')
            
            # Send update if progress changed
            if progress != last_update or status in ['completed', 'error']:
                data = {
                    'download_id': download_id,
                    'status': status,
                    'progress': progress,
                    'errors': session.get('errors', [])
                }
                yield f"data: {json.dumps(data)}\n\n"
                last_update = progress
            
            # Exit if completed or error
            if status in ['completed', 'error', 'failed']:
                break
            
            # Sleep before next check
            import time
            time.sleep(1)
    
    return generate(), 200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    }


@app.route('/api/download-status/<download_id>')
def get_download_status(download_id):
    """Get current download session status."""
    try:
        session = download_sessions.get(download_id)
        manager_session = download_manager.get_session_info(download_id)
        
        if not session and not manager_session:
            return jsonify({'error': 'Download session not found'}), 404
        
        # Combine info
        status_info = {
            'download_id': download_id,
            'status': session.get('status', 'unknown') if session else 'unknown',
            'progress': session.get('progress', 0) if session else 0,
            'errors': session.get('errors', []) if session else [],
            'files': manager_session.get('downloaded_files', []) if manager_session else [],
            'total_size': manager_session.get('total_size', 0) if manager_session else 0
        }
        
        return jsonify(status_info)
    
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/confirm-storage/<download_id>', methods=['POST'])
def confirm_storage(download_id):
    """
    Confirm storage decision for downloaded imagery.
    
    Expects JSON with:
    - action: 'keep' (permanent), 'analyze' (temp), or 'delete'
    - auto_analyze: Whether to start analysis
    """
    try:
        data = request.json
        action = data.get('action', 'delete')
        auto_analyze = data.get('auto_analyze', False)
        
        session = download_sessions.get(download_id)
        if not session:
            return jsonify({'error': 'Download session not found'}), 404
        
        # Handle storage action
        result = {'download_id': download_id, 'action': action}
        
        if action == 'keep':
            # Move to permanent storage
            permanent_path = download_manager.move_to_permanent(download_id)
            storage_manager.mark_as_kept(download_id)
            result['permanent_path'] = str(permanent_path)
            logger.info(f"Imagery moved to permanent storage: {permanent_path}")
        
        elif action == 'analyze':
            # Keep temp for analysis - mark for deletion after analysis
            storage_manager.mark_for_deletion(download_id, reason="analysis_complete")
            result['temp_path'] = session.get('folder')
            logger.info(f"Session {download_id} marked for deletion after analysis")
        
        elif action == 'delete':
            # Delete downloads immediately
            download_manager.cleanup_session(download_id, force=True)
            result['deleted'] = True
            logger.info(f"Download cleaned up: {download_id}")
        
        # Start analysis if requested
        if auto_analyze and action in ['keep', 'analyze']:
            imagery_folder = download_manager.get_session_info(download_id).get('folder')
            
            # Create analysis session
            analysis_id = str(uuid.uuid4())
            analysis_status[analysis_id] = {
                'session_id': analysis_id,
                'status': 'queued',
                'progress': 0,
                'imagery_source': 'stac_download',
                'download_id': download_id,
                'created': datetime.now().isoformat()
            }
            
            # Track analysis in storage manager
            storage_manager.mark_as_analyzed(download_id, analysis_id)
            
            result['analysis_id'] = analysis_id
            logger.info(f"Analysis session created from STAC download: {analysis_id}")
        
        # Clean up session tracking
        if download_id in download_sessions:
            del download_sessions[download_id]
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Storage confirmation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cleanup-downloads', methods=['POST'])
def cleanup_old_downloads():
    """
    Clean up old temporary downloads.
    
    Expects JSON with optional 'hours' parameter (default 24).
    """
    try:
        data = request.json or {}
        hours = data.get('hours', 24)
        
        count = download_manager.cleanup_old_sessions(hours=hours)
        
        return jsonify({
            'message': f'Cleaned up {count} old download session(s)',
            'cleaned': count
        })
    
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache-info', methods=['GET'])
def get_cache_info():
    """
    Get information about current cache (temporary downloads).
    
    Returns cache statistics including total size, file count, breakdown by satellite,
    age distribution, and sessions awaiting cleanup.
    """
    try:
        cache_info = storage_manager.get_cache_info()
        return jsonify(cache_info)
    
    except Exception as e:
        logger.error(f"Cache info error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cache-cleanup', methods=['POST'])
def cache_cleanup():
    """
    Clean up marked and old cache files.
    
    Expects JSON with optional parameters:
    - force_old: Whether to force delete old sessions (default False)
    - days_old: Age threshold for force deletion (default 1)
    """
    try:
        data = request.json or {}
        force_old = data.get('force_old', False)
        days_old = data.get('days_old', 1)
        
        report = storage_manager.cleanup_marked_sessions(
            force_delete_old=force_old,
            days_old=days_old
        )
        
        logger.info(f"Cache cleanup report: {len(report['deleted_sessions'])} sessions deleted, "
                   f"{report['size_freed_mb']:.2f} MB freed")
        
        return jsonify(report)
    
    except Exception as e:
        logger.error(f"Cache cleanup error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/session-metadata/<session_id>', methods=['GET'])
def get_session_metadata(session_id):
    """
    Get metadata for a specific download session.
    
    Returns scene metadata including source, acquisition date, satellite, cloud cover, etc.
    """
    try:
        metadata = storage_manager.get_session_metadata(session_id)
        
        if not metadata:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify(metadata)
    
    except Exception as e:
        logger.error(f"Metadata retrieval error: {str(e)}")
        return jsonify({'error': str(e)}), 500


def query_stac_catalog(geometry, satellite='sentinel2', start_date=None, end_date=None, 
                       max_cloud=100, sar_options=None):
    """
    Query Planetary Computer STAC catalog for matching imagery.
    
    Requires pystac-client to be installed (see requirements.txt).
    
    Args:
        geometry: GeoJSON geometry dict
        satellite: 'sentinel2', 'sentinel1', 'landsat8', or 'landsat9'
        start_date: ISO date string (YYYY-MM-DD)
        end_date: ISO date string (YYYY-MM-DD)
        max_cloud: Max cloud coverage percentage (0-100)
        sar_options: Dict with SAR-specific filters
    
    Returns:
        List of feature objects with id and properties
    
    Raises:
        ImportError: If pystac-client is not installed
        Exception: If STAC query fails
    """
    import pystac_client
    from datetime import datetime as dt
    
    # Planetary Computer STAC API endpoint
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    
    # Map satellite names to collection IDs
    collection_map = {
        'sentinel2': 'sentinel-2-l2a',
        'sentinel1': 'sentinel-1-rtc',
        'landsat8': 'landsat-c2-l2',
        'landsat9': 'landsat-c2-l2'
    }
    
    collection = collection_map.get(satellite, 'sentinel-2-l2a')
    
    # Build search query
    search = pystac_client.Client.open(catalog_url).search(
        collections=[collection],
        intersects=geometry,
        datetime=f"{start_date}T00:00:00Z/{end_date}T23:59:59Z" if start_date and end_date else None,
        max_items=100
    )
    
    # Get matching items
    items = search.get_all_items()
    
    # Process and filter results
    results = []
    for item in items:
        props = item.properties.copy()
        
        # Add item ID and asset info
        props['id'] = item.id
        props['datetime'] = props.get('datetime') or item.datetime.isoformat() if item.datetime else 'Unknown'
        
        # Normalize metadata for frontend display
        props['cloud_cover'] = props.get('eo:cloud_cover', props.get('s2:medium_proba_clouds_percentage', 0))
        
        # Solar Elevation (for Sentinel-2 it's 90 - Zenith)
        sun_elev = props.get('view:sun_elevation')
        if sun_elev is None and 's2:mean_solar_zenith' in props:
            sun_elev = 90 - props['s2:mean_solar_zenith']
        props['sun_elevation'] = sun_elev
        
        # Nadir Angle
        props['off_nadir'] = props.get('view:off_nadir', props.get('view:incidence_angle', 0))
        
        props['collection'] = item.collection_id
        props['platform_name'] = props.get('platform', props.get('platform_id', 'Unknown'))
        props['satellite_name'] = satellite.upper()

        if item.collection_id == 'sentinel-2-l2a':
             props['platform_name'] = props.get('platform', 'Sentinel-2')
             props['satellite_name'] = 'SENTINEL-2'

        # Filter by cloud cover if optical
        if satellite.startswith('sentinel2') or satellite.startswith('landsat'):
            if props['cloud_cover'] > max_cloud:
                continue
        
        # Add asset URLs if available
        props['assets'] = {key: {
            'href': asset.href,
            'type': asset.media_type,
            'roles': asset.roles
        } for key, asset in item.assets.items()}
        
        # Priority for preview sources
        preview_url = None
        if 'rendered_preview' in props['assets']:
            preview_url = props['assets']['rendered_preview']['href']
        elif 'thumbnail' in props['assets']:
            preview_url = props['assets']['thumbnail']['href']
        elif 'overview' in props['assets']:
            preview_url = props['assets']['overview']['href']
        elif 'visual' in props['assets']:
             # Only use visual as preview if it is an overview/thumbnail role
             if 'overview' in (props['assets']['visual'].get('roles') or []):
                 preview_url = props['assets']['visual']['href']
        
        props['preview_url'] = preview_url
        
        results.append({'id': item.id, 'properties': props})
    
    logger.info(f"STAC search returned {len(results)} results for {satellite}")
    return results


if __name__ == '__main__':
    print("=" * 70)
    print("🌍 Analyz Web Application")
    print("=" * 70)
    print("Starting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("Press CTRL+C to stop the server")
    print("=" * 70)
    
    # Run Flask with reloader disabled to prevent forrtl errors from SNAP/GDAL libraries
    # The auto-reloader causes issues with Fortran runtime libraries when processing SAR data
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
