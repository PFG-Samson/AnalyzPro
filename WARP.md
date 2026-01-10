# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common commands

- Install dependencies
  - pip install -r requirements.txt

- CLI usage (run analyses on local GeoTIFFs)
  - Optical NDVI
    - python app.py --image path/to/optical.tif --image-type optical --analysis ndvi --output outputs/ndvi
  - Optical Classification (custom clusters and bands)
    - python app.py --image path/to/optical.tif --image-type optical --analysis classification --n-clusters 7 --band-indices "red:2,nir:3,green:1,blue:0" --output outputs/classification
  - SAR Flood Mapping
    - python app.py --image path/to/sar.tif --image-type sar --analysis flood_mapping --output outputs/flood
  - With study-area boundary (clips raster before analysis)
    - python app.py --image path/to/image.tif --image-type optical --analysis ndvi --boundary path/to/aoi.geojson --output outputs/ndvi_clipped

- Examples (end-to-end runnable scripts)
  - python examples/optical_analysis_example.py
  - python examples/sar_analysis_example.py

- Web app (Flask UI)
  - Windows launcher
    - .\start_webapp.bat
  - Or directly
    - cd webapp && python app_web.py
  - Then open http://localhost:5000

- Lint, tests, build
  - No linter, test suite, or packaging/build configs are present in the repo.

## Architecture and code structure (big picture)

- Core data flow (CLI and API)
  1) analyz.utils.FileHandler reads rasters/vectors; optional analyz.processing.BoundaryHandler clips data to the AOI.
  2) analyz.core.OpticalAnalyzer and SARAnalyzer compute analysis arrays and statistics (e.g., NDVI with stress categories; NDWI; NDBI; EVI; SAVI; SAR oil/ship/flood/polarimetric/biomass/crop/terrain; etc.).
  3) analyz.visualization.Plotter produces PNGs (histograms, maps, comparisons); analyz.visualization.InsightsGenerator derives plain-language insights and summaries.
  4) Results are written via FileHandler.write_raster and saved alongside plots and insights in outputs/.

- Python package layout (analyz/)
  - core/: analyzers for optical and SAR.
    - OpticalAnalyzer: indices (NDVI/NDWI/NDBI/EVI/SAVI), unsupervised land-cover classification with semantic labels, band stacks, RGB composites, change detection.
    - SARAnalyzer: application-focused analyses (oil spill, ship detection via CFAR-style thresholding, crop monitoring via RVI, land cover, biomass, flood mapping, polarimetric ratio, wildfire burn mapping, geology/terrain), with internal speckle filtering and texture features.
  - processing/: utilities applied across workflows.
    - BoundaryHandler: vector-based clipping and CRS handling; works with files or in-memory arrays.
    - Preprocessor: normalization, nodata filling, contrast enhancement, resampling.
  - utils/: cross-cutting utilities.
    - FileHandler: raster/vector I/O (rasterio/geopandas), safe writers, output path helpers.
    - logger.py with setup_logger/get_logger (loguru-based) used across modules.
    - SatellitePreprocessor: handles satellite archives (Sentinel-1/2, Landsat 8/9), extracts and stacks bands to GeoTIFF; used by the web app for auto-processing uploaded archives.
  - visualization/: rendering and reporting.
    - Plotter: saves static PNGs (300 DPI) for maps, histograms, multi-figure layouts, and land-cover legends.
    - InsightsGenerator: generates domain-specific textual insights for optical and SAR outputs (e.g., vegetation stress alerts, flood coverage, ship counts).
  - analyz/__init__.py re-exports the primary API surface: OpticalAnalyzer, SARAnalyzer, BoundaryHandler, Preprocessor, Plotter, InsightsGenerator, setup_logger, FileHandler.

- CLI (app.py)
  - Argparse-based entry point. Loads defaults from config.yaml (e.g., optical.default_bands). Orchestrates I/O, boundary clipping, analysis selection, plotting, and insights; writes artifacts to the chosen output directory.

- Web application (webapp/app_web.py)
  - Flask server with pages and JSON endpoints. Uploads are stored per-session; analyses run in a background thread (run_analysis_async), with progress tracked in analysis_status.json. Integrates SatellitePreprocessor to accept satellite archives (.zip/.tar) and auto-produce a GeoTIFF before analysis. Results per session include result.tif, {analysis}_plot.png, {analysis}_histogram.png, statistics.json, insights.txt.

- Configuration (config.yaml)
  - Central defaults for processing (threads, resampling), optical (default bands, NDVI thresholds), SAR (filtering, coherence, change detection), visualization (dpi, figure size, colormap), insights, export (compression/tiling/overviews), and study_area formats.

## Notes for future agents

- Heavy geospatial dependencies (rasterio, GDAL, fiona, geopandas) are required; ensure the environment has these installed via requirements.txt before running CLI or the web app.
- The repository does not currently include tests or lint rules; focus on running the examples and CLI to validate changes. If you add tests later, prefer pytest and place them under tests/.
