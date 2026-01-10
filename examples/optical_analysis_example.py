"""Example script for optical image analysis."""

from pathlib import Path
from analyz import (
    OpticalAnalyzer, BoundaryHandler, Plotter, 
    InsightsGenerator, setup_logger, FileHandler
)

# Setup logging
logger = setup_logger("INFO", "optical_example.log")

# Example configuration
IMAGE_PATH = "path/to/your/optical_image.tif"
BOUNDARY_PATH = "path/to/your/boundary.geojson"  # Optional
OUTPUT_DIR = "outputs/optical_example"

def main():
    """Run optical analysis example."""
    
    # 1. Load optical image
    logger.info(f"Loading image: {IMAGE_PATH}")
    data, profile = FileHandler.read_raster(IMAGE_PATH)
    print(f"Loaded image with shape: {data.shape}")
    
    # 2. (Optional) Apply study area boundary
    if Path(BOUNDARY_PATH).exists():
        logger.info("Applying study area boundary")
        boundary_handler = BoundaryHandler(BOUNDARY_PATH)
        data, profile = boundary_handler.clip_array(data, profile)
        print(f"Clipped to boundary. New shape: {data.shape}")
    
    # 3. Define band indices for your sensor
    # Example for Sentinel-2: Band 2=Blue, 3=Green, 4=Red, 8=NIR, 11=SWIR1, 12=SWIR2
    band_indices = {
        'blue': 0,    # Adjust these indices based on your data
        'green': 1,
        'red': 2,
        'nir': 3,
        'swir1': 4,
        'swir2': 5
    }
    
    # 4. Initialize analyzer
    analyzer = OpticalAnalyzer(data, band_indices)
    
    # 5. Generate RGB True Color Composite
    print("\n=== Generating RGB True Color Composite ===")
    rgb_composite, rgb_stats = analyzer.rgb_composite(stretch_method='percentile', percentile_range=(2, 98))
    print(f"RGB Composite Shape: {rgb_stats['shape']}")
    print(f"Stretch Method: {rgb_stats['method']}")
    print(f"Value Range: {rgb_stats['stretched_range']}")
    
    # Save RGB composite as GeoTIFF
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For saving RGB composite, we need to transpose it to (bands, height, width)
    rgb_for_save = np.transpose(rgb_composite, (2, 0, 1))
    FileHandler.write_raster(output_dir / "rgb_composite.tif", rgb_for_save, profile)
    
    # Visualize RGB composite
    Plotter.plot_rgb_composite(rgb_composite, 
                              title="True Color Composite (RGB)",
                              output_path=output_dir / "rgb_composite.png")
    
    # Visualize RGB composite with histograms
    Plotter.plot_rgb_composite(rgb_composite, 
                              title="True Color Composite with Histograms",
                              output_path=output_dir / "rgb_composite_with_histograms.png",
                              show_histograms=True)
    
    # 6. Run NDVI analysis
    print("\n=== Running NDVI Analysis ===")
    ndvi, ndvi_stats = analyzer.ndvi()
    print(f"NDVI Mean: {ndvi_stats['mean']:.3f}")
    print(f"Vegetation Coverage: {ndvi_stats['vegetation_cover_percent']:.2f}%")
    print(f"Healthy Vegetation: {ndvi_stats['healthy_vegetation_percent']:.2f}%")
    
    FileHandler.write_raster(output_dir / "ndvi_result.tif", ndvi, profile)
    
    # Generate NDVI visualization
    Plotter.plot_ndvi_classification(ndvi, output_path=output_dir / "ndvi_classification.png")
    Plotter.plot_histogram(ndvi, "NDVI Distribution", output_path=output_dir / "ndvi_histogram.png")
    
    # Generate insights
    ndvi_insights = InsightsGenerator.generate_ndvi_insights(ndvi, ndvi_stats)
    report = InsightsGenerator.format_insights_report(ndvi_insights, 
                                                      output_dir / "ndvi_insights.txt")
    print("\n" + report)
    
    # 7. Run NDWI analysis
    print("\n=== Running NDWI Analysis ===")
    ndwi, ndwi_stats = analyzer.ndwi()
    print(f"NDWI Mean: {ndwi_stats['mean']:.3f}")
    print(f"Water Coverage: {ndwi_stats['water_cover_percent']:.2f}%")
    
    FileHandler.write_raster(output_dir / "ndwi_result.tif", ndwi, profile)
    Plotter.plot_raster(ndwi, "NDWI Analysis", cmap='Blues', 
                       output_path=output_dir / "ndwi_plot.png")
    
    # 8. Run NDBI analysis
    print("\n=== Running NDBI Analysis ===")
    ndbi, ndbi_stats = analyzer.ndbi()
    print(f"NDBI Mean: {ndbi_stats['mean']:.3f}")
    print(f"Urban Coverage: {ndbi_stats['urban_cover_percent']:.2f}%")
    
    FileHandler.write_raster(output_dir / "ndbi_result.tif", ndbi, profile)
    Plotter.plot_raster(ndbi, "NDBI Analysis", cmap='Reds',
                       output_path=output_dir / "ndbi_plot.png")
    
    # 9. Run EVI analysis
    print("\n=== Running EVI Analysis ===")
    evi, evi_stats = analyzer.evi()
    print(f"EVI Mean: {evi_stats['mean']:.3f}")
    
    FileHandler.write_raster(output_dir / "evi_result.tif", evi, profile)
    Plotter.plot_raster(evi, "EVI Analysis", cmap='RdYlGn',
                       output_path=output_dir / "evi_plot.png")
    
    # 10. Run land cover classification
    print("\n=== Running Land Cover Classification ===")
    classified, class_stats = analyzer.classify_kmeans(n_clusters=5)
    print("Class Distribution:")
    for class_name, class_info in class_stats['class_distribution'].items():
        print(f"  {class_name}: {class_info['percent']:.2f}% ({class_info['count']} pixels)")
    
    FileHandler.write_raster(output_dir / "classification_result.tif", classified, profile)
    Plotter.plot_raster(classified, "Land Cover Classification", cmap='tab10',
                       output_path=output_dir / "classification_plot.png")
    
    # 11. Create multi-figure comparison
    print("\n=== Creating Multi-Figure Comparison ===")
    Plotter.plot_multifigure(
        [ndvi, ndwi, ndbi, evi],
        ['NDVI', 'NDWI', 'NDBI', 'EVI'],
        ['RdYlGn', 'Blues', 'Reds', 'RdYlGn'],
        output_path=output_dir / "all_indices_comparison.png"
    )
    
    print(f"\n✓ All analyses complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
