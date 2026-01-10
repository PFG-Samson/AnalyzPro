"""Example script for SAR image analysis."""

from pathlib import Path
from analyz import (
    SARAnalyzer, BoundaryHandler, Plotter, 
    InsightsGenerator, setup_logger, FileHandler
)

# Setup logging
logger = setup_logger("INFO", "sar_example.log")

# Example configuration
IMAGE_PATH = "path/to/your/sar_image.tif"
BOUNDARY_PATH = "path/to/your/boundary.geojson"  # Optional
OUTPUT_DIR = "outputs/sar_example"

def main():
    """Run SAR analysis example."""
    
    # 1. Load SAR image
    logger.info(f"Loading SAR image: {IMAGE_PATH}")
    data, profile = FileHandler.read_raster(IMAGE_PATH)
    print(f"Loaded SAR image with shape: {data.shape}")
    
    # 2. (Optional) Apply study area boundary
    if Path(BOUNDARY_PATH).exists():
        logger.info("Applying study area boundary")
        boundary_handler = BoundaryHandler(BOUNDARY_PATH)
        data, profile = boundary_handler.clip_array(data, profile)
        print(f"Clipped to boundary. New shape: {data.shape}")
    
    # 3. Initialize SAR analyzer
    analyzer = SARAnalyzer(data, profile=profile)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Oil Spill Detection
    print("\n=== Oil Spill Detection ===")
    oil_spill_mask, oil_stats = analyzer.oil_spill_detection(window_size=51, k_threshold=1.5, min_area_pixels=100)
    print(f"Detected Slicks: {oil_stats['num_detected_slicks']}")
    print(f"Total Slick Area: {oil_stats['total_slick_area_km2']:.2f} km²")
    print(f"Coverage: {oil_stats['coverage_percent']:.2f}%")
    
    FileHandler.write_raster(output_dir / "oil_spill_mask.tif", oil_spill_mask, profile)
    Plotter.plot_raster(oil_spill_mask, "Oil Spill Detection", cmap='Reds',
                       output_path=output_dir / "oil_spill_map.png")
    
    # 5. Ship Detection
    print("\n=== Ship Detection ===")
    ship_mask, ship_stats = analyzer.ship_detection(cfar_window=50, false_alarm_rate=1e-5)
    print(f"Detected Ships: {ship_stats['num_detected_ships']}")
    if ship_stats['num_detected_ships'] > 0:
        print(f"Largest Ship: {ship_stats['largest_ship_m2']:.0f} m²")
        print(f"Mean Ship Size: {ship_stats['mean_ship_size_m2']:.0f} m²")
    
    FileHandler.write_raster(output_dir / "ship_mask.tif", ship_mask, profile)
    Plotter.plot_raster(ship_mask, "Ship Detection", cmap='hot',
                       output_path=output_dir / "ship_detection_map.png")
    
    # 6. Crop Monitoring (requires dual-pol VV+VH)
    if data.shape[0] > 1:
        print("\n=== Crop Monitoring ===")
        crop_vigor, crop_stats = analyzer.crop_monitoring()
        print(f"Mean Crop Vigor (RVI): {crop_stats['mean']:.3f}")
        print(f"High Vigor Coverage: {crop_stats['high_vigor_percent']:.2f}%")
        print(f"Vegetated Area: {crop_stats['vegetated_area_km2']:.2f} km²")
        
        FileHandler.write_raster(output_dir / "crop_vigor.tif", crop_vigor, profile)
        Plotter.plot_raster(crop_vigor, "Crop Vigor Index (RVI)", cmap='YlGn',
                           output_path=output_dir / "crop_vigor_map.png")
    
    # 7. Land Cover Classification
    print("\n=== Land Cover Classification ===")
    lc_map, lc_stats = analyzer.land_cover_classification(num_classes=4)
    print(f"Number of Classes: {lc_stats['num_classes']}")
    for i in range(lc_stats['num_classes']):
        print(f"  Class {i}: {lc_stats[f'class_{i}_percent']:.2f}% ({lc_stats[f'class_{i}_area_km2']:.2f} km²)")
    
    FileHandler.write_raster(output_dir / "land_cover_map.tif", lc_map, profile)
    Plotter.plot_raster(lc_map, "Land Cover Classification", cmap='tab10',
                       output_path=output_dir / "land_cover_map.png")
    
    # 8. Biomass Estimation
    print("\n=== Biomass Estimation ===")
    biomass_idx, biomass_stats = analyzer.biomass_estimation()
    print(f"Mean Biomass Index: {biomass_stats['mean']:.3f}")
    print(f"Forest Area: {biomass_stats['forest_area_km2']:.2f} km²")
    print(f"High Biomass: {biomass_stats['high_biomass_percent']:.2f}%")
    
    FileHandler.write_raster(output_dir / "biomass_index.tif", biomass_idx, profile)
    Plotter.plot_raster(biomass_idx, "Biomass Index", cmap='Greens',
                       output_path=output_dir / "biomass_map.png")
    
    # 9. Flood mapping
    print("\n=== Flood Mapping ===")
    flood_mask, flood_stats = analyzer.flood_mapping(threshold_method='otsu')
    print(f"Flood Threshold: {flood_stats['threshold']:.2f} dB")
    print(f"Water Coverage: {flood_stats['water_percent']:.2f}%")
    print(f"Water Pixels: {flood_stats['water_pixels']}")
    
    FileHandler.write_raster(output_dir / "flood_mask.tif", flood_mask, profile)
    Plotter.plot_raster(flood_mask, "Flood Mapping", cmap='Blues',
                       output_path=output_dir / "flood_map.png")
    
    # Generate flood insights
    flood_insights = InsightsGenerator.generate_sar_insights("Flood Mapping", flood_mask, flood_stats)
    flood_report = InsightsGenerator.format_insights_report(flood_insights,
                                                            output_dir / "flood_insights.txt")
    print("\n" + flood_report)
    
    # 10. Polarimetric Analysis (requires dual-pol VV+VH)
    if data.shape[0] > 1:
        print("\n=== Polarimetric Decomposition ===")
        pol_ratio, pol_stats = analyzer.polarimetric_decomposition()
        print(f"VV/VH Ratio: {pol_stats['mean']:.2f} dB")
        print(f"Forest: {pol_stats['forest_percent']:.2f}%")
        print(f"Urban: {pol_stats['urban_percent']:.2f}%")
        print(f"Agricultural: {pol_stats['agricultural_percent']:.2f}%")
        
        FileHandler.write_raster(output_dir / "polarimetric_ratio.tif", pol_ratio, profile)
        Plotter.plot_raster(pol_ratio, "VV/VH Polarimetric Ratio (dB)", cmap='RdYlBu',
                           output_path=output_dir / "polarimetric_map.png")
    
    # 11. Soil Moisture Estimation
    print("\n=== Soil Moisture Estimation ===")
    soil_moisture, sm_stats = analyzer.soil_moisture_estimation(incidence_angle=39.0)
    print(f"Mean Soil Moisture Index: {sm_stats['mean']:.3f}")
    print(f"Very Dry: {sm_stats['very_dry_percent']:.2f}%")
    print(f"Moist/Very Moist: {sm_stats['moist_percent'] + sm_stats['very_moist_percent']:.2f}%")
    
    FileHandler.write_raster(output_dir / "soil_moisture.tif", soil_moisture, profile)
    Plotter.plot_raster(soil_moisture, "Soil Moisture Index", cmap='Blues',
                       output_path=output_dir / "soil_moisture_map.png")
    
    # 12. Wildfire Burn Mapping (if you have pre/post-fire scenes)
    # Uncomment if you have pre-fire and post-fire images
    """
    print("\n=== Wildfire Burn Mapping ===")
    POST_FIRE_PATH = "path/to/post_fire_sar.tif"
    post_fire_data, _ = FileHandler.read_raster(POST_FIRE_PATH)
    
    # Use current data as pre-fire, load post-fire separately
    burn_severity, burn_stats = analyzer.wildfire_burn_mapping(
        pre_fire_data=data,
        post_fire_data=post_fire_data,
        use_dual_pol=True
    )
    print(f"Burned Area: {burn_stats['burned_area_km2']:.2f} km²")
    print(f"High Severity: {burn_stats['high_severity_percent']:.2f}%")
    
    FileHandler.write_raster(output_dir / "burn_severity.tif", burn_severity, profile)
    Plotter.plot_raster(burn_severity, "Wildfire Burn Severity", cmap='YlOrRd',
                       output_path=output_dir / "burn_severity_map.png")
    """
    
    # 13. Geology & Terrain Analysis
    print("\n=== Geology & Terrain Analysis ===")
    terrain_idx, terrain_stats = analyzer.geology_terrain_analysis()
    print(f"Mean Roughness: {terrain_stats['mean']:.3f}")
    print(f"Lineament Density: {terrain_stats['lineament_density_percent']:.2f}%")
    print(f"Mountainous Area: {terrain_stats['mountainous_area_km2']:.2f} km²")
    
    FileHandler.write_raster(output_dir / "terrain_roughness.tif", terrain_idx, profile)
    Plotter.plot_raster(terrain_idx, "Terrain Roughness Index", cmap='terrain',
                       output_path=output_dir / "terrain_map.png")
    
    print(f"\n✓ All SAR analyses complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
