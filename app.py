"""Main application CLI for Analyz."""

import argparse
import yaml
from pathlib import Path
import sys

from analyz import (
    OpticalAnalyzer, SARAnalyzer, BoundaryHandler, Preprocessor,
    Plotter, InsightsGenerator, setup_logger, FileHandler
)


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_optical_analysis(args, config):
    """Run optical image analysis."""
    logger = setup_logger(
        config['app']['log_level'],
        f"outputs/{args.analysis}_log.txt"
    )
    
    # Load image
    logger.info(f"Loading optical image: {args.image}")
    data, profile = FileHandler.read_raster(args.image)
    
    # Apply boundary if provided
    if args.boundary:
        logger.info(f"Applying study area boundary: {args.boundary}")
        boundary_handler = BoundaryHandler(args.boundary)
        data, profile = boundary_handler.clip_array(data, profile)
    
    # Define band indices (customize based on your data)
    band_indices = config['optical']['default_bands']
    if args.band_indices:
        # Parse custom band indices: "red:0,nir:3,green:1,blue:2"
        band_indices = dict(item.split(':') for item in args.band_indices.split(','))
        band_indices = {k: int(v) for k, v in band_indices.items()}
    
    # Initialize analyzer
    analyzer = OpticalAnalyzer(data, band_indices, sensor=args.sensor)
    
    # Run analysis
    if args.analysis == 'ndvi':
        result, stats = analyzer.ndvi()
        cmap = 'RdYlGn'
    elif args.analysis == 'ndwi':
        result, stats = analyzer.ndwi()
        cmap = 'Blues'
    elif args.analysis == 'ndbi':
        result, stats = analyzer.ndbi()
        cmap = 'Reds'
    elif args.analysis == 'evi':
        result, stats = analyzer.evi()
        cmap = 'RdYlGn'
    elif args.analysis == 'savi':
        result, stats = analyzer.savi()
        cmap = 'RdYlGn'
    elif args.analysis == 'classification':
        result, stats = analyzer.classify_kmeans(n_clusters=args.n_clusters or 5)
        cmap = 'tab10'
    else:
        # Try spectral index by name
        indices = set(analyzer.available_indices().keys())
        if args.analysis.lower() in indices:
            result, stats = analyzer.compute_index(args.analysis)
            # Choose colormap heuristically
            name = args.analysis.lower()
            if 'ndwi' in name or 'mndwi' in name or 'water' in name:
                cmap = 'Blues'
            elif 'nbr' in name or 'bai' in name or 'burn' in name:
                cmap = 'inferno'
            elif 'ndbi' in name or 'bsi' in name:
                cmap = 'Greys'
            else:
                cmap = 'RdYlGn'
        else:
            logger.error(f"Unknown analysis type: {args.analysis}")
            sys.exit(1)
    
    # Save result
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = output_dir / f"{args.analysis}_result.tif"
    FileHandler.write_raster(result_path, result, profile)
    
    # Generate visualizations
    plot_path = output_dir / f"{args.analysis}_plot.png"
    Plotter.plot_raster(result, title=f"{args.analysis.upper()} Analysis",
                       cmap=cmap, output_path=plot_path)
    
    hist_path = output_dir / f"{args.analysis}_histogram.png"
    Plotter.plot_histogram(result, title=f"{args.analysis.upper()} Distribution",
                          output_path=hist_path)
    
    # Generate insights
    if args.analysis == 'ndvi':
        insights = InsightsGenerator.generate_ndvi_insights(result, stats)
    elif args.analysis == 'ndwi':
        insights = InsightsGenerator.generate_ndwi_insights(result, stats)
    else:
        insights = {'analysis_type': args.analysis.upper(), 
                   'summary': f"Analysis complete. See statistics for details.",
                   'stats': stats}
    
    # Save insights report
    report_path = output_dir / f"{args.analysis}_insights.txt"
    report = InsightsGenerator.format_insights_report(insights, report_path)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    print("\n" + report)
    print(f"\nResults saved to: {output_dir}")


def run_sar_analysis(args, config):
    """Run SAR image analysis."""
    logger = setup_logger(
        config['app']['log_level'],
        f"outputs/{args.analysis}_log.txt"
    )
    
    # Load image
    logger.info(f"Loading SAR image: {args.image}")
    data, profile = FileHandler.read_raster(args.image)
    
    # Apply boundary if provided
    if args.boundary:
        logger.info(f"Applying study area boundary: {args.boundary}")
        boundary_handler = BoundaryHandler(args.boundary)
        data, profile = boundary_handler.clip_array(data, profile)
    
    # Initialize analyzer
    analyzer = SARAnalyzer(data, profile=profile)
    
    # Run analysis
    if args.analysis == 'lee_filter':
        result, stats = analyzer.lee_filter(
            window_size=args.window_size or 5,
            num_looks=args.num_looks or 1
        )
        cmap = 'gray'
    elif args.analysis == 'frost_filter':
        result, stats = analyzer.frost_filter(window_size=args.window_size or 5)
        cmap = 'gray'
    elif args.analysis == 'median_filter':
        result, stats = analyzer.median_filter(window_size=args.window_size or 5)
        cmap = 'gray'
    elif args.analysis == 'backscatter':
        result, stats = analyzer.backscatter_analysis()
        cmap = 'viridis'
    elif args.analysis == 'flood_mapping':
        result, stats = analyzer.flood_mapping()
        cmap = 'Blues'
    else:
        logger.error(f"Unknown SAR analysis type: {args.analysis}")
        sys.exit(1)
    
    # Save result
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = output_dir / f"{args.analysis}_result.tif"
    FileHandler.write_raster(result_path, result, profile)
    
    # Generate visualizations
    plot_path = output_dir / f"{args.analysis}_plot.png"
    Plotter.plot_raster(result, title=f"{args.analysis.upper()} Analysis",
                       cmap=cmap, output_path=plot_path)
    
    # Generate insights
    insights = InsightsGenerator.generate_sar_insights(args.analysis.upper(), result, stats)
    
    # Save insights report
    report_path = output_dir / f"{args.analysis}_insights.txt"
    report = InsightsGenerator.format_insights_report(insights, report_path)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    print("\n" + report)
    print(f"\nResults saved to: {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyz - Optical and SAR Image Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--image-type', choices=['optical', 'sar'], required=True,
                       help='Image type')
    parser.add_argument('--analysis', required=True,
                       help='Analysis type (ndvi, ndwi, ndbi, evi, savi, classification, '
                            'lee_filter, frost_filter, median_filter, backscatter, flood_mapping)')
    parser.add_argument('--boundary', help='Study area boundary file (GeoJSON, Shapefile)')
    parser.add_argument('--output', default='outputs', help='Output directory')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    # Optical-specific options
    parser.add_argument('--band-indices', help='Band indices mapping (e.g., "red:0,nir:3,green:1,blue:2")')
    parser.add_argument('--sensor', choices=['sentinel2','landsat8','landsat9'], help='Sensor for validation (optional)')
    parser.add_argument('--n-clusters', type=int, help='Number of clusters for classification')
    
    # SAR-specific options
    parser.add_argument('--window-size', type=int, help='Filter window size')
    parser.add_argument('--num-looks', type=int, help='Number of looks for SAR')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run appropriate analysis
    if args.image_type == 'optical':
        run_optical_analysis(args, config)
    elif args.image_type == 'sar':
        run_sar_analysis(args, config)


if __name__ == '__main__':
    main()
