"""
Command-line tool to preprocess satellite data archives.

Usage:
    python preprocess_satellite.py input_archive.tar output.tif
    python preprocess_satellite.py landsat.zip landsat_processed.tif
    python preprocess_satellite.py sentinel.zip sentinel_processed.tif --bands B02 B03 B04 B08
"""

import argparse
import sys
from pathlib import Path
from analyz.utils import SatellitePreprocessor, setup_logger

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess satellite data archives (Landsat/Sentinel)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and process
  python preprocess_satellite.py landsat.tar landsat.tif
  
  # Process Landsat with specific bands
  python preprocess_satellite.py landsat.tar landsat.tif --bands B2 B3 B4 B5
  
  # Process Sentinel-2 at 10m resolution
  python preprocess_satellite.py sentinel.zip sentinel.tif --resolution 10m
  
  # Process Sentinel-2 with specific bands
  python preprocess_satellite.py sentinel.zip sentinel.tif --bands B02 B03 B04 B08
        """
    )
    
    parser.add_argument('input', type=str, help='Input satellite archive (.tar, .tar.gz, .zip) or .SAFE directory')
    parser.add_argument('output', type=str, help='Output GeoTIFF file path')
    parser.add_argument('--bands', nargs='+', help='Bands to include (e.g., B2 B3 B4 for Landsat, B02 B03 B04 for Sentinel)')
    parser.add_argument('--resolution', default='10m', choices=['10m', '20m', '60m'], 
                       help='Resolution for Sentinel-2 (default: 10m)')
    parser.add_argument('--satellite', choices=['landsat', 'sentinel2', 'auto'], default='auto',
                       help='Satellite type (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(log_level)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Processing {input_path.name}...")
        
        # Build kwargs
        kwargs = {}
        if args.bands:
            kwargs['bands'] = args.bands
        if args.satellite == 'sentinel2' or 'sentinel' in input_path.name.lower():
            kwargs['resolution'] = args.resolution
        
        # Process based on type
        if args.satellite == 'auto':
            result = SatellitePreprocessor.process_auto(input_path, output_path, **kwargs)
        elif args.satellite == 'landsat':
            result = SatellitePreprocessor.process_landsat(input_path, output_path, **kwargs)
        elif args.satellite == 'sentinel2':
            result = SatellitePreprocessor.process_sentinel2(input_path, output_path, **kwargs)
        
        print(f"✓ Success! Processed file saved to: {result}")
        print(f"\nYou can now use this file in the web app or with the CLI:")
        print(f"  python webapp/app_web.py")
        print(f"  python app.py --image {result} --analysis ndvi")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
