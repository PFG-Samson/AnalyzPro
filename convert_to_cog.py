"""
Convert any raster to COG (Cloud-Optimized GeoTIFF).

Examples:
  python convert_to_cog.py input.tif output_cog.tif
  python convert_to_cog.py input.tif output_webmercator_cog.tif --web-mercator
"""

import argparse
from pathlib import Path
from analyz.utils.cog import convert_to_cog
from analyz.utils import setup_logger


def main():
    p = argparse.ArgumentParser(description="Convert raster to COG (Cloud-Optimized GeoTIFF)")
    p.add_argument("src", help="Input raster (GeoTIFF or GDAL-readable)")
    p.add_argument("dst", help="Output COG path (.tif)")
    p.add_argument("--web-mercator", action="store_true", help="Reproject to EPSG:3857 before COG conversion")
    p.add_argument("--epsg", type=int, help="Reproject to specific EPSG (overrides --web-mercator)")
    p.add_argument("--blocksize", type=int, default=512, help="Internal tile size (default: 512)")
    p.add_argument("--compress", default="DEFLATE", choices=["DEFLATE", "LZW", "JPEG"], help="Compression")
    p.add_argument("--predictor", type=int, default=2, help="Predictor for DEFLATE/LZW (2 is good for 8-bit)")
    p.add_argument("--zlevel", type=int, default=9, help="DEFLATE compression level (1-9)")
    p.add_argument("--threads", default="ALL_CPUS", help="GDAL thread setting (e.g., ALL_CPUS)")
    p.add_argument("--resampling", default="LANCZOS", help="Overview resampling (e.g., NEAREST, BILINEAR, CUBIC, LANCZOS)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = p.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    epsg = args.epsg if args.epsg else (3857 if args.web_mercator else None)

    out = convert_to_cog(
        src,
        dst,
        reproject_epsg=epsg,
        blocksize=args.blocksize,
        compress=args.compress,
        predictor=args.predictor,
        zlevel=args.zlevel,
        num_threads=args.threads,
        overview_resampling=args.resampling,
    )

    print(f"COG created: {out}")


if __name__ == "__main__":
    main()
