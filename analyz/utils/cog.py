"""COG (Cloud-Optimized GeoTIFF) conversion utilities.

This module wraps GDAL commands to convert any GeoTIFF (or other GDAL-readable raster)
into a COG suitable for web mapping. It optionally reprojects to Web Mercator (EPSG:3857).

Requirements:
- GDAL must be installed and available on PATH (gdal_translate, gdalwarp).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from .logger import get_logger

logger = get_logger(__name__)


def _run(cmd: Sequence[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        logger.error(proc.stdout)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    if proc.stdout:
        logger.debug(proc.stdout)


def _check_gdal() -> None:
    for tool in ("gdal_translate", "gdalwarp"):
        if shutil.which(tool) is None:
            raise EnvironmentError(f"{tool} not found on PATH. Please install GDAL and ensure {tool} is available.")


def convert_to_cog(
    src_path: os.PathLike | str,
    dst_path: os.PathLike | str,
    *,
    reproject_epsg: Optional[int] = None,
    blocksize: int = 512,
    compress: str = "DEFLATE",
    predictor: int = 2,
    zlevel: int = 9,
    num_threads: str = "ALL_CPUS",
    overview_resampling: str = "LANCZOS",
    bigtiff: str = "IF_SAFER",
) -> Path:
    """
    Convert a raster to a Cloud-Optimized GeoTIFF (COG).

    Args:
        src_path: Input raster path (GeoTIFF or any GDAL-readable)
        dst_path: Output COG path
        reproject_epsg: Optional EPSG code to reproject prior to COG creation (e.g., 3857)
        blocksize: Internal tile size (both X and Y)
        compress: Compression (DEFLATE, LZW, JPEG for 8-bit RGB, etc.)
        predictor: Predictor for DEFLATE/LZW (2 for horizontal differencing works for 8-bit)
        zlevel: DEFLATE compression level (1-9)
        num_threads: GDAL threads setting (e.g., ALL_CPUS)
        overview_resampling: Resampling used for overviews
        bigtiff: BIGTIFF policy (YES/NO/IF_NEEDED/IF_SAFER)

    Returns:
        Path to the created COG
    """
    _check_gdal()

    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix="cog_")
    tmp_reproj = Path(tmpdir) / "reprojected.tif"

    try:
        intermediate_src = src

        # Optional reprojection step (e.g., to EPSG:3857 for web maps)
        if reproject_epsg is not None:
            warp_cmd = [
                "gdalwarp",
                "-t_srs",
                f"EPSG:{reproject_epsg}",
                "-r",
                "cubic",
                "-multi",
                "-wo",
                f"NUM_THREADS={num_threads}",
                # Create a tiled, compressed intermediate GeoTIFF
                "-co",
                "TILED=YES",
                "-co",
                f"BLOCKXSIZE={blocksize}",
                "-co",
                f"BLOCKYSIZE={blocksize}",
                "-co",
                f"COMPRESS={compress}",
                "-co",
                f"PREDICTOR={predictor}",
                "-co",
                f"BIGTIFF={bigtiff}",
                str(src),
                str(tmp_reproj),
            ]
            _run(warp_cmd)
            intermediate_src = tmp_reproj

        # Create COG
        # Use GDAL COG driver which builds internal overviews and organizes layout
        translate_cmd = [
            "gdal_translate",
            "-of",
            "COG",
            "-co",
            f"BLOCKSIZE={blocksize}",
            "-co",
            f"COMPRESS={compress}",
            "-co",
            f"PREDICTOR={predictor}",
            "-co",
            f"ZLEVEL={zlevel}",
            "-co",
            f"BIGTIFF={bigtiff}",
            "-co",
            f"NUM_THREADS={num_threads}",
            "-co",
            "OVERVIEWS=AUTO",
            "-co",
            f"RESAMPLING={overview_resampling}",
            str(intermediate_src),
            str(dst),
        ]
        _run(translate_cmd)

        logger.info("COG created: %s", dst)
        return dst
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
