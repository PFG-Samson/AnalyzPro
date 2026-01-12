"""
Thumbnail and metadata generator for STAC scenes.

Generates preview thumbnails and extracts metadata for display
in the web interface before downloading full imagery.
"""

import json
import base64
from pathlib import Path
from io import BytesIO
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from urllib.parse import urljoin

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analyz.utils.logger import setup_logger

logger = setup_logger("INFO")


class ThumbnailGenerator:
    """Generate and cache thumbnails for STAC scenes."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize thumbnail generator.
        
        Args:
            cache_dir: Directory to cache generated thumbnails
        """
        self.cache_dir = cache_dir or Path(__file__).parent.parent / 'static' / 'thumbnails'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_size = (300, 200)
        self.quality = 85
    
    def generate_thumbnail(
        self, 
        scene: Dict, 
        quicklook_url: Optional[str] = None,
        rgb_urls: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Generate thumbnail for a STAC scene.
        
        Args:
            scene: STAC scene GeoJSON feature
            quicklook_url: URL to quicklook image (faster)
            rgb_urls: URLs to RGB band data (slower, better quality)
        
        Returns:
            Dict with thumbnail info:
            {
                'success': bool,
                'base64': str (base64 encoded image),
                'url': str (cached file URL if saved),
                'error': str (if failed)
            }
        """
        try:
            scene_id = scene.get('id', 'unknown')
            cache_file = self.cache_dir / f"{scene_id}_thumb.jpg"
            
            # Return cached thumbnail if exists
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    img_data = f.read()
                return {
                    'success': True,
                    'base64': base64.b64encode(img_data).decode('utf-8'),
                    'url': f'/static/thumbnails/{scene_id}_thumb.jpg'
                }
            
            # Try quicklook first (fastest)
            if quicklook_url:
                img = self._fetch_image(quicklook_url)
                if img:
                    return self._save_and_encode(cache_file, img, scene_id)
            
            # Try RGB composition (slower but better quality)
            if rgb_urls:
                img = self._create_rgb_composite(rgb_urls)
                if img:
                    return self._save_and_encode(cache_file, img, scene_id)
            
            # Fallback: Generate placeholder with metadata
            img = self._create_placeholder(scene)
            return self._save_and_encode(cache_file, img, scene_id)
            
        except Exception as e:
            logger.error(f"Thumbnail generation failed for {scene.get('id')}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _fetch_image(self, url: str) -> Optional[Image.Image]:
        """Fetch image from URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            # Resize to standard dimensions
            img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            logger.debug(f"Failed to fetch image from {url}: {e}")
            return None
    
    def _create_rgb_composite(self, rgb_urls: Dict[str, str]) -> Optional[Image.Image]:
        """Create RGB composite from individual band URLs."""
        try:
            bands = {}
            for band, url in rgb_urls.items():
                if band in ['red', 'green', 'blue']:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    # Assume GeoTIFF - would need rasterio for proper handling
                    # For now, just try basic image formats
                    try:
                        img = Image.open(BytesIO(response.content))
                        bands[band] = np.array(img)
                    except:
                        logger.debug(f"Could not parse {band} band image")
                        return None
            
            if len(bands) == 3:
                # Stack bands into RGB
                r = self._normalize_band(bands.get('red'))
                g = self._normalize_band(bands.get('green'))
                b = self._normalize_band(bands.get('blue'))
                rgb = np.stack([r, g, b], axis=2)
                img = Image.fromarray(rgb.astype('uint8'))
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                return img
        except Exception as e:
            logger.debug(f"RGB composite creation failed: {e}")
        return None
    
    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        """Normalize band data to 0-255."""
        if band is None:
            return np.zeros_like(band)
        band = band.astype(float)
        vmin, vmax = np.percentile(band[band > 0], [2, 98]) if np.any(band > 0) else (0, 1)
        normalized = 255 * (band - vmin) / (vmax - vmin + 1e-8)
        return np.clip(normalized, 0, 255)
    
    def _create_placeholder(self, scene: Dict) -> Image.Image:
        """Create informative placeholder with scene metadata."""
        img = Image.new('RGB', self.thumbnail_size, color=(240, 248, 255))
        draw = ImageDraw.Draw(img)
        
        # Get metadata
        props = scene.get('properties', {})
        platform = props.get('platform_id', 'Unknown')
        date = props.get('datetime', 'N/A')
        cloud = props.get('eo_cloud_cover', 'N/A')
        
        # Format date
        if date != 'N/A':
            try:
                dt = datetime.fromisoformat(date.replace('Z', '+00:00'))
                date = dt.strftime('%Y-%m-%d')
            except:
                pass
        
        # Draw text
        try:
            # Try to use default font, fall back to default if not available
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_large = font_small = ImageFont.load_default()
        
        y = 20
        draw.text((10, y), "🛰️ " + platform, fill=(70, 130, 180), font=font_large)
        y += 40
        draw.text((10, y), f"📅 {date}", fill=(70, 130, 180), font=font_small)
        y += 30
        draw.text((10, y), f"☁️ Cloud: {cloud if cloud != 'N/A' else 'N/A'}%", 
                 fill=(70, 130, 180), font=font_small)
        
        return img
    
    def _save_and_encode(self, cache_file: Path, img: Image.Image, scene_id: str) -> Dict:
        """Save image to cache and return as base64."""
        try:
            img.save(cache_file, format='JPEG', quality=self.quality)
            with open(cache_file, 'rb') as f:
                img_data = f.read()
            return {
                'success': True,
                'base64': base64.b64encode(img_data).decode('utf-8'),
                'url': f'/static/thumbnails/{scene_id}_thumb.jpg'
            }
        except Exception as e:
            logger.error(f"Failed to save thumbnail: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_cache(self, older_than_days: int = 7):
        """Clear old cached thumbnails."""
        import time
        current_time = time.time()
        cutoff = current_time - (older_than_days * 86400)
        
        for thumb_file in self.cache_dir.glob('*_thumb.jpg'):
            if thumb_file.stat().st_mtime < cutoff:
                try:
                    thumb_file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {thumb_file}: {e}")


class SceneMetadataExtractor:
    """Extract and format metadata from STAC scenes for display."""
    
    @staticmethod
    def extract_metadata(scene: Dict) -> Dict:
        """
        Extract key metadata from STAC scene.
        
        Returns:
            Dict with extracted metadata
        """
        props = scene.get('properties', {})
        
        return {
            'id': scene.get('id', 'unknown'),
            'platform': props.get('platform_id', 'Unknown'),
            'instrument': props.get('instruments', [None])[0] if props.get('instruments') else 'Unknown',
            'datetime': props.get('datetime') or props.get('start_datetime', 'Unknown'),
            'cloud_cover': props.get('eo_cloud_cover', props.get('cloud_cover', None)),
            'area_km2': props.get('area', 0) / 1000000,
            'sun_elevation': props.get('sun_elevation', None),
            'view_angle': props.get('view:off_nadir', None),
            'product_type': props.get('product_type', props.get('type', 'Unknown')),
            'collection': props.get('collection', 'Unknown'),
            'geometry': scene.get('geometry'),
            'assets': {
                name: {
                    'title': asset.get('title', name),
                    'type': asset.get('type', 'unknown'),
                    'roles': asset.get('roles', [])
                }
                for name, asset in scene.get('assets', {}).items()
            }
        }
    
    @staticmethod
    def format_metadata_display(metadata: Dict) -> str:
        """Format metadata as readable HTML string."""
        html = f"""
        <div class="metadata-display">
            <h4>{metadata['platform']}</h4>
            <dl>
                <dt>Date:</dt>
                <dd>{metadata['datetime']}</dd>
                <dt>Instrument:</dt>
                <dd>{metadata['instrument']}</dd>
                <dt>Cloud Cover:</dt>
                <dd>{metadata['cloud_cover']:.1f}% if metadata['cloud_cover'] else 'N/A'</dd>
                <dt>Area:</dt>
                <dd>{metadata['area_km2']:.2f} km²</dd>
                <dt>Collection:</dt>
                <dd>{metadata['collection']}</dd>
        """
        
        if metadata.get('sun_elevation'):
            html += f"<dt>Sun Elevation:</dt><dd>{metadata['sun_elevation']:.1f}°</dd>"
        
        if metadata.get('view_angle'):
            html += f"<dt>View Angle:</dt><dd>{metadata['view_angle']:.1f}°</dd>"
        
        html += "</dl></div>"
        return html


class ProgressTracker:
    """Track progress of individual scene downloads."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize progress tracker.
        
        Args:
            storage_dir: Directory to store progress files
        """
        self.storage_dir = storage_dir or Path(__file__).parent.parent / 'download_progress'
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize_download(
        self, 
        download_id: str, 
        scene_ids: List[str],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Initialize download tracking for a batch of scenes.
        
        Args:
            download_id: Unique download session ID
            scene_ids: List of scene IDs being downloaded
            metadata: Optional metadata about the download
        
        Returns:
            Path to progress file
        """
        progress_file = self.storage_dir / f"{download_id}_progress.json"
        
        progress_data = {
            'download_id': download_id,
            'start_time': datetime.now().isoformat(),
            'total_scenes': len(scene_ids),
            'scenes': {
                scene_id: {
                    'status': 'queued',  # queued, downloading, completed, failed
                    'progress': 0,
                    'bytes_downloaded': 0,
                    'bytes_total': 0,
                    'error': None,
                    'start_time': None,
                    'end_time': None
                }
                for scene_id in scene_ids
            },
            'metadata': metadata or {}
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        return str(progress_file)
    
    def update_scene_progress(
        self, 
        download_id: str, 
        scene_id: str,
        status: str,
        progress: int = None,
        bytes_downloaded: int = None,
        bytes_total: int = None,
        error: str = None
    ):
        """
        Update progress for a specific scene.
        
        Args:
            download_id: Download session ID
            scene_id: Scene ID being downloaded
            status: Current status (queued, downloading, completed, failed)
            progress: Progress percentage (0-100)
            bytes_downloaded: Bytes downloaded so far
            bytes_total: Total bytes to download
            error: Error message if failed
        """
        progress_file = self.storage_dir / f"{download_id}_progress.json"
        
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
            
            if scene_id in data['scenes']:
                scene = data['scenes'][scene_id]
                scene['status'] = status
                
                if progress is not None:
                    scene['progress'] = progress
                if bytes_downloaded is not None:
                    scene['bytes_downloaded'] = bytes_downloaded
                if bytes_total is not None:
                    scene['bytes_total'] = bytes_total
                if error is not None:
                    scene['error'] = error
                
                if status == 'downloading' and scene['start_time'] is None:
                    scene['start_time'] = datetime.now().isoformat()
                elif status in ['completed', 'failed'] and scene['end_time'] is None:
                    scene['end_time'] = datetime.now().isoformat()
            
            with open(progress_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to update progress for {scene_id}: {e}")
    
    def get_progress(self, download_id: str) -> Optional[Dict]:
        """
        Get current progress for a download.
        
        Returns:
            Dict with complete progress information, or None if not found
        """
        progress_file = self.storage_dir / f"{download_id}_progress.json"
        
        try:
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read progress file: {e}")
        
        return None
    
    def get_summary(self, download_id: str) -> Optional[Dict]:
        """
        Get summary statistics for a download.
        
        Returns:
            Dict with overall progress statistics
        """
        progress_data = self.get_progress(download_id)
        if not progress_data:
            return None
        
        scenes = progress_data['scenes']
        statuses = [s['status'] for s in scenes.values()]
        
        total_bytes = sum(s['bytes_total'] for s in scenes.values())
        downloaded_bytes = sum(s['bytes_downloaded'] for s in scenes.values())
        
        return {
            'download_id': download_id,
            'total_scenes': len(scenes),
            'queued': statuses.count('queued'),
            'downloading': statuses.count('downloading'),
            'completed': statuses.count('completed'),
            'failed': statuses.count('failed'),
            'overall_progress': (
                statuses.count('completed') / len(scenes) * 100 
                if scenes else 0
            ),
            'bytes_downloaded': downloaded_bytes,
            'bytes_total': total_bytes,
            'start_time': progress_data['start_time'],
            'elapsed_seconds': (
                (datetime.fromisoformat(progress_data['metadata'].get('end_time'))
                 - datetime.fromisoformat(progress_data['start_time'])).total_seconds()
                if progress_data['metadata'].get('end_time')
                else (datetime.now() - datetime.fromisoformat(progress_data['start_time'])).total_seconds()
            )
        }
    
    def cleanup(self, download_id: str, older_than_days: int = 7):
        """Clean up old progress files."""
        import time
        progress_file = self.storage_dir / f"{download_id}_progress.json"
        
        if progress_file.exists():
            mtime = progress_file.stat().st_mtime
            current_time = time.time()
            
            if current_time - mtime > (older_than_days * 86400):
                try:
                    progress_file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete progress file: {e}")
