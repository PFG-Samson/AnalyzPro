"""
STAC Imagery Download Manager

Handles fetching, downloading, and managing satellite imagery from STAC catalogs.
Provides progress tracking and integration with analysis pipeline.
"""

import os
import json
import requests
import threading
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable
from urllib.parse import urlparse
import pystac_client

from .logger import get_logger

logger = get_logger("STAC_DOWNLOADER")


class STACDownloadManager:
    """Manages STAC imagery downloads with progress tracking and storage management."""
    
    def __init__(self, temp_folder: Path, permanent_folder: Path, chunk_size: int = 8192):
        """
        Initialize download manager.
        
        Args:
            temp_folder: Path to temporary downloads folder
            permanent_folder: Path to permanent storage folder
            chunk_size: Size of download chunks in bytes
        """
        self.temp_folder = Path(temp_folder)
        self.permanent_folder = Path(permanent_folder)
        self.chunk_size = chunk_size
        
        # Ensure folders exist
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        self.permanent_folder.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, session_id: str, scenes: List[Dict], keep_imagery: bool = False) -> Dict:
        """
        Create a download session.
        
        Args:
            session_id: Unique session identifier
            scenes: List of scene objects from STAC query
            keep_imagery: Whether to keep imagery after analysis
        
        Returns:
            Session info dict
        """
        session_folder = self.temp_folder / session_id
        session_folder.mkdir(exist_ok=True)
        
        session_info = {
            'session_id': session_id,
            'folder': str(session_folder),
            'scenes': scenes,
            'keep_imagery': keep_imagery,
            'status': 'initialized',
            'progress': 0,
            'total_size': 0,
            'downloaded_size': 0,
            'downloaded_files': [],
            'errors': [],
            'created': datetime.now().isoformat(),
            'metadata': {}
        }
        
        self.sessions[session_id] = session_info
        self._save_session_metadata(session_id)
        
        logger.info(f"Session created: {session_id}")
        return session_info
    
    def download_scene(self, session_id: str, scene: Dict, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Download a single scene from STAC.
        
        Args:
            session_id: Download session ID
            scene: Scene object with properties and assets
            progress_callback: Function to call with progress updates
        
        Returns:
            Download result with file paths and metadata
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session_folder = Path(session['folder'])
        result = {
            'scene_id': scene.get('id', 'unknown'),
            'success': False,
            'files': [],
            'errors': []
        }
        
        try:
            # Get scene metadata
            props = scene.get('properties', {})
            datetime_str = props.get('datetime', 'unknown')
            platform = props.get('platform_id', 'unknown')
            
            # Create scene subfolder
            scene_folder = session_folder / f"{result['scene_id']}"
            scene_folder.mkdir(exist_ok=True)
            
            # Save scene metadata
            metadata = {
                'scene_id': result['scene_id'],
                'datetime': datetime_str,
                'platform': platform,
                'properties': props,
                'downloaded': datetime.now().isoformat()
            }
            
            metadata_file = scene_folder / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            result['files'].append(str(metadata_file))
            
            # Download assets
            assets = scene.get('assets', {})
            for asset_key, asset_info in assets.items():
                try:
                    asset_url = asset_info.get('href') if isinstance(asset_info, dict) else asset_info
                    if not asset_url:
                        continue
                    
                    # Download asset
                    file_path = self._download_asset(
                        asset_url,
                        scene_folder,
                        asset_key,
                        progress_callback
                    )
                    
                    if file_path:
                        result['files'].append(str(file_path))
                        logger.info(f"Downloaded: {asset_key} from {result['scene_id']}")
                
                except Exception as e:
                    error_msg = f"Failed to download asset {asset_key}: {str(e)}"
                    result['errors'].append(error_msg)
                    logger.error(error_msg)
            
            result['success'] = bool(result['files'])
            session['downloaded_files'].extend(result['files'])
            
        except Exception as e:
            error_msg = f"Scene download failed: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
            session['errors'].append(error_msg)
        
        self._save_session_metadata(session_id)
        return result
    
    def _download_asset(self, url: str, dest_folder: Path, asset_name: str, 
                       progress_callback: Optional[Callable] = None) -> Optional[Path]:
        """
        Download a single asset file.
        
        Args:
            url: Asset URL
            dest_folder: Destination folder
            asset_name: Asset identifier
            progress_callback: Progress callback function
        
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            # Get filename from URL
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename:
                filename = f"{asset_name}"
            
            file_path = dest_folder / filename
            
            # Download with progress tracking
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size:
                            progress_callback({
                                'asset': asset_name,
                                'downloaded': downloaded,
                                'total': total_size,
                                'percent': (downloaded / total_size) * 100
                            })
            
            logger.info(f"Asset downloaded: {filename} ({downloaded} bytes)")
            return file_path
        
        except requests.RequestException as e:
            logger.error(f"Download error for {url}: {str(e)}")
            return None
        except IOError as e:
            logger.error(f"File write error: {str(e)}")
            return None
    
    def finalize_session(self, session_id: str) -> Dict:
        """
        Finalize download session.
        
        Updates session status and prepares for analysis.
        
        Args:
            session_id: Session ID
        
        Returns:
            Finalized session info
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        session['status'] = 'completed'
        session['completed'] = datetime.now().isoformat()
        
        # Calculate total size
        session_folder = Path(session['folder'])
        total_size = sum(f.stat().st_size for f in session_folder.rglob('*') if f.is_file())
        session['total_size'] = total_size
        
        self._save_session_metadata(session_id)
        logger.info(f"Session finalized: {session_id} ({total_size} bytes)")
        
        return session
    
    def move_to_permanent(self, session_id: str, sub_folder: Optional[str] = None) -> Path:
        """
        Move downloaded imagery from temp to permanent storage.
        
        Args:
            session_id: Session ID
            sub_folder: Optional subfolder name in permanent storage
        
        Returns:
            Path to permanent storage location
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        temp_folder = Path(session['folder'])
        if not temp_folder.exists():
            raise ValueError(f"Temp folder not found: {temp_folder}")
        
        # Create permanent storage path
        if sub_folder:
            permanent_path = self.permanent_folder / sub_folder
        else:
            permanent_path = self.permanent_folder / session_id
        
        permanent_path.mkdir(parents=True, exist_ok=True)
        
        # Move files
        try:
            for file_path in temp_folder.rglob('*'):
                if file_path.is_file():
                    rel_path = file_path.relative_to(temp_folder)
                    dest_path = permanent_path / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(dest_path))
            
            logger.info(f"Session moved to permanent: {session_id} → {permanent_path}")
            session['permanent_path'] = str(permanent_path)
            self._save_session_metadata(session_id)
            
            return permanent_path
        
        except Exception as e:
            logger.error(f"Failed to move to permanent: {str(e)}")
            raise
    
    def cleanup_session(self, session_id: str, force: bool = False) -> bool:
        """
        Clean up session temporary files.
        
        Args:
            session_id: Session ID
            force: Force cleanup even if keep_imagery is True
        
        Returns:
            True if cleanup successful
        """
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found for cleanup: {session_id}")
            return False
        
        # Check if should keep
        if session.get('keep_imagery') and not force:
            logger.info(f"Session marked to keep, skipping cleanup: {session_id}")
            return False
        
        temp_folder = Path(session['folder'])
        if temp_folder.exists():
            try:
                shutil.rmtree(temp_folder)
                logger.info(f"Session cleaned up: {session_id}")
                
                # Remove from sessions dict
                if session_id in self.sessions:
                    del self.sessions[session_id]
                
                return True
            except Exception as e:
                logger.error(f"Cleanup failed for {session_id}: {str(e)}")
                return False
        
        return True
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information."""
        return self.sessions.get(session_id)
    
    def _save_session_metadata(self, session_id: str) -> None:
        """Save session metadata to file."""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        metadata_file = Path(session['folder']) / '.session_metadata.json'
        try:
            with open(metadata_file, 'w') as f:
                # Create serializable copy
                serializable = {
                    k: v for k, v in session.items()
                    if k not in ['scenes']  # scenes might not be serializable
                }
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save metadata: {str(e)}")
    
    def cleanup_old_sessions(self, hours: int = 24) -> int:
        """
        Clean up old temporary sessions.
        
        Args:
            hours: Sessions older than this many hours will be deleted
        
        Returns:
            Number of sessions cleaned up
        """
        from datetime import timedelta
        
        now = datetime.now()
        cutoff = now - timedelta(hours=hours)
        cleaned = 0
        
        for folder in self.temp_folder.iterdir():
            if not folder.is_dir():
                continue
            
            try:
                mtime = datetime.fromtimestamp(folder.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(folder)
                    cleaned += 1
                    logger.info(f"Cleaned old session: {folder.name}")
            except Exception as e:
                logger.warning(f"Could not clean {folder.name}: {str(e)}")
        
        return cleaned


def download_imagery_async(manager: STACDownloadManager, session_id: str, 
                          scenes: List[Dict], status_callback: Optional[Callable] = None) -> None:
    """
    Download imagery in background thread.
    
    Args:
        manager: STACDownloadManager instance
        session_id: Session ID
        scenes: List of scenes to download
        status_callback: Function to call with status updates
    """
    try:
        total_scenes = len(scenes)
        
        for idx, scene in enumerate(scenes):
            try:
                # Download scene
                result = manager.download_scene(session_id, scene)
                
                # Update progress
                progress = {
                    'type': 'scene_complete',
                    'scene_id': scene.get('id', 'unknown'),
                    'current': idx + 1,
                    'total': total_scenes,
                    'percent': ((idx + 1) / total_scenes) * 100,
                    'success': result['success']
                }
                
                if status_callback:
                    status_callback(progress)
            
            except Exception as e:
                logger.error(f"Scene download error: {str(e)}")
                if status_callback:
                    status_callback({
                        'type': 'error',
                        'scene_id': scene.get('id', 'unknown'),
                        'error': str(e)
                    })
        
        # Finalize
        manager.finalize_session(session_id)
        
        if status_callback:
            status_callback({
                'type': 'complete',
                'session_id': session_id,
                'percent': 100
            })
    
    except Exception as e:
        logger.error(f"Download thread error: {str(e)}")
        if status_callback:
            status_callback({
                'type': 'fatal_error',
                'error': str(e)
            })
