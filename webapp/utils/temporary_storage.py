"""
Temporary Storage Manager for downloaded imagery.

Manages metadata tracking, cleanup, and cache management for online-downloaded
satellite imagery with integration into the analysis workflow.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading


class TemporaryStorageManager:
    """Manages temporary storage of downloaded imagery with metadata tracking."""
    
    def __init__(self, temp_folder: Path, metadata_file: Optional[Path] = None):
        """
        Initialize temporary storage manager.
        
        Args:
            temp_folder: Path to temporary downloads folder
            metadata_file: Optional path to global metadata index file
        """
        self.temp_folder = Path(temp_folder)
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        
        # Metadata index file tracks all downloads
        self.metadata_file = metadata_file or (self.temp_folder / '.cache_metadata.json')
        self.cache_index = self._load_cache_index()
        
    def register_download(self, session_id: str, scene_id: str, metadata: Dict) -> Dict:
        """
        Register a downloaded scene in cache metadata.
        
        Args:
            session_id: Download session ID
            scene_id: Scene identifier
            metadata: Scene metadata (source, date, satellite, cloud_cover, etc.)
        
        Returns:
            Updated cache entry
        """
        if 'downloads' not in self.cache_index:
            self.cache_index['downloads'] = {}
        
        cache_entry = {
            'session_id': session_id,
            'scene_id': scene_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'folder': str(self.temp_folder / session_id / scene_id),
            'size_bytes': 0,
            'kept': False,  # If user chose to keep
            'analyzed': False,
            'analysis_session_id': None
        }
        
        # Store in index
        self.cache_index['downloads'][f"{session_id}_{scene_id}"] = cache_entry
        self._save_cache_index()
        
        return cache_entry
    
    def mark_for_deletion(self, session_id: str, reason: str = "analysis_complete") -> bool:
        """
        Mark downloaded files for deletion after analysis.
        
        Args:
            session_id: Session ID to mark
            reason: Reason for deletion (analysis_complete, manual_cleanup, etc.)
        
        Returns:
            True if marked successfully
        """
        session_folder = self.temp_folder / session_id
        if not session_folder.exists():
            return False
        
        # Create marker file
        delete_marker = session_folder / '.delete_after_analysis'
        with open(delete_marker, 'w') as f:
            json.dump({
                'reason': reason,
                'marked_at': datetime.now().isoformat()
            }, f)
        
        return True
    
    def mark_as_kept(self, session_id: str) -> bool:
        """
        Mark session files to be kept (not deleted).
        
        Args:
            session_id: Session ID to keep
        
        Returns:
            True if marked successfully
        """
        key = next((k for k in self.cache_index['downloads'].keys() 
                   if k.startswith(session_id)), None)
        if key:
            self.cache_index['downloads'][key]['kept'] = True
            self._save_cache_index()
            return True
        return False
    
    def mark_as_analyzed(self, session_id: str, analysis_id: str) -> bool:
        """
        Mark session as having been analyzed.
        
        Args:
            session_id: Session ID that was analyzed
            analysis_id: Analysis session ID
        
        Returns:
            True if marked successfully
        """
        for key, entry in self.cache_index['downloads'].items():
            if entry['session_id'] == session_id:
                entry['analyzed'] = True
                entry['analysis_session_id'] = analysis_id
        self._save_cache_index()
        return True
    
    def cleanup_marked_sessions(self, force_delete_old: bool = False, 
                               days_old: int = 1) -> Dict:
        """
        Clean up sessions marked for deletion and optionally old sessions.
        
        Args:
            force_delete_old: Delete sessions older than days_old regardless
            days_old: Age threshold for force deletion
        
        Returns:
            Cleanup report with sessions deleted and sizes freed
        """
        report = {
            'deleted_sessions': [],
            'size_freed_bytes': 0,
            'size_freed_mb': 0,
            'sessions_kept': []
        }
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for session_dir in self.temp_folder.iterdir():
            if not session_dir.is_dir() or session_dir.name.startswith('.'):
                continue
            
            session_id = session_dir.name
            
            # Check if marked for deletion
            delete_marker = session_dir / '.delete_after_analysis'
            should_delete = delete_marker.exists()
            
            # Check if old
            session_created = datetime.fromtimestamp(session_dir.stat().st_mtime)
            is_old = session_created < cutoff_date
            
            # Check if kept by user
            is_kept = any(
                entry['session_id'] == session_id and entry['kept']
                for entry in self.cache_index['downloads'].values()
            )
            
            # Determine action
            if is_kept:
                report['sessions_kept'].append(session_id)
            elif should_delete or (force_delete_old and is_old):
                try:
                    size = sum(
                        f.stat().st_size for f in session_dir.rglob('*')
                        if f.is_file()
                    )
                    shutil.rmtree(session_dir)
                    report['deleted_sessions'].append(session_id)
                    report['size_freed_bytes'] += size
                except Exception as e:
                    print(f"Error deleting {session_id}: {str(e)}")
        
        report['size_freed_mb'] = report['size_freed_bytes'] / (1024 * 1024)
        
        # Remove deleted from index
        self.cache_index['downloads'] = {
            k: v for k, v in self.cache_index['downloads'].items()
            if v['session_id'] not in report['deleted_sessions']
        }
        self._save_cache_index()
        
        return report
    
    def get_cache_info(self) -> Dict:
        """
        Get information about current cache contents.
        
        Returns:
            Cache info with total size, file count, breakdown by satellite
        """
        info = {
            'total_sessions': 0,
            'total_size_bytes': 0,
            'total_size_mb': 0,
            'total_files': 0,
            'by_satellite': {},
            'by_age': {
                'less_than_1_day': 0,
                '1_to_7_days': 0,
                'more_than_7_days': 0
            },
            'kept_sessions': [],
            'sessions_awaiting_cleanup': []
        }
        
        now = datetime.now()
        
        for session_dir in self.temp_folder.iterdir():
            if not session_dir.is_dir() or session_dir.name.startswith('.'):
                continue
            
            session_id = session_dir.name
            size = sum(f.stat().st_size for f in session_dir.rglob('*') if f.is_file())
            file_count = len(list(session_dir.rglob('*')))
            
            info['total_sessions'] += 1
            info['total_size_bytes'] += size
            info['total_files'] += file_count
            
            # Check if marked for deletion
            delete_marker = session_dir / '.delete_after_analysis'
            if delete_marker.exists():
                info['sessions_awaiting_cleanup'].append(session_id)
            
            # Track kept sessions
            for entry in self.cache_index['downloads'].values():
                if entry['session_id'] == session_id and entry['kept']:
                    info['kept_sessions'].append(session_id)
            
            # Track by satellite
            for entry in self.cache_index['downloads'].values():
                if entry['session_id'] == session_id:
                    sat = entry['metadata'].get('platform', 'unknown')
                    if sat not in info['by_satellite']:
                        info['by_satellite'][sat] = {'count': 0, 'size_mb': 0}
                    info['by_satellite'][sat]['count'] += 1
                    info['by_satellite'][sat]['size_mb'] += size / (1024 * 1024)
            
            # Track by age
            session_created = datetime.fromtimestamp(session_dir.stat().st_mtime)
            age_days = (now - session_created).days
            if age_days < 1:
                info['by_age']['less_than_1_day'] += 1
            elif age_days < 7:
                info['by_age']['1_to_7_days'] += 1
            else:
                info['by_age']['more_than_7_days'] += 1
        
        info['total_size_mb'] = info['total_size_bytes'] / (1024 * 1024)
        
        return info
    
    def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific session.
        
        Args:
            session_id: Session to retrieve
        
        Returns:
            Session metadata or None if not found
        """
        for entry in self.cache_index['downloads'].values():
            if entry['session_id'] == session_id:
                return entry
        return None
    
    def _load_cache_index(self) -> Dict:
        """Load cache metadata index from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {'downloads': {}, 'last_cleanup': None}
    
    def _save_cache_index(self) -> None:
        """Save cache metadata index to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            print(f"Error saving cache index: {str(e)}")
    
    def cleanup_old_cache_async(self, days: int = 1) -> None:
        """
        Asynchronously cleanup old cache files.
        
        Args:
            days: Age threshold for cleanup
        """
        def cleanup_task():
            self.cleanup_marked_sessions(force_delete_old=True, days_old=days)
        
        thread = threading.Thread(target=cleanup_task, daemon=True)
        thread.start()
