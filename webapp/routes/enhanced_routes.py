"""
Flask API endpoints for enhanced online imagery workflow.

Supports thumbnail generation, input validation, progress tracking, and metadata display.
Add these routes to the main Flask app.
"""

from flask import jsonify, request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from webapp.utils.thumbnail_generator import (
    ThumbnailGenerator, SceneMetadataExtractor, ProgressTracker
)
from webapp.utils.input_validator import InputValidator, ValidationErrorFormatter

# Initialize utilities
thumbnail_generator = ThumbnailGenerator()
metadata_extractor = SceneMetadataExtractor()
progress_tracker = ProgressTracker()


def register_enhanced_routes(app):
    """Register all enhanced workflow routes to Flask app."""
    
    # ===== THUMBNAIL ENDPOINTS =====
    
    @app.route('/api/generate-thumbnail', methods=['POST'])
    def generate_thumbnail():
        """
        Generate thumbnail for a STAC scene.
        
        JSON body:
        {
            'scene_id': str,
            'scene': dict (STAC feature),
            'quicklook_url': str (optional)
        }
        """
        try:
            data = request.get_json()
            scene_id = data.get('scene_id')
            scene = data.get('scene', {})
            quicklook_url = data.get('quicklook_url')
            
            if not scene_id:
                return jsonify({'success': False, 'error': 'Missing scene_id'}), 400
            
            result = thumbnail_generator.generate_thumbnail(
                scene, 
                quicklook_url=quicklook_url
            )
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    
    @app.route('/api/scene-metadata', methods=['POST'])
    def get_scene_metadata():
        """
        Extract and format metadata for a scene.
        
        JSON body:
        {
            'scene': dict (STAC feature)
        }
        """
        try:
            data = request.get_json()
            scene = data.get('scene')
            
            if not scene:
                return jsonify({'success': False, 'error': 'Missing scene'}), 400
            
            metadata = metadata_extractor.extract_metadata(scene)
            html_display = metadata_extractor.format_metadata_display(metadata)
            
            return jsonify({
                'success': True,
                'metadata': metadata,
                'html': html_display
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    
    # ===== VALIDATION ENDPOINTS =====
    
    @app.route('/api/validate-roi', methods=['POST'])
    def validate_roi():
        """
        Validate ROI geometry.
        
        JSON body:
        {
            'geometry': dict (GeoJSON geometry)
        }
        """
        try:
            data = request.get_json()
            geometry = data.get('geometry')
            
            is_valid, error_msg, area_km2 = InputValidator.validate_roi(geometry)
            
            result = {
                'valid': is_valid,
                'error': error_msg,
                'area_km2': area_km2
            }
            
            if is_valid:
                result['formatted'] = ValidationErrorFormatter.format_info(
                    f"✓ ROI valid ({area_km2:.2f} km²)"
                )
            else:
                result['formatted'] = ValidationErrorFormatter.format_error(error_msg)
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({
                'valid': False,
                'error': str(e),
                'formatted': ValidationErrorFormatter.format_error(str(e))
            }), 500
    
    
    @app.route('/api/validate-dates', methods=['POST'])
    def validate_dates():
        """
        Validate date range.
        
        JSON body:
        {
            'start_date': str (YYYY-MM-DD),
            'end_date': str (YYYY-MM-DD),
            'satellite': str
        }
        """
        try:
            data = request.get_json()
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            satellite = data.get('satellite', 'sentinel2')
            
            is_valid, error_msg = InputValidator.validate_date_range(
                start_date, end_date, satellite
            )
            
            result = {
                'valid': is_valid,
                'error': error_msg
            }
            
            if not is_valid:
                result['formatted'] = ValidationErrorFormatter.format_error(error_msg)
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({
                'valid': False,
                'error': str(e),
                'formatted': ValidationErrorFormatter.format_error(str(e))
            }), 500
    
    
    @app.route('/api/validate-cloud-coverage', methods=['POST'])
    def validate_cloud_coverage():
        """
        Validate cloud coverage parameter.
        
        JSON body:
        {
            'cloud_coverage': float,
            'satellite': str
        }
        """
        try:
            data = request.get_json()
            cloud_coverage = data.get('cloud_coverage')
            satellite = data.get('satellite', 'sentinel2')
            
            is_valid, error_msg = InputValidator.validate_cloud_coverage(
                cloud_coverage, satellite
            )
            
            result = {
                'valid': is_valid,
                'error': error_msg
            }
            
            if not is_valid:
                result['formatted'] = ValidationErrorFormatter.format_error(error_msg)
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({
                'valid': False,
                'error': str(e),
                'formatted': ValidationErrorFormatter.format_error(str(e))
            }), 500
    
    
    @app.route('/api/validate-search', methods=['POST'])
    def validate_search():
        """
        Validate complete search parameters.
        
        JSON body:
        {
            'geometry': dict,
            'start_date': str,
            'end_date': str,
            'satellite': str,
            'max_cloud': float (optional)
        }
        """
        try:
            data = request.get_json()
            
            is_valid, validation_results = InputValidator.validate_complete_search(
                geometry=data.get('geometry'),
                start_date=data.get('start_date'),
                end_date=data.get('end_date'),
                satellite=data.get('satellite'),
                cloud_coverage=data.get('max_cloud')
            )
            
            formatted_result = ValidationErrorFormatter.format_validation_results(
                validation_results
            )
            
            return jsonify(formatted_result)
        
        except Exception as e:
            return jsonify({
                'valid': False,
                'errors': [ValidationErrorFormatter.format_error(str(e))],
                'info': {}
            }), 500
    
    
    # ===== PROGRESS TRACKING ENDPOINTS =====
    
    @app.route('/api/initialize-download-progress', methods=['POST'])
    def initialize_download_progress():
        """
        Initialize progress tracking for a download.
        
        JSON body:
        {
            'download_id': str,
            'scene_ids': list,
            'metadata': dict (optional)
        }
        """
        try:
            data = request.get_json()
            download_id = data.get('download_id')
            scene_ids = data.get('scene_ids', [])
            metadata = data.get('metadata')
            
            if not download_id or not scene_ids:
                return jsonify({
                    'success': False,
                    'error': 'Missing download_id or scene_ids'
                }), 400
            
            progress_file = progress_tracker.initialize_download(
                download_id, scene_ids, metadata
            )
            
            return jsonify({
                'success': True,
                'download_id': download_id,
                'progress_file': progress_file
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    
    @app.route('/api/update-download-progress', methods=['POST'])
    def update_download_progress():
        """
        Update progress for a specific scene download.
        
        JSON body:
        {
            'download_id': str,
            'scene_id': str,
            'status': str (queued|downloading|completed|failed),
            'progress': int (0-100),
            'bytes_downloaded': int,
            'bytes_total': int,
            'error': str (optional)
        }
        """
        try:
            data = request.get_json()
            
            progress_tracker.update_scene_progress(
                download_id=data.get('download_id'),
                scene_id=data.get('scene_id'),
                status=data.get('status'),
                progress=data.get('progress'),
                bytes_downloaded=data.get('bytes_downloaded'),
                bytes_total=data.get('bytes_total'),
                error=data.get('error')
            )
            
            return jsonify({'success': True})
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    
    @app.route('/api/download-progress', methods=['GET'])
    def get_download_progress():
        """
        Get current progress for a download.
        
        Query parameters:
        - id: download_id
        """
        try:
            download_id = request.args.get('id')
            
            if not download_id:
                return jsonify({
                    'success': False,
                    'error': 'Missing download id'
                }), 400
            
            progress_data = progress_tracker.get_progress(download_id)
            
            if not progress_data:
                return jsonify({
                    'success': False,
                    'error': 'Progress data not found'
                }), 404
            
            # Calculate summary
            summary = progress_tracker.get_summary(download_id)
            
            return jsonify({
                'success': True,
                'progress': progress_data,
                'summary': summary
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    
    @app.route('/api/download-progress-summary', methods=['GET'])
    def get_download_progress_summary():
        """
        Get summary statistics for a download (lightweight endpoint).
        
        Query parameters:
        - id: download_id
        """
        try:
            download_id = request.args.get('id')
            
            if not download_id:
                return jsonify({
                    'success': False,
                    'error': 'Missing download id'
                }), 400
            
            summary = progress_tracker.get_summary(download_id)
            
            if not summary:
                return jsonify({
                    'success': False,
                    'error': 'Progress data not found'
                }), 404
            
            return jsonify({
                'success': True,
                'summary': summary
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    
    # ===== CLEANUP ENDPOINTS =====
    
    @app.route('/api/cleanup-progress', methods=['POST'])
    def cleanup_progress():
        """
        Clean up old progress files.
        
        JSON body:
        {
            'download_id': str (optional, specific file),
            'older_than_days': int (default: 7)
        }
        """
        try:
            data = request.get_json() or {}
            download_id = data.get('download_id')
            older_than_days = data.get('older_than_days', 7)
            
            if download_id:
                progress_tracker.cleanup(download_id, older_than_days)
            else:
                # Clean up all old files
                import glob
                for file in Path(progress_tracker.storage_dir).glob('*_progress.json'):
                    progress_tracker.cleanup(file.stem.replace('_progress', ''), older_than_days)
            
            return jsonify({'success': True})
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    
    @app.route('/api/cleanup-thumbnails', methods=['POST'])
    def cleanup_thumbnails():
        """
        Clean up old cached thumbnails.
        
        JSON body:
        {
            'older_than_days': int (default: 7)
        }
        """
        try:
            data = request.get_json() or {}
            older_than_days = data.get('older_than_days', 7)
            
            thumbnail_generator.clear_cache(older_than_days)
            
            return jsonify({'success': True})
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500


# ===== USAGE IN app_web.py =====
"""
# At the end of app_web.py, add:

from webapp.routes import enhanced_routes

if __name__ == '__main__':
    # Register enhanced workflow routes
    enhanced_routes.register_enhanced_routes(app)
    
    app.run(debug=True)
"""
