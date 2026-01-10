"""Unit tests for the Flask web application."""

import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp.app_web import app, run_analysis_async, analysis_status

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Use temporary folders for testing
    test_base = Path(__file__).parent / 'test_env'
    upload_dir = test_base / 'uploads'
    results_dir = test_base / 'results'
    
    upload_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    app.config['UPLOAD_FOLDER'] = upload_dir
    app.config['RESULTS_FOLDER'] = results_dir
    
    with app.test_client() as client:
        yield client
        
    # Cleanup after all tests in this fixture
    if test_base.exists():
        shutil.rmtree(test_base, ignore_errors=True)

def test_index_route(client):
    """Test the home page."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Analyz" in response.data

def test_upload_route_get(client):
    """Test the upload page GET request."""
    response = client.get('/upload')
    assert response.status_code == 200
    assert b"Upload" in response.data

def test_robust_cleanup_on_success(client):
    """Verify that session folders are cleaned up after successful analysis."""
    session_id = "test_success_session"
    session_folder = app.config['UPLOAD_FOLDER'] / session_id
    session_folder.mkdir(exist_ok=True)
    
    # Create a dummy image file
    image_path = session_folder / "image.tif"
    image_path.write_text("dummy image data")
    
    # Initialize status entry
    analysis_status[session_id] = {'status': 'queued'}
    
    # Mock analysis dependencies to avoid heavy processing
    with patch('webapp.app_web.FileHandler.read_raster', return_value=(MagicMock(), {'transform': None})), \
         patch('webapp.app_web.run_optical_analysis', return_value=(MagicMock(), {}, {"analysis_type": "NDVI", "summary": "Success", "key_findings": [], "recommendations": []})), \
         patch('webapp.app_web.FileHandler.write_raster'), \
         patch('webapp.app_web.save_analysis_status'):
        
        # Run cleanup-logic-heavy function synchronously
        run_analysis_async(
            session_id=session_id,
            image_path=str(image_path),
            boundary_path=None,
            image_type='optical',
            analysis_type='ndvi',
            params={}
        )
    
    # ASSERTIONS
    # 1. Status should be completed
    if analysis_status[session_id]['status'] != 'completed':
        status_info = analysis_status[session_id]
        pytest.fail(f"Analysis failed with status '{status_info['status']}' and error: {status_info.get('error')}")
    
    # 2. Session folder should be GONE
    assert not session_folder.exists(), "Session folder was not cleaned up on success"

def test_robust_cleanup_on_failure(client):
    """Verify that session folders are cleaned up even if analysis fails."""
    session_id = "test_fail_session"
    session_folder = app.config['UPLOAD_FOLDER'] / session_id
    session_folder.mkdir(exist_ok=True)
    
    image_path = session_folder / "image.tif"
    image_path.write_text("dummy image data")
    
    analysis_status[session_id] = {'status': 'queued'}
    
    # Mock failure during processing
    with patch('webapp.app_web.FileHandler.read_raster', side_effect=Exception("Critical Failure")), \
         patch('webapp.app_web.save_analysis_status'):
        
        run_analysis_async(
            session_id=session_id,
            image_path=str(image_path),
            boundary_path=None,
            image_type='optical',
            analysis_type='ndvi',
            params={}
        )
    
    # ASSERTIONS
    # 1. Session folder should be GONE even on failure
    assert not session_folder.exists(), "Session folder was not cleaned up on failure"
    # 2. Status should be failed
    assert analysis_status[session_id]['status'] == 'failed'
    assert "Critical Failure" in analysis_status[session_id]['error']
