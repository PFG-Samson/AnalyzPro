<!-- Enhanced snippet to add to online_imagery.html -->
<!-- Add after {% block extra_css %} and before the closing </style> tag -->

<!-- ADD THIS TO THE <head> SECTION: -->
<link rel="stylesheet" href="{{ url_for('static', filename='css/enhancements.css') }}">

<!-- ADD THIS VALIDATION MESSAGES CONTAINER after the </style> closing tag: -->

<!-- ADD THIS TO THE MAIN CONTENT, right after the map container: -->

<!-- ROI Info Overlay (added by JavaScript) -->
<div class="map-info-overlay" id="roi-info-overlay" style="display: none;">
    <h4>ROI Information</h4>
    <div class="info-item">
        <span class="info-label">Area:</span>
        <span class="info-value" id="roi-area-display">Select area</span>
    </div>
    <div class="info-item">
        <span class="info-label">Bounds:</span>
        <span class="info-value" id="roi-bounds-display">--</span>
    </div>
    <div class="roi-area-display" id="roi-area-badge" style="display: none;">
        <span class="roi-area-value" id="roi-area-value">0</span>
        <span class="roi-area-unit">km²</span>
    </div>
</div>

<!-- SCENE CARD ENHANCEMENT - Replace scene card rendering in JavaScript with: -->
<div class="scene-card" data-id="{scene_id}">
    <!-- Thumbnail with loading state -->
    <div class="scene-thumbnail scene-thumbnail-enhanced">
        <div class="thumbnail-container loading" id="thumb-{scene_id}">
            <div class="spinner-mini"></div>
        </div>
        <div class="scene-badge">{cloud}% cloud</div>
    </div>
    
    <!-- Scene information with metadata -->
    <div class="scene-info">
        <div class="scene-date">{datetime}</div>
        
        <!-- Extended metadata display -->
        <div class="scene-metadata">
            <div class="metadata-item">
                <span class="metadata-label">Platform:</span>
                <span class="metadata-value">{platform}</span>
            </div>
            <div class="metadata-item">
                <span class="metadata-label">Cloud:</span>
                <span class="metadata-value">{cloud}%</span>
            </div>
            {if area}
            <div class="metadata-item">
                <span class="metadata-label">Area:</span>
                <span class="metadata-value">{area} km²</span>
            </div>
            {endif}
        </div>
        
        <div class="scene-checkbox">
            <input type="checkbox" data-id="{scene_id}" class="scene-check" />
            <label>Select</label>
        </div>
    </div>
</div>

<!-- VALIDATION MESSAGES CONTAINER - Add near top of content section -->
<div id="validation-messages" class="validation-container"></div>

<!-- PROGRESS TRACKING SECTION - Add when download starts -->
<div class="progress-section" id="progress-{download_id}" style="display: none;">
    <!-- Rendered by ProgressTracker -->
</div>

<!-- FORM VALIDATION EXAMPLE - Add data attributes to form inputs -->
<div class="form-group">
    <label for="start-date">From:</label>
    <input type="date" id="start-date" data-validate="date" data-required="true" />
    <div class="form-input-status"></div>
</div>

<!-- ADD THESE SCRIPTS AT END OF {% block extra_js %}: -->
<script src="{{ url_for('static', filename='js/enhancements.js') }}"></script>

<!-- EXAMPLE: Enhanced renderResults function to include thumbnails -->
/*
function renderResults() {
    const grid = document.getElementById('scenes-grid');
    const list = document.getElementById('scene-list');
    
    searchResults.forEach((scene, idx) => {
        const props = scene.properties;
        const id = scene.id || idx;
        
        // Create grid card with thumbnail
        const gridCard = document.createElement('div');
        gridCard.className = 'scene-card';
        gridCard.dataset.id = id;
        
        // Add thumbnail
        const thumbnail = document.createElement('div');
        thumbnail.className = 'scene-thumbnail';
        thumbnail.id = `thumb-${id}`;
        thumbnail.innerHTML = '<div class="spinner-mini"></div>';
        gridCard.appendChild(thumbnail);
        
        // Generate thumbnail async
        const quicklookUrl = findQuicklookURL(scene.assets);
        if (quicklookUrl) {
            thumbnailManager.generateThumbnail(id, quicklookUrl).then(result => {
                if (result.success) {
                    thumbnail.innerHTML = `
                        <img src="data:image/jpeg;base64,${result.base64}" 
                             alt="Scene thumbnail" 
                             class="thumbnail-image" />
                    `;
                    gridCard.classList.remove('loading');
                }
            });
        }
        
        // ... rest of card content ...
        grid.appendChild(gridCard);
    });
}
*/

<!-- HELPER FUNCTION: Find quicklook URL in STAC assets -->
/*
function findQuicklookURL(assets) {
    if (!assets) return null;
    
    // Try common quicklook names
    const quicklookNames = ['thumbnail', 'quicklook', 'preview', 'overview'];
    
    for (const name of quicklookNames) {
        if (assets[name] && assets[name].href) {
            return assets[name].href;
        }
    }
    
    return null;
}
*/

<!-- INPUT VALIDATION EXAMPLE: -->
/*
// Validate input on change
document.getElementById('start-date').addEventListener('change', function() {
    const satellite = document.getElementById('satellite-select').value;
    const result = validationUI.validateAndDisplay('dates', 
        this.value, 
        document.getElementById('end-date').value,
        satellite
    );
});

// Validate ROI before search
document.getElementById('search-btn').addEventListener('click', async function() {
    const validation = await fetch('/api/validate-search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            geometry: roiGeometry,
            start_date: document.getElementById('start-date').value,
            end_date: document.getElementById('end-date').value,
            satellite: document.getElementById('satellite-select').value,
            max_cloud: parseInt(document.getElementById('cloud-slider').value)
        })
    });
    
    const validationResult = await validation.json();
    
    if (!validationResult.valid) {
        validationResult.errors.forEach(err => {
            validationUI.showError(err.message);
        });
        return;
    }
    
    // Proceed with search
    performSearch();
});

// Start progress tracking
progressTracker.startTracking(downloadId, 1000); // Poll every 1 second
*/
