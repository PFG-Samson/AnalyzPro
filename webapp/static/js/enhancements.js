/**
 * Enhanced JavaScript for online imagery workflow
 * - Thumbnails with metadata
 * - Real-time progress tracking
 * - Input validation
 * - Interactive map features
 */

// ===== INPUT VALIDATION MODULE =====

class InputValidator {
    constructor() {
        this.validators = {
            roi: this.validateROI,
            dates: this.validateDates,
            cloudCoverage: this.validateCloudCoverage,
            satellite: this.validateSatellite
        };
    }
    
    /**
     * Validate ROI geometry
     */
    validateROI(geometry) {
        if (!geometry) {
            return {
                valid: false,
                error: "ROI is empty. Please draw an area on the map.",
                icon: "❌"
            };
        }
        
        if (!geometry.coordinates || geometry.coordinates.length === 0) {
            return {
                valid: false,
                error: "Geometry has no coordinates.",
                icon: "❌"
            };
        }
        
        return {
            valid: true,
            icon: "✓"
        };
    }
    
    /**
     * Validate date range
     */
    validateDates(startDate, endDate, satellite) {
        if (!startDate || !endDate) {
            return {
                valid: false,
                error: "Both start and end dates are required.",
                icon: "❌"
            };
        }
        
        const start = new Date(startDate);
        const end = new Date(endDate);
        const now = new Date();
        
        if (start > end) {
            return {
                valid: false,
                error: "Start date cannot be after end date",
                icon: "❌"
            };
        }
        
        if (end > now) {
            return {
                valid: false,
                error: "End date cannot be in the future",
                icon: "❌"
            };
        }
        
        const daysDiff = Math.floor((end - start) / (1000 * 60 * 60 * 24));
        if (daysDiff < 1) {
            return {
                valid: false,
                error: "Date range must be at least 1 day",
                icon: "❌"
            };
        }
        
        if (daysDiff > 3650) {
            return {
                valid: false,
                error: "Date range cannot exceed 10 years",
                icon: "❌"
            };
        }
        
        return {
            valid: true,
            info: `${daysDiff} days selected`,
            icon: "✓"
        };
    }
    
    /**
     * Validate cloud coverage
     */
    validateCloudCoverage(value, satellite) {
        const num = parseFloat(value);
        
        if (isNaN(num)) {
            return {
                valid: false,
                error: "Cloud coverage must be a number",
                icon: "❌"
            };
        }
        
        if (num < 0 || num > 100) {
            return {
                valid: false,
                error: "Cloud coverage must be between 0 and 100%",
                icon: "❌"
            };
        }
        
        if (satellite === 'sentinel1') {
            return {
                valid: false,
                error: "Cloud coverage filter not applicable for SAR data",
                icon: "❌"
            };
        }
        
        return {
            valid: true,
            icon: "✓"
        };
    }
    
    /**
     * Validate satellite selection
     */
    validateSatellite(satellite) {
        const valid = ['sentinel2', 'sentinel1', 'landsat8', 'landsat9'].includes(satellite);
        return {
            valid: valid,
            error: valid ? null : "Invalid satellite selection",
            icon: valid ? "✓" : "❌"
        };
    }
}

// ===== VALIDATION UI MODULE =====

class ValidationUI {
    constructor() {
        this.messageContainer = this.createMessageContainer();
        this.validator = new InputValidator();
    }
    
    createMessageContainer() {
        let container = document.getElementById('validation-messages');
        if (!container) {
            container = document.createElement('div');
            container.id = 'validation-messages';
            container.style.cssText = 'position: fixed; top: 80px; right: 20px; max-width: 400px; z-index: 1000;';
            document.body.appendChild(container);
        }
        return container;
    }
    
    showError(message, dismissible = true) {
        this.showMessage(message, 'error', '❌', dismissible);
    }
    
    showWarning(message, dismissible = true) {
        this.showMessage(message, 'warning', '⚠️', dismissible);
    }
    
    showInfo(message, dismissible = true) {
        this.showMessage(message, 'info', 'ℹ️', dismissible);
    }
    
    showSuccess(message, dismissible = true) {
        this.showMessage(message, 'success', '✓', dismissible);
    }
    
    showMessage(message, severity, icon, dismissible) {
        const msgEl = document.createElement('div');
        msgEl.className = `validation-message ${severity}`;
        msgEl.innerHTML = `
            <div class="validation-icon">${icon}</div>
            <div class="validation-content">
                <p>${message}</p>
            </div>
            ${dismissible ? '<button class="close-validation">×</button>' : ''}
        `;
        
        this.messageContainer.appendChild(msgEl);
        
        if (dismissible) {
            msgEl.querySelector('.close-validation').onclick = () => msgEl.remove();
            setTimeout(() => msgEl.remove(), 5000);
        }
    }
    
    clearMessages() {
        this.messageContainer.innerHTML = '';
    }
    
    validateAndDisplay(type, ...args) {
        const validator = this.validator.validators[type];
        if (!validator) return { valid: false };
        
        const result = validator.call(this.validator, ...args);
        
        if (!result.valid && result.error) {
            this.showError(result.error);
        } else if (result.info) {
            this.showInfo(result.info, true);
        }
        
        return result;
    }
}

// ===== THUMBNAIL MODULE =====

class ThumbnailManager {
    constructor() {
        this.cache = new Map();
        this.loadingStates = new Map();
    }
    
    async generateThumbnail(sceneId, quicklookUrl) {
        if (this.cache.has(sceneId)) {
            return this.cache.get(sceneId);
        }
        
        this.loadingStates.set(sceneId, true);
        
        try {
            const response = await fetch('/api/generate-thumbnail', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scene_id: sceneId,
                    quicklook_url: quicklookUrl
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.cache.set(sceneId, data);
                return data;
            }
        } catch (error) {
            console.error('Thumbnail generation failed:', error);
        }
        
        this.loadingStates.delete(sceneId);
        return { success: false };
    }
    
    createThumbnailElement(sceneId, base64Data) {
        const container = document.createElement('div');
        container.className = 'thumbnail-container';
        
        if (base64Data) {
            const img = document.createElement('img');
            img.src = `data:image/jpeg;base64,${base64Data}`;
            img.className = 'thumbnail-image';
            img.alt = `Scene ${sceneId} thumbnail`;
            container.appendChild(img);
        } else {
            container.innerHTML = '<div class="thumbnail-placeholder">🛰️</div>';
        }
        
        return container;
    }
}

// ===== PROGRESS TRACKING MODULE =====

class ProgressTracker {
    constructor() {
        this.progressData = new Map();
        this.pollInterval = null;
    }
    
    startTracking(downloadId, pollInterval = 1000) {
        this.stopTracking();
        
        this.pollInterval = setInterval(() => {
            this.updateProgress(downloadId);
        }, pollInterval);
        
        // Initial update
        this.updateProgress(downloadId);
    }
    
    stopTracking() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    
    async updateProgress(downloadId) {
        try {
            const response = await fetch(`/api/download-progress?id=${downloadId}`);
            const data = await response.json();
            
            if (!data.success) return;
            
            this.progressData.set(downloadId, data.progress);
            this.renderProgress(downloadId, data.progress);
        } catch (error) {
            console.error('Progress update failed:', error);
        }
    }
    
    renderProgress(downloadId, progressData) {
        const container = document.getElementById(`progress-${downloadId}`);
        if (!container) return;
        
        const scenes = progressData.scenes || {};
        const summary = progressData.summary || {};
        
        let html = `
            <div class="progress-header">
                <h3>Download Progress</h3>
                <span class="progress-percentage">${summary.overall_progress?.toFixed(0) || 0}%</span>
            </div>
            
            <div class="progress-stats">
                <div class="stat-card">
                    <div class="stat-number">${summary.completed || 0}</div>
                    <div class="stat-label">Completed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${summary.downloading || 0}</div>
                    <div class="stat-label">Downloading</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${summary.queued || 0}</div>
                    <div class="stat-label">Queued</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${summary.failed || 0}</div>
                    <div class="stat-label">Failed</div>
                </div>
            </div>
            
            <div class="scenes-progress">
        `;
        
        Object.entries(scenes).forEach(([sceneId, sceneData]) => {
            const status = sceneData.status || 'queued';
            const progress = sceneData.progress || 0;
            const icon = this.getStatusIcon(status);
            
            html += `
                <div class="scene-progress-item ${status}">
                    <div class="scene-progress-header">
                        <div class="scene-progress-title">${sceneId}</div>
                        <div class="scene-progress-status ${status}">
                            <span>${icon}</span>
                            <span>${this.formatStatus(status)}</span>
                        </div>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar-fill ${status}" style="width: ${progress}%">
                            ${progress > 10 ? `${progress}%` : ''}
                        </div>
                    </div>
                    ${sceneData.error ? `<div class="scene-error">${sceneData.error}</div>` : ''}
                </div>
            `;
        });
        
        html += '</div>';
        container.innerHTML = html;
    }
    
    getStatusIcon(status) {
        const icons = {
            'queued': '⏳',
            'downloading': '⬇️',
            'completed': '✓',
            'failed': '✕'
        };
        return icons[status] || '•';
    }
    
    formatStatus(status) {
        return status.charAt(0).toUpperCase() + status.slice(1);
    }
}

// ===== MAP ENHANCEMENTS =====

class MapEnhancer {
    constructor(map) {
        this.map = map;
        this.roiArea = null;
    }
    
    addROIInfoOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'map-info-overlay';
        overlay.id = 'roi-info-overlay';
        overlay.innerHTML = `
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
        `;
        
        const mapContainer = document.querySelector('.map-wrapper');
        if (mapContainer) {
            mapContainer.appendChild(overlay);
        }
    }
    
    updateROIDisplay(area) {
        this.roiArea = area;
        const display = document.getElementById('roi-area-display');
        const badge = document.getElementById('roi-area-badge');
        const value = document.getElementById('roi-area-value');
        
        if (display && area) {
            const formatted = area > 1000 
                ? `${(area / 1000).toFixed(1)} km²` 
                : `${area.toFixed(2)} km²`;
            display.textContent = formatted;
            value.textContent = formatted.split(' ')[0];
            badge.style.display = 'block';
        }
    }
    
    addMapInstructions() {
        const instructions = document.querySelector('.map-instructions');
        if (!instructions) return;
        
        instructions.innerHTML = `
            <strong>🗺️ Interactive Map</strong>
            <div style="font-size: 0.8rem; margin-top: 0.5rem; line-height: 1.4;">
                <p>• Use rectangle/polygon tool to draw ROI</p>
                <p>• Click edit icon to modify ROI</p>
                <p>• Use +/- to zoom in/out</p>
                <p>• Drag to pan around map</p>
            </div>
        `;
    }
}

// ===== INITIALIZATION =====

document.addEventListener('DOMContentLoaded', function() {
    // Initialize validators and UI
    const validationUI = new ValidationUI();
    const thumbnailManager = new ThumbnailManager();
    const progressTracker = new ProgressTracker();
    
    // Validate on form changes
    const satelliteSelect = document.getElementById('satellite-select');
    const startDate = document.getElementById('start-date');
    const endDate = document.getElementById('end-date');
    const cloudSlider = document.getElementById('cloud-slider');
    
    if (satelliteSelect) {
        satelliteSelect.addEventListener('change', function() {
            const result = validationUI.validateAndDisplay('satellite', this.value);
            if (result.valid) {
                validationUI.showSuccess('Satellite selection valid');
            }
        });
    }
    
    if (startDate && endDate) {
        const validateDateRange = () => {
            const satellite = satelliteSelect?.value || 'sentinel2';
            validationUI.validateAndDisplay('dates', startDate.value, endDate.value, satellite);
        };
        
        startDate.addEventListener('change', validateDateRange);
        endDate.addEventListener('change', validateDateRange);
    }
    
    if (cloudSlider) {
        cloudSlider.addEventListener('input', function() {
            const satellite = satelliteSelect?.value || 'sentinel2';
            validationUI.validateAndDisplay('cloudCoverage', this.value, satellite);
        });
    }
    
    // Make validators globally available
    window.validationUI = validationUI;
    window.progressTracker = progressTracker;
    window.thumbnailManager = thumbnailManager;
});
