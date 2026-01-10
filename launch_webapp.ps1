# Analyz Web Application Launch Script
# PowerShell script to start the web application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🌍 Analyz Web Application Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check if Flask is installed
try {
    python -c "import flask" 2>$null
    Write-Host "✓ Flask is installed" -ForegroundColor Green
} catch {
    Write-Host "⚠ Flask not found. Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "Starting Analyz Web Server..." -ForegroundColor Cyan
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Access the application at:" -ForegroundColor White
Write-Host "  http://localhost:5000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press CTRL+C to stop the server" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to webapp directory and start server
Set-Location webapp
python app_web.py
