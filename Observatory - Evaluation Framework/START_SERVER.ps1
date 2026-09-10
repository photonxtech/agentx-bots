# ============================================================
#   Observatory RAG Evaluation - Server Starter (PowerShell)
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Starting Observatory RAG Evaluation Server" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Change to script's directory
Set-Location $PSScriptRoot

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.12 or add it to your PATH" -ForegroundColor Red
    pause
    exit 1
}
Write-Host ""

# Check if backend directory exists
if (-not (Test-Path "backend")) {
    Write-Host "ERROR: backend directory not found!" -ForegroundColor Red
    Write-Host "Make sure you're running this from the Observatory - Evaluation Framework folder" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "Backend directory: OK" -ForegroundColor Green
Write-Host ""

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found!" -ForegroundColor Yellow
    Write-Host "The server may not work without API keys configured" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Starting server..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Server will be available at:" -ForegroundColor Green
Write-Host "  - http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  - http://localhost:8000" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
