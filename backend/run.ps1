# Backend startup script for Windows PowerShell

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING PEDESTRIAN DETECTION BACKEND" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (!(Test-Path ".env")) {
    Write-Host "⚠️  .env file not found!" -ForegroundColor Yellow
    Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✅ .env created. Please edit it with your MongoDB URI!" -ForegroundColor Green
    Write-Host ""
}

# Check virtual environment
if (!(Test-Path "venv")) {
    Write-Host "📦 Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Install/upgrade dependencies
Write-Host "📚 Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# Run backend
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✅ BACKEND STARTING" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "📍 API URL: http://localhost:8000" -ForegroundColor Green
Write-Host "📍 Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
