@echo off
REM ============================================================
REM   Observatory RAG Evaluation - Server Starter
REM ============================================================

echo.
echo ============================================================
echo   Starting Observatory RAG Evaluation Server
echo ============================================================
echo.

REM Change to the script's directory
cd /d "%~dp0"

echo Current directory: %CD%
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.12 or add it to your PATH
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Check if backend directory exists
if not exist "backend" (
    echo ERROR: backend directory not found!
    echo Make sure you're running this from the Observatory - Evaluation Framework folder
    pause
    exit /b 1
)

echo Backend directory: OK
echo.

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo The server may not work without API keys configured
    echo.
)

echo Starting server...
echo.
echo Server will be available at:
echo   - http://127.0.0.1:8000
echo   - http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
echo.

python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

pause
