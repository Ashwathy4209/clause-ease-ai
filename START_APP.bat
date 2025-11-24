@echo off
echo ==============================================
echo      CLAUSE EASE AI - CONTAINER LAUNCHER
echo ==============================================

:: Check if Docker is running
docker info >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Desktop is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b
)

echo [1/3] Stopping existing containers...
docker-compose down

echo [2/3] Building and Starting container (this may take time on first run)...
docker-compose up --build -d

echo [3/3] Launching Application...
echo Waiting for Streamlit to initialize...
timeout /t 5 /nobreak >nul

:: Open default browser
start http://localhost:8501

echo ==============================================
echo      SUCCESS! App is running in background.
echo      You can close this window.
echo ==============================================
pause