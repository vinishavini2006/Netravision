@echo off
title NetraVision - Global Public Internet Access Launcher
color 0a

echo =====================================================================
echo         NetraVision AI: Instant Global Internet Access
echo   Makes your local retinal screening app accessible worldwide!
echo =====================================================================
echo.

cd /d "%~dp0"

:: Check if cloudflared exists, if not download it
if not exist "cloudflared.exe" (
    echo [*] Downloading secure Cloudflare Tunnel utility...
    curl.exe -L -o cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
)

:: Check if python venv exists
if not exist ".venv\Scripts\python.exe" (
    echo [*] Initializing virtual environment...
    ..\uv.exe venv --python 3.11 .venv
    ..\uv.exe pip install -r requirements.txt --python .venv\Scripts\python.exe
)

echo [*] Launching NetraVision Local AI Diagnostic Server in background...
start /b .venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 > server_output.log 2>&1

echo [*] Waiting for server to initialize...
timeout /t 4 /nobreak >nul

echo.
echo =====================================================================
echo  CONNECTING SECURE GLOBAL PUBLIC TUNNEL...
echo  Look for the line below starting with:
echo  https://...trycloudflare.com
echo.
echo  Share that HTTPS link with ANY user on ANY device, anywhere in the world!
echo =====================================================================
echo.

cloudflared.exe tunnel --url http://127.0.0.1:8000

pause
