@echo off
title NetraVision - Rural Retinal Screening System (Network Edition)
color 0b

echo =====================================================================
echo           NetraVision (Explainable AI Retinal Screening)
echo            Multi-Device & Network Diagnostic Web Server
echo =====================================================================
echo.

cd /d "%~dp0"

:: Initialize virtual environment if needed
if not exist ".venv\Scripts\python.exe" (
    echo [*] Initializing virtual environment...
    ..\uv.exe venv --python 3.11 .venv
    echo [*] Installing dependencies...
    ..\uv.exe pip install -r requirements.txt --python .venv\Scripts\python.exe
)

:: Find current machine local network IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do set LOCAL_IP=%%b
)

echo [*] Server is configured for MULTI-DEVICE / CROSS-NETWORK ACCESS!
echo.
echo =====================================================================
echo   [1] Access from THIS computer:
echo       -> http://localhost:8000  or  http://127.0.0.1:8000
echo.
if defined LOCAL_IP (
    echo   [2] Access from ANY OTHER COMPUTER, TABLET, or PHONE on Wi-Fi:
    echo       -> http://%LOCAL_IP%:8000
) else (
    echo   [2] Access from other devices on your local Wi-Fi / network:
    echo       -> http://[YOUR-IP-ADDRESS]:8000
)
echo =====================================================================
echo.
echo [*] Press Ctrl+C in this terminal to stop the server.
echo.

start http://localhost:8000
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

pause
