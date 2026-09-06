@echo off
setlocal enabledelayedexpansion
title Clinique-Agent-AI Local Server

echo =======================================================
echo    Starting Clinique-Agent-AI Local Server
echo =======================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

:: Ensure dependencies are installed
echo [*] Checking and installing required dependencies...
python -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Dependency check encountered issues. Proceeding anyway...
)

echo.
echo [*] Launching application on http://127.0.0.1:8000 ...
echo [*] Press Ctrl+C in this console window to stop the server.
echo.

:: Automatically open browser after 2 seconds in background
start /b cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

:: Start Uvicorn via run.py
python run.py

pause
