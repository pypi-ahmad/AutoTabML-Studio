@echo off
setlocal
title AutoTabML Studio

cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo AutoTabML Studio requires uv, but uv was not found on PATH.
    echo Install uv, then double-click this file again.
    echo See README.md for setup instructions.
    pause
    exit /b 1
)

echo Starting AutoTabML Studio...
uv run streamlit run app/main.py
set "launch_exit=%errorlevel%"

if "%launch_exit%"=="0" exit /b 0

echo.
echo AutoTabML Studio exited with an error.
pause
exit /b %launch_exit%
