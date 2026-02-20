@echo off
REM DOE Toolkit Launcher
REM Double-click this file to start the application.

REM Set working directory to the folder containing this script.
REM This ensures src/ is found regardless of where the user extracted the zip.
cd /d "%~dp0"

REM Check the bundled environment exists
if not exist "env\Scripts\streamlit.exe" (
    echo ERROR: Bundled environment not found.
    echo Please re-download and extract DOE-Toolkit again.
    pause
    exit /b 1
)

REM Check app source exists
if not exist "src\ui\app.py" (
    echo ERROR: Application source not found.
    echo Please re-download and extract DOE-Toolkit again.
    pause
    exit /b 1
)

echo ============================================================
echo  DOE Toolkit - Design of Experiments Software
echo ============================================================
echo.
echo  Starting... your browser will open automatically.
echo  To stop the app, close this window.
echo.
echo ============================================================

REM Launch Streamlit using the bundled environment.
REM No subprocess spawning, no recursion - streamlit.exe is a real executable.
"env\Scripts\streamlit.exe" run src\ui\app.py ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --server.enableCORS false ^
    --theme.base light