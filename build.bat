@echo off
REM Build script for DOE Toolkit Windows Executable
REM Requires: Anaconda environment "doe-toolkit" with all dependencies installed

echo ============================================================
echo DOE Toolkit - Build Script for Windows
echo ============================================================
echo.

REM Activate conda environment
echo Activating conda environment: doe-toolkit
call conda activate doe-toolkit
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    echo Please ensure 'doe-toolkit' environment exists
    pause
    exit /b 1
)

REM Install PyInstaller if not already installed
echo.
echo Installing PyInstaller...
pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller
    pause
    exit /b 1
)

REM Clean previous builds
echo.
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist DOE-Toolkit.spec del DOE-Toolkit.spec

REM Run PyInstaller
echo.
echo Building executable with PyInstaller...
echo This may take several minutes...
echo.
pyinstaller build_config.spec
if errorlevel 1 (
    echo.
    echo ERROR: Build failed
    pause
    exit /b 1
)

REM Success message
echo.
echo ============================================================
echo BUILD SUCCESSFUL!
echo ============================================================
echo.
echo Executable location: dist\DOE-Toolkit\DOE-Toolkit.exe
echo.
echo To distribute:
echo   1. Zip the entire 'dist\DOE-Toolkit' folder
echo   2. Users extract and run DOE-Toolkit.exe
echo.
echo First launch may take 15-30 seconds while dependencies load.
echo ============================================================
echo.
pause
