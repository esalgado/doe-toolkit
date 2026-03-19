#!/bin/bash
# Build script for DOE Toolkit Windows Executable
# Requires: Anaconda environment "doe-toolkit" with all dependencies installed

echo "============================================================"
echo "DOE Toolkit - Build Script for Windows"
echo "============================================================"
echo ""

# Activate conda environment
echo "Activating conda environment: doe-toolkit"
eval "$(conda shell.bash hook)"
conda activate doe-toolkit

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    echo "Please ensure 'doe-toolkit' environment exists"
    read -p "Press Enter to exit..."
    exit 1
fi

# Install PyInstaller if not already installed
echo ""
echo "Installing PyInstaller..."
pip install pyinstaller pyinstaller-hooks-contrib --quiet

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install PyInstaller"
    read -p "Press Enter to exit..."
    exit 1
fi

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build dist DOE-Toolkit.spec

# Run PyInstaller
echo ""
echo "Building executable with PyInstaller..."
echo "This may take several minutes..."
echo ""
pyinstaller build_config.spec

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed"
    read -p "Press Enter to exit..."
    exit 1
fi

# Success message
echo ""
echo "============================================================"
echo "BUILD SUCCESSFUL!"
echo "============================================================"
echo ""
echo "Executable location: dist/DOE-Toolkit/DOE-Toolkit.exe"
echo ""
echo "To distribute:"
echo "  1. Zip the entire 'dist/DOE-Toolkit' folder"
echo "  2. Users extract and run DOE-Toolkit.exe"
echo ""
echo "First launch may take 15-30 seconds while dependencies load."
echo "============================================================"
echo ""
read -p "Press Enter to exit..."
