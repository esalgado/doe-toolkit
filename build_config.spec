"""
PyInstaller Build Configuration for DOE Toolkit

This configuration file specifies how to package the application into a standalone executable.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get project root
project_root = Path(__file__).parent

# Application metadata
APP_NAME = "DOE-Toolkit"
VERSION = "0.1.0"
AUTHOR = "DOE-Toolkit Project"

# Main script
MAIN_SCRIPT = project_root / "app_launcher.py"

# Collect all source files
src_dir = project_root / "src"

# Data files to include
datas = [
    # Include all Python source files from src/
    (str(src_dir), "src"),
    
    # Include Streamlit configuration if exists
    (str(project_root / ".streamlit"), ".streamlit") if (project_root / ".streamlit").exists() else None,
]

# Remove None entries
datas = [d for d in datas if d is not None]

# Add Streamlit data files
datas += collect_data_files("streamlit")
datas += collect_data_files("altair")
datas += collect_data_files("plotly")

# Hidden imports (modules not automatically detected)
hiddenimports = [
    # Streamlit and dependencies
    "streamlit",
    "streamlit.runtime.scriptrunner.magic_funcs",
    "streamlit.components.v1",
    
    # Scientific libraries
    "numpy",
    "scipy",
    "scipy.stats",
    "scipy.optimize",
    "scipy.special",
    "scipy.linalg",
    "pandas",
    "matplotlib",
    "plotly",
    "plotly.graph_objs",
    
    # Statistical modeling
    "statsmodels",
    "statsmodels.api",
    "statsmodels.formula.api",
    "sklearn",
    "sklearn.preprocessing",
    
    # Optimization
    "cvxpy",
    
    # All submodules from our package
    "src.core",
    "src.core.optimal",
    "src.core.diagnostics",
    "src.core.augmentation",
    "src.core.candidates",
    "src.ui",
    "src.ui.pages",
    "src.ui.components",
    "src.ui.utils",
]

# Collect all submodules automatically
hiddenimports += collect_submodules("src")

# Excluded modules (to reduce size)
excludes = [
    "tkinter",
    "IPython",
    "jupyter",
    "notebook",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
]

# Binary includes/excludes
binaries = []

# Analysis
a = Analysis(
    [str(MAIN_SCRIPT)],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PYZ (archive)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# EXE (executable)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Show console for logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon file here if you have one: str(project_root / "icon.ico")
)

# COLLECT (bundle everything)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
