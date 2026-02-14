# DOE Toolkit - Installation & Distribution Guide

## For Users: Running the Application

### Option 1: Download Pre-built Executable (Easiest)
1. Download `DOE-Toolkit.zip` from the releases page
2. Extract the ZIP file to your desired location (e.g., `C:\Program Files\DOE-Toolkit`)
3. Double-click `DOE-Toolkit.exe` to launch
4. The application will open in your default web browser
5. **First launch may take 15-30 seconds** while loading dependencies

**System Requirements:**
- Windows 10 or 11 (64-bit)
- 4 GB RAM minimum, 8 GB recommended
- 500 MB free disk space
- Internet browser (Chrome, Edge, Firefox)

### Option 2: Run from Source (For Developers)
See "For Developers" section below.

---

## For Developers: Building the Executable

### Prerequisites
1. **Anaconda/Miniconda** installed
2. **doe-toolkit conda environment** set up:
   ```bash
   conda env create -f environment.yml
   # OR manually:
   conda create -n doe-toolkit python=3.11
   conda activate doe-toolkit
   pip install -r requirements.txt
   ```

### Build Steps

#### Windows:
```bash
# From project root directory
build.bat
```

The script will:
1. Activate the conda environment
2. Install PyInstaller
3. Clean previous builds
4. Build the executable
5. Create `dist/DOE-Toolkit/` folder with executable

**Build output:**
- Executable: `dist/DOE-Toolkit/DOE-Toolkit.exe`
- Supporting files: All bundled in `dist/DOE-Toolkit/`
- **Do not separate the .exe from its folder** - it needs the bundled files

#### Manual Build:
```bash
conda activate doe-toolkit
pip install pyinstaller
pyinstaller build_config.spec
```

### Build Configuration

The build is configured in `build_config.spec`:
- **One-folder mode:** Creates a folder with .exe and dependencies
- **Console mode:** Shows console window for logs (helpful for debugging)
- **Includes:** All source code, Streamlit files, scientific libraries
- **Excludes:** Jupyter, Qt libraries (reduces size)

To modify:
- **Add icon:** Set `icon='path/to/icon.ico'` in the EXE section
- **Hide console:** Change `console=True` to `console=False`
- **One-file mode:** Change to use `onefile=True` (slower startup, easier distribution)

### Testing the Build

```bash
# Navigate to build output
cd dist/DOE-Toolkit

# Run the executable
DOE-Toolkit.exe
```

Expected behavior:
1. Console window opens showing startup logs
2. Browser opens automatically to `http://localhost:8501`
3. Application loads in browser

### Distribution

To share with users:
1. Zip the entire `dist/DOE-Toolkit` folder
2. Upload to GitHub releases or file sharing
3. Users extract and run `DOE-Toolkit.exe`

**Important:** 
- Include the entire folder, not just the .exe
- File size will be ~200-400 MB (includes Python, NumPy, SciPy, etc.)

---

## For Developers: Running from Source

### Setup
```bash
# Clone repository
git clone https://github.com/bpimentel3/doe-toolkit.git
cd doe-toolkit

# Create environment
conda create -n doe-toolkit python=3.11
conda activate doe-toolkit

# Install dependencies
pip install -r requirements.txt
```

### Run Application
```bash
# Option 1: Using Streamlit directly
streamlit run src/ui/app.py

# Option 2: Using the launcher (simulates .exe behavior)
python app_launcher.py
```

### Development Workflow
```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Type checking
mypy src/

# Linting
black src/ tests/
flake8 src/ tests/
```

---

## Troubleshooting

### Build Issues

**Problem:** PyInstaller fails with "module not found"
- **Solution:** Add missing module to `hiddenimports` in `build_config.spec`

**Problem:** Build succeeds but .exe crashes immediately
- **Solution:** Run with `console=True` to see error messages

**Problem:** .exe is too large (>500 MB)
- **Solution:** Add more modules to `excludes` in `build_config.spec`

### Runtime Issues

**Problem:** Browser doesn't open automatically
- **Solution:** Manually navigate to `http://localhost:8501`

**Problem:** Port 8501 already in use
- **Solution:** The launcher automatically finds a free port. Check console output.

**Problem:** Streamlit shows "module not found" errors
- **Solution:** Rebuild with missing module added to `hiddenimports`

---

## Technical Details

### Architecture
```
DOE-Toolkit.exe (launcher)
  â†' Starts Python interpreter (embedded)
  â†' Launches Streamlit server on localhost
  â†' Opens browser to localhost:8501
  â†' Streamlit serves the UI
  â†' User interacts via browser
```

### Why Not One-File Mode?
- **Faster startup:** No need to extract to temp directory
- **Better compatibility:** Some libraries (NumPy, SciPy) work better in folder mode
- **Easier debugging:** Can inspect bundled files if issues arise

### File Structure After Build
```
dist/
â""â"€â"€ DOE-Toolkit/
    â"œâ"€â"€ DOE-Toolkit.exe          # Main executable
    â"œâ"€â"€ python311.dll            # Python runtime
    â"œâ"€â"€ _internal/               # Bundled libraries
    â"‚   â"œâ"€â"€ numpy/
    â"‚   â"œâ"€â"€ scipy/
    â"‚   â"œâ"€â"€ streamlit/
    â"‚   â""â"€â"€ ...
    â""â"€â"€ src/                     # Your source code
        â"œâ"€â"€ core/
        â""â"€â"€ ui/
```

---

## Version Information

- **Python:** 3.11
- **Streamlit:** 1.28+
- **NumPy:** 1.24+
- **SciPy:** 1.10+
- **PyInstaller:** Latest

Built with ❤️ for the DOE community
