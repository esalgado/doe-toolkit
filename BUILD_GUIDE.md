# Building DOE Toolkit Executable - Step by Step

This guide walks you through building the Windows executable from scratch.

## Prerequisites Check

Before you start, verify you have:

- [ ] **Windows 10 or 11** (64-bit)
- [ ] **Anaconda or Miniconda** installed
- [ ] **Git** installed (to clone repository)
- [ ] **~2 GB free disk space** (for build artifacts)

## Step 1: Set Up Environment

```bash
# Clone the repository
git clone https://github.com/bpimentel3/doe-toolkit.git
cd doe-toolkit

# Create conda environment
conda create -n doe-toolkit python=3.11
conda activate doe-toolkit

# Install runtime dependencies
pip install -r requirements.txt

# Install build dependencies
pip install -r requirements-build.txt
```

**Expected result:** All packages install without errors.

## Step 2: Test the Application

Before building, make sure the app works:

```bash
# Test the launcher script
python app_launcher.py
```

**Expected result:** 
- Console shows "Starting application on http://localhost:8501"
- Browser opens automatically
- Application loads correctly
- You can navigate through pages

Press Ctrl+C to stop the server.

## Step 3: Build the Executable

### Option A: Automated Build (Recommended)

Simply run the build script:

```bash
build.bat
```

**What it does:**
1. Activates conda environment
2. Installs PyInstaller if needed
3. Cleans old builds
4. Runs PyInstaller with configuration
5. Shows success message

**Expected duration:** 3-5 minutes on a modern PC

### Option B: Manual Build

```bash
# Activate environment
conda activate doe-toolkit

# Install PyInstaller
pip install pyinstaller pyinstaller-hooks-contrib

# Clean previous builds
rmdir /s /q build dist
del *.spec

# Run PyInstaller
pyinstaller build_config.spec
```

## Step 4: Verify the Build

```bash
# Navigate to output
cd dist\DOE-Toolkit

# Run the executable
DOE-Toolkit.exe
```

**Expected result:**
- Console window opens with startup logs
- After 5-10 seconds, browser opens
- Application loads and works correctly

**Check:**
- [ ] Application starts
- [ ] Browser opens automatically
- [ ] All pages are accessible
- [ ] Can create designs
- [ ] Can import CSV files
- [ ] Plots render correctly

## Step 5: Test the Package

Before distributing, test that the package is truly standalone:

1. **Copy to different location:**
   ```bash
   # Copy entire folder
   xcopy dist\DOE-Toolkit C:\Temp\DOE-Test /E /I
   
   # Run from new location
   cd C:\Temp\DOE-Test
   DOE-Toolkit.exe
   ```

2. **Test on clean machine (optional but recommended):**
   - Copy folder to USB drive
   - Test on another Windows PC without Python/Anaconda
   - Verify it runs without dependencies

## Step 6: Package for Distribution

```bash
# Navigate to dist folder
cd dist

# Create ZIP file
# Windows: Right-click DOE-Toolkit folder → Send to → Compressed (zipped) folder
# OR use command line:
powershell Compress-Archive -Path DOE-Toolkit -DestinationPath DOE-Toolkit-v0.1.0-win64.zip
```

**Result:** `DOE-Toolkit-v0.1.0-win64.zip` ready for distribution

## Common Build Issues

### Issue: "ModuleNotFoundError: No module named 'X'"

**Cause:** PyInstaller didn't detect a required module

**Fix:** Add to `hiddenimports` in `build_config.spec`:
```python
hiddenimports = [
    # ... existing imports ...
    "X",  # Add missing module
]
```

Then rebuild.

### Issue: Build succeeds but .exe crashes immediately

**Cause:** Missing runtime dependency or path issue

**Fix:** 
1. Check console output for errors (ensure `console=True` in spec file)
2. Run in command prompt to see error messages:
   ```bash
   DOE-Toolkit.exe
   ```
3. Check if missing DLLs or data files

### Issue: "FileNotFoundError" for Streamlit files

**Cause:** Streamlit data files not included

**Fix:** Already handled by `collect_data_files("streamlit")` in spec file. If still occurs, verify:
```python
datas += collect_data_files("streamlit")
```

### Issue: .exe is too large (>500 MB)

**Cause:** Including unnecessary packages

**Fix:** Add to `excludes` in `build_config.spec`:
```python
excludes = [
    # ... existing excludes ...
    "IPython",      # Not needed
    "jupyter",      # Not needed
    "notebook",     # Not needed
]
```

### Issue: Antivirus flags the .exe

**Cause:** Common with PyInstaller executables (false positive)

**Solutions:**
1. **Code signing:** Sign the executable (requires certificate, $$$)
2. **Submit to antivirus:** Upload to VirusTotal, report as false positive
3. **Document:** Include note in README about false positives
4. **Alternative distribution:** Provide Python source as alternative

## Build Output Structure

After successful build:

```
dist/
└── DOE-Toolkit/
    ├── DOE-Toolkit.exe          # Main executable (~50 KB)
    ├── python311.dll            # Python runtime (~5 MB)
    ├── _internal/               # Bundled dependencies (~300 MB)
    │   ├── numpy.libs/
    │   ├── scipy.libs/
    │   ├── streamlit/
    │   └── ... (many more)
    └── src/                     # Your source code
        ├── core/
        └── ui/
```

**Total size:** ~350-400 MB (compressed ZIP: ~120-150 MB)

## Size Optimization Tips

If you need to reduce size:

1. **Use UPX compression** (already enabled):
   - Already in spec: `upx=True`
   - Can reduce by ~30%

2. **Exclude large unused packages:**
   ```python
   excludes = [
       "tkinter",        # GUI toolkit (not needed)
       "IPython",        # Interactive shell
       "jupyter",        # Notebook
       "PyQt5", "PyQt6", # Qt frameworks
       "PySide2", "PySide6",
   ]
   ```

3. **Remove debug symbols:**
   ```python
   strip=True  # In EXE section
   ```

4. **Use one-file mode** (slower startup but smaller distribution):
   ```python
   exe = EXE(
       # ...
       onefile=True,  # Single file instead of folder
   )
   ```

## Advanced: Custom Icon

To add a custom icon:

1. Create or download a `.ico` file (256x256 or multi-resolution)
2. Place in project root (e.g., `icon.ico`)
3. Update `build_config.spec`:
   ```python
   exe = EXE(
       # ...
       icon='icon.ico',  # Path to icon file
   )
   ```

## Publishing the Release

Once built and tested:

1. **Create GitHub Release:**
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. **Upload ZIP to release:**
   - Go to GitHub → Releases → Create new release
   - Upload `DOE-Toolkit-v0.1.0-win64.zip`
   - Add release notes

3. **Include in release:**
   - QUICKSTART.md (user guide)
   - INSTALL.md (installation instructions)
   - CHANGELOG.md (what's new)
   - LICENSE.txt

## Maintenance

**When to rebuild:**
- After adding new features
- After fixing bugs
- After updating dependencies
- After changing UI significantly

**Version numbering:**
- v0.1.0 → v0.2.0 (minor features)
- v0.2.0 → v1.0.0 (major release, stable)
- v1.0.0 → v1.0.1 (bug fixes)

---

**Build successfully completed? Great!** 🎉

You now have a standalone executable that can be distributed to users without Python or Anaconda installed.

**Next steps:**
- Share with beta testers
- Gather feedback
- Iterate and improve
- Build community

Happy building! 🔨
