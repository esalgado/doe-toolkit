# DOE Toolkit - Installation & Distribution Guide

## For Users: Running the Application

### Download and Run (Windows)
1. Go to the [Releases page](https://github.com/bpimentel3/doe-toolkit/releases)
2. Download `DOE-Toolkit-v0.1.0-win64.zip` from the **Assets** section
3. Extract the ZIP to any folder (right-click → Extract All)
4. Double-click `DOE-Toolkit.bat`
5. Your browser will open automatically with the application

**No Python required. No installation. No admin rights needed.**

**System Requirements:**
- Windows 10 or 11 (64-bit)
- 4 GB RAM minimum, 8 GB recommended
- 500 MB free disk space
- Any modern browser (Chrome, Edge, Firefox)

> **Note:** Do not download the auto-generated "Source code" ZIP from GitHub.
> That does not include the bundled Python environment and will not run.
> Always download the named release asset (`DOE-Toolkit-v0.1.0-win64.zip`).

---

## For Developers: Running from Source

### Setup
```powershell
git clone https://github.com/bpimentel3/doe-toolkit.git
cd doe-toolkit
conda create -n doe-toolkit python=3.11
conda activate doe-toolkit
pip install -r requirements.txt
```

### Run
```powershell
streamlit run src/ui/app.py
```

### Development Workflow
```powershell
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

## For Developers: Building the Release

DOE Toolkit is distributed as a self-contained folder built with `conda-pack`.
Users receive a ZIP, extract it, and double-click `DOE-Toolkit.bat`.

See [BUILD_GUIDE.md](BUILD_GUIDE.md) for full build instructions.

### Quick Summary

**Prerequisites:**
- Anaconda or Miniconda installed
- `doe-toolkit` conda environment with all dependencies
- `conda-pack` installed in the base environment:
  ```powershell
  conda install conda-pack
  ```

**Build:**
```powershell
.\build.ps1
# or
build.bat
```

**Package:**
```powershell
Compress-Archive -Path dist\DOE-Toolkit -DestinationPath dist\DOE-Toolkit-v0.1.0-win64.zip
```

**Distribute:**
Upload `DOE-Toolkit-v0.1.0-win64.zip` as a release asset on GitHub.
Do not rely on GitHub's auto-generated source archives — they do not contain the bundled environment.
