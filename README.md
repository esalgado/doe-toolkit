# DOE Toolkit

Free, open-source Design of Experiments software for engineers.

Provides professional statistical design capabilities — without the expensive licenses.

---

## Features

- Full factorial and fractional factorial designs
- Response surface methods (Central Composite, Box-Behnken)
- D-optimal designs with linear and nonlinear constraints
- Split-plot designs for hard-to-change factors
- Categorical and discrete numeric factors
- Design augmentation (foldover, axial points, optimal augmentation)
- Complete analysis workflow: design → analyze → optimize
- Stepwise model selection with hierarchy enforcement
- Response optimization using desirability functions
- HTML report export and project save/load

---

## Who Is This For?

- Manufacturing engineers optimizing processes
- Pharmaceutical researchers designing experiments
- Quality engineers conducting investigations
- Anyone who needs JMP or Design-Expert capabilities on a $0 budget

---

## Getting Started

### For Users (Windows)
1. Download `DOE-Toolkit.zip` from the [Releases page](https://github.com/bpimentel3/doe-toolkit/releases)
2. Extract to any folder
3. Double-click `DOE-Toolkit.bat`
4. The app opens in your browser automatically

No Python required. No installation. No admin rights needed.

See [QUICKSTART.md](QUICKSTART.md) for a full walkthrough.

### For Developers
```bash
git clone https://github.com/bpimentel3/doe-toolkit.git
cd doe-toolkit
conda create -n doe-toolkit python=3.11
conda activate doe-toolkit
pip install -r requirements.txt
streamlit run src/ui/app.py
```

See [INSTALL.md](INSTALL.md) for build instructions and development workflow.

---

## Design Types

| Design | Best For |
|---|---|
| Full Factorial | All factor combinations, complete information |
| Fractional Factorial | Screening many factors with fewer runs |
| Central Composite (CCD) | Response surface, finding optima |
| Box-Behnken | Response surface without extreme corners |
| D-Optimal | Constrained regions, irregular design spaces |
| Split-Plot | Hard-to-change factors in practice |
| Latin Hypercube | Space-filling, simulation experiments |

---

## Why DOE Toolkit?

**vs. JMP / Design-Expert:** Free and open source (they cost $1,000–8,400/year)

**vs. Python libraries (pyDOE3, etc.):** No coding required — full GUI with complete workflow

**vs. Online tools:** Works offline, your data never leaves your machine

---

## System Requirements

- Windows 10 or 11 (64-bit)
- 4 GB RAM minimum, 8 GB recommended
- 500 MB free disk space
- Any modern browser (Chrome, Edge, Firefox)

---

## Contributing

Bug reports, feature requests, and pull requests are welcome.

- [Open an issue](https://github.com/bpimentel3/doe-toolkit/issues)
- [Read the build guide](BUILD_GUIDE.md)
- [Browse the docs](docs/)

---

## License

MIT License — see [LICENSE](LICENSE.txt) file for details.
