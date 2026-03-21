# DOE Toolkit - Quick Start Guide

## What is DOE Toolkit?

DOE Toolkit is a **free, open-source Design of Experiments software** that provides professional statistical design capabilities without expensive licenses.

**Perfect for:**
- Manufacturing engineers optimizing processes
- Pharmaceutical researchers designing experiments
- Quality engineers conducting investigations
- Anyone who needs JMP/Design-Expert capabilities but has a $0 budget

---

## Installation (Windows)

1. Go to the [Releases page](https://github.com/bpimentel3/doe-toolkit/releases)
2. Download `DOE-Toolkit-v0.1.0-win64.zip` from the **Assets** section
3. Extract to any folder (right-click → Extract All)
4. Double-click `DOE-Toolkit.bat`
5. Your browser will open automatically with the application

**That's it!** No installation, no admin rights needed, no Python required.

> **Note:** Do not download the auto-generated "Source code" ZIP from GitHub.
> Always download the named release asset (`DOE-Toolkit-v0.1.0-win64.zip`).

---

## Basic Workflow

1. **Define Factors** → Set up your experimental factors (temperature, pressure, etc.)
2. **Select Model** → Choose which model terms to estimate
3. **Choose Design** → Select design type (full factorial, fractional, response surface, etc.)
4. **Preview Design** → Review and export your experimental runs
5. **Import Results** → Upload your data after running experiments
6. **Analyze** → Fit models, check diagnostics, identify significant effects
7. **Augment** *(optional)* → Add runs to an existing design
8. **Optimize** → Find optimal settings for your responses

---

## Example: Simple 2³ Factorial

**Goal:** Optimize yield based on Temperature, Pressure, and Time

1. **Define Factors (Page 1)**
   - Factor 1: Temperature (150–200°C)
   - Factor 2: Pressure (50–100 psi)
   - Factor 3: Time (10–30 min)
   - Response: Yield (%)

2. **Choose Design (Page 3)**
   - Design Type: Full Factorial
   - Add 3 center points
   - Enable randomization

3. **Preview & Export (Page 4)**
   - Review 11 runs (8 factorial + 3 center points)
   - Download CSV with run conditions

4. **Run Experiments**
   - Perform experiments in the lab
   - Record Yield values in the CSV

5. **Import Results (Page 5)**
   - Upload completed CSV
   - Verify data loaded correctly

6. **Analyze (Page 6)**
   - Fit model (main effects + interactions)
   - Check R² and p-values
   - Review diagnostic plots
   - Identify significant factors

7. **Optimize (Page 8)**
   - Set target: Maximize Yield
   - Find optimal settings
   - Review prediction plots

---

## Design Types Available

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

## Tips & Tricks

**Factor Selection:**
- Use factors you can actually control
- Choose realistic ranges (safe operating region)
- Start with screening before optimization

**Design Selection:**
- Screening → fractional factorial
- Optimization → response surface (CCD, Box-Behnken)
- Constraints → D-optimal
- Hard-to-change factors → split-plot

**Model Building:**
- Start simple (main effects only)
- Add interactions if significant
- Check diagnostics (normality, constant variance)
- Remove insignificant terms while respecting hierarchy

---

## Example Datasets

The toolkit includes test datasets in `test_data/` covering full factorial, fractional factorial, CCD, D-optimal, split-plot, multi-response, Latin hypercube, and more. Try importing these to learn the workflow.

---

## Need Help?

- **Issues/Bugs:** [GitHub Issues](https://github.com/bpimentel3/doe-toolkit/issues)
- **Algorithm details:** See `docs/algorithms/`
- **Build instructions:** See [BUILD_GUIDE.md](BUILD_GUIDE.md)

### Learning Resources
- *NIST Engineering Statistics Handbook* — free online DOE reference
- Montgomery, *Design and Analysis of Experiments* — classic textbook
- Box, Hunter & Hunter, *Statistics for Experimenters* — practical approach

---

## Why DOE Toolkit?

**vs. JMP / Design-Expert:** Free and open source (they cost $1,000–8,400/year)

**vs. Python libraries (pyDOE3, etc.):** No coding required — full GUI with complete workflow

**vs. Online tools:** Works offline, your data never leaves your machine

---

**Happy Experimenting!** 🔬📊
