# DOE Toolkit - Quick Start Guide

## What is DOE Toolkit?

DOE Toolkit is a **free, open-source Design of Experiments software** that provides professional statistical design capabilities without expensive licenses.

**Perfect for:**
- Manufacturing engineers optimizing processes
- Pharmaceutical researchers designing experiments
- Quality engineers conducting investigations
- Anyone who needs JMP/Design-Expert capabilities but has a $0 budget

---

## Installation (For Users)

### Windows Installation
1. Download `DOE-Toolkit.zip` from the releases page
2. Extract to any folder (e.g., `C:\Program Files\DOE-Toolkit`)
3. Double-click `DOE-Toolkit.exe`
4. Wait 15-30 seconds for first launch (loading libraries)
5. Your browser will open automatically with the application

**That's it!** No installation, no admin rights needed, no Python required.

---

## First Steps

### Basic Workflow
1. **Define Factors** → Set up your experimental factors (temperature, pressure, etc.)
2. **Choose Design** → Select design type (full factorial, fractional, response surface, etc.)
3. **Preview Design** → Review and download your experimental runs
4. **Import Results** → Upload your data after running experiments
5. **Analyze** → Fit models, check diagnostics, identify significant effects
6. **Optimize** → Find optimal settings for your responses

### Example: Simple 2^3 Factorial

**Goal:** Optimize yield based on Temperature, Pressure, and Time

1. **Define Factors (Page 1)**
   - Factor 1: Temperature (150-200°C)
   - Factor 2: Pressure (50-100 psi)  
   - Factor 3: Time (10-30 min)
   - Response: Yield (%)

2. **Choose Design (Page 3)**
   - Design Type: Full Factorial
   - Add 3 center points
   - Enable randomization

3. **Preview & Export (Page 4)**
   - Review 11 runs (8 factorial + 3 center)
   - Download CSV with run conditions
   - Print for lab use

4. **Run Experiments**
   - Perform experiments in lab
   - Record Yield values
   - Fill in CSV file

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
   - Generate prediction plots

---

## Design Types Available

### Full Factorial
- **Use when:** Need to study all factor combinations
- **Pros:** Complete information, can estimate all interactions
- **Cons:** Number of runs = 2^k (grows quickly)
- **Example:** 2^3 = 8 runs, 2^5 = 32 runs

### Fractional Factorial  
- **Use when:** Many factors, limited budget
- **Pros:** Fewer runs, screen factors efficiently
- **Cons:** Some interactions confounded (aliased)
- **Example:** 2^(5-1) = 16 runs instead of 32

### Response Surface (CCD, Box-Behnken)
- **Use when:** Need to fit quadratic models, find optimal point
- **Pros:** Estimate curvature, find sweet spot
- **Cons:** More runs than screening designs
- **Example:** 3-factor CCD = 20 runs

### D-Optimal
- **Use when:** Have constraints (e.g., ingredients must sum to 100%)
- **Pros:** Handles constraints, irregular regions, any model
- **Cons:** Requires algorithm (not classical design)
- **Example:** Mixture with process variables

### Split-Plot
- **Use when:** Some factors are hard to change (require whole-plot structure)
- **Pros:** Practical, reflects real constraints
- **Cons:** More complex analysis (multiple error terms)
- **Example:** Furnace temperature (hard) + holding time (easy)

---

## Tips & Tricks

### Getting Good Results

**Factor Selection:**
- ✅ Use factors you can actually control
- ✅ Choose realistic ranges (safe operating region)
- ✅ Include factors likely to be important
- ❌ Don't include too many factors initially (start with screening)

**Design Selection:**
- **Screening:** Use fractional factorial (2^(k-p))
- **Optimization:** Use response surface (CCD, Box-Behnken)
- **Constraints:** Use D-optimal
- **Hard-to-change factors:** Use split-plot

**Model Building:**
- ✅ Start simple (main effects only)
- ✅ Add interactions if significant
- ✅ Check diagnostics (normality, constant variance)
- ✅ Remove insignificant terms (unless hierarchy)
- ❌ Don't overfit (too many terms for too few runs)

**Validation:**
- ✅ Run center points (check curvature)
- ✅ Run confirmation runs at optimal settings
- ✅ Check predicted vs actual values
- ❌ Don't extrapolate beyond design space

### Common Pitfalls

**Problem:** "My model R² is low"
- Check if you included the right factors
- Look for outliers or measurement errors
- Consider adding interaction terms
- May need quadratic model (response surface)

**Problem:** "Nothing is significant"
- Factor ranges may be too narrow
- Measurement error may be too large
- May need more replicates
- Check if you randomized properly

**Problem:** "Everything is significant"
- May be overfitting
- Check if effects are practically significant (not just statistically)
- Simplify model (remove small effects)

---

## Example Datasets

The toolkit includes 10 test datasets in `test_data/`:

1. **Full Factorial** - Basic 2^3 with center points
2. **Fractional Factorial** - 2^(5-1) resolution V
3. **CCD** - Central composite design
4. **D-Optimal** - Design with constraints
5. **Split-Plot** - Hard and easy factors
6. **Multi-Response** - Multiple objectives
7. **Latin Hypercube** - Space-filling design
8. **Hierarchy** - Tests hierarchy enforcement
9. **Outlier** - Outlier detection example
10. **Blocking** - Nuisance variable handling

Try importing these to learn the workflow!

---

## Need Help?

### Documentation
- **Algorithm Details:** See `docs/algorithms/` for mathematical explanations
- **API Reference:** See docstrings in source code
- **Issues/Bugs:** Report on GitHub issues page

### Learning Resources
- **NIST Engineering Statistics Handbook:** Free online DOE guide
- **Montgomery "Design and Analysis of Experiments":** Classic textbook
- **Box, Hunter, Hunter "Statistics for Experimenters":** Practical approach

### Community
- **GitHub:** https://github.com/bpimentel3/doe-toolkit
- **Issues:** Report bugs or request features
- **Discussions:** Ask questions, share use cases

---

## What Makes DOE Toolkit Different?

**vs. JMP/Design-Expert:**
- ✅ Free (they cost $1,000-8,400/year)
- ✅ Open source
- ✅ Local (your data never leaves your computer)
- ❌ Fewer advanced features (for now)

**vs. Python libraries (pyDOE3, etc.):**
- ✅ No coding required (GUI-based)
- ✅ Complete workflow (design + analyze + optimize)
- ✅ Split-plot and optimal designs included
- ✅ Professional output and exports

**vs. Online tools:**
- ✅ Works offline
- ✅ No data uploaded to cloud
- ✅ Unlimited use (no trials or limits)
- ✅ Full control of your data

---

## Support the Project

DOE Toolkit is free and open source. Ways to help:

- ⭐ Star the GitHub repository
- 🐛 Report bugs or suggest features
- 📝 Share your success stories
- 💻 Contribute code (see CONTRIBUTING.md)
- 📚 Improve documentation

---

**Happy Experimenting!** 🔬📊

*DOE Toolkit - Professional design of experiments for everyone*
