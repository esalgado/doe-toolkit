"""
Synthetic Test Data Generator for DOE-Toolkit End-to-End Testing.

Generates 10 CSV files in DOE-Toolkit format with known relationships to
validate the complete workflow. Each test case covers a different design
scenario to maximize analysis coverage.

All factor columns are written in natural units. Response values are
pre-computed synthetic data using coded values internally, then written
to the CSV as filled response columns alongside the factor data.

Run this script to regenerate all test data files in a ``test_data/``
directory relative to the working directory.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.ui.utils.csv_parser import generate_doe_csv

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir() -> Path:
    """Create and return the test_data output directory."""
    path = Path("test_data")
    path.mkdir(exist_ok=True)
    return path


def _add_noise(values: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    Add proportional Gaussian noise to an array of values.

    Parameters
    ----------
    values : np.ndarray
        True response values.
    noise_level : float
        Noise as a fraction of the absolute value (default 0.05).

    Returns
    -------
    np.ndarray
        Noisy response values.
    """
    return values + np.random.normal(0, noise_level * np.abs(values))


def _write_readme(path: Path, content: str) -> None:
    """Write a README sidecar file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_csv(path: Path, csv_content: str) -> None:
    """Write a CSV file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(csv_content)


# ---------------------------------------------------------------------------
# Test Case 1: Full Factorial – Basic Analysis + Optimization
# ---------------------------------------------------------------------------

def generate_test_case_1(output_dir: Path) -> None:
    """
    Generate 2^3 full factorial with 3 center points (11 runs).

    True model: Yield = 75 + 5*A + 3*B - 2*C + 4*A*B
    Optimal: Temperature=200, Pressure=100, Time=10 → Yield ≈ 91%
    """
    temperature = [150, 200, 150, 200, 150, 200, 150, 200, 175, 175, 175]
    pressure    = [ 50,  50, 100, 100,  50,  50, 100, 100,  75,  75,  75]
    time        = [ 10,  10,  10,  10,  30,  30,  30,  30,  20,  20,  20]

    a = (np.array(temperature) - 175) / 25
    b = (np.array(pressure)    -  75) / 25
    c = (np.array(time)        -  20) / 10

    yield_vals = _add_noise(75 + 5*a + 3*b - 2*c + 4*a*b, 0.03)

    design = pd.DataFrame({
        "StdOrder":    range(1, 12),
        "RunOrder":    [8, 3, 11, 1, 6, 9, 2, 7, 4, 10, 5],
        "Temperature": temperature,
        "Pressure":    pressure,
        "Time":        time,
        "Yield":       yield_vals,
    })

    factors = [
        {"name": "Temperature", "type": "continuous", "changeability": "easy",
         "levels": [150.0, 200.0], "units": "°C"},
        {"name": "Pressure",    "type": "continuous", "changeability": "easy",
         "levels": [50.0,  100.0], "units": "psi"},
        {"name": "Time",        "type": "continuous", "changeability": "easy",
         "levels": [10.0,   30.0], "units": "min"},
    ]
    responses = [{"name": "Yield", "units": "%"}]

    from src.core.factors import Factor, FactorType, ChangeabilityLevel
    factor_objs = [
        Factor(name=fac["name"], factor_type=FactorType(fac["type"]),
               changeability=ChangeabilityLevel(fac["changeability"]),
               levels=fac["levels"], units=fac.get("units"))
        for fac in factors
    ]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="full_factorial",
        design_metadata={
            "center_points": 3,
            "true_model": "Yield = 75 + 5*A + 3*B - 2*C + 4*A*B",
        },
    )
    _write_csv(output_dir / "test_case_1_full_factorial.csv", csv_content)
    _write_readme(
        output_dir / "test_case_1_README.txt",
        "Test Case 1: Basic Full Factorial\n"
        "Design: 2^3 + 3 center points (11 runs)\n"
        "Model: Yield = 75 + 5*A + 3*B - 2*C + 4*A*B\n"
        "Optimal: Temperature=200°C, Pressure=100psi, Time=10min → Yield ≈ 91%\n",
    )


# ---------------------------------------------------------------------------
# Test Case 2: Fractional Factorial Resolution V
# ---------------------------------------------------------------------------

def generate_test_case_2(output_dir: Path) -> None:
    """
    Generate 2^(5-1) Resolution V fractional factorial (16 runs, E=ABCD).

    True model: Strength = 50 + 4*A + 3*B - 2*C + 1.5*D - E
    """
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    # Natural-unit factor centers and half-ranges
    centers   = [50.0, 100.0,  10.0, 200.0,  5.0]
    half_span = [25.0,  50.0,   5.0, 100.0,  2.5]
    names     = ["A", "B", "C", "D", "E"]

    # Base factorial for A–D (16 runs, coded ±1)
    coded_abcd = np.array([
        [-1, -1, -1, -1],
        [ 1, -1, -1, -1],
        [-1,  1, -1, -1],
        [ 1,  1, -1, -1],
        [-1, -1,  1, -1],
        [ 1, -1,  1, -1],
        [-1,  1,  1, -1],
        [ 1,  1,  1, -1],
        [-1, -1, -1,  1],
        [ 1, -1, -1,  1],
        [-1,  1, -1,  1],
        [ 1,  1, -1,  1],
        [-1, -1,  1,  1],
        [ 1, -1,  1,  1],
        [-1,  1,  1,  1],
        [ 1,  1,  1,  1],
    ])
    # Generator E = ABCD
    e_coded = np.prod(coded_abcd, axis=1)
    coded_all = np.column_stack([coded_abcd, e_coded])

    # Convert to natural units
    natural = coded_all * half_span + centers

    a, b, c, d, e = coded_all[:, 0], coded_all[:, 1], coded_all[:, 2], coded_all[:, 3], coded_all[:, 4]
    strength = _add_noise(50 + 4*a + 3*b - 2*c + 1.5*d - e, 0.04)

    design = pd.DataFrame({
        "StdOrder": range(1, 17),
        "RunOrder": np.random.permutation(range(1, 17)),
        "A": natural[:, 0],
        "B": natural[:, 1],
        "C": natural[:, 2],
        "D": natural[:, 3],
        "E": natural[:, 4],
        "Strength": strength,
    })

    factor_objs = [
        Factor(name=names[i], factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[i] - half_span[i], centers[i] + half_span[i]])
        for i in range(5)
    ]
    responses = [{"name": "Strength", "units": "MPa"}]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="fractional_factorial",
        design_metadata={
            "resolution": "V",
            "generator": "E=ABCD",
            "true_model": "Strength = 50 + 4*A + 3*B - 2*C + 1.5*D - E",
        },
    )
    _write_csv(output_dir / "test_case_2_fractional_factorial.csv", csv_content)
    _write_readme(
        output_dir / "test_case_2_README.txt",
        "Test Case 2: Fractional Factorial Res V\n"
        "Design: 2^(5-1), generator E=ABCD (16 runs)\n"
        "Model: Strength = 50 + 4*A + 3*B - 2*C + 1.5*D - E\n",
    )


# ---------------------------------------------------------------------------
# Test Case 3: CCD with Quadratic Effects
# ---------------------------------------------------------------------------

def generate_test_case_3(output_dir: Path) -> None:
    """
    Generate rotatable CCD (8 factorial + 6 axial + 6 center = 20 runs).

    True model: Quality = 80 + 5*A - 3*B + 2*C - 4*A² - 2*B² + 3*A*B
    """
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    alpha = 1.682
    centers   = [100.0, 50.0, 5.0]
    half_span = [ 20.0, 10.0, 2.0]

    factorial_coded = [[-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],
                       [-1,-1,1], [1,-1,1], [-1,1,1], [1,1,1]]
    axial_coded = [
        [alpha,0,0],[-alpha,0,0],
        [0,alpha,0],[0,-alpha,0],
        [0,0,alpha],[0,0,-alpha],
    ]
    center_coded = [[0,0,0]] * 6

    all_coded = np.array(factorial_coded + axial_coded + center_coded)
    natural = all_coded * half_span + centers

    a, b, c = all_coded[:,0], all_coded[:,1], all_coded[:,2]
    quality = _add_noise(80 + 5*a - 3*b + 2*c - 4*a**2 - 2*b**2 + 3*a*b, 0.03)

    design = pd.DataFrame({
        "StdOrder": range(1, 21),
        "RunOrder": np.random.permutation(range(1, 21)),
        "A": natural[:, 0],
        "B": natural[:, 1],
        "C": natural[:, 2],
        "Quality": quality,
    })

    factor_objs = [
        Factor(name="A", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[0]-half_span[0], centers[0]+half_span[0]]),
        Factor(name="B", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[1]-half_span[1], centers[1]+half_span[1]]),
        Factor(name="C", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[2]-half_span[2], centers[2]+half_span[2]]),
    ]
    responses = [{"name": "Quality", "units": "score"}]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="ccd",
        design_metadata={
            "alpha": str(alpha),
            "center_points": 6,
            "true_model": "Quality = 80 + 5*A - 3*B + 2*C - 4*A^2 - 2*B^2 + 3*A*B",
        },
    )
    _write_csv(output_dir / "test_case_3_ccd.csv", csv_content)
    _write_readme(
        output_dir / "test_case_3_README.txt",
        "Test Case 3: CCD Rotatable\n"
        "Design: 8 factorial + 6 axial + 6 center (20 runs)\n"
        "Model: Quality = 80 + 5*A - 3*B + 2*C - 4*A^2 - 2*B^2 + 3*A*B\n",
    )


# ---------------------------------------------------------------------------
# Test Case 4: D-Optimal with Linear Constraint
# ---------------------------------------------------------------------------

def generate_test_case_4(output_dir: Path) -> None:
    """
    Generate D-optimal design (20 runs) with constraint A+B+C ≤ 1.5 (coded).

    Points span the full [-1, 1] coded range subject to the constraint,
    ensuring the design is well-conditioned and the true model coefficients
    are recoverable from a fit.

    True model: Conversion = 60 + 8*A + 5*B + 3*C - 2*A² + 4*A*B
    """
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    # Structured candidate points spanning [-1, 1] per factor with A+B+C <= 1.5.
    # Includes extreme low-corner points (which trivially satisfy the constraint
    # since their sum is at most -3) to ensure the design is not biased toward
    # the positive coded half-space.
    all_coded = np.array([
        # constraint-active boundary (sum = 1.5)
        [ 1.0,  0.5,  0.0],
        [ 1.0,  0.0,  0.5],
        [ 0.5,  1.0,  0.0],
        [ 0.0,  1.0,  0.5],
        [ 0.5,  0.0,  1.0],
        [ 0.0,  0.5,  1.0],
        # low-corner extreme points (sum << 1.5, always feasible)
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        # mid-range interior points
        [ 0.0,  0.0,  0.0],
        [ 0.5,  0.5,  0.0],
        [ 0.0,  0.5,  0.5],
        [ 0.5,  0.0,  0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 1.0, -1.0,  0.0],
        [ 1.0,  0.0, -1.0],
        [-1.0,  1.0,  1.0],  # sum = 1.0, satisfies constraint
    ])

    centers   = [10.0, 20.0, 30.0]
    half_span = [ 5.0, 10.0, 15.0]
    natural   = all_coded * half_span + centers

    a, b, c = all_coded[:,0], all_coded[:,1], all_coded[:,2]
    conversion = _add_noise(60 + 8*a + 5*b + 3*c - 2*a**2 + 4*a*b, 0.04)

    design = pd.DataFrame({
        "StdOrder":   range(1, 21),
        "RunOrder":   np.random.permutation(range(1, 21)),
        "A":          natural[:, 0],
        "B":          natural[:, 1],
        "C":          natural[:, 2],
        "Conversion": conversion,
    })

    factor_objs = [
        Factor(name="A", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[0]-half_span[0], centers[0]+half_span[0]]),
        Factor(name="B", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[1]-half_span[1], centers[1]+half_span[1]]),
        Factor(name="C", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[2]-half_span[2], centers[2]+half_span[2]]),
    ]
    responses = [{"name": "Conversion", "units": "%"}]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="d_optimal",
        design_metadata={
            "constraint": "A+B+C <= 1.5 (coded)",
            "true_model": "Conversion = 60 + 8*A + 5*B + 3*C - 2*A^2 + 4*A*B",
        },
    )
    _write_csv(output_dir / "test_case_4_d_optimal.csv", csv_content)
    _write_readme(
        output_dir / "test_case_4_README.txt",
        "Test Case 4: D-Optimal with Constraint\n"
        "Constraint: A+B+C <= 1.5 (coded)\n"
        "Model: Conversion = 60 + 8*A + 5*B + 3*C - 2*A^2 + 4*A*B\n",
    )


# ---------------------------------------------------------------------------
# Test Case 5: Split-Plot Design
# ---------------------------------------------------------------------------

def generate_test_case_5(output_dir: Path) -> None:
    """
    Generate split-plot design (2 hard + 2 easy factors, 32 runs, 2 replicates).

    True model: Yield = 70 + 5*T + 3*P + 4*Ti + 2*C + 3*T*Ti + errors
    """
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    whole_plots = [
        {"WholePlot": 1, "Temperature": 125, "Pressure": 25},
        {"WholePlot": 2, "Temperature": 175, "Pressure": 25},
        {"WholePlot": 3, "Temperature": 125, "Pressure": 75},
        {"WholePlot": 4, "Temperature": 175, "Pressure": 75},
    ]
    sub_plots = [
        {"Time":  0, "Catalyst": "A"},
        {"Time": 20, "Catalyst": "A"},
        {"Time":  0, "Catalyst": "B"},
        {"Time": 20, "Catalyst": "B"},
    ]

    rows = []
    run_id = 1
    for rep in [1, 2]:
        for wp in whole_plots:
            for sp in sub_plots:
                rows.append({
                    "WholePlot":   wp["WholePlot"],
                    "Replicate":   rep,
                    "Temperature": wp["Temperature"],
                    "Pressure":    wp["Pressure"],
                    "Time":        sp["Time"],
                    "Catalyst":    sp["Catalyst"],
                    "RunOrder":    run_id,
                })
                run_id += 1

    design = pd.DataFrame(rows)
    design.insert(0, "StdOrder", range(1, 33))

    t_coded   = (design["Temperature"] - 150) / 25
    p_coded   = (design["Pressure"]    -  50) / 25
    ti_coded  = (design["Time"]        -  10) / 10
    cat_coded = design["Catalyst"].map({"A": -1, "B": 1})

    wp_error  = np.random.normal(0, 2, 4)
    wp_err    = design["WholePlot"].map(dict(enumerate(wp_error, start=1)))
    sp_error  = np.random.normal(0, 1, 32)

    yield_true = 70 + 5*t_coded + 3*p_coded + 4*ti_coded + 2*cat_coded + 3*t_coded*ti_coded
    design["Yield"] = yield_true + wp_err + sp_error

    factor_objs = [
        Factor(name="Temperature", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.HARD, levels=[125.0, 175.0], units="°C"),
        Factor(name="Pressure",    factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.HARD, levels=[25.0,   75.0], units="psi"),
        Factor(name="Time",        factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[0.0,    20.0], units="min"),
        Factor(name="Catalyst",    factor_type=FactorType.CATEGORICAL,
               changeability=ChangeabilityLevel.EASY, levels=["A", "B"]),
    ]
    responses = [{"name": "Yield", "units": "%"}]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="split_plot",
        design_metadata={
            "hard_factors": "Temperature, Pressure",
            "easy_factors": "Time, Catalyst",
            "true_model":   "Yield = 70 + 5*T + 3*P + 4*Ti + 2*C + 3*T*Ti + errors",
        },
    )
    _write_csv(output_dir / "test_case_5_split_plot.csv", csv_content)
    _write_readme(
        output_dir / "test_case_5_README.txt",
        "Test Case 5: Split-Plot\n"
        "HARD: Temperature, Pressure | EASY: Time, Catalyst\n"
        "Design: 4 whole plots × 4 sub-plots × 2 replicates (32 runs)\n"
        "Model: Yield = 70 + 5*T + 3*P + 4*Ti + 2*C + 3*T*Ti + errors\n",
    )


# ---------------------------------------------------------------------------
# Test Case 6: Box-Behnken with Multiple Responses
# ---------------------------------------------------------------------------

def generate_test_case_6(output_dir: Path) -> None:
    """
    Generate Box-Behnken design (15 runs) with two conflicting responses.

    Yield model: Yield = 85 + 6*A + 4*B + 2*C - 3*A² - 2*B² + 2*A*B
    Cost model:  Cost  = 15 + 3*A - 2*B + C + 1.5*A²
    """
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    centers   = [50.0, 100.0, 10.0]
    half_span = [20.0,  50.0,  5.0]

    bb_coded = np.array([
        [-1,-1, 0],[ 1,-1, 0],[-1, 1, 0],[ 1, 1, 0],
        [-1, 0,-1],[ 1, 0,-1],[-1, 0, 1],[ 1, 0, 1],
        [ 0,-1,-1],[ 0, 1,-1],[ 0,-1, 1],[ 0, 1, 1],
        [ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0],
    ], dtype=float)

    natural = bb_coded * half_span + centers
    a, b, c = bb_coded[:,0], bb_coded[:,1], bb_coded[:,2]

    yield_vals = _add_noise(85 + 6*a + 4*b + 2*c - 3*a**2 - 2*b**2 + 2*a*b, 0.02)
    cost_vals  = _add_noise(15 + 3*a - 2*b + c + 1.5*a**2, 0.03)

    design = pd.DataFrame({
        "StdOrder": range(1, 16),
        "RunOrder": np.random.permutation(range(1, 16)),
        "A":     natural[:, 0],
        "B":     natural[:, 1],
        "C":     natural[:, 2],
        "Yield": yield_vals,
        "Cost":  cost_vals,
    })

    factor_objs = [
        Factor(name="A", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[0]-half_span[0], centers[0]+half_span[0]]),
        Factor(name="B", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[1]-half_span[1], centers[1]+half_span[1]]),
        Factor(name="C", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[2]-half_span[2], centers[2]+half_span[2]]),
    ]
    responses = [{"name": "Yield", "units": "%"}, {"name": "Cost", "units": "$/kg"}]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="box_behnken",
        design_metadata={
            "center_points": 3,
            "yield_model": "Yield = 85 + 6*A + 4*B + 2*C - 3*A^2 - 2*B^2 + 2*A*B",
            "cost_model":  "Cost = 15 + 3*A - 2*B + C + 1.5*A^2",
        },
    )
    _write_csv(output_dir / "test_case_6_multi_response.csv", csv_content)
    _write_readme(
        output_dir / "test_case_6_README.txt",
        "Test Case 6: Box-Behnken Multi-Response\n"
        "Design: 15 runs, 3 center points\n"
        "Yield = 85 + 6*A + 4*B + 2*C - 3*A^2 - 2*B^2 + 2*A*B\n"
        "Cost  = 15 + 3*A - 2*B + C + 1.5*A^2\n"
        "Conflicting objectives: maximize Yield, minimize Cost\n",
    )


# ---------------------------------------------------------------------------
# Test Case 7: Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def generate_test_case_7(output_dir: Path) -> None:
    """
    Generate LHS design (5 factors, 30 runs).

    True model: Performance = 50 + 8*A + 5*B + C - 0.5*D + 6*E
    """
    from scipy.stats import qmc
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    centers   = [10.0, 50.0, 100.0,  5.0, 20.0]
    half_span = [ 5.0, 25.0,  50.0,  2.0, 10.0]
    names     = ["A", "B", "C", "D", "E"]

    sampler   = qmc.LatinHypercube(d=5, seed=42)
    coded_lhs = 2 * sampler.random(n=30) - 1  # scale to [-1, 1]
    natural   = coded_lhs * half_span + centers

    a, b, c, d, e = (coded_lhs[:, i] for i in range(5))
    performance = _add_noise(50 + 8*a + 5*b + c - 0.5*d + 6*e, 0.05)

    design = pd.DataFrame({
        "StdOrder":    range(1, 31),
        "RunOrder":    np.random.permutation(range(1, 31)),
        "A":           natural[:, 0],
        "B":           natural[:, 1],
        "C":           natural[:, 2],
        "D":           natural[:, 3],
        "E":           natural[:, 4],
        "Performance": performance,
    })

    factor_objs = [
        Factor(name=names[i], factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY,
               levels=[centers[i]-half_span[i], centers[i]+half_span[i]])
        for i in range(5)
    ]
    responses = [{"name": "Performance", "units": "score"}]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="lhs",
        design_metadata={
            "sampling":   "Latin Hypercube",
            "true_model": "Performance = 50 + 8*A + 5*B + C - 0.5*D + 6*E",
        },
    )
    _write_csv(output_dir / "test_case_7_lhs.csv", csv_content)
    _write_readme(
        output_dir / "test_case_7_README.txt",
        "Test Case 7: Latin Hypercube Sampling\n"
        "Design: 5 factors, 30 runs\n"
        "Model: Performance = 50 + 8*A + 5*B + C - 0.5*D + 6*E\n"
        "Significant: A, B, E | Weak: C, D\n",
    )


# ---------------------------------------------------------------------------
# Test Case 8: Hierarchy Enforcement
# ---------------------------------------------------------------------------

def generate_test_case_8(output_dir: Path) -> None:
    """
    Generate 2^3 full factorial (8 runs) to test hierarchy enforcement.

    B is not significant but A*B interaction is — tests that stepwise and
    the model builder correctly enforce strong heredity.

    True model: Response = 60 + 5*A + 8*A*B + 2*C
    """
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    a_vals = [10, 20, 10, 20, 10, 20, 10, 20]
    b_vals = [100, 100, 200, 200, 100, 100, 200, 200]
    c_vals = [5, 5, 5, 5, 15, 15, 15, 15]

    a = (np.array(a_vals) - 15) / 5
    b = (np.array(b_vals) - 150) / 50
    c = (np.array(c_vals) - 10) / 5

    response = _add_noise(60 + 5*a + 8*a*b + 2*c, 0.03)

    design = pd.DataFrame({
        "StdOrder": range(1, 9),
        "RunOrder": [5, 2, 7, 1, 8, 3, 6, 4],
        "A":        a_vals,
        "B":        b_vals,
        "C":        c_vals,
        "Response": response,
    })

    factor_objs = [
        Factor(name="A", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[10.0, 20.0]),
        Factor(name="B", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[100.0, 200.0]),
        Factor(name="C", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[5.0, 15.0]),
    ]
    responses = [{"name": "Response", "units": ""}]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="full_factorial",
        design_metadata={
            "true_model": "Response = 60 + 5*A + 8*A*B + 2*C",
            "note": "B not significant but A*B is - tests hierarchy enforcement",
        },
    )
    _write_csv(output_dir / "test_case_8_hierarchy.csv", csv_content)
    _write_readme(
        output_dir / "test_case_8_README.txt",
        "Test Case 8: Hierarchy Enforcement\n"
        "Design: 2^3 full factorial (8 runs)\n"
        "Model: Response = 60 + 5*A + 8*A*B + 2*C\n"
        "B not significant but A*B is — verify hierarchy keeps B in model\n",
    )


# ---------------------------------------------------------------------------
# Test Case 9: Outlier Detection
# ---------------------------------------------------------------------------

def generate_test_case_9(output_dir: Path) -> None:
    """
    Generate 2^3 full factorial (8 runs) with one planted outlier (run 5).

    True model: Quality = 70 + 6*A + 4*B - 2*C + 3*A*B
    Run 5 value set to 40 (expected ≈ 78) to create a detectable outlier.
    """
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    a_vals = [100, 200, 100, 200, 100, 200, 100, 200]
    b_vals = [ 50,  50, 150, 150,  50,  50, 150, 150]
    c_vals = [ 10,  10,  10,  10,  30,  30,  30,  30]

    a = (np.array(a_vals) - 150) / 50
    b = (np.array(b_vals) - 100) / 50
    c = (np.array(c_vals) -  20) / 10

    quality = _add_noise(70 + 6*a + 4*b - 2*c + 3*a*b, 0.02)
    quality[4] = 40.0  # Planted outlier (expected ≈ 78)

    design = pd.DataFrame({
        "StdOrder": range(1, 9),
        "RunOrder": range(1, 9),
        "A":        a_vals,
        "B":        b_vals,
        "C":        c_vals,
        "Quality":  quality,
    })

    factor_objs = [
        Factor(name="A", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[100.0, 200.0]),
        Factor(name="B", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[50.0,  150.0]),
        Factor(name="C", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[10.0,   30.0]),
    ]
    responses = [{"name": "Quality", "units": "score"}]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="full_factorial",
        design_metadata={
            "true_model": "Quality = 70 + 6*A + 4*B - 2*C + 3*A*B",
            "outlier":    "Run 5 is outlier (40 vs expected ~78)",
        },
    )
    _write_csv(output_dir / "test_case_9_outlier.csv", csv_content)
    _write_readme(
        output_dir / "test_case_9_README.txt",
        "Test Case 9: Outlier Detection\n"
        "Design: 2^3 full factorial (8 runs)\n"
        "Model: Quality = 70 + 6*A + 4*B - 2*C + 3*A*B\n"
        "Run 5 is outlier (40 vs expected ~78)\n"
        "Exclude run 5 and verify R² improves\n",
    )


# ---------------------------------------------------------------------------
# Test Case 10: Blocked Design
# ---------------------------------------------------------------------------

def generate_test_case_10(output_dir: Path) -> None:
    """
    Generate 2^4 full factorial with 2 blocks (16 runs, 8 per block).

    True model: Efficiency = 75 + 5*A + 3*B - 2*C + 4*D + 2*A*B + Block
    Block effect: Block 1 = -2, Block 2 = +2.
    """
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    a_vals = [30, 70, 30, 70, 30, 70, 30, 70, 30, 70, 30, 70, 30, 70, 30, 70]
    b_vals = [80, 80,120,120, 80, 80,120,120, 80, 80,120,120, 80, 80,120,120]
    c_vals = [2.5,2.5,2.5,2.5,7.5,7.5,7.5,7.5,2.5,2.5,2.5,2.5,7.5,7.5,7.5,7.5]
    d_vals = [150]*8 + [250]*8
    blocks = [1]*8 + [2]*8

    a = (np.array(a_vals) -  50) / 10
    b = (np.array(b_vals) - 100) / 20
    c = (np.array(c_vals) -   5) / 2.5
    d = (np.array(d_vals) - 200) / 50

    block_effect = np.where(np.array(blocks) == 1, -2.0, 2.0)
    efficiency_true = 75 + 5*a + 3*b - 2*c + 4*d + 2*a*b
    efficiency = efficiency_true + block_effect + _add_noise(efficiency_true, 0.03)

    design = pd.DataFrame({
        "StdOrder":    range(1, 17),
        "RunOrder":    np.random.permutation(range(1, 17)),
        "Block":       blocks,
        "A":           a_vals,
        "B":           b_vals,
        "C":           c_vals,
        "D":           d_vals,
        "Efficiency":  efficiency,
    })

    factor_objs = [
        Factor(name="A", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[40.0,  60.0]),
        Factor(name="B", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[80.0, 120.0]),
        Factor(name="C", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[2.5,    7.5]),
        Factor(name="D", factor_type=FactorType.CONTINUOUS,
               changeability=ChangeabilityLevel.EASY, levels=[150.0, 250.0]),
    ]
    responses = [{"name": "Efficiency", "units": "%"}]

    csv_content = generate_doe_csv(
        design=design,
        factors=factor_objs,
        response_definitions=responses,
        design_type="full_factorial",
        design_metadata={
            "blocks":       2,
            "block_effect": "±2",
            "true_model":   "Efficiency = 75 + 5*A + 3*B - 2*C + 4*D + 2*A*B + Block",
        },
    )
    _write_csv(output_dir / "test_case_10_blocking.csv", csv_content)
    _write_readme(
        output_dir / "test_case_10_README.txt",
        "Test Case 10: Blocked Design\n"
        "Design: 2^4 full factorial, 2 blocks of 8 runs\n"
        "Model: Efficiency = 75 + 5*A + 3*B - 2*C + 4*D + 2*A*B + Block\n"
        "Block effect = ±2\n",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out = _output_dir()

    print("Generating DOE-Toolkit test data...")
    print(f"Output directory: {out.absolute()}\n")

    generators = [
        ("Test Case 1:  Full Factorial",          generate_test_case_1),
        ("Test Case 2:  Fractional Factorial",     generate_test_case_2),
        ("Test Case 3:  CCD",                      generate_test_case_3),
        ("Test Case 4:  D-Optimal",                generate_test_case_4),
        ("Test Case 5:  Split-Plot",               generate_test_case_5),
        ("Test Case 6:  Multi-Response (BBD)",     generate_test_case_6),
        ("Test Case 7:  Latin Hypercube",          generate_test_case_7),
        ("Test Case 8:  Hierarchy Enforcement",    generate_test_case_8),
        ("Test Case 9:  Outlier Detection",        generate_test_case_9),
        ("Test Case 10: Blocking",                 generate_test_case_10),
    ]

    for name, func in generators:
        print(f"  ✓ {name}")
        func(out)

    print(f"\nAll 10 test datasets written to: {out.absolute()}")
