"""
Synthetic Test Data Generator for DOE-Toolkit End-to-End Testing

Generates 10 CSV files in DOE-Toolkit format with known relationships to validate
the complete workflow. Each test case includes different scenarios to maximize coverage.

Run this script to generate all test data files in a 'test_data/' directory.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)


def create_test_data_directory():
    """Create directory for test data files."""
    Path("test_data").mkdir(exist_ok=True)
    return Path("test_data")


def add_noise(values, noise_level=0.05):
    """Add proportional random noise to values."""
    noise = np.random.normal(0, noise_level * np.abs(values))
    return values + noise


def generate_doe_toolkit_csv(
    design, factors_metadata, responses_metadata, design_type, design_metadata=None
):
    """
    Generate DOE-Toolkit CSV format with metadata header.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor and response columns
    factors_metadata : list of dict
        Each dict has: name, type, changeability, levels, units
    responses_metadata : list of dict
        Each dict has: name, units
    design_type : str
        Type of design (e.g., 'full_factorial', 'ccd')
    design_metadata : dict, optional
        Additional metadata (resolution, alpha, etc.)

    Returns
    -------
    str
        CSV content with DOE-Toolkit format
    """
    lines = []

    # Header
    lines.append("# DOE-TOOLKIT DESIGN")
    lines.append("# Version: 1.0")
    lines.append(f"# Design Type: {design_type}")
    lines.append(f"# Generated: {datetime.now().isoformat()}")

    # Additional metadata
    if design_metadata:
        for key, value in design_metadata.items():
            formatted_key = key.replace("_", " ").title()
            lines.append(f"# {formatted_key}: {value}")

    # Factor definitions
    lines.append("#")
    lines.append("# FACTOR DEFINITIONS")
    lines.append("# Name,Type,Changeability,Levels,Units")

    for factor in factors_metadata:
        name = factor["name"]
        type_str = factor["type"]
        changeability = factor["changeability"]
        levels = factor["levels"]
        units = factor.get("units", "")

        # Format levels
        if isinstance(levels, list):
            if isinstance(levels[0], str):
                levels_str = "|".join(levels)
            else:
                levels_str = "|".join(str(v) for v in levels)
        else:
            levels_str = levels

        lines.append(
            f"# {name},{type_str},{changeability},{levels_str},{units}"
        )

    # Response definitions
    lines.append("#")
    lines.append("# RESPONSE DEFINITIONS")
    lines.append("# Name,Units")

    for response in responses_metadata:
        name = response["name"]
        units = response.get("units", "")
        lines.append(f"# {name},{units}")

    # Design data
    lines.append("#")
    lines.append("# DESIGN DATA")

    csv_data = design.to_csv(index=False)
    lines.append(csv_data.rstrip())

    return "\n".join(lines)


# ============================================================================
# Test Case 1: Basic Full Factorial → Analysis → Optimization
# ============================================================================
def generate_test_case_1(output_dir):
    """3 factors, 2^3 + 3 center = 11 runs, Single response"""
    design = pd.DataFrame(
        {
            "StdOrder": range(1, 12),
            "RunOrder": [8, 3, 11, 1, 6, 9, 2, 7, 4, 10, 5],
            "Temperature": [150, 200, 150, 200, 150, 200, 150, 200, 175, 175, 175],
            "Pressure": [50, 50, 100, 100, 50, 50, 100, 100, 75, 75, 75],
            "Time": [10, 10, 10, 10, 30, 30, 30, 30, 20, 20, 20],
        }
    )

    # Calculate response using coded values
    temp_coded = (design["Temperature"] - 175) / 25
    press_coded = (design["Pressure"] - 75) / 25
    time_coded = (design["Time"] - 20) / 10

    yield_true = (
        75 + 5 * temp_coded + 3 * press_coded - 2 * time_coded + 4 * temp_coded * press_coded
    )
    design["Yield"] = add_noise(yield_true, 0.03)

    # Factor metadata
    factors = [
        {
            "name": "Temperature",
            "type": "continuous",
            "changeability": "easy",
            "levels": "150|200",
            "units": "°C",
        },
        {
            "name": "Pressure",
            "type": "continuous",
            "changeability": "easy",
            "levels": "50|100",
            "units": "psi",
        },
        {
            "name": "Time",
            "type": "continuous",
            "changeability": "easy",
            "levels": "10|30",
            "units": "min",
        },
    ]

    responses = [{"name": "Yield", "units": "%"}]

    metadata = {"center_points": 3, "true_model": "Yield = 75 + 5*A + 3*B - 2*C + 4*A*B"}

    csv_content = generate_doe_toolkit_csv(design, factors, responses, "full_factorial", metadata)

    with open(output_dir / "test_case_1_full_factorial.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_1_README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Test Case 1: Basic Full Factorial\n"
            "Model: Yield = 75 + 5*A + 3*B - 2*C + 4*A*B\n"
            "Optimal: A=+1, B=+1, C=-1 -> Yield ~= 91%\n"
        )


# ============================================================================
# Test Case 2: Fractional Factorial Resolution V
# ============================================================================
def generate_test_case_2(output_dir):
    """5 factors, 2^(5-1) = 16 runs, E=ABCD generator"""
    design = pd.DataFrame(
        {
            "StdOrder": range(1, 17),
            "RunOrder": np.random.permutation(range(1, 17)),
            "A": [25, 75, 25, 75, 25, 75, 25, 75, 25, 75, 25, 75, 25, 75, 25, 75],
            "B": [50, 50, 150, 150, 50, 50, 150, 150, 50, 50, 150, 150, 50, 50, 150, 150],
            "C": [5, 5, 5, 5, 15, 15, 15, 15, 5, 5, 5, 5, 15, 15, 15, 15],
            "D": [100, 100, 100, 100, 100, 100, 100, 100, 300, 300, 300, 300, 300, 300, 300, 300],
            "E": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5],
        }
    )

    # Code factors
    a_coded = (design["A"] - 50) / 25
    b_coded = (design["B"] - 100) / 50
    c_coded = (design["C"] - 10) / 5
    d_coded = (design["D"] - 200) / 100
    # E = ABCD
    design["E"] = np.where(
        a_coded * b_coded * c_coded * d_coded > 0, 7.5, 2.5
    )
    e_coded = (design["E"] - 5) / 2.5

    strength_true = 50 + 4 * a_coded + 3 * b_coded - 2 * c_coded + 1.5 * d_coded - e_coded
    design["Strength"] = add_noise(strength_true, 0.04)

    factors = [
        {"name": "A", "type": "continuous", "changeability": "easy", "levels": "25|75", "units": ""},
        {
            "name": "B",
            "type": "continuous",
            "changeability": "easy",
            "levels": "50|150",
            "units": "",
        },
        {"name": "C", "type": "continuous", "changeability": "easy", "levels": "5|15", "units": ""},
        {
            "name": "D",
            "type": "continuous",
            "changeability": "easy",
            "levels": "100|300",
            "units": "",
        },
        {
            "name": "E",
            "type": "continuous",
            "changeability": "easy",
            "levels": "2.5|7.5",
            "units": "",
        },
    ]

    responses = [{"name": "Strength", "units": "MPa"}]

    metadata = {
        "resolution": "V",
        "generator": "E=ABCD",
        "true_model": "Strength = 50 + 4*A + 3*B - 2*C + 1.5*D - E",
    }

    csv_content = generate_doe_toolkit_csv(
        design, factors, responses, "fractional_factorial", metadata
    )

    with open(output_dir / "test_case_2_fractional_factorial.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_2_README.txt", "w") as f:
        f.write(
            "Test Case 2: Fractional Factorial Res V\n"
            "Generator: E=ABCD\n"
            "Model: Strength = 50 + 4*A + 3*B - 2*C + 1.5*D - E\n"
        )


# ============================================================================
# Test Case 3: CCD with Quadratic
# ============================================================================
def generate_test_case_3(output_dir):
    """3 factors, CCD: 8+6+6=20 runs"""
    alpha = 1.682
    factorial_coded = [
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
    ]
    axial_coded = [
        [alpha, 0, 0],
        [-alpha, 0, 0],
        [0, alpha, 0],
        [0, -alpha, 0],
        [0, 0, alpha],
        [0, 0, -alpha],
    ]
    center_coded = [[0, 0, 0]] * 6

    all_coded = factorial_coded + axial_coded + center_coded

    design = pd.DataFrame(
        {
            "StdOrder": range(1, 21),
            "RunOrder": np.random.permutation(range(1, 21)),
            "A": [100 + x[0] * 20 for x in all_coded],
            "B": [50 + x[1] * 10 for x in all_coded],
            "C": [5 + x[2] * 2 for x in all_coded],
        }
    )

    a_coded = (design["A"] - 100) / 20
    b_coded = (design["B"] - 50) / 10
    c_coded = (design["C"] - 5) / 2

    quality_true = (
        80 + 5 * a_coded - 3 * b_coded + 2 * c_coded - 4 * a_coded**2 - 2 * b_coded**2 + 3 * a_coded * b_coded
    )
    design["Quality"] = add_noise(quality_true, 0.03)

    factors = [
        {
            "name": "A",
            "type": "continuous",
            "changeability": "easy",
            "levels": "80|120",
            "units": "",
        },
        {
            "name": "B",
            "type": "continuous",
            "changeability": "easy",
            "levels": "40|60",
            "units": "",
        },
        {"name": "C", "type": "continuous", "changeability": "easy", "levels": "3|7", "units": ""},
    ]

    responses = [{"name": "Quality", "units": "score"}]

    metadata = {
        "alpha": str(alpha),
        "center_points": 6,
        "true_model": "Quality = 80 + 5*A - 3*B + 2*C - 4*A^2 - 2*B^2 + 3*A*B",
    }

    csv_content = generate_doe_toolkit_csv(design, factors, responses, "ccd", metadata)

    with open(output_dir / "test_case_3_ccd.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_3_README.txt", "w") as f:
        f.write(
            "Test Case 3: CCD Rotatable\n"
            "Model: Quality = 80 + 5*A - 3*B + 2*C - 4*A^2 - 2*B^2 + 3*A*B\n"
        )


# ============================================================================
# Test Case 4: D-Optimal with Constraints
# ============================================================================
def generate_test_case_4(output_dir):
    """3 factors, constraint A+B+C≤1.5, 20 runs"""
    np.random.seed(42)
    points_coded = [
        [1, 0.5, 0],
        [0.5, 1, 0],
        [0, 1, 0.5],
        [1, 0, 0.5],
        [0.5, 0, 1],
        [0, 0.5, 1],
        [0.75, 0.75, 0],
        [0.5, 0.5, 0.5],
        [0, 0.75, 0.75],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    for _ in range(8):
        while True:
            pt = np.random.uniform(-0.5, 1, 3)
            if np.sum(pt) <= 1.5:
                points_coded.append(pt.tolist())
                break

    design = pd.DataFrame(
        {
            "StdOrder": range(1, 21),
            "RunOrder": np.random.permutation(range(1, 21)),
            "A": [10 + x[0] * 5 for x in points_coded[:20]],
            "B": [20 + x[1] * 10 for x in points_coded[:20]],
            "C": [30 + x[2] * 15 for x in points_coded[:20]],
        }
    )

    a_coded = (design["A"] - 10) / 5
    b_coded = (design["B"] - 20) / 10
    c_coded = (design["C"] - 30) / 15

    conversion_true = (
        60 + 8 * a_coded + 5 * b_coded + 3 * c_coded - 2 * a_coded**2 + 4 * a_coded * b_coded
    )
    design["Conversion"] = add_noise(conversion_true, 0.04)

    factors = [
        {
            "name": "A",
            "type": "continuous",
            "changeability": "easy",
            "levels": "7.5|15",
            "units": "",
        },
        {
            "name": "B",
            "type": "continuous",
            "changeability": "easy",
            "levels": "15|30",
            "units": "",
        },
        {
            "name": "C",
            "type": "continuous",
            "changeability": "easy",
            "levels": "22.5|45",
            "units": "",
        },
    ]

    responses = [{"name": "Conversion", "units": "%"}]

    metadata = {
        "constraint": "A+B+C <= 1.5 (coded)",
        "true_model": "Conversion = 60 + 8*A + 5*B + 3*C - 2*A^2 + 4*A*B",
    }

    csv_content = generate_doe_toolkit_csv(design, factors, responses, "d_optimal", metadata)

    with open(output_dir / "test_case_4_d_optimal.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_4_README.txt", "w") as f:
        f.write(
            "Test Case 4: D-Optimal\n"
            "Constraint: A+B+C <= 1.5\n"
            "Model: Conversion = 60 + 8*A + 5*B + 3*C - 2*A^2 + 4*A*B\n"
        )


# ============================================================================
# Test Case 5: Split-Plot
# ============================================================================
def generate_test_case_5(output_dir):
    """2 HARD + 2 EASY factors, 32 runs (2 replicates)"""
    whole_plots_coded = [
        {"WholePlot": 1, "Temp": -1, "Press": -1},
        {"WholePlot": 2, "Temp": 1, "Press": -1},
        {"WholePlot": 3, "Temp": -1, "Press": 1},
        {"WholePlot": 4, "Temp": 1, "Press": 1},
    ]

    sub_plots_coded = [
        {"Time": -1, "Cat": -1},
        {"Time": 1, "Cat": -1},
        {"Time": -1, "Cat": 1},
        {"Time": 1, "Cat": 1},
    ]

    rows = []
    run_id = 1
    for rep in [1, 2]:
        for wp in whole_plots_coded:
            for sp in sub_plots_coded:
                rows.append(
                    {
                        "WholePlot": wp["WholePlot"],
                        "Replicate": rep,
                        "Temperature": 150 + wp["Temp"] * 25,
                        "Pressure": 50 + wp["Press"] * 25,
                        "Time": 10 + sp["Time"] * 10,
                        "Catalyst": "A" if sp["Cat"] == -1 else "B",
                        "RunOrder": run_id,
                    }
                )
                run_id += 1

    design = pd.DataFrame(rows)
    design.insert(0, "StdOrder", range(1, 33))

    # Calculate response
    temp_coded = (design["Temperature"] - 150) / 25
    press_coded = (design["Pressure"] - 50) / 25
    time_coded = (design["Time"] - 10) / 10
    cat_coded = design["Catalyst"].map({"A": -1, "B": 1})

    wp_error = np.random.normal(0, 2, 4)
    wp_map = {1: wp_error[0], 2: wp_error[1], 3: wp_error[2], 4: wp_error[3]}
    wp_err = design["WholePlot"].map(wp_map)
    sp_error = np.random.normal(0, 1, 32)

    yield_true = (
        70 + 5 * temp_coded + 3 * press_coded + 4 * time_coded + 2 * cat_coded + 3 * temp_coded * time_coded
    )
    design["Yield"] = yield_true + wp_err + sp_error

    factors = [
        {
            "name": "Temperature",
            "type": "continuous",
            "changeability": "hard",
            "levels": "125|175",
            "units": "°C",
        },
        {
            "name": "Pressure",
            "type": "continuous",
            "changeability": "hard",
            "levels": "25|75",
            "units": "psi",
        },
        {
            "name": "Time",
            "type": "continuous",
            "changeability": "easy",
            "levels": "0|20",
            "units": "min",
        },
        {
            "name": "Catalyst",
            "type": "categorical",
            "changeability": "easy",
            "levels": "A|B",
            "units": "",
        },
    ]

    responses = [{"name": "Yield", "units": "%"}]

    metadata = {
        "hard_factors": "Temperature, Pressure",
        "easy_factors": "Time, Catalyst",
        "true_model": "Yield = 70 + 5*T + 3*P + 4*Ti + 2*C + 3*T*Ti + errors",
    }

    csv_content = generate_doe_toolkit_csv(design, factors, responses, "split_plot", metadata)

    with open(output_dir / "test_case_5_split_plot.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_5_README.txt", "w") as f:
        f.write(
            "Test Case 5: Split-Plot\n"
            "HARD: Temp, Press | EASY: Time, Cat\n"
            "Model: Yield = 70 + 5*T + 3*P + 4*Ti + 2*C + 3*T*Ti + errors\n"
        )


# ============================================================================
# Test Case 6: Multi-Response
# ============================================================================
def generate_test_case_6(output_dir):
    """Box-Behnken, 15 runs, 2 responses"""
    points_coded = [
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        [1, 1, 0],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        [1, 0, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
        [0, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    design = pd.DataFrame(
        {
            "StdOrder": range(1, 16),
            "RunOrder": np.random.permutation(range(1, 16)),
            "A": [50 + x[0] * 20 for x in points_coded],
            "B": [100 + x[1] * 50 for x in points_coded],
            "C": [10 + x[2] * 5 for x in points_coded],
        }
    )

    a_coded = (design["A"] - 50) / 20
    b_coded = (design["B"] - 100) / 50
    c_coded = (design["C"] - 10) / 5

    yield_true = (
        85 + 6 * a_coded + 4 * b_coded + 2 * c_coded - 3 * a_coded**2 - 2 * b_coded**2 + 2 * a_coded * b_coded
    )
    cost_true = 15 + 3 * a_coded - 2 * b_coded + 1 * c_coded + 1.5 * a_coded**2
    design["Yield"] = add_noise(yield_true, 0.02)
    design["Cost"] = add_noise(cost_true, 0.03)

    factors = [
        {
            "name": "A",
            "type": "continuous",
            "changeability": "easy",
            "levels": "30|70",
            "units": "",
        },
        {
            "name": "B",
            "type": "continuous",
            "changeability": "easy",
            "levels": "50|150",
            "units": "",
        },
        {
            "name": "C",
            "type": "continuous",
            "changeability": "easy",
            "levels": "5|15",
            "units": "",
        },
    ]

    responses = [{"name": "Yield", "units": "%"}, {"name": "Cost", "units": "$/kg"}]

    metadata = {
        "center_points": 3,
        "yield_model": "Yield = 85 + 6*A + 4*B + 2*C - 3*A^2 - 2*B^2 + 2*A*B",
        "cost_model": "Cost = 15 + 3*A - 2*B + C + 1.5*A^2",
    }

    csv_content = generate_doe_toolkit_csv(
        design, factors, responses, "box_behnken", metadata
    )

    with open(output_dir / "test_case_6_multi_response.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_6_README.txt", "w") as f:
        f.write(
            "Test Case 6: Multi-Response\n"
            "Yield = 85 + 6*A + 4*B + 2*C - 3*A^2 - 2*B^2 + 2*A*B\n"
            "Cost = 15 + 3*A - 2*B + C + 1.5*A^2\n"
            "Conflicting objectives\n"
        )


# ============================================================================
# Test Case 7: Latin Hypercube
# ============================================================================
def generate_test_case_7(output_dir):
    """5 factors, 30 runs, LHS"""
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=5, seed=42)
    sample = sampler.random(n=30)
    lhs_coded = 2 * sample - 1

    design = pd.DataFrame(
        {
            "StdOrder": range(1, 31),
            "RunOrder": np.random.permutation(range(1, 31)),
            "A": 10 + lhs_coded[:, 0] * 5,
            "B": 50 + lhs_coded[:, 1] * 25,
            "C": 100 + lhs_coded[:, 2] * 50,
            "D": 5 + lhs_coded[:, 3] * 2,
            "E": 20 + lhs_coded[:, 4] * 10,
        }
    )

    a_coded = (design["A"] - 10) / 5
    b_coded = (design["B"] - 50) / 25
    c_coded = (design["C"] - 100) / 50
    d_coded = (design["D"] - 5) / 2
    e_coded = (design["E"] - 20) / 10

    performance_true = (
        50 + 8 * a_coded + 5 * b_coded + 1 * c_coded - 0.5 * d_coded + 6 * e_coded
    )
    design["Performance"] = add_noise(performance_true, 0.05)

    factors = [
        {"name": "A", "type": "continuous", "changeability": "easy", "levels": "5|15", "units": ""},
        {
            "name": "B",
            "type": "continuous",
            "changeability": "easy",
            "levels": "25|75",
            "units": "",
        },
        {
            "name": "C",
            "type": "continuous",
            "changeability": "easy",
            "levels": "50|150",
            "units": "",
        },
        {"name": "D", "type": "continuous", "changeability": "easy", "levels": "3|7", "units": ""},
        {
            "name": "E",
            "type": "continuous",
            "changeability": "easy",
            "levels": "10|30",
            "units": "",
        },
    ]

    responses = [{"name": "Performance", "units": "score"}]

    metadata = {
        "sampling": "Latin Hypercube",
        "true_model": "Performance = 50 + 8*A + 5*B + C - 0.5*D + 6*E",
    }

    csv_content = generate_doe_toolkit_csv(design, factors, responses, "lhs", metadata)

    with open(output_dir / "test_case_7_lhs.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_7_README.txt", "w") as f:
        f.write(
            "Test Case 7: LHS\n"
            "Model: Performance = 50 + 8*A + 5*B + C - 0.5*D + 6*E\n"
            "Significant: A, B, E | Weak: C, D\n"
        )


# ============================================================================
# Test Case 8: Hierarchy Testing
# ============================================================================
def generate_test_case_8(output_dir):
    """2^3 for hierarchy enforcement testing"""
    design = pd.DataFrame(
        {
            "StdOrder": range(1, 9),
            "RunOrder": [5, 2, 7, 1, 8, 3, 6, 4],
            "A": [10, 20, 10, 20, 10, 20, 10, 20],
            "B": [100, 100, 200, 200, 100, 100, 200, 200],
            "C": [5, 5, 5, 5, 15, 15, 15, 15],
        }
    )

    a_coded = (design["A"] - 15) / 5
    b_coded = (design["B"] - 150) / 50
    c_coded = (design["C"] - 10) / 5

    response_true = 60 + 5 * a_coded + 8 * a_coded * b_coded + 2 * c_coded
    design["Response"] = add_noise(response_true, 0.03)

    factors = [
        {
            "name": "A",
            "type": "continuous",
            "changeability": "easy",
            "levels": "10|20",
            "units": "",
        },
        {
            "name": "B",
            "type": "continuous",
            "changeability": "easy",
            "levels": "100|200",
            "units": "",
        },
        {
            "name": "C",
            "type": "continuous",
            "changeability": "easy",
            "levels": "5|15",
            "units": "",
        },
    ]

    responses = [{"name": "Response", "units": ""}]

    metadata = {
        "true_model": "Response = 60 + 5*A + 8*A*B + 2*C",
        "note": "B not significant but A*B is - test hierarchy enforcement",
    }

    csv_content = generate_doe_toolkit_csv(
        design, factors, responses, "full_factorial", metadata
    )

    with open(output_dir / "test_case_8_hierarchy.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_8_README.txt", "w") as f:
        f.write(
            "Test Case 8: Hierarchy\n"
            "Model: Response = 60 + 5*A + 8*A*B + 2*C\n"
            "B not significant but A*B is - test hierarchy enforcement\n"
        )


# ============================================================================
# Test Case 9: Outlier Detection
# ============================================================================
def generate_test_case_9(output_dir):
    """2^3 with one obvious outlier"""
    design = pd.DataFrame(
        {
            "StdOrder": range(1, 9),
            "RunOrder": range(1, 9),
            "A": [100, 200, 100, 200, 100, 200, 100, 200],
            "B": [50, 50, 150, 150, 50, 50, 150, 150],
            "C": [10, 10, 10, 10, 30, 30, 30, 30],
        }
    )

    a_coded = (design["A"] - 150) / 50
    b_coded = (design["B"] - 100) / 50
    c_coded = (design["C"] - 20) / 10

    quality_true = 70 + 6 * a_coded + 4 * b_coded - 2 * c_coded + 3 * a_coded * b_coded
    quality = add_noise(quality_true, 0.02)
    quality.iloc[4] = 40  # Obvious outlier (should be ~78)
    design["Quality"] = quality

    factors = [
        {
            "name": "A",
            "type": "continuous",
            "changeability": "easy",
            "levels": "100|200",
            "units": "",
        },
        {
            "name": "B",
            "type": "continuous",
            "changeability": "easy",
            "levels": "50|150",
            "units": "",
        },
        {
            "name": "C",
            "type": "continuous",
            "changeability": "easy",
            "levels": "10|30",
            "units": "",
        },
    ]

    responses = [{"name": "Quality", "units": "score"}]

    metadata = {
        "true_model": "Quality = 70 + 6*A + 4*B - 2*C + 3*A*B",
        "outlier": "Run 5 is outlier (40 vs expected 78)",
    }

    csv_content = generate_doe_toolkit_csv(
        design, factors, responses, "full_factorial", metadata
    )

    with open(output_dir / "test_case_9_outlier.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_9_README.txt", "w") as f:
        f.write(
            "Test Case 9: Outlier\n"
            "Model: Quality = 70 + 6*A + 4*B - 2*C + 3*A*B\n"
            "Run 5 is outlier (40 vs expected 78)\n"
            "Exclude and verify R² improves\n"
        )


# ============================================================================
# Test Case 10: Blocking
# ============================================================================
def generate_test_case_10(output_dir):
    """2^4 with 2 blocks, 16 runs"""
    design = pd.DataFrame(
        {
            "StdOrder": range(1, 17),
            "RunOrder": np.random.permutation(range(1, 17)),
            "Block": [1] * 8 + [2] * 8,
            "A": [30, 70, 30, 70, 30, 70, 30, 70, 30, 70, 30, 70, 30, 70, 30, 70],
            "B": [80, 80, 120, 120, 80, 80, 120, 120, 80, 80, 120, 120, 80, 80, 120, 120],
            "C": [
                2.5,
                2.5,
                2.5,
                2.5,
                7.5,
                7.5,
                7.5,
                7.5,
                2.5,
                2.5,
                2.5,
                2.5,
                7.5,
                7.5,
                7.5,
                7.5,
            ],
            "D": [150] * 8 + [250] * 8,
        }
    )

    a_coded = (design["A"] - 50) / 10
    b_coded = (design["B"] - 100) / 20
    c_coded = (design["C"] - 5) / 2.5
    d_coded = (design["D"] - 200) / 50

    block_effect = design["Block"].map({1: -2, 2: 2})

    efficiency_true = (
        75 + 5 * a_coded + 3 * b_coded - 2 * c_coded + 4 * d_coded + 2 * a_coded * b_coded
    )
    design["Efficiency"] = efficiency_true + block_effect + add_noise(efficiency_true, 0.03)

    factors = [
        {
            "name": "A",
            "type": "continuous",
            "changeability": "easy",
            "levels": "40|60",
            "units": "",
        },
        {
            "name": "B",
            "type": "continuous",
            "changeability": "easy",
            "levels": "80|120",
            "units": "",
        },
        {
            "name": "C",
            "type": "continuous",
            "changeability": "easy",
            "levels": "2.5|7.5",
            "units": "",
        },
        {
            "name": "D",
            "type": "continuous",
            "changeability": "easy",
            "levels": "150|250",
            "units": "",
        },
    ]

    responses = [{"name": "Efficiency", "units": "%"}]

    metadata = {
        "blocks": 2,
        "block_effect": "±2",
        "true_model": "Efficiency = 75 + 5*A + 3*B - 2*C + 4*D + 2*A*B + Block",
    }

    csv_content = generate_doe_toolkit_csv(
        design, factors, responses, "full_factorial", metadata
    )

    with open(output_dir / "test_case_10_blocking.csv", "w", encoding="utf-8") as f:
        f.write(csv_content)

    with open(output_dir / "test_case_10_README.txt", "w") as f:
        f.write(
            "Test Case 10: Blocking\n"
            "Model: Efficiency = 75 + 5*A + 3*B - 2*C + 4*D + 2*A*B + Block\n"
            "Block effect = ±2\n"
            "2 blocks of 8 runs each\n"
        )


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    output_dir = create_test_data_directory()

    print("Generating DOE-Toolkit test data...")
    print(f"Output directory: {output_dir.absolute()}")
    print()

    generators = [
        ("Test Case 1: Full Factorial", generate_test_case_1),
        ("Test Case 2: Fractional Factorial", generate_test_case_2),
        ("Test Case 3: CCD", generate_test_case_3),
        ("Test Case 4: D-Optimal", generate_test_case_4),
        ("Test Case 5: Split-Plot", generate_test_case_5),
        ("Test Case 6: Multi-Response", generate_test_case_6),
        ("Test Case 7: LHS", generate_test_case_7),
        ("Test Case 8: Hierarchy", generate_test_case_8),
        ("Test Case 9: Outlier", generate_test_case_9),
        ("Test Case 10: Blocking", generate_test_case_10),
    ]

    for name, func in generators:
        print(f"✓ {name}")
        func(output_dir)

    print()
    print("=" * 60)
    print("SUCCESS: All 10 test datasets generated in DOE-Toolkit format!")
    print("=" * 60)
    print(f"\nFiles created in: {output_dir.absolute()}")
    print("\nEach test case includes:")
    print("  - CSV file with DOE-Toolkit metadata format")
    print("  - Factor definitions (type, changeability, levels, units)")
    print("  - Response definitions")
    print("  - Design data")
    print("  - README.txt with true model and testing notes")
    print("\nCSV files can now be imported directly into DOE-Toolkit!")
