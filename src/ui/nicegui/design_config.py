"""Design-type availability, configuration schemas and run estimation (pure).

This module is the NiceGUI side's source of truth for the Step 3
(configuration) logic. It is Streamlit-free so it can be unit-tested directly.

NOTE: the port deliberately FIXES the Streamlit Step 3 -> Step 4 config handoff:
in the Streamlit app, ``3_choose_design.py`` writes ``design_config`` to session
state but ``4_preview_design.py`` reads legacy top-level keys that nothing ever
writes. Here the config dict produced on this page is the ONLY input consumed by
``design_generation.generate_design`` (see that module for the mapping).
"""

from typing import Dict, List, Optional, Tuple

from src.core.factors import ChangeabilityLevel, Factor

DESIGN_TYPES: List[str] = [
    "Full Factorial",
    "Fractional Factorial",
    "Response Surface (CCD)",
    "Response Surface (Box-Behnken)",
    "D-Optimal",
    "Latin Hypercube",
    "Split-Plot",
]

_DESCRIPTIONS = {
    "Full Factorial": (
        "All possible combinations of factor levels. Best for small experiments.",
        "2-4 factors, want to estimate all interactions",
        "2^k for 2-level factors (grows exponentially)",
    ),
    "Fractional Factorial": (
        "Subset of full factorial. Efficient screening for many factors.",
        "4+ factors, willing to sacrifice some interactions",
        "2^(k-p) where p is the fraction",
    ),
    "Response Surface (CCD)": (
        "Central Composite Design for quadratic models and optimization.",
        "Continuous factors, need to model curvature",
        "2^k + 2k + center points",
    ),
    "Response Surface (Box-Behnken)": (
        "Efficient response surface design (no corner points).",
        "3+ continuous factors, avoid extreme combinations",
        "Fewer than CCD, no axial points at +/-alpha",
    ),
    "D-Optimal": (
        "Computer-generated optimal design with constraints.",
        "Constrained design space, irregular regions, mixed factors",
        "User-specified (typically p+1 to 2p where p=parameters)",
    ),
    "Latin Hypercube": (
        "Space-filling design for exploration and screening.",
        "Initial exploration, many factors, computer experiments",
        "User-specified (flexible)",
    ),
    "Split-Plot": (
        "Hierarchical design for hard-to-change factors.",
        "Some factors are expensive or slow to change",
        "Based on whole-plot structure",
    ),
}


def classify_factors(factors: List[Factor]) -> Tuple[bool, bool, bool]:
    """Return (all_continuous, has_categorical, has_hard_factors)."""
    all_continuous = all(f.is_continuous() for f in factors)
    has_categorical = any(f.is_categorical() for f in factors)
    has_hard_factors = any(
        f.changeability != ChangeabilityLevel.EASY for f in factors
    )
    return all_continuous, has_categorical, has_hard_factors


def design_options(factors: List[Factor]) -> List[Dict]:
    """Describe the seven design types with availability + reason for the /design page."""
    all_continuous, _has_categorical, has_hard_factors = classify_factors(factors)
    n = len(factors)

    availability: Dict[str, Tuple[bool, Optional[str]]] = {
        "Full Factorial": (True, None),
        "Fractional Factorial": (
            n >= 4, None if n >= 4 else "Requires at least 4 factors",
        ),
        "Response Surface (CCD)": (
            all_continuous and n >= 2,
            None if (all_continuous and n >= 2) else (
                "Requires all continuous factors" if n >= 2
                else "Requires at least 2 factors"
            ),
        ),
        "Response Surface (Box-Behnken)": (
            all_continuous and n >= 3,
            None if (all_continuous and n >= 3) else (
                "Requires all continuous factors" if n >= 3
                else "Requires at least 3 factors"
            ),
        ),
        "D-Optimal": (True, None),
        "Latin Hypercube": (
            all_continuous, None if all_continuous else "Requires all continuous factors",
        ),
        "Split-Plot": (
            has_hard_factors,
            None if has_hard_factors else "Requires at least one hard-to-change factor",
        ),
    }

    options: List[Dict] = []
    for name in DESIGN_TYPES:
        enabled, reason = availability[name]
        description, when_to_use, runs = _DESCRIPTIONS[name]
        options.append({
            'name': name,
            'enabled': enabled,
            'disabled_reason': reason,
            'description': description,
            'when_to_use': when_to_use,
            'runs': runs,
        })
    return options


def fraction_to_p(fraction: str) -> int:
    """Convert a fraction label ('1/2', '1/4', ...) to the generator count p."""
    denominator = int(fraction.split('/')[1])
    return denominator.bit_length() - 1


def valid_fractions(n_factors: int) -> List[str]:
    """Fractions whose design keeps k - p >= 3 (need at least a Resolution III)."""
    fractions = ["1/2", "1/4", "1/8", "1/16"]
    return [
        frac for frac in fractions
        if n_factors - fraction_to_p(frac) >= 3
    ]


CCD_ALPHA_CHOICES: List[str] = ["Face-centered (α=1)", "Orthogonal", "Rotatable"]


def alpha_for_label(label: str) -> str:
    """Map the CCD alpha dropdown label to the semantic value for the core generator."""
    if "Face-centered" in label:
        return 'face'
    if "Orthogonal" in label:
        return 'orthogonal'
    return 'rotatable'


def derive_model_type(model_terms: Optional[List[str]]) -> str:
    """Infer the design model type from analysis model terms.

    Returns one of 'linear', 'interaction', 'quadratic' for D-Optimal generation.
    """
    if not model_terms:
        return 'linear'
    if any(t.startswith('I(') and '**2' in t for t in model_terms):
        return 'quadratic'
    if any('*' in t and not t.startswith('I(') for t in model_terms):
        return 'interaction'
    return 'linear'


def estimate_runs(
    design_type: str,
    config: Dict,
    n_factors: int,
    factor_levels: Optional[List[int]] = None,
) -> Optional[int]:
    """Estimate the number of runs a config will produce (informational only).

    ``factor_levels`` is only needed for Split-Plot (number of levels per factor).
    """
    try:
        if design_type == "Full Factorial":
            # Core produces 2-level factorials (n_levels selector is a no-op).
            base = 2 ** n_factors
            center = int(config.get('n_center_points', 0))
            reps = int(config.get('n_replicates', 1))
            return (base + center) * reps

        if design_type == "Fractional Factorial":
            fraction = config.get('fraction', '1/2')
            p = fraction_to_p(fraction)
            center = int(config.get('n_center_points', 0))
            return (2 ** (n_factors - p)) + center

        if design_type == "Response Surface (CCD)":
            k = n_factors
            center = int(config.get('n_center_points', 5))
            return (2 ** k) + (2 * k) + center

        if design_type == "Response Surface (Box-Behnken)":
            k = n_factors
            center = int(config.get('n_center_points', 5))
            return (2 * k * (k - 1)) + center

        if design_type == "D-Optimal":
            return int(config.get('n_runs', 0)) or None

        if design_type == "Latin Hypercube":
            return int(config.get('n_runs', 0)) or None

        if design_type == "Split-Plot":
            reps = int(config.get('n_replicates', 1))
            blocks = int(config.get('n_blocks', 1))
            center = int(config.get('n_center_points', 0))
            n_levels = factor_levels or ([2] * n_factors)
            base = 1
            for count in n_levels:
                base *= int(count)
            return (base * reps + center) * blocks
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def split_plot_factors(factors: List[Factor]) -> Dict[str, List[str]]:
    """Group factor names by changeability for the Split-Plot configuration panel."""
    hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.HARD]
    very_hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.VERY_HARD]
    easy = [f.name for f in factors if f.changeability == ChangeabilityLevel.EASY]
    return {
        'hard': hard + very_hard,
        'easy': easy,
        'all_hard': hard + very_hard,
    }


def ensure_full_factorial_config(config: Dict) -> Dict:
    """Seed defaults that a freshly-rendered Full Factorial form implies.

    Widgets persist their values into ``config`` only on change, so an untouched
    form would otherwise leave ``design_config`` missing keys and fail
    validation on first click. ``setdefault`` never clobbers user edits.
    """
    config.setdefault('n_levels', 2)
    return config


def ensure_fractional_config(config: Dict, n_factors: int) -> Dict:
    """Seed defaults for a freshly-rendered Fractional Factorial form."""
    valid = valid_fractions(n_factors)
    config.setdefault('fraction', valid[0] if valid else '1/2')
    config.setdefault('resolution', 5)
    config.setdefault('generator_mode', 'Standard (Recommended)')
    return config


def ensure_ccd_config(config: Dict) -> Dict:
    """Seed defaults for a freshly-rendered Response Surface (CCD) form."""
    config.setdefault('alpha_label', 'Rotatable')
    config.setdefault('alpha', 'rotatable')
    return config


def ensure_d_optimal_config(config: Dict, min_runs: int) -> Dict:
    """Seed the default run count for a freshly-rendered D-Optimal form."""
    config.setdefault('n_runs', min_runs * 2)
    return config


def ensure_lhs_config(config: Dict, n_factors: int) -> Dict:
    """Seed defaults for a freshly-rendered Latin Hypercube form."""
    config.setdefault('n_runs', n_factors * 10)
    config.setdefault('criterion', 'Maximin')
    return config


def validate_config(
    design_type: str,
    config: Dict,
    n_factors: int,
    model_terms: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """Validate a design configuration, returning (ok, errors)."""
    errors: List[str] = []

    if design_type == "Fractional Factorial":
        fraction = config.get('fraction')
        valid = valid_fractions(n_factors)
        if fraction is None or fraction not in (valid + ["1/16"]):
            errors.append(f"Fraction {fraction} not supported for {n_factors} factors")
        if model_terms and any(t.startswith('I(') and '**2' in t for t in model_terms):
            errors.append(
                "Model includes quadratic terms, but fractional factorial designs "
                "cannot estimate quadratic effects"
            )

    elif design_type == "Response Surface (CCD)":
        if 'alpha' not in config or config['alpha'] not in ('face', 'orthogonal', 'rotatable'):
            errors.append("CCD requires an alpha selection")

    elif design_type == "Response Surface (Box-Behnken)":
        if n_factors < 3:
            errors.append("Box-Behnken requires at least 3 factors")

    elif design_type == "D-Optimal":
        model_type = derive_model_type(model_terms)
        model_terms = model_terms or []
        if not model_terms:
            errors.append("No analysis model defined — return to Step 2 (Select Model)")
        min_runs = len(set(model_terms)) or (n_factors + 1)
        n_runs = int(config.get('n_runs', 0) or 0)
        if n_runs < min_runs:
            errors.append(
                f"D-Optimal needs at least {min_runs} runs for {len(model_terms)} model terms"
            )

    elif design_type == "Latin Hypercube":
        n_runs = int(config.get('n_runs', 0) or 0)
        if n_runs < max(2, n_factors + 1):
            errors.append(f"Latin Hypercube needs at least {max(2, n_factors + 1)} runs")

    elif design_type == "Split-Plot":
        reps = int(config.get('n_replicates', 1) or 1)
        if reps < 1:
            errors.append("Split-Plot requires at least 1 replicate")

    return (len(errors) == 0, errors)