"""Design generation dispatch for the NiceGUI app (pure, Streamlit-free).

This module FIXES the Streamlit Step 3 -> Step 4 config handoff. In the
Streamlit app ``4_preview_design.py`` reads legacy top-level session keys
(``random_seed``, ``randomize``, ``fraction``, ``resolution``, ...) that
``3_choose_design.py`` never writes — only ``design_config['n_center_points']``
actually flows. The NiceGUI port instead passes the ``design_config`` dict
straight through to the core generators, so every configured option is honored.
It also generates a real Box-Behnken design (the Streamlit preview built a CCD
for the Box-Behnken selection).

Returns designs in NATURAL units and a JSON-serializable metadata dict, mirroring
``src/ui/pages/4_preview_design.py`` (with the fixes above).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.factors import Factor
from src.core.optimal.constraints import LinearConstraint
from src.ui.nicegui.design_config import alpha_for_label, derive_model_type, fraction_to_p


def generate_design(
    factors: List[Factor],
    design_type: str,
    config: Optional[Dict] = None,
    constraints: Optional[List[LinearConstraint]] = None,
    seed: Optional[int] = None,
    model_term_list: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate a design and its metadata for the given type/config.

    Parameters
    ----------
    factors : factor definitions (natural units on output)
    design_type : one of ``design_config.DESIGN_TYPES``
    config : the JSON-safe config dict produced by the /design page
    constraints : LinearConstraint objects (D-Optimal only)
    seed : optional random seed (applies to every type)
    model_term_list : optional list of design-level terms (D-Optimal sizing)

    Returns
    -------
    (design : pd.DataFrame in natural units, metadata : Dict)

    Raises
    ------
    ValueError with a user-facing message on invalid configuration/generation.
    """
    config = config or {}
    metadata: Dict = {'design_type': design_type, 'is_split_plot': False}

    if design_type == "Full Factorial":
        design = _full_factorial(factors, config, metadata, seed)

    elif design_type == "Fractional Factorial":
        design = _fractional_factorial(factors, config, metadata, seed)

    elif design_type == "Response Surface (CCD)":
        design = _ccd(factors, config, metadata, seed)

    elif design_type == "Response Surface (Box-Behnken)":
        design = _box_behnken(factors, config, metadata, seed)

    elif design_type == "D-Optimal":
        design = _d_optimal(factors, config, metadata, seed, constraints, model_term_list)

    elif design_type == "Latin Hypercube":
        design = _latin_hypercube(factors, config, metadata, seed)

    elif design_type == "Split-Plot":
        design = _split_plot(factors, config, metadata, seed)

    else:
        raise ValueError(f"Design type '{design_type}' not implemented yet")

    if not isinstance(design, pd.DataFrame) or design is None or len(design) == 0:
        raise ValueError("Design generation returned no runs")

    return design, metadata


# --- Individual generators --------------------------------------------------


def _full_factorial(factors, config, metadata, seed) -> pd.DataFrame:
    from src.core.full_factorial import full_factorial

    n_blocks = int(config.get('n_blocks', 1))
    design = full_factorial(
        factors=factors,
        n_center_points=int(config.get('n_center_points', 0)),
        n_replicates=int(config.get('n_replicates', 1)),
        randomize=bool(config.get('randomize', True)),
        random_seed=seed,
        n_blocks=None if n_blocks <= 1 else n_blocks,
    )
    metadata.update({'n_levels': int(config.get('n_levels', 2))})
    return design


def _fractional_factorial(factors, config, metadata, seed) -> pd.DataFrame:
    from src.core.fractional_factorial import FractionalFactorial

    fraction = config.get('fraction')
    if not fraction:
        raise ValueError("Fractional Factorial requires a fraction selection")

    generator_mode = config.get('generator_mode', 'Standard (Recommended)')
    custom_generators = config.get('custom_generators')
    if generator_mode == 'Custom' and custom_generators:
        ff = FractionalFactorial(
            factors=factors, fraction=fraction, generators=list(custom_generators)
        )
    else:
        ff = FractionalFactorial(
            factors=factors,
            fraction=fraction,
            resolution=int(config.get('resolution', 4)),
        )

    n_blocks = int(config.get('n_blocks', 1))
    design = ff.generate(
        randomize=bool(config.get('randomize', True)),
        random_seed=seed,
        n_blocks=None if n_blocks <= 1 else n_blocks,
    )
    metadata.update({
        'resolution': ff.resolution,
        'generators': ff.generators_algebraic or [],
        'alias_structure': ff.alias_structure,
        'fraction': fraction,
    })
    return design


def _ccd(factors, config, metadata, seed) -> pd.DataFrame:
    from src.core.response_surface import CentralCompositeDesign

    alpha_label = config.get('alpha_label') or 'Rotatable'
    alpha = config.get('alpha') or alpha_for_label(alpha_label)
    rsm = CentralCompositeDesign(
        factors=factors,
        alpha=alpha,
        center_points=int(config.get('n_center_points', 5)) or None,
    )
    design = rsm.generate(
        randomize=bool(config.get('randomize', True)),
        random_seed=seed,
    )
    metadata.update({'variant': 'ccd', 'alpha': alpha, 'alpha_label': alpha_label})
    return design


def _box_behnken(factors, config, metadata, seed) -> pd.DataFrame:
    from src.core.response_surface import BoxBehnkenDesign

    rsm = BoxBehnkenDesign(
        factors=factors,
        center_points=int(config.get('n_center_points', 5)) or 3,
    )
    design = rsm.generate(
        randomize=bool(config.get('randomize', True)),
        random_seed=seed,
    )
    metadata.update({'variant': 'box_behnken'})
    return design


def _d_optimal(factors, config, metadata, seed, constraints, model_term_list) -> pd.DataFrame:
    from src.core.optimal import generate_d_optimal_design

    model_terms = config.get('model_terms') or model_term_list
    model_type = config.get('model_type') or derive_model_type(model_terms)
    n_runs = int(config.get('n_runs', 0))
    if n_runs <= 1:
        raise ValueError("D-Optimal requires a number of runs")

    result = generate_d_optimal_design(
        factors=factors,
        model_type=model_type,
        n_runs=n_runs,
        constraints=[c for c in (constraints or [])],
        seed=seed,
    )
    metadata.update({
        'model_type': model_type,
        'converged_by': result.converged_by,
        'n_parameters': result.n_parameters,
        'condition_number': result.condition_number,
        'd_efficiency': result.d_efficiency_vs_benchmark,
    })
    return result.design_actual


def _latin_hypercube(factors, config, metadata, seed) -> pd.DataFrame:
    from src.core.latin_hypercube import generate_latin_hypercube

    criterion_label = config.get('criterion', 'Maximin')
    if criterion_label == 'None':
        criterion, n_candidates = 'maximin', 1
    else:
        criterion, n_candidates = criterion_label.lower(), int(config.get('n_candidates', 10))

    result = generate_latin_hypercube(
        factors=factors,
        n_runs=int(config.get('n_runs', 20)),
        criterion=criterion,
        n_candidates=n_candidates,
        seed=seed,
    )
    metadata.update({
        'criterion': result.criterion,
        'criterion_value': float(result.criterion_value),
        'n_runs': result.n_runs,
    })
    return result.design


def _split_plot(factors, config, metadata, seed) -> pd.DataFrame:
    from src.core.split_plot import generate_split_plot_design

    result = generate_split_plot_design(
        factors=factors,
        n_replicates=int(config.get('n_replicates', 1)),
        n_center_points=int(config.get('n_center_points', 0)),
        n_blocks=int(config.get('n_blocks', 1)),
        randomize_whole_plots=bool(config.get('randomize_whole_plots', True)),
        randomize_sub_plots=bool(config.get('randomize_subplots', True)),
        seed=seed,
    )
    metadata.update({
        'is_split_plot': True,
        'n_whole_plots': result.n_whole_plots,
        'n_sub_plots_per_whole_plot': result.n_sub_plots_per_whole_plot,
        'whole_plot_factors': list(result.whole_plot_factors),
        'sub_plot_factors': list(result.sub_plot_factors),
        'has_very_hard_factors': bool(result.has_very_hard_factors),
    })
    return result.design


# --- D-efficiency -----------------------------------------------------------


def compute_d_efficiency(
    design: pd.DataFrame, factors: List[Factor], metadata: Dict
) -> Optional[float]:
    """Compute the design's D-efficiency vs. the theoretical benchmark (%)."""
    try:
        from src.core.optimal.utils import (
            compute_benchmark_criterion,
            compute_d_efficiency_vs_benchmark,
        )
        from src.core.optimal.criteria import create_polynomial_builder
        from src.core.analysis import generate_model_terms
        from src.core.coding import DesignSpace

        model_type = metadata.get('model_type', 'linear')
        if 'variant' in metadata:
            model_type = 'quadratic'
        elif int(metadata.get('resolution', 0) or 0) >= 5:
            model_type = 'interaction'

        model_builder = create_polynomial_builder(factors, model_type)
        factor_names = [f.name for f in factors]
        if not all(col in design.columns for col in factor_names):
            return None

        design_space = DesignSpace.from_factors(factors)
        design_coded = design_space.encode_dataframe(
            design[factor_names].copy()
        ).values

        X = model_builder(design_coded)
        XtX = X.T @ X
        det_benchmark, _ = compute_benchmark_criterion(
            factors, model_type, model_builder, criterion_type='D'
        )
        det_design = np.linalg.det(XtX)
        model_terms = generate_model_terms(
            factors, model_type, include_intercept=True
        )
        n_params = len(model_terms)
        return compute_d_efficiency_vs_benchmark(
            det_design, len(design), n_params, det_benchmark
        )
    except Exception:
        return None