"""
Optimal Design Augmentation for Model Extension.

This module implements optimal design augmentation (D and I) to add runs that 
improve model precision or enable estimation of additional model terms.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.augmentation.plan import (
    AugmentationPlan,
    AugmentedDesign,
    OptimalAugmentConfig,
)
from src.core.candidates.generators import (
    CandidatePoolConfig,
    generate_augmentation_candidates,
)
from src.core.factors import ChangeabilityLevel, Factor, FactorType


def _candidates_to_coded_df(
    candidates: np.ndarray,
    factors: List[Factor],
) -> pd.DataFrame:
    """
    Convert a raw numeric candidates array to a DataFrame in **coded** space.

    Used internally by the coordinate-exchange optimizer, which requires coded
    values in [-1, 1] for continuous/discrete factors and string labels for
    categoricals (so build_model_matrix can apply effects coding).

    Parameters
    ----------
    candidates : np.ndarray, shape (N, k)
        Candidate points in coded space.
    factors : List[Factor]
        Factor definitions.

    Returns
    -------
    pd.DataFrame
        Coded DataFrame — string labels for categoricals, raw coded floats
        for continuous/discrete numerics.
    """
    n, k = candidates.shape
    data: Dict[str, list] = {f.name: [] for f in factors}

    for row in candidates:
        for j, factor in enumerate(factors):
            val = float(row[j])
            if factor.factor_type == FactorType.CATEGORICAL:
                levels = factor.levels
                n_levels = len(levels)
                if n_levels == 1:
                    data[factor.name].append(levels[0])
                else:
                    idx = (val + 1.0) / 2.0 * (n_levels - 1)
                    idx = int(round(max(0.0, min(float(n_levels - 1), idx))))
                    data[factor.name].append(levels[idx])
            else:
                # Keep as coded value for the optimizer
                data[factor.name].append(val)

    return pd.DataFrame(data)


def _candidates_to_design_df(
    candidates: np.ndarray,
    factors: List[Factor],
) -> pd.DataFrame:
    """
    Convert a raw numeric candidates array to a properly-typed design DataFrame.

    Continuous factors are decoded from coded [-1, 1] space to actual values.
    Discrete-numeric factors are snapped to the nearest declared level.
    Categorical factors: the numeric value is snapped to the nearest level
    by treating the k levels as evenly spaced in [-1, 1].

    Parameters
    ----------
    candidates : np.ndarray, shape (N, k)
        Candidate points in coded space.
    factors : List[Factor]
        Factor definitions.

    Returns
    -------
    pd.DataFrame
        DataFrame with correct dtypes — string labels for categoricals,
        actual values for continuous/discrete — matching what the rest of
        the app expects (actual-value convention).
    """
    from src.core.coding import decode_value

    n, k = candidates.shape
    data: Dict[str, list] = {f.name: [] for f in factors}

    for row in candidates:
        for j, factor in enumerate(factors):
            val = float(row[j])
            if factor.factor_type == FactorType.CATEGORICAL:
                levels = factor.levels  # list of strings
                n_levels = len(levels)
                if n_levels == 1:
                    data[factor.name].append(levels[0])
                else:
                    # Map [-1, 1] linearly to level index [0, n_levels-1]
                    idx = (val + 1.0) / 2.0 * (n_levels - 1)
                    idx = int(round(max(0.0, min(float(n_levels - 1), idx))))
                    data[factor.name].append(levels[idx])
            elif factor.factor_type == FactorType.DISCRETE_NUMERIC:
                # Snap coded value to the nearest declared discrete level
                levels = factor.levels  # list of floats
                n_levels = len(levels)
                if n_levels == 1:
                    data[factor.name].append(float(levels[0]))
                else:
                    idx = (val + 1.0) / 2.0 * (n_levels - 1)
                    idx = int(round(max(0.0, min(float(n_levels - 1), idx))))
                    data[factor.name].append(float(levels[idx]))
            else:
                # Continuous: decode from coded [-1, 1] to actual range
                min_val, max_val = float(factor.levels[0]), float(factor.levels[1])
                actual = decode_value(val, min_val, max_val)
                data[factor.name].append(actual)

    return pd.DataFrame(data)


def _infer_subplot_template(
    original_design: pd.DataFrame,
    easy_factors: List[Factor],
) -> pd.DataFrame:
    """
    Extract the sub-plot combination template from an existing split-plot design.

    Reads the easy-factor settings from the first whole-plot and returns them as
    a DataFrame with one row per sub-plot run.  This template is replicated for
    every new whole-plot during augmentation so that the sub-plot structure of
    augmented whole-plots matches the original design.

    Parameters
    ----------
    original_design : pd.DataFrame
        Original design in actual values, must contain a ``WholePlot`` column.
    easy_factors : List[Factor]
        Easy-to-change factors (sub-plot level).

    Returns
    -------
    pd.DataFrame
        One row per sub-plot run, columns = easy factor names, actual values.
    """
    easy_names = [f.name for f in easy_factors]
    first_wp = int(original_design["WholePlot"].min())
    wp_rows = original_design[original_design["WholePlot"] == first_wp]
    return wp_rows[easy_names].reset_index(drop=True)


def _augment_split_plot(
    original_design: pd.DataFrame,
    factors: List[Factor],
    current_model_terms: List[str],
    new_model_terms: List[str],
    n_runs_to_add: int,
    criterion: str = "D",
    seed: Optional[int] = None,
) -> AugmentedDesign:
    """
    D/I-optimal augmentation that respects split-plot whole-plot structure.

    In a split-plot design, new runs must be added as complete whole-plot
    blocks — you cannot add individual sub-plot runs in isolation because the
    hard (whole-plot) factors are expensive to change.  This function:

    1. Infers the sub-plot template (easy-factor combinations) from the first
       whole-plot of the existing design.
    2. Computes ``n_whole_plots_to_add = n_runs_to_add // n_subplots_per_wp``.
    3. Runs the coordinate-exchange optimizer **only in hard-factor space** to
       select the best new whole-plot settings.
    4. Expands each selected whole-plot setting into a full block of sub-plot
       runs using the template from step 1.
    5. Assigns sequential ``WholePlot`` numbers starting from
       ``max(original WholePlot) + 1``.

    Parameters
    ----------
    original_design : pd.DataFrame
        Original design in actual values with ``WholePlot`` column.
    factors : List[Factor]
        All factor definitions.
    current_model_terms : List[str]
        Terms in the current model.
    new_model_terms : List[str]
        Complete set of terms for the (possibly extended) model.
    n_runs_to_add : int
        Total runs to add; must be a multiple of ``n_subplots_per_wp``.
    criterion : {'D', 'I'}, default='D'
        Optimality criterion.
    seed : int, optional
        Random seed.

    Returns
    -------
    AugmentedDesign
        Combined design with proper whole-plot blocking.

    Raises
    ------
    ValueError
        If ``n_runs_to_add`` is not a multiple of the sub-plot block size, or
        if fewer whole-plot candidates are available than needed.
    """
    from src.core.coding import decode_value, encode_design
    from src.core.diagnostics.variance import build_model_matrix

    rng = np.random.default_rng(seed)

    # Partition factors by changeability.
    hard_factors = [
        f for f in factors
        if f.changeability in (ChangeabilityLevel.HARD, ChangeabilityLevel.VERY_HARD)
    ]
    easy_factors = [
        f for f in factors
        if f.changeability == ChangeabilityLevel.EASY
    ]
    hard_names = [f.name for f in hard_factors]
    easy_names = [f.name for f in easy_factors]
    factor_names = [f.name for f in factors]

    # Determine sub-plot block size from existing design.
    wp_sizes = original_design.groupby("WholePlot").size()
    n_subplots_per_wp = int(wp_sizes.iloc[0])  # Assume balanced design.

    # Round up to the nearest whole-plot multiple so the caller never has to
    # do this arithmetic themselves.
    if n_runs_to_add % n_subplots_per_wp != 0:
        n_runs_to_add = (
            (n_runs_to_add // n_subplots_per_wp) + 1
        ) * n_subplots_per_wp

    n_whole_plots_to_add = n_runs_to_add // n_subplots_per_wp

    # Extract sub-plot template (easy-factor settings) from first whole-plot.
    subplot_template = _infer_subplot_template(original_design, easy_factors)

    # --- Optimise in hard-factor space only ---
    # Build the model matrix for the full original design (coded) to use as
    # the fixed part while we optimise new whole-plot settings.
    original_design_coded = encode_design(original_design[factor_names], factors)
    X_original, _ = build_model_matrix(original_design_coded, factors, new_model_terms)

    # Generate hard-factor-only candidates (coded).
    from src.core.candidates.generators import generate_candidate_pool, CandidatePoolConfig
    from itertools import product as iproduct

    # Existing hard-factor settings for exclusion — must be in actual values
    # because generate_candidate_pool exclusion is compared against decoded
    # hard_candidates_actual which is in actual space.
    from src.core.coding import is_design_coded, encode_design as _encode_design
    existing_hard_raw = (
        original_design[hard_names]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # _candidates_to_design_df always returns actual values; match that convention
    # for the exclusion check by decoding if the design is in coded space.
    if is_design_coded(existing_hard_raw, hard_factors):
        from src.core.coding import decode_design as _decode_design
        existing_hard = _decode_design(existing_hard_raw, hard_factors)
    else:
        existing_hard = existing_hard_raw

    hp_config = CandidatePoolConfig(
        include_vertices=True,
        include_axial=True,
        include_center=True,
        alpha_axial=1.0,
        lhs_multiplier=10,
        exclude_existing_runs=True,
        min_distance=0.02,
    )
    hard_candidates_coded = generate_candidate_pool(
        factors=hard_factors,
        n_runs=n_whole_plots_to_add,
        config=hp_config,
        existing_design=existing_hard,
        seed=seed,
    )

    if len(hard_candidates_coded) < n_whole_plots_to_add:
        raise ValueError(
            f"Insufficient whole-plot candidates: found {len(hard_candidates_coded)}, "
            f"need {n_whole_plots_to_add}."
        )

    # Decode hard candidates to actual values.
    hard_candidates_actual = _candidates_to_design_df(hard_candidates_coded, hard_factors)

    # For each hard-factor candidate, build the full block (hard + every sub-plot
    # row) and compute its contribution to the model matrix.
    # We select whole-plot candidates by coordinate exchange: each "move" swaps
    # one whole-plot (i.e. n_subplots_per_wp rows) for another candidate.

    N_hp_cand = len(hard_candidates_coded)

    # Pre-build model-matrix blocks for each hard candidate.
    # block_X[i] has shape (n_subplots_per_wp, p).
    block_X: List[np.ndarray] = []
    for i in range(N_hp_cand):
        hard_row = hard_candidates_actual.iloc[i]
        block_rows = []
        for _, subplot_row in subplot_template.iterrows():
            # Assemble a full run: hard settings + easy settings.
            run: Dict[str, object] = {}
            for fname in hard_names:
                run[fname] = hard_row[fname]
            for fname in easy_names:
                run[fname] = subplot_row[fname]
            block_rows.append(run)

        block_df = pd.DataFrame(block_rows)
        # Reorder to match factors order.
        block_df = block_df[[f.name for f in factors]]
        block_coded = encode_design(block_df, factors)
        X_block, _ = build_model_matrix(block_coded, factors, new_model_terms)
        block_X.append(X_block)

    # Coordinate exchange over whole-plot slots.
    # selected_wp_indices: which hard candidate occupies each new WP slot.
    selected_wp_indices = rng.choice(N_hp_cand, size=n_whole_plots_to_add, replace=False)

    def _criterion_value(selected: np.ndarray) -> float:
        X_new = np.vstack([block_X[i] for i in selected])
        X_combined = np.vstack([X_original, X_new])
        XtX = X_combined.T @ X_combined
        try:
            if criterion == "D":
                sign, logdet = np.linalg.slogdet(XtX)
                return float(logdet) if sign > 0 else -1e18
            else:  # I-optimal: minimise trace(inv(XtX)), so maximise its negative.
                return -float(np.trace(np.linalg.inv(XtX)))
        except np.linalg.LinAlgError:
            return -1e18

    best_val = _criterion_value(selected_wp_indices)

    for _iteration in range(100):
        improved = False
        for slot in range(n_whole_plots_to_add):
            current_idx = selected_wp_indices[slot]
            best_idx = current_idx
            # Try swapping with all candidates not already selected.
            for cand_idx in rng.permutation(N_hp_cand):
                if cand_idx in selected_wp_indices:
                    continue
                trial = selected_wp_indices.copy()
                trial[slot] = cand_idx
                val = _criterion_value(trial)
                if val > best_val:
                    best_val = val
                    best_idx = cand_idx
                    improved = True
            if best_idx != current_idx:
                selected_wp_indices[slot] = best_idx
        if not improved:
            break

    # --- Expand selected whole-plot settings into full run blocks ---
    max_existing_wp = int(original_design["WholePlot"].max())
    new_runs_rows: List[Dict] = []

    for wp_offset, hp_idx in enumerate(selected_wp_indices):
        wp_number = max_existing_wp + wp_offset + 1
        hard_row = hard_candidates_actual.iloc[hp_idx]
        for _, subplot_row in subplot_template.iterrows():
            run: Dict[str, object] = {"WholePlot": wp_number}
            for fname in hard_names:
                run[fname] = hard_row[fname]
            for fname in easy_names:
                run[fname] = subplot_row[fname]
            new_runs_rows.append(run)

    new_runs = pd.DataFrame(new_runs_rows)
    # Reorder columns: WholePlot first, then factors in definition order.
    new_runs = new_runs[["WholePlot"] + factor_names]
    new_runs["Phase"] = 2

    # Combine — ensure new_runs is in the same coordinate space as original_design.
    # hard_candidates_actual and subplot_template are decoded to actual values;
    # if the original is coded we must re-encode new_runs to match.
    if is_design_coded(original_design[factor_names], factors):
        from src.core.coding import encode_design as _enc
        new_runs_factor_cols = new_runs[factor_names]
        new_runs_encoded = _enc(new_runs_factor_cols, factors)
        new_runs = pd.concat(
            [new_runs[["WholePlot"]], new_runs_encoded, new_runs[["Phase"]]],
            axis=1,
        )

    original_with_phase = original_design.copy()
    original_with_phase["Phase"] = 1
    combined = pd.concat([original_with_phase, new_runs], ignore_index=True)

    for col in ("StdOrder", "RunOrder"):
        if col in combined.columns:
            combined = combined.drop(columns=[col])
    combined.insert(0, "StdOrder", range(1, len(combined) + 1))
    combined.insert(1, "RunOrder", range(1, len(combined) + 1))

    # Compute condition number of combined model matrix.
    combined_coded = encode_design(combined[factor_names], factors)
    X_combined_full, _ = build_model_matrix(combined_coded, factors, new_model_terms)
    XtX_full = X_combined_full.T @ X_combined_full
    try:
        condition_number = float(np.linalg.cond(XtX_full))
    except Exception:
        condition_number = np.inf

    return AugmentedDesign(
        combined_design=combined,
        new_runs_only=new_runs,
        block_column="Phase",
        n_runs_original=len(original_design),
        n_runs_added=n_runs_to_add,
        n_runs_total=len(combined),
        achieved_improvements={
            "whole_plots_added": f"{n_whole_plots_to_add} new whole-plot(s)",
            "n_runs": f"{len(original_design)} → {len(combined)}",
        },
        condition_number=condition_number,
    )


def augment_for_model_extension(
    original_design: pd.DataFrame,
    factors: List[Factor],
    current_model_terms: List[str],
    new_model_terms: List[str],
    n_runs_to_add: int,
    criterion: str = "D",
    prediction_grid_config: Optional[Dict] = None,
    seed: Optional[int] = None,
) -> AugmentedDesign:
    """
    Add runs optimized for extended model.

    Use case: Model needs additional terms (e.g., linear → quadratic).
    Strategy: Fix existing runs, optimize new runs to maximize criterion
    value of extended model.

    Parameters
    ----------
    original_design : pd.DataFrame
        Original design matrix (coded levels)
    factors : List[Factor]
        Factor definitions
    current_model_terms : List[str]
        Terms in current model
    new_model_terms : List[str]
        Complete set of terms for extended model (includes current)
    n_runs_to_add : int
        Number of runs to add
    criterion : {'D', 'I'}, default='D'
        Optimality criterion:
        - 'D': D-optimal (maximize det(X'X), best for parameter estimates)
        - 'I': I-optimal (minimize avg prediction variance, best for prediction)
    prediction_grid_config : dict, optional
        Configuration for I-optimal prediction grid (ignored for D-optimal)
    seed : int, optional
        Random seed

    Returns
    -------
    AugmentedDesign
        Combined design optimized for extended model

    Examples
    --------
    >>> # Original model: linear
    >>> current_terms = ['1', 'A', 'B', 'C']
    >>>
    >>> # Extended model: quadratic
    >>> new_terms = ['1', 'A', 'B', 'C', 'A*B', 'A*C', 'B*C',
    ...              'I(A**2)', 'I(B**2)', 'I(C**2)']
    >>>
    >>> # D-optimal augmentation
    >>> augmented_d = augment_for_model_extension(
    ...     design, factors, current_terms, new_terms,
    ...     n_runs_to_add=10, criterion='D'
    ... )
    >>>
    >>> # I-optimal augmentation
    >>> augmented_i = augment_for_model_extension(
    ...     design, factors, current_terms, new_terms,
    ...     n_runs_to_add=10, criterion='I',
    ...     prediction_grid_config={'n_points_per_dim': 7}
    ... )

    Notes
    -----
    The algorithm:
    1. Build model matrix for existing runs with new terms
    2. Generate candidate pool (excluding existing runs)
    3. Use coordinate exchange to select new runs maximizing criterion([X_old; X_new])
    4. Combine original and new runs

    This reuses the CEXCH optimizer from optimal_design.py but with
    existing runs fixed.
    """
    if seed is not None:
        np.random.seed(seed)

    # Dispatch to split-plot-aware augmentation when the design has a
    # WholePlot column and hard factors are present.
    is_split_plot = (
        "WholePlot" in original_design.columns
        and any(
            f.changeability in (ChangeabilityLevel.HARD, ChangeabilityLevel.VERY_HARD)
            for f in factors
        )
    )
    if is_split_plot:
        return _augment_split_plot(
            original_design=original_design,
            factors=factors,
            current_model_terms=current_model_terms,
            new_model_terms=new_model_terms,
            n_runs_to_add=n_runs_to_add,
            criterion=criterion,
            seed=seed,
        )

    from src.core.diagnostics.variance import build_model_matrix
    from src.core.coding import encode_design

    # Extract factor columns
    factor_names = [f.name for f in factors]

    # Build model matrix for original runs with new terms.
    # build_model_matrix expects continuous factors in coded [-1, 1] space,
    # so encode the original design (which is in actual values) first.
    original_design_coded = encode_design(original_design[factor_names], factors)
    X_model_original, _ = build_model_matrix(
        original_design_coded, factors, new_model_terms
    )

    n_original = len(original_design)
    p = len(new_model_terms)

    # Check that augmented design will have enough runs
    if n_original + n_runs_to_add < p:
        raise ValueError(
            f"Augmented design will be supersaturated: "
            f"{n_original + n_runs_to_add} runs for {p} parameters"
        )

    # Generate candidate pool (excluding existing runs)
    config = CandidatePoolConfig(
        include_vertices=True,
        include_axial=True,
        include_center=True,
        alpha_axial=1.0,
        lhs_multiplier=5,
        exclude_existing_runs=True,
        min_distance=0.01,
    )

    # Candidate exclusion uses coded space; ensure we pass a coded design
    # regardless of whether original_design is stored coded or actual.
    from src.core.coding import is_design_coded, encode_design
    if is_design_coded(original_design[factor_names], factors):
        original_design_for_exclusion = original_design[factor_names]
    else:
        original_design_for_exclusion = encode_design(original_design[factor_names], factors)

    candidates = generate_augmentation_candidates(
        factors=factors,
        original_design=original_design_for_exclusion,
        n_candidates=max(n_runs_to_add * 10, 200),
        seed=seed,
    )

    if len(candidates) < n_runs_to_add:
        raise ValueError(
            f"Insufficient candidates: found {len(candidates)}, " f"need {n_runs_to_add}"
        )

    # Select best new runs using augmented coordinate exchange
    new_run_indices = _augmented_coordinate_exchange(
        X_model_original=X_model_original,
        candidates=candidates,
        factors=factors,
        model_terms=new_model_terms,
        n_runs_to_add=n_runs_to_add,
        criterion=criterion,
        prediction_grid_config=prediction_grid_config,
        seed=seed,
    )

    new_runs_coded = candidates[new_run_indices]

    # Match the coordinate convention of the original design.
    # Designs generated by this app may be stored in either coded [-1, 1] space
    # (e.g. CCD, full/fractional factorial) or in actual-value space (e.g. after
    # CSV import with decoded values).  Candidates are always in coded space;
    # we decode to actual values only when the original design is in actual space.
    from src.core.coding import is_design_coded
    if is_design_coded(original_design[factor_names], factors):
        # Keep new runs in coded space so they match the original
        new_runs = _candidates_to_coded_df(new_runs_coded, factors)
    else:
        # Decode to actual values to match the original
        new_runs = _candidates_to_design_df(new_runs_coded, factors)
    new_runs["Phase"] = 2

    # Combine original and new
    original_with_phase = original_design.copy()
    original_with_phase["Phase"] = 1

    combined = pd.concat([original_with_phase, new_runs], ignore_index=True)

    # Add standard order and run order (remove if they already exist)
    if "StdOrder" in combined.columns:
        combined = combined.drop(columns=["StdOrder"])
    if "RunOrder" in combined.columns:
        combined = combined.drop(columns=["RunOrder"])
    
    combined.insert(0, "StdOrder", range(1, len(combined) + 1))
    combined.insert(1, "RunOrder", range(1, len(combined) + 1))

    # Compute D-efficiency
    # Encode combined design (actual values) to coded space for model matrix
    combined_coded_for_metrics = encode_design(combined[factor_names], factors)
    X_model_combined, _ = build_model_matrix(combined_coded_for_metrics, factors, new_model_terms)

    XtX = X_model_combined.T @ X_model_combined
    try:
        det_combined = np.linalg.det(XtX)
        d_efficiency = 100.0  # Relative to itself (could compare to ideal)
    except:
        det_combined = 0.0
        d_efficiency = 0.0

    # Compute condition number
    try:
        condition_number = np.linalg.cond(XtX)
    except:
        condition_number = np.inf

    # Build result
    augmented = AugmentedDesign(
        combined_design=combined,
        new_runs_only=new_runs,
        block_column="Phase",
        n_runs_original=n_original,
        n_runs_added=n_runs_to_add,
        n_runs_total=len(combined),
        achieved_improvements={
            "model_terms": f"{len(current_model_terms)} → {len(new_model_terms)}",
            "n_runs": f"{n_original} → {len(combined)}",
        },
        d_efficiency=d_efficiency,
        condition_number=condition_number,
    )

    return augmented


def _augmented_coordinate_exchange(
    X_model_original: np.ndarray,
    candidates: np.ndarray,
    factors: List[Factor],
    model_terms: List[str],
    n_runs_to_add: int,
    criterion: str = "D",
    prediction_grid_config: Optional[Dict] = None,
    seed: Optional[int] = None,
    max_iterations: int = 100,
) -> np.ndarray:
    """
    Coordinate exchange for augmented design.

    This is a simplified version of the CEXCH algorithm that:
    1. Fixes existing runs (X_model_original)
    2. Optimizes selection of new runs from candidates
    3. Maximizes criterion value for [X_original; X_new]

    Parameters
    ----------
    X_model_original : np.ndarray
        Model matrix for existing runs
    candidates : np.ndarray
        Candidate points (coded space)
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms
    n_runs_to_add : int
        Number of runs to select
    criterion : {'D', 'I'}, default='D'
        Optimality criterion
    prediction_grid_config : dict, optional
        I-optimal prediction grid configuration
    seed : int, optional
        Random seed
    max_iterations : int
        Maximum iterations

    Returns
    -------
    np.ndarray
        Indices of selected candidates
    """
    from src.core.diagnostics.variance import build_model_matrix
    from src.core.optimal.criteria import (
        create_optimality_criterion,
        create_polynomial_builder,
    )

    rng = np.random.default_rng(seed)
    N_cand = len(candidates)
    p = X_model_original.shape[1]

    # Initialize with random selection
    selected_indices = rng.choice(N_cand, size=n_runs_to_add, replace=False)

    # Build model matrix for candidates — use coded DataFrame so categorical
    # columns contain string labels (for effects coding) and continuous factors
    # remain in [-1, 1] coded space as build_model_matrix expects.
    candidates_coded_df = _candidates_to_coded_df(candidates, factors)
    X_model_candidates, _ = build_model_matrix(candidates_coded_df, factors, model_terms)

    # Create criterion object
    # The builder receives numeric rows in coded space; convert categoricals only.
    def simple_builder(X_points: np.ndarray) -> np.ndarray:
        df = _candidates_to_coded_df(X_points, factors)
        X, _ = build_model_matrix(df, factors, model_terms)
        return X

    criterion_obj = create_optimality_criterion(
        criterion_type=criterion,
        model_builder=simple_builder,
        factors=factors,
        prediction_grid_config=prediction_grid_config,
    )

    # Compute initial objective
    X_current = X_model_candidates[selected_indices]
    X_combined = np.vstack([X_model_original, X_current])

    try:
        objective = criterion_obj.objective(X_combined)
    except:
        return selected_indices

    # Coordinate exchange
    for iteration in range(max_iterations):
        improved = False

        for i in range(n_runs_to_add):
            current_idx = selected_indices[i]
            best_idx = current_idx
            best_objective = objective

            # Try swapping with candidates
            candidate_subset = rng.choice(N_cand, size=min(50, N_cand), replace=False)

            for cand_idx in candidate_subset:
                if cand_idx in selected_indices:
                    continue

                # Swap
                trial_indices = selected_indices.copy()
                trial_indices[i] = cand_idx

                X_trial = X_model_candidates[trial_indices]
                X_trial_combined = np.vstack([X_model_original, X_trial])

                try:
                    objective_trial = criterion_obj.objective(X_trial_combined)
                    if objective_trial > best_objective:
                        best_objective = objective_trial
                        best_idx = cand_idx
                        improved = True
                except:
                    continue

            # Apply best swap
            if best_idx != current_idx:
                selected_indices[i] = best_idx
                X_current = X_model_candidates[selected_indices]
                X_combined = np.vstack([X_model_original, X_current])
                objective = best_objective

        if not improved:
            break

    return selected_indices


def augment_for_orthogonality(
    original_design: pd.DataFrame,
    factors: List[Factor],
    current_model_terms: List[str],
    n_runs_to_add: int,
    seed: Optional[int] = None,
) -> AugmentedDesign:
    """
    Add runs to improve orthogonality of the existing model.

    Use case: High VIF / collinearity in current model — no new terms needed,
    just better support for the terms already in the model.

    Parameters
    ----------
    original_design : pd.DataFrame
        Original design matrix (coded levels)
    factors : List[Factor]
        Factor definitions
    current_model_terms : List[str]
        Terms already in the model (these are supported, not extended)
    n_runs_to_add : int
        Number of runs to add
    seed : int, optional
        Random seed

    Returns
    -------
    AugmentedDesign
        Combined design with improved orthogonality

    Notes
    -----
    This calls `augment_for_model_extension` with new_model_terms == current_model_terms,
    so no model extension occurs — the optimizer simply finds runs that maximise
    D-efficiency for the existing model.
    """
    return augment_for_model_extension(
        original_design=original_design,
        factors=factors,
        current_model_terms=current_model_terms,
        new_model_terms=current_model_terms,
        n_runs_to_add=n_runs_to_add,
        criterion="D",
        seed=seed,
    )


def execute_optimal_plan(plan: AugmentationPlan) -> AugmentedDesign:
    """
    Execute optimal design augmentation plan (D or I).

    Called by AugmentationPlan.execute() for d_optimal strategies.

    Parameters
    ----------
    plan : AugmentationPlan
        Plan to execute

    Returns
    -------
    AugmentedDesign
        Result of execution
    """
    config = plan.strategy_config

    if not isinstance(config, OptimalAugmentConfig):
        raise TypeError(f"Expected OptimalAugmentConfig, got {type(config)}")

    # Validate required metadata
    if "current_model_terms" not in plan.metadata:
        raise ValueError(
            "Plan metadata missing 'current_model_terms'. "
            "Cannot perform model extension without current model specification."
        )

    current_model_terms = plan.metadata["current_model_terms"]

    if not current_model_terms:
        raise ValueError("current_model_terms cannot be empty")

    # plan.n_runs_to_add is the authoritative value — the user may have
    # adjusted it in the UI after the plan was created, so always prefer
    # it over config.n_runs_to_add which reflects the original recommendation.
    n_runs_to_add = plan.n_runs_to_add

    # Orthogonality mode: no model extension, just better support for existing terms
    augmentation_purpose = plan.metadata.get("augmentation_purpose", "model_extension")
    if augmentation_purpose == "orthogonality":
        augmented = augment_for_orthogonality(
            original_design=plan.original_design,
            factors=plan.factors,
            current_model_terms=current_model_terms,
            n_runs_to_add=n_runs_to_add,
            seed=plan.metadata.get("seed"),
        )
    else:
        # Extract criterion from config (default to 'D' for backward compatibility)
        criterion = getattr(config, "criterion", "D")
        prediction_grid_config = getattr(config, "prediction_grid_config", None)

        augmented = augment_for_model_extension(
            original_design=plan.original_design,
            factors=plan.factors,
            current_model_terms=current_model_terms,
            new_model_terms=config.new_model_terms,
            n_runs_to_add=n_runs_to_add,
            criterion=criterion,
            prediction_grid_config=prediction_grid_config,
            seed=plan.metadata.get("seed"),
        )

    # Attach plan provenance
    augmented.plan_executed = plan

    return augmented