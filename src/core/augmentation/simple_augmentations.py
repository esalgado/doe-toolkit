"""
Simple Augmentation Strategies: Center Points and Replicates.

These strategies do not require optimisation — they add fixed-structure runs
to the design.  They are used by Mode A when lack-of-fit or insufficient
replication issues are detected.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.augmentation.plan import AugmentationPlan, AugmentedDesign
from src.core.factors import Factor, FactorType


# ---------------------------------------------------------------------------
# Center Points
# ---------------------------------------------------------------------------


def augment_with_center_points(
    original_design: pd.DataFrame,
    factors: List[Factor],
    n_center_points: int = 3,
    seed: Optional[int] = None,
) -> AugmentedDesign:
    """
    Add center points (all-zeros in coded space) to an existing design.

    Center points allow curvature detection without a full response-surface
    upgrade.  They also provide a pure-error estimate when replicated.

    Parameters
    ----------
    original_design : pd.DataFrame
        Original design matrix (coded levels).
    factors : List[Factor]
        Factor definitions.  Only continuous factors get a midpoint (0);
        categorical factors are excluded from center-point runs.
    n_center_points : int, default=3
        Number of center-point runs to add.  Three is the typical minimum
        for a useful pure-error estimate.
    seed : int, optional
        Random seed (used only for run-order randomisation).

    Returns
    -------
    AugmentedDesign
        Combined design with center-point runs appended as Phase 2.

    Examples
    --------
    >>> augmented = augment_with_center_points(design, factors, n_center_points=3)
    >>> print(f"Added {augmented.n_runs_added} center points")

    Notes
    -----
    Categorical factors cannot have a meaningful center point.  If the design
    contains categorical factors, this function raises ``ValueError``.

    References
    ----------
    .. [1] Montgomery, D. C. (2017). Design and Analysis of Experiments, 9th Ed.
    """
    rng = np.random.default_rng(seed)

    factor_names = [f.name for f in factors]

    # Reject if any categorical factors present — no natural center point
    categorical = [f.name for f in factors if f.factor_type == FactorType.CATEGORICAL]
    if categorical:
        raise ValueError(
            f"Cannot add center points when categorical factors are present: "
            f"{categorical}"
        )

    # Build center-point rows using actual midpoint values (not coded 0).
    center_row: Dict[str, object] = {}
    for f in factors:
        if f.factor_type == FactorType.CONTINUOUS:
            min_val, max_val = float(f.levels[0]), float(f.levels[1])
            center_row[f.name] = (min_val + max_val) / 2.0
        elif f.factor_type == FactorType.DISCRETE_NUMERIC:
            levels_sorted = sorted(float(v) for v in f.levels)
            mid_idx = len(levels_sorted) // 2
            center_row[f.name] = levels_sorted[mid_idx]
        else:
            # Categorical — already blocked above, but defensive default
            center_row[f.name] = f.levels[0]

    center_runs = pd.DataFrame([center_row] * n_center_points)
    center_runs["Phase"] = 2

    # Propagate WholePlot column if present (assign a new whole-plot group
    # for center points, continuing the numbering from the original design).
    if "WholePlot" in original_design.columns:
        max_wp = int(original_design["WholePlot"].max())
        center_runs["WholePlot"] = max_wp + 1

    # Randomise run order within the new block
    center_runs = center_runs.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(
        drop=True
    )

    # Tag original runs
    original_with_phase = original_design.copy()
    original_with_phase["Phase"] = 1

    combined = pd.concat([original_with_phase, center_runs], ignore_index=True)

    # Re-index StdOrder / RunOrder
    for col in ("StdOrder", "RunOrder"):
        if col in combined.columns:
            combined = combined.drop(columns=[col])
    combined.insert(0, "StdOrder", range(1, len(combined) + 1))
    combined.insert(1, "RunOrder", range(1, len(combined) + 1))

    # Condition number of model matrix (intercept + main effects)
    try:
        from src.core.diagnostics.variance import build_model_matrix

        main_terms = ["1"] + factor_names
        X, _ = build_model_matrix(combined[factor_names], factors, main_terms)
        condition_number = float(np.linalg.cond(X.T @ X))
    except Exception:
        condition_number = np.inf

    return AugmentedDesign(
        combined_design=combined,
        new_runs_only=center_runs,
        block_column="Phase",
        n_runs_original=len(original_design),
        n_runs_added=n_center_points,
        n_runs_total=len(combined),
        achieved_improvements={
            "curvature_detection": "Center points enable LOF / curvature test",
            "pure_error": f"{n_center_points} replicates provide pure-error estimate",
            "n_runs": f"{len(original_design)} → {len(combined)}",
        },
        condition_number=condition_number,
    )


def execute_center_points_plan(plan: AugmentationPlan) -> AugmentedDesign:
    """
    Execute a center-points augmentation plan.

    Called by ``AugmentationPlan.execute()`` for ``center_points`` strategies.

    Parameters
    ----------
    plan : AugmentationPlan
        Plan to execute.

    Returns
    -------
    AugmentedDesign
        Result of execution.
    """
    augmented = augment_with_center_points(
        original_design=plan.original_design,
        factors=plan.factors,
        n_center_points=plan.n_runs_to_add,
        seed=plan.metadata.get("seed"),
    )
    augmented.plan_executed = plan
    return augmented


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------


def augment_with_replicates(
    original_design: pd.DataFrame,
    factors: List[Factor],
    n_replicates: int,
    seed: Optional[int] = None,
) -> AugmentedDesign:
    """
    Add replicate runs to an existing design.

    Replicates provide a pure-error estimate, which is required for a formal
    lack-of-fit F-test.  Runs are selected to spread replication across the
    design space rather than concentrating it at one point.

    Parameters
    ----------
    original_design : pd.DataFrame
        Original design matrix (coded levels).
    factors : List[Factor]
        Factor definitions.
    n_replicates : int
        Number of replicate runs to add.  These are sampled (without replacement
        where possible) from the existing runs.
    seed : int, optional
        Random seed.

    Returns
    -------
    AugmentedDesign
        Combined design with replicate runs appended as Phase 2.

    Examples
    --------
    >>> augmented = augment_with_replicates(design, factors, n_replicates=4)
    >>> print(f"Added {augmented.n_runs_added} replicates")

    Notes
    -----
    Selection strategy: choose runs spread across the design space by
    maximising the minimum pairwise distance between selected replicates.
    Falls back to random selection if the distance calculation fails.

    References
    ----------
    .. [1] Montgomery, D. C. (2017). Design and Analysis of Experiments, 9th Ed.
    """
    rng = np.random.default_rng(seed)

    factor_names = [f.name for f in factors]

    n_original = len(original_design)

    # Select which original runs to replicate (spread across space)
    replicate_indices = _select_spread_replicates(
        original_design[factor_names].values,
        n_replicates=n_replicates,
        rng=rng,
    )

    # Carry forward WholePlot if present so replicated runs inherit the
    # whole-plot assignment of the run they are replicating.
    cols_to_copy = factor_names + (["WholePlot"] if "WholePlot" in original_design.columns else [])
    replicate_runs = original_design.iloc[replicate_indices][cols_to_copy].copy()
    replicate_runs = replicate_runs.sample(
        frac=1, random_state=int(rng.integers(1e6))
    ).reset_index(drop=True)
    replicate_runs["Phase"] = 2

    original_with_phase = original_design.copy()
    original_with_phase["Phase"] = 1

    combined = pd.concat([original_with_phase, replicate_runs], ignore_index=True)

    for col in ("StdOrder", "RunOrder"):
        if col in combined.columns:
            combined = combined.drop(columns=[col])
    combined.insert(0, "StdOrder", range(1, len(combined) + 1))
    combined.insert(1, "RunOrder", range(1, len(combined) + 1))

    try:
        from src.core.diagnostics.variance import build_model_matrix

        main_terms = ["1"] + factor_names
        X, _ = build_model_matrix(combined[factor_names], factors, main_terms)
        condition_number = float(np.linalg.cond(X.T @ X))
    except Exception:
        condition_number = np.inf

    return AugmentedDesign(
        combined_design=combined,
        new_runs_only=replicate_runs,
        block_column="Phase",
        n_runs_original=n_original,
        n_runs_added=n_replicates,
        n_runs_total=len(combined),
        achieved_improvements={
            "pure_error": f"{n_replicates} replicates enable formal LOF F-test",
            "n_runs": f"{n_original} → {len(combined)}",
        },
        condition_number=condition_number,
    )


def _select_spread_replicates(
    X: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Select run indices spread across the design space.

    Uses a greedy maximin strategy: pick the first point randomly, then
    iteratively pick the point that maximises the minimum distance to
    already-selected points.

    Parameters
    ----------
    X : np.ndarray
        Factor matrix (n_runs × n_factors), coded levels.
    n_replicates : int
        Number of runs to select.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    List[int]
        Indices of selected runs.
    """
    n = len(X)
    n_select = min(n_replicates, n)

    try:
        from scipy.spatial.distance import cdist

        distances = cdist(X, X)
        np.fill_diagonal(distances, np.inf)

        selected: List[int] = [int(rng.integers(n))]

        while len(selected) < n_select:
            min_dists = np.min(distances[:, selected], axis=1)
            min_dists[selected] = -np.inf  # exclude already selected
            selected.append(int(np.argmax(min_dists)))

        return selected

    except Exception:
        # Fallback: random selection
        return list(rng.choice(n, size=n_select, replace=False))


def execute_replicates_plan(plan: AugmentationPlan) -> AugmentedDesign:
    """
    Execute a replicates augmentation plan.

    Called by ``AugmentationPlan.execute()`` for ``replicates`` strategies.

    Parameters
    ----------
    plan : AugmentationPlan
        Plan to execute.

    Returns
    -------
    AugmentedDesign
        Result of execution.
    """
    augmented = augment_with_replicates(
        original_design=plan.original_design,
        factors=plan.factors,
        n_replicates=plan.n_runs_to_add,
        seed=plan.metadata.get("seed"),
    )
    augmented.plan_executed = plan
    return augmented
