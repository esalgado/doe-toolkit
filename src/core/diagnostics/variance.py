"""
Prediction Variance Diagnostics for Design Quality Assessment.

This module computes prediction variance statistics across the design space
to identify regions with poor precision.

Categorical factor handling
---------------------------
Categorical factors are encoded using **sum-to-zero (effects) coding**, which
is the convention used by JMP and Design-Expert.  For a factor with k levels:

  - k-1 indicator columns are added to the model matrix.
  - The last level is the reference: it receives -1 on every column.
  - All other levels receive +1 on their own column and 0 elsewhere.

Example: Catalyst with levels [A, B, C]
  Run A  → [+1,  0]
  Run B  → [ 0, +1]
  Run C  → [-1, -1]   ← reference level

This makes main-effect estimates interpretable in the presence of interactions
and mirrors what commercial DOE tools report.

`build_model_matrix` returns **both** the numeric matrix and a list of column
names so that callers (VIF, leverage, …) can map columns back to terms.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.core.factors import Factor, FactorType


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_term_type(term: str) -> Tuple[str, List[str], Optional[int]]:
    """
    Parse term string into type, factor names, and optional power.

    Parameters
    ----------
    term : str
        Model term (e.g., '1', 'A', 'A*B', 'I(A**2)', 'A^2')

    Returns
    -------
    term_type : str
        One of: 'intercept', 'main', 'interaction', 'power'
    factors : List[str]
        Factor names involved in the term.
    power : int or None
        Power for polynomial terms; None for other types.

    Examples
    --------
    >>> _parse_term_type('1')
    ('intercept', [], None)
    >>> _parse_term_type('A')
    ('main', ['A'], None)
    >>> _parse_term_type('A*B')
    ('interaction', ['A', 'B'], None)
    >>> _parse_term_type('I(A**2)')
    ('power', ['A'], 2)
    >>> _parse_term_type('A^2')
    ('power', ['A'], 2)
    """
    if term == "1":
        return "intercept", [], None

    # I(A**2) notation (patsy style)
    if term.startswith("I(") and "**" in term:
        inner = term[2:-1]
        parts = inner.split("**")
        return "power", [parts[0].strip()], int(parts[1].strip())

    # A^2 or A**2 notation
    if "^" in term or "**" in term:
        sep = "^" if "^" in term else "**"
        parts = term.split(sep)
        return "power", [parts[0].strip()], int(parts[1].strip())

    # Interaction: A*B
    if "*" in term:
        factor_names = [f.strip() for f in term.split("*")]
        return "interaction", factor_names, None

    # Main effect
    return "main", [term.strip()], None


def _effects_encode(
    raw: np.ndarray,
    factor_name: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply sum-to-zero (effects) coding to a categorical column.

    For k unique levels the last level (in order of first appearance) is the
    reference.  k-1 columns are returned.

    Parameters
    ----------
    raw : np.ndarray
        Raw string values for one factor (shape: n_runs,).
    factor_name : str
        Factor name used to build column labels.

    Returns
    -------
    encoded : np.ndarray
        Shape (n_runs, k-1).  dtype float64.
    col_names : List[str]
        Column labels, e.g. ``['Catalyst[A]', 'Catalyst[B]']``.

    Notes
    -----
    Sum-to-zero coding:
      - Non-reference level j  →  +1 in column j, 0 elsewhere.
      - Reference level (last) →  -1 in all k-1 columns.

    References
    ----------
    .. [1] Venables, W. N. & Ripley, B. D. (2002). *Modern Applied Statistics
           with S*, 4th ed.  Section 6.2.

    Examples
    --------
    >>> raw = np.array(['A', 'B', 'C', 'A', 'C'])
    >>> enc, names = _effects_encode(raw, 'Cat')
    >>> names
    ['Cat[A]', 'Cat[B]']
    >>> enc
    array([[ 1.,  0.],
           [ 0.,  1.],
           [-1., -1.],
           [ 1.,  0.],
           [-1., -1.]])
    """
    # Preserve insertion order so the reference level is deterministic
    unique_levels: List[str] = list(dict.fromkeys(raw.tolist()))
    k = len(unique_levels)

    if k < 2:
        raise ValueError(
            f"Categorical factor '{factor_name}' has fewer than 2 levels."
        )

    # Non-reference levels (all except the last)
    non_ref_levels = unique_levels[:-1]
    ref_level = unique_levels[-1]

    n = len(raw)
    encoded = np.zeros((n, k - 1), dtype=float)

    for col_idx, level in enumerate(non_ref_levels):
        is_level = raw == level
        is_ref = raw == ref_level
        encoded[is_level, col_idx] = 1.0
        encoded[is_ref, col_idx] = -1.0
        # All other levels remain 0

    col_names = [f"{factor_name}[{lvl}]" for lvl in non_ref_levels]
    return encoded, col_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_model_matrix(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the model matrix X and its column names from a design and term list.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with one column per factor.
    factors : List[Factor]
        Factor definitions (used to detect categorical factors).
    model_terms : List[str]
        Model terms, e.g. ``['1', 'A', 'B', 'A*B', 'I(A**2)', 'Cat']``.

    Returns
    -------
    X : np.ndarray
        Model matrix of shape (n_runs, n_cols).  dtype float64.
    col_names : List[str]
        Column label for every column of X.  Categorical main effects expand
        to ``k-1`` labels of the form ``'Factor[Level]'``.

    Notes
    -----
    **Continuous / discrete-numeric factors** are used as-is (coded values).

    **Categorical factors** receive sum-to-zero (effects) coding (see module
    docstring).  A main-effect term for a factor with k levels expands to k-1
    columns.  An interaction term that involves at least one categorical factor
    also expands: each categorical is replaced by its k-1 indicator columns
    and element-wise multiplied with the continuous or categorical columns of
    the other factor(s) in the interaction.

    Examples
    --------
    >>> X, names = build_model_matrix(design, factors, ['1', 'A', 'B', 'A*B'])
    >>> X.shape
    (16, 4)
    >>> names
    ['Intercept', 'A', 'B', 'A*B']

    With a 3-level categorical ``Cat`` (levels A, B, C):

    >>> X, names = build_model_matrix(design, factors, ['1', 'Cat'])
    >>> X.shape
    (12, 3)
    >>> names
    ['Intercept', 'Cat[A]', 'Cat[B]']
    """
    n_runs = len(design)

    # Build a lookup: factor_name -> (is_categorical, numeric_or_encoded_array)
    # For categorical factors we store the full (n, k-1) encoded block.
    # For numeric factors we store a 1-D array.
    factor_lookup: Dict[str, Tuple[bool, np.ndarray, List[str]]] = {}

    cat_names = {f.name for f in factors if f.factor_type == FactorType.CATEGORICAL}

    for f in factors:
        raw = design[f.name].values
        if f.name in cat_names:
            encoded_block, enc_col_names = _effects_encode(raw, f.name)
            factor_lookup[f.name] = (True, encoded_block, enc_col_names)
        else:
            numeric_col = raw.astype(float)
            factor_lookup[f.name] = (False, numeric_col, [f.name])

    columns: List[np.ndarray] = []
    col_names: List[str] = []

    for term in model_terms:
        term_type, factors_in_term, power = _parse_term_type(term)

        if term_type == "intercept":
            columns.append(np.ones(n_runs))
            col_names.append("Intercept")

        elif term_type == "main":
            fname = factors_in_term[0]
            if fname not in factor_lookup:
                raise ValueError(f"Unknown factor: '{fname}'")

            is_cat, data, names = factor_lookup[fname]
            if is_cat:
                # data is (n, k-1); add each column individually
                for c_idx, c_name in enumerate(names):
                    columns.append(data[:, c_idx])
                    col_names.append(c_name)
            else:
                columns.append(data)
                col_names.append(fname)

        elif term_type == "power":
            fname = factors_in_term[0]
            if fname not in factor_lookup:
                raise ValueError(f"Unknown factor in term '{term}': {fname}")

            is_cat, data, _ = factor_lookup[fname]
            if is_cat:
                raise ValueError(
                    f"Polynomial term '{term}' is not valid for categorical "
                    f"factor '{fname}'."
                )
            columns.append(data**power)
            col_names.append(term)

        elif term_type == "interaction":
            # Build the set of expanded columns for each factor in the term,
            # then take the outer product across factors.
            factor_column_sets: List[Tuple[np.ndarray, List[str]]] = []

            for fname in factors_in_term:
                if fname not in factor_lookup:
                    raise ValueError(
                        f"Unknown factor in term '{term}': {fname}"
                    )
                is_cat, data, names = factor_lookup[fname]
                if is_cat:
                    # Each categorical column as separate 1-D array
                    factor_column_sets.append(
                        (
                            np.array([data[:, i] for i in range(data.shape[1])]),
                            names,
                        )
                    )
                else:
                    factor_column_sets.append(
                        (data.reshape(1, -1), [fname])
                    )

            # Enumerate all combinations across factors
            # factor_column_sets[i] = (array of shape (n_cols_i, n_runs), [names])
            # We want the Cartesian product of columns across factors
            import itertools

            all_col_arrays = [fcs[0] for fcs in factor_column_sets]
            all_col_names_lists = [fcs[1] for fcs in factor_column_sets]

            for combo in itertools.product(
                *[range(arr.shape[0]) for arr in all_col_arrays]
            ):
                combined_col = np.ones(n_runs)
                combined_name_parts = []
                for factor_idx, col_idx in enumerate(combo):
                    combined_col = combined_col * all_col_arrays[factor_idx][col_idx]
                    combined_name_parts.append(
                        all_col_names_lists[factor_idx][col_idx]
                    )
                columns.append(combined_col)
                col_names.append("*".join(combined_name_parts))

        else:
            raise ValueError(f"Unknown term type for '{term}'")

    X = np.column_stack(columns)
    return X, col_names


def compute_prediction_variance(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    sigma_squared: float = 1.0,
) -> np.ndarray:
    """
    Compute prediction variance at each design point.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms.
    sigma_squared : float, default=1.0
        Residual variance estimate.

    Returns
    -------
    np.ndarray
        Prediction variance at each design point (shape: n_runs,).

    Notes
    -----
    Prediction variance at point x:

        Var(ŷ) = σ² · x'(X'X)⁻¹x

    References
    ----------
    .. [1] Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016).
           *Response surface methodology*, 4th ed.  Wiley.
    """
    X, _ = build_model_matrix(design, factors, model_terms)

    XtX = X.T @ X

    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.inv(XtX + 1e-8 * np.eye(XtX.shape[0]))

    pred_var = np.sum((X @ XtX_inv) * X, axis=1) * sigma_squared
    return pred_var


def prediction_variance_stats(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    sigma_squared: float = 1.0,
) -> Dict[str, float]:
    """
    Compute summary statistics of prediction variance across design points.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms.
    sigma_squared : float, default=1.0
        Residual variance estimate.

    Returns
    -------
    Dict[str, float]
        Keys: ``'min'``, ``'max'``, ``'mean'``, ``'std'``, ``'range'``,
        ``'max_ratio'``.

    Examples
    --------
    >>> stats = prediction_variance_stats(design, factors, model_terms, sigma_squared=2.5)
    >>> print(f"Max prediction variance: {stats['max']:.2f}")
    >>> print(f"Max/Min ratio: {stats['max_ratio']:.2f}")
    """
    pred_var = compute_prediction_variance(
        design, factors, model_terms, sigma_squared
    )

    result: Dict[str, float] = {
        "min": float(np.min(pred_var)),
        "max": float(np.max(pred_var)),
        "mean": float(np.mean(pred_var)),
        "std": float(np.std(pred_var)),
        "range": float(np.ptp(pred_var)),
    }

    if result["min"] > 0:
        result["max_ratio"] = result["max"] / result["min"]
    else:
        result["max_ratio"] = np.inf

    return result


def identify_high_variance_regions(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    sigma_squared: float = 1.0,
    threshold: float = 2.0,
) -> List[Dict]:
    """
    Identify design runs with prediction variance above ``threshold × mean``.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms.
    sigma_squared : float, default=1.0
        Residual variance estimate.
    threshold : float, default=2.0
        Multiplier on mean variance to define "high".

    Returns
    -------
    List[Dict]
        Each dict has keys ``'run_index'``, ``'variance'``,
        ``'variance_ratio'``, plus one key per factor name.

    Examples
    --------
    >>> regions = identify_high_variance_regions(design, factors, model_terms, threshold=2.5)
    >>> for r in regions:
    ...     print(f"Run {r['run_index']}: {r['variance']:.2f}")
    """
    pred_var = compute_prediction_variance(
        design, factors, model_terms, sigma_squared
    )
    mean_var = np.mean(pred_var)

    high_var_indices = np.where(pred_var > threshold * mean_var)[0]

    regions = []
    for idx in high_var_indices:
        region: Dict = {
            "run_index": int(idx),
            "variance": float(pred_var[idx]),
            "variance_ratio": float(pred_var[idx] / mean_var),
        }
        for factor in factors:
            region[factor.name] = design[factor.name].iloc[idx]
        regions.append(region)

    return regions


def compute_fraction_of_design_space(
    pred_var: np.ndarray,
    threshold: float,
    n_grid_points: int = 1000,
) -> float:
    """
    Estimate fraction of design space with prediction variance ≤ threshold.

    Parameters
    ----------
    pred_var : np.ndarray
        Prediction variances at design points.
    threshold : float
        Variance threshold.
    n_grid_points : int
        Unused in this simple version; reserved for future Monte-Carlo
        implementation.

    Returns
    -------
    float
        Empirical fraction of design points with variance ≤ threshold.

    Notes
    -----
    This uses the empirical CDF of the design points as an approximation.
    A Monte-Carlo grid evaluation would be more accurate but is out of scope
    for this helper.
    """
    return float(np.mean(pred_var <= threshold))


def compute_scaled_prediction_variance(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
) -> np.ndarray:
    """
    Compute scaled prediction variance (SPV = n · x'(X'X)⁻¹x).

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms.

    Returns
    -------
    np.ndarray
        SPV at each design point.

    Notes
    -----
    SPV is dimensionless and useful for comparing designs of different sizes.
    For an ideal design, mean(SPV) equals the number of model parameters p.

    References
    ----------
    .. [1] Box, G. E., & Draper, N. R. (2007). *Response surfaces, mixtures,
           and ridge analyses*, 2nd ed.  Wiley.
    """
    n = len(design)
    pred_var = compute_prediction_variance(
        design, factors, model_terms, sigma_squared=1.0
    )
    return n * pred_var


def assess_variance_uniformity(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
) -> Tuple[bool, str]:
    """
    Assess whether prediction variance is acceptably uniform across the design.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms.

    Returns
    -------
    is_uniform : bool
        True when max(SPV) / min(SPV) < 3.
    message : str
        Human-readable description of variance uniformity.

    Notes
    -----
    Thresholds (max/min SPV ratio):
      - < 3   → uniform (acceptable)
      - 3–5   → moderately non-uniform
      - ≥ 5   → highly non-uniform
    """
    spv = compute_scaled_prediction_variance(design, factors, model_terms)

    min_spv = float(np.min(spv))
    max_spv = float(np.max(spv))

    ratio = (max_spv / min_spv) if min_spv > 0 else np.inf

    if ratio < 3.0:
        return True, f"Prediction variance is uniform (max/min ratio: {ratio:.2f})"
    elif ratio < 5.0:
        return (
            False,
            f"Prediction variance is moderately non-uniform (ratio: {ratio:.2f})",
        )
    else:
        return (
            False,
            f"Prediction variance is highly non-uniform (ratio: {ratio:.2f})",
        )


def compute_i_criterion(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    prediction_grid_config: Optional[dict] = None,
) -> float:
    """
    Compute the I-optimality criterion (average prediction variance).

    I = trace((X'X)⁻¹ · M)

    where M is the moment matrix over a prediction grid.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms.
    prediction_grid_config : dict, optional
        Passed to ``generate_prediction_grid``.

    Returns
    -------
    float
        I-criterion value (lower = better prediction uniformity).

    Notes
    -----
    D-optimal designs minimise parameter variance; I-optimal designs minimise
    average prediction variance.  Both criteria are complementary.

    References
    ----------
    .. [1] Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).
           *Optimum experimental designs, with SAS*.  Oxford University Press.
    .. [2] Jones, B., & Goos, P. (2012). I-optimal versus D-optimal
           split-plot response-surface designs. *Journal of Quality
           Technology*, 44(2), 85-101.

    Examples
    --------
    >>> i_val = compute_i_criterion(design, factors, model_terms)
    >>> print(f"Average prediction variance: {i_val:.3f}")
    """
    from src.core.optimal.criteria import generate_prediction_grid

    X, _ = build_model_matrix(design, factors, model_terms)

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(XtX.shape[0]))
    except np.linalg.LinAlgError:
        return np.inf

    factor_names = [f.name for f in factors]
    prediction_points = generate_prediction_grid(
        factors, prediction_grid_config or {}
    )
    pred_df = pd.DataFrame(prediction_points, columns=factor_names)
    X_pred, _ = build_model_matrix(pred_df, factors, model_terms)

    M = (X_pred.T @ X_pred) / len(prediction_points)
    return float(np.trace(XtX_inv @ M))


def compute_design_quality_metrics(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    include_i_optimal: bool = True,
    prediction_grid_config: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive design quality metrics (D and I optimality).

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms.
    include_i_optimal : bool, default=True
        Whether to compute the I-criterion.
    prediction_grid_config : dict, optional
        Passed to ``compute_i_criterion``.

    Returns
    -------
    Dict[str, float]
        Keys: ``'d_efficiency'``, ``'condition_number'``,
        ``'avg_prediction_variance'``, ``'max_prediction_variance'``,
        ``'prediction_variance_ratio'``, and optionally ``'i_criterion'``.

    Examples
    --------
    >>> metrics = compute_design_quality_metrics(design, factors, model_terms)
    >>> print(f"D-efficiency: {metrics['d_efficiency']:.1f}%")
    """
    X, _ = build_model_matrix(design, factors, model_terms)
    XtX = X.T @ X

    metrics: Dict[str, float] = {}

    try:
        det_XtX = np.linalg.det(XtX)
        metrics["d_efficiency"] = 100.0 if det_XtX > 0 else 0.0
        metrics["condition_number"] = float(np.linalg.cond(XtX))
    except Exception:
        metrics["d_efficiency"] = 0.0
        metrics["condition_number"] = np.inf

    try:
        pv_stats = prediction_variance_stats(
            design, factors, model_terms, sigma_squared=1.0
        )
        metrics["avg_prediction_variance"] = pv_stats["mean"]
        metrics["max_prediction_variance"] = pv_stats["max"]
        metrics["prediction_variance_ratio"] = pv_stats["max_ratio"]
    except Exception:
        metrics["avg_prediction_variance"] = np.nan
        metrics["max_prediction_variance"] = np.nan
        metrics["prediction_variance_ratio"] = np.nan

    if include_i_optimal:
        try:
            metrics["i_criterion"] = compute_i_criterion(
                design, factors, model_terms, prediction_grid_config
            )
        except Exception:
            metrics["i_criterion"] = np.nan

    return metrics
