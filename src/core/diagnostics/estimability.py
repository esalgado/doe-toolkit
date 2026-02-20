"""
Estimability Diagnostics for Design Quality Assessment.

This module computes Variance Inflation Factors (VIF), identifies collinearity
issues, and assesses model matrix conditioning.

Categorical VIF aggregation
---------------------------
When a model term is a categorical factor with k levels, it expands to k-1
columns in the model matrix.  VIF is computed per column and then aggregated
to a single per-term value using the **maximum** across the k-1 columns.
This is the conservative choice: it flags the worst collinearity within a
categorical term and mirrors how JMP reports VIF for categorical effects.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import warnings

from src.core.factors import Factor


def _term_to_col_indices(
    model_terms: List[str],
    col_names: List[str],
) -> Dict[str, List[int]]:
    """
    Map each model term to the column indices it occupies in the model matrix.

    Because categorical factors expand to k-1 columns, a single term can span
    multiple indices.  The mapping uses the column-name convention produced by
    ``build_model_matrix``:

    - Continuous / intercept terms have a single matching column name.
    - Categorical main effects produce columns named ``'Factor[Level]'``;
      all columns that start with ``'Factor['`` are assigned to term
      ``'Factor'``.
    - Interactions that involve categoricals produce columns like
      ``'Factor[A]*Temp'``; columns whose names contain all the original
      factor names separated by ``*`` are assigned to that interaction term.

    Parameters
    ----------
    model_terms : List[str]
        Original model terms (e.g. ``['1', 'A', 'Cat', 'A*Cat']``).
    col_names : List[str]
        Column names returned by ``build_model_matrix``.

    Returns
    -------
    Dict[str, List[int]]
        Mapping from term → list of column indices.

    Notes
    -----
    The intercept column is always named ``'Intercept'`` and maps to term
    ``'1'``.
    """
    mapping: Dict[str, List[int]] = {t: [] for t in model_terms}

    for col_idx, col_name in enumerate(col_names):
        for term in model_terms:
            if _col_belongs_to_term(col_name, term):
                mapping[term].append(col_idx)
                break  # each column belongs to exactly one term

    return mapping


def _col_belongs_to_term(col_name: str, term: str) -> bool:
    """
    Return True if *col_name* was generated from *term*.

    Rules
    -----
    - ``term='1'``         matches ``col_name='Intercept'``.
    - ``term='A'``         matches ``col_name='A'`` or ``col_name='A[...]'``.
    - ``term='A*B'``       matches ``col_name='A*B'`` or any column that
                           contains exactly the factor sub-names from the
                           term when split by ``*``.

    Parameters
    ----------
    col_name : str
        Single column name from the model matrix.
    term : str
        Original model term string.

    Returns
    -------
    bool
    """
    # Intercept
    if term == "1":
        return col_name == "Intercept"

    # Exact match (continuous main effect or interaction without categoricals)
    if col_name == term:
        return True

    # Categorical main effect: col looks like 'Factor[Level]'
    if "[" in col_name and "*" not in term:
        factor_part = col_name.split("[")[0]
        return factor_part == term

    # Interaction: split both term and col_name by '*' and compare factor parts
    if "*" in term:
        term_factors = [t.strip() for t in term.split("*")]
        col_factors = [c.strip() for c in col_name.split("*")]

        # Strip level suffixes from col factors: 'Cat[A]' -> 'Cat'
        col_factors_base = [
            c.split("[")[0] if "[" in c else c for c in col_factors
        ]

        return sorted(term_factors) == sorted(col_factors_base)

    return False


def compute_vif(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
) -> Dict[str, float]:
    """
    Compute Variance Inflation Factors (VIF) for model terms.

    VIF measures multicollinearity via auxiliary regressions:
        VIF_j = 1 / (1 - R²_j)
    where R²_j is the R² from regressing column j on all other columns.

    For categorical terms that expand to k-1 columns, the **maximum** VIF
    across those columns is returned as the term-level VIF.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms (intercept ``'1'`` is included to anchor the regression
        but excluded from the returned dict).

    Returns
    -------
    Dict[str, float]
        VIF per term (intercept excluded).  ``np.inf`` signals perfect
        collinearity; ``np.nan`` signals a saturated / singular design.

    Notes
    -----
    Auxiliary regression for column j:
        X_j ~ X_{-j}   (all other columns, including the intercept)

    VIF > 10 is commonly flagged as problematic.

    Examples
    --------
    >>> vif = compute_vif(design, factors, ['1', 'A', 'B', 'A*B'])
    >>> print(vif['A*B'])
    2.5
    """
    from src.core.diagnostics.variance import build_model_matrix

    X, col_names = build_model_matrix(design, factors, model_terms)

    terms_no_intercept = [t for t in model_terms if t != "1"]

    if X.shape[1] == 0:
        return {}

    n, p = X.shape

    if n <= p:
        warnings.warn(
            f"Cannot compute VIF: n_runs ({n}) ≤ n_terms ({p}). "
            "Design is saturated or supersaturated."
        )
        return {term: np.nan for term in terms_no_intercept}

    # Map each original term to its column indices in X
    term_col_map = _term_to_col_indices(model_terms, col_names)

    # Compute per-column VIF for all non-intercept columns
    intercept_col_idx = col_names.index("Intercept") if "Intercept" in col_names else None
    non_intercept_cols = [
        i for i in range(p)
        if intercept_col_idx is None or i != intercept_col_idx
    ]

    col_vif: Dict[int, float] = {}

    for col_idx in non_intercept_cols:
        X_j = X[:, col_idx].reshape(-1, 1)
        X_not_j = np.delete(X, col_idx, axis=1)

        try:
            beta, _, _, _ = np.linalg.lstsq(X_not_j, X_j, rcond=None)
            y_pred = X_not_j @ beta
            ss_res = float(np.sum((X_j.flatten() - y_pred.flatten()) ** 2))
            ss_tot = float(np.sum((X_j.flatten() - np.mean(X_j)) ** 2))

            if ss_tot > 0:
                r_squared = 1.0 - ss_res / ss_tot
            else:
                r_squared = 0.0

            if r_squared >= 0.9999:
                col_vif[col_idx] = np.inf
            elif r_squared < 0:
                col_vif[col_idx] = 1.0
            else:
                col_vif[col_idx] = 1.0 / (1.0 - r_squared)

        except (np.linalg.LinAlgError, ValueError):
            col_vif[col_idx] = np.inf

    # Aggregate per-column VIF → per-term VIF (max across categorical columns)
    vif_values: Dict[str, float] = {}

    for term in terms_no_intercept:
        indices = term_col_map.get(term, [])
        if not indices:
            vif_values[term] = np.nan
            continue

        term_vifs = [col_vif.get(i, np.nan) for i in indices]
        finite_vifs = [v for v in term_vifs if np.isfinite(v)]

        if any(np.isinf(v) for v in term_vifs):
            vif_values[term] = np.inf
        elif finite_vifs:
            vif_values[term] = float(max(finite_vifs))
        else:
            vif_values[term] = np.nan

    return vif_values


def check_collinearity(
    vif_values: Dict[str, float],
    threshold: float = 10.0,
) -> List[str]:
    """
    Return terms whose VIF exceeds *threshold*.

    Parameters
    ----------
    vif_values : Dict[str, float]
        Output of ``compute_vif``.
    threshold : float, default=10.0
        VIF threshold above which a term is flagged.

    Returns
    -------
    List[str]
        Term names with VIF > threshold.
    """
    return [
        term
        for term, vif in vif_values.items()
        if not np.isnan(vif) and not np.isinf(vif) and vif > threshold
    ]


def compute_condition_number(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
) -> float:
    """
    Compute the condition number κ of the model matrix X.

    κ = σ_max / σ_min  (ratio of largest to smallest singular value)

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
    float
        Condition number.  ``np.inf`` if the matrix is singular.

    Notes
    -----
    κ > 1000 indicates severe numerical ill-conditioning.
    """
    from src.core.diagnostics.variance import build_model_matrix

    X, _ = build_model_matrix(design, factors, model_terms)

    try:
        sv = np.linalg.svd(X, compute_uv=False)
        return float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    except (np.linalg.LinAlgError, ValueError):
        return np.inf


def assess_estimability(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
) -> Tuple[bool, List[str]]:
    """
    Assess whether all model terms are estimable given the design.

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
    all_estimable : bool
        False if any hard estimability failure is found.
    issues : List[str]
        Descriptions of detected problems.
    """
    from src.core.diagnostics.variance import build_model_matrix

    X, _ = build_model_matrix(design, factors, model_terms)
    n, p = X.shape

    issues: List[str] = []

    if n < p:
        issues.append(
            f"Design is supersaturated: {n} runs for {p} terms. "
            "Cannot estimate all effects."
        )
        return False, issues

    if n == p:
        issues.append(
            f"Design is saturated: {n} runs for {p} terms. "
            "No degrees of freedom for error. Cannot test effects."
        )

    rank = np.linalg.matrix_rank(X)
    if rank < p:
        issues.append(
            f"Model matrix is rank-deficient: rank = {rank}, expected {p}. "
            f"{p - rank} term(s) are linearly dependent."
        )
        return False, issues

    vif_values = compute_vif(design, factors, model_terms)
    high_vif = [t for t, v in vif_values.items() if np.isfinite(v) and v > 50]

    if high_vif:
        issues.append(
            f"Severe collinearity (VIF > 50): {', '.join(high_vif)}. "
            "Coefficient estimates may be unreliable."
        )

    kappa = compute_condition_number(design, factors, model_terms)
    if kappa > 1000:
        issues.append(
            f"Severely ill-conditioned design (κ = {kappa:.1e}). "
            "Numerical instability likely."
        )

    all_estimable = len(issues) == 0 or (
        len(issues) == 1 and "saturated" in issues[0]
    )
    return all_estimable, issues


def identify_redundant_terms(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    tolerance: float = 1e-6,
) -> List[str]:
    """
    Identify model terms that are linearly dependent (via QR decomposition).

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms.
    tolerance : float, default=1e-6
        Diagonal elements of R below this value are treated as zero.

    Returns
    -------
    List[str]
        Terms whose corresponding column is linearly dependent.

    Notes
    -----
    For categorical terms that span multiple columns, the *term* is flagged
    if **any** of its columns are linearly dependent.
    """
    from src.core.diagnostics.variance import build_model_matrix

    X, col_names = build_model_matrix(design, factors, model_terms)

    try:
        _, R = np.linalg.qr(X)
        diag_R = np.abs(np.diag(R))
        dependent_col_indices = set(np.where(diag_R < tolerance)[0].tolist())
    except (np.linalg.LinAlgError, ValueError):
        return []

    term_col_map = _term_to_col_indices(model_terms, col_names)
    redundant_terms = []

    for term, indices in term_col_map.items():
        if any(i in dependent_col_indices for i in indices):
            redundant_terms.append(term)

    return redundant_terms


def compute_leverage(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
) -> np.ndarray:
    """
    Compute leverage (hat-matrix diagonal) for each observation.

    h_i = x_i' (X'X)⁻¹ x_i

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
        Leverage values (shape: n_runs,).  ``np.nan`` if computation fails.

    Notes
    -----
    The sum of all leverage values equals the number of model-matrix columns p,
    not the number of original terms (because categoricals expand).
    """
    from src.core.diagnostics.variance import build_model_matrix

    X, _ = build_model_matrix(design, factors, model_terms)
    n, p = X.shape

    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(p))
        return np.sum((X @ XtX_inv) * X, axis=1)
    except (np.linalg.LinAlgError, ValueError):
        return np.full(n, np.nan)


def identify_high_leverage_points(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    threshold_multiplier: float = 2.0,
) -> List[int]:
    """
    Return indices of observations whose leverage exceeds the threshold.

    threshold = threshold_multiplier × (p / n)

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms.
    threshold_multiplier : float, default=2.0
        Multiplier on average leverage p/n.

    Returns
    -------
    List[int]
        Zero-based row indices of high-leverage observations.
    """
    from src.core.diagnostics.variance import build_model_matrix

    X, _ = build_model_matrix(design, factors, model_terms)
    n, p = X.shape

    leverage = compute_leverage(design, factors, model_terms)

    if np.any(np.isnan(leverage)):
        return []

    threshold = threshold_multiplier * (p / n)
    return list(np.where(leverage > threshold)[0])
