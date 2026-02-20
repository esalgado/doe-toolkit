"""
Shared primitives for ANOVA analysis modules.

This module holds the types and pure-utility functions that are needed by
*both* ``analysis.py`` (fixed-effects / mixed-effects ANOVA) and
``split_plot_analysis.py`` (two-strata split-plot ANOVA).  Keeping them here
breaks the circular import that would otherwise exist between those two
modules.

Contents
--------
- ``ANOVAResults``     – dataclass container for all ANOVA outputs.
- ``parse_model_term`` – parse a patsy-notation term into constituent factors.
- ``enforce_hierarchy``– ensure hierarchical completeness of a model term list.
- ``quadratic``        – convenience helper for quadratic term notation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ANOVAResults:
    """
    Container for ANOVA model results.

    Attributes
    ----------
    anova_table : pd.DataFrame
        ANOVA table.  For split-plot models this is the two-strata table with
        a ``Stratum`` column; for fixed-effects models it is the standard
        Type II table from statsmodels.
    effect_estimates : pd.DataFrame
        Coefficient table with columns ``Coefficient``, ``Std_Error``,
        ``t_value``, ``p_value``.
    logworth : pd.DataFrame
        Per-term ``LogWorth`` (``-log10(p)``).
    residuals : np.ndarray
        Model residuals (subplot level for split-plot).
    fitted_values : np.ndarray
        Fitted/predicted values corresponding to each observation.
    fitted_model : object
        The underlying statsmodels result object (OLS or MixedLM).
    diagnostics : Dict[str, any]
        Diagnostic test results, e.g. ``{'shapiro_wilk': {'statistic': ..., 'p_value': ...}}``.
    model_terms : List[str]
        Model terms used in the fit (patsy notation, including ``'1'`` for intercept).
    is_split_plot : bool
        ``True`` when the two-strata split-plot estimator was used.
    r_squared : float
        Coefficient of determination.
    adj_r_squared : float
        Adjusted R².
    rmse : float
        Root mean squared error of residuals.
    """

    anova_table: pd.DataFrame
    effect_estimates: pd.DataFrame
    logworth: pd.DataFrame
    residuals: np.ndarray
    fitted_values: np.ndarray
    fitted_model: object
    diagnostics: Dict[str, object]
    model_terms: List[str]
    is_split_plot: bool
    r_squared: float
    adj_r_squared: float
    rmse: float

    def predict_from_settings(self, settings: Dict[str, object]) -> float:
        """
        Predict the response for a dict of factor settings without calling patsy.

        Builds the model matrix row directly from ``self.model_terms`` and
        ``self.effect_estimates``, so it works regardless of which underlying
        statsmodels object is stored in ``fitted_model`` and never requires
        nuisance columns such as ``WholePlot`` to be present.

        Supported term types
        --------------------
        - Intercept  : ``'1'``
        - Main effect: ``'A'``
        - Interaction: ``'A*B'``
        - Quadratic  : ``'I(A**2)'``
        - Categorical main effect via dummy coding (patsy ``FactorName[T.Level]`` convention).

        Parameters
        ----------
        settings : Dict[str, object]
            Factor name -> value mapping (coded or actual; must match the
            scale used when the model was fitted).

        Returns
        -------
        float
            Predicted response value.

        Notes
        -----
        Categorical reference levels contribute only through the intercept.
        Non-reference levels are matched via ``FactorName[T.Level]`` keys
        in ``effect_estimates``.

        Examples
        --------
        >>> results.predict_from_settings({'Temperature': -1.0, 'Time': 0.5})
        42.7
        """
        coefficients = self.effect_estimates["Coefficient"]
        prediction = 0.0

        for term in self.model_terms:
            if term == "1":
                intercept_key = next(
                    (k for k in coefficients.index if k in ("Intercept", "const")),
                    None,
                )
                if intercept_key is not None:
                    prediction += float(coefficients[intercept_key])
                continue

            factor_list, operator = parse_model_term(term)

            if operator == "**":
                # Quadratic: I(A**2)
                fname = factor_list[0]
                val = float(settings.get(fname, 0))  # type: ignore[arg-type]
                coef_key = next(
                    (k for k in coefficients.index
                     if k == term or f"I({fname} ** 2)" in k),
                    None,
                )
                if coef_key is not None:
                    prediction += float(coefficients[coef_key]) * val ** 2

            elif operator == "*":
                # Two-way interaction (A*B -> patsy stores as A:B)
                vals = [float(settings.get(f, 0)) for f in factor_list]  # type: ignore[arg-type]
                product = vals[0] * vals[1]
                coef_key = next(
                    (k for k in coefficients.index
                     if k == term or k == term.replace("*", ":")),
                    None,
                )
                if coef_key is not None:
                    prediction += float(coefficients[coef_key]) * product

            else:
                # Main effect - continuous or categorical
                fname = factor_list[0]
                val = settings.get(fname)
                if val is None:
                    continue

                if fname in coefficients.index:
                    # Numeric (continuous / discrete)
                    prediction += float(coefficients[fname]) * float(val)  # type: ignore[arg-type]
                else:
                    # Categorical dummy: "Catalyst[T.B]"
                    dummy_key = f"{fname}[T.{val}]"
                    if dummy_key in coefficients.index:
                        prediction += float(coefficients[dummy_key])
                    # Reference level contributes 0 (absorbed in intercept)

        return prediction


# ---------------------------------------------------------------------------
# Term parsing
# ---------------------------------------------------------------------------


def parse_model_term(term: str) -> Tuple[List[str], str]:
    """
    Parse a patsy-notation model term into its constituent factors and operator.

    Recognised term forms
    ---------------------
    - Main effect  : ``'A'``   → ``(['A'], '')``
    - Interaction  : ``'A*B'`` → ``(['A', 'B'], '*')``
    - Quadratic    : ``'I(A**2)'`` → ``(['A'], '**')``

    Parameters
    ----------
    term : str
        A single model term in patsy notation.

    Returns
    -------
    factor_list : List[str]
        Names of the constituent factor(s).
    operator : str
        ``'*'`` for interactions, ``'**'`` for quadratic terms, ``''`` for
        main effects.

    Examples
    --------
    >>> parse_model_term('Temperature')
    (['Temperature'], '')
    >>> parse_model_term('Temperature*Pressure')
    (['Temperature', 'Pressure'], '*')
    >>> parse_model_term('I(Temperature**2)')
    (['Temperature'], '**')
    """
    if '*' in term and not term.startswith('I('):
        # Interaction: A*B
        factor_list = [f.strip() for f in term.split('*')]
        return factor_list, '*'
    elif term.startswith('I(') and '**' in term:
        # Quadratic: I(A**2)
        inner = term[2:-1]
        base = inner.split('**')[0].strip()
        return [base], '**'
    else:
        # Main effect
        return [term.strip()], ''


# ---------------------------------------------------------------------------
# Hierarchy enforcement
# ---------------------------------------------------------------------------


def enforce_hierarchy(
    terms: List[str],
    factor_names: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Ensure hierarchical completeness and canonical ordering of model terms.

    For every interaction or quadratic term present, all lower-order parents
    (main effects) are added if missing.  The returned list is ordered as:
    intercept → main effects → two-way interactions → quadratic terms →
    any remaining terms.

    Parameters
    ----------
    terms : List[str]
        Current model terms (patsy notation).
    factor_names : List[str]
        All factor names in the design (used to validate that auto-added
        parents actually exist in the design).

    Returns
    -------
    ordered_terms : List[str]
        Deduplicated, hierarchy-complete, canonically ordered term list.
    added_terms : List[str]
        Terms that were inserted to satisfy hierarchy (may be empty).

    Examples
    --------
    >>> enforce_hierarchy(['1', 'A*B'], ['A', 'B'])
    (['1', 'A', 'B', 'A*B'], ['A', 'B'])
    """
    complete_terms = list(terms)
    added_terms: List[str] = []
    complete_terms_set: set = set(terms)
    factor_names_set: set = set(factor_names)

    for term in terms:
        if term == '1':
            continue

        factor_list, operator = parse_model_term(term)

        if operator in ('*', '**'):
            for factor_name in factor_list:
                if (
                    factor_name not in complete_terms_set
                    and factor_name in factor_names_set
                ):
                    complete_terms.append(factor_name)
                    added_terms.append(factor_name)
                    complete_terms_set.add(factor_name)

    # Deduplicate while preserving first-seen order
    seen: set = set()
    unique_terms = [t for t in complete_terms if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

    # Canonical ordering: intercept → main → interaction → quadratic → other
    ordered: List[str] = []

    if '1' in unique_terms:
        ordered.append('1')

    for term in unique_terms:
        if term == '1':
            continue
        factor_list, op = parse_model_term(term)
        if op == '' and len(factor_list) == 1:
            ordered.append(term)

    for term in unique_terms:
        if term == '1':
            continue
        factor_list, op = parse_model_term(term)
        if op == '*' and len(factor_list) == 2:
            ordered.append(term)

    for term in unique_terms:
        if term == '1':
            continue
        factor_list, op = parse_model_term(term)
        if op == '**':
            ordered.append(term)

    for term in unique_terms:
        if term not in ordered:
            ordered.append(term)

    return ordered, added_terms


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def compute_actual_coefficients(
    effect_estimates: pd.DataFrame,
    factors: List[Factor],
) -> pd.DataFrame:
    """
    Convert coded-unit coefficients and standard errors to actual units.

    The model is fit on coded factors where each continuous factor x is
    transformed via ``x_coded = (x_actual - center) / half_range``.
    This function inverts that scaling so the coefficients reflect the
    original factor units.

    Transformation rules
    --------------------
    - **Intercept**: adjusted for all centering shifts.
    - **Main effect** (continuous): ``b_actual = b_coded / half_range``
    - **Interaction A:B** (both continuous): ``b_actual = b_coded / (hr_A * hr_B)``
    - **Quadratic I(A**2)**: ``b_actual = b_coded / hr_A**2``
    - **Categorical / discrete-numeric** terms: passed through unchanged.

    Standard errors scale by the same factor as their corresponding
    coefficient (no centering correction needed for errors).

    Parameters
    ----------
    effect_estimates : pd.DataFrame
        Coefficient table with at minimum ``Coefficient`` and ``Std_Error``
        columns and term names as the index (patsy convention).
    factors : List[Factor]
        Factor definitions used to retrieve ``min_value`` / ``max_value``
        for continuous factors.

    Returns
    -------
    pd.DataFrame
        Copy of ``effect_estimates`` with two additional columns:
        ``Actual_Coefficient`` and ``Actual_Std_Error``.  Rows for
        categorical dummies or unrecognised terms are filled with ``NaN``.

    Notes
    -----
    Discrete-numeric factors are treated as categorical for this conversion
    because their levels are not necessarily evenly spaced, and a single
    half-range scale factor would be misleading.

    Examples
    --------
    >>> compute_actual_coefficients(effect_estimates, factors)
       Coefficient  Std_Error  ...  Actual_Coefficient  Actual_Std_Error
    """
    # Build lookup: factor_name -> (center, half_range) for continuous only
    scale: Dict[str, Tuple[float, float]] = {}
    for f in factors:
        if f.is_continuous() and f.min_value is not None and f.max_value is not None:
            center = (f.min_value + f.max_value) / 2.0
            half_range = (f.max_value - f.min_value) / 2.0
            if half_range > 0:
                scale[f.name] = (center, half_range)

    result = effect_estimates.copy()
    actual_coefs: List[float] = []
    actual_ses: List[float] = []

    # Pre-compute intercept adjustment from all continuous main effects
    intercept_adjustment = 0.0
    for term_name in effect_estimates.index:
        if term_name in ("Intercept", "const"):
            continue
        _factors, _op = _parse_index_term(term_name)
        if _op == "" and len(_factors) == 1:
            fname = _factors[0]
            if fname in scale:
                center, half_range = scale[fname]
                b_coded = float(effect_estimates.loc[term_name, "Coefficient"])
                intercept_adjustment += b_coded * (-center / half_range)

    for term_name in effect_estimates.index:
        b_coded = float(effect_estimates.loc[term_name, "Coefficient"])
        se_coded = float(effect_estimates.loc[term_name, "Std_Error"])

        if term_name in ("Intercept", "const"):
            actual_coefs.append(b_coded + intercept_adjustment)
            actual_ses.append(se_coded)  # SE of intercept is not rescaled
            continue

        factor_names_parsed, op = _parse_index_term(term_name)

        if op == "**":  # Quadratic: I(A ** 2)
            fname = factor_names_parsed[0]
            if fname in scale:
                _, half_range = scale[fname]
                s = 1.0 / (half_range ** 2)
                actual_coefs.append(b_coded * s)
                actual_ses.append(se_coded * s)
            else:
                actual_coefs.append(float("nan"))
                actual_ses.append(float("nan"))

        elif op == ":":  # Interaction A:B (patsy uses ":" separator)
            if len(factor_names_parsed) == 2:
                fa, fb = factor_names_parsed
                if fa in scale and fb in scale:
                    _, hr_a = scale[fa]
                    _, hr_b = scale[fb]
                    s = 1.0 / (hr_a * hr_b)
                    actual_coefs.append(b_coded * s)
                    actual_ses.append(se_coded * s)
                else:
                    # Mixed continuous/categorical interaction — not converted
                    actual_coefs.append(float("nan"))
                    actual_ses.append(float("nan"))
            else:
                actual_coefs.append(float("nan"))
                actual_ses.append(float("nan"))

        elif op == "":  # Main effect or categorical dummy
            fname = factor_names_parsed[0]
            if fname in scale:
                _, half_range = scale[fname]
                s = 1.0 / half_range
                actual_coefs.append(b_coded * s)
                actual_ses.append(se_coded * s)
            else:
                # Categorical: pass through as NaN (no single scale factor)
                actual_coefs.append(float("nan"))
                actual_ses.append(float("nan"))

        else:
            actual_coefs.append(float("nan"))
            actual_ses.append(float("nan"))

    result["Actual_Coefficient"] = actual_coefs
    result["Actual_Std_Error"] = actual_ses
    return result


def _parse_index_term(term_name: str) -> Tuple[List[str], str]:
    """
    Parse a statsmodels/patsy coefficient index label into factor names and operator.

    Handles the following patsy output conventions:
    - ``'Intercept'`` / ``'const'``  → ``([], '')``
    - ``'A'``                         → ``(['A'], '')``
    - ``'A[T.level]'``                → ``(['A'], '')``  (categorical dummy)
    - ``'A:B'``                        → ``(['A', 'B'], ':')``
    - ``'I(A ** 2)'``                 → ``(['A'], '**')``

    Parameters
    ----------
    term_name : str
        A single index label from ``fitted_model.params``.

    Returns
    -------
    factor_names : List[str]
        Extracted factor name(s).
    operator : str
        ``':'`` for interactions, ``'**'`` for quadratics, ``''`` otherwise.
    """
    import re

    if term_name in ("Intercept", "const"):
        return [], ""

    # Quadratic: "I(A ** 2)"
    quad_match = re.match(r"I\((.+?)\s*\*\*\s*2\)", term_name)
    if quad_match:
        return [quad_match.group(1).strip()], "**"

    # Interaction: "A:B" (patsy uses ":" not "*")
    if ":" in term_name and not term_name.startswith("I("):
        parts = term_name.split(":")
        # Strip categorical encoding e.g. "A[T.level]" -> "A"
        clean = [re.sub(r"\[T\..*?\]", "", p).strip() for p in parts]
        return clean, ":"

    # Categorical dummy: "A[T.level]" -> factor name is "A"
    cat_match = re.match(r"^([^\[]+)\[T\.", term_name)
    if cat_match:
        return [cat_match.group(1).strip()], ""

    # Plain main effect
    return [term_name.strip()], ""


def quadratic(factor_name: str) -> str:
    """
    Return the patsy quadratic term notation for a factor.

    Parameters
    ----------
    factor_name : str
        Name of a continuous factor.

    Returns
    -------
    str
        Patsy identity-function notation, e.g. ``'I(Temperature**2)'``.

    Examples
    --------
    >>> quadratic('Temperature')
    'I(Temperature**2)'
    """
    return f"I({factor_name}**2)"
