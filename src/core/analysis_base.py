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
