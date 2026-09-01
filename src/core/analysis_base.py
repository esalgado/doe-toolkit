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
                # Two-way interaction (A*B -> patsy stores as A:B).
                # Continuous factors enter as scalar * value; categorical
                # factors match their dummy key F[T.level]. patsy expands a
                # categorical interaction into dummies with the all-reference
                # baseline suppressed.
                prediction += self._predict_interaction(
                    factor_list, term, settings, coefficients
                )

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

    def _predict_interaction(
        self,
        factor_list: List[str],
        term: str,
        settings: Dict[str, object],
        coefficients: pd.Series,
    ) -> float:
        """
        Predict the contribution of a two-way interaction term.

        Handles continuous/continuous (single ``A:B`` coefficient) as well as
        categorical interactions (patsy expands into ``C(A)[T.x]:C(B)[T.y]``
        dummies with the all-reference baseline suppressed; the plain ``A:B``
        key corresponds to a categorical main effect evaluated at its
        reference level in the other dimension).

        Each factor in the interaction is matched against the levels in
        ``settings``.  Continuous factors are treated as scalar variables; a
        match is required for categorical ones.
        """
        total = 0.0
        f1, f2 = factor_list
        v1 = settings.get(f1, 0)
        v2 = settings.get(f2, 0)

        # Classify each side as continuous (scalar) or categorical (has
        # dummy keys of the form F[T.level] in the coefficient index).
        def is_categorical(name: str) -> bool:
            cat_level_key = f"{name}[T."
            return any(
                str(k).startswith(cat_level_key)
                for k in coefficients.index
            )

        cat1 = is_categorical(f1)
        cat2 = is_categorical(f2)

        if not cat1 and not cat2:
            # Both continuous: single coefficient A:B
            coef_key = next(
                (k for k in coefficients.index
                 if k == term or k == term.replace("*", ":")),
                None,
            )
            if coef_key is not None:
                return float(coefficients[coef_key]) * float(v1) * float(v2)
            return 0.0

        # At least one categorical. Build the exact patsy interaction key(s)
        # for the selected levels and sum the matching coefficients.
        #   categorical-only : A[T.x]:B[T.y]   (order matches term)
        #   mixed            : A[T.x]:B  or  A:B[T.y]
        def side_key(name: str, val: object, cat: bool) -> str:
            return f"{name}[T.{val}]" if cat else name

        k1 = side_key(f1, v1, cat1)
        k2 = side_key(f2, v2, cat2)

        candidates = {f"{k1}:{k2}", f"{k2}:{k1}"}
        for k in coefficients.index:
            if str(k) in candidates:
                total += float(coefficients[k])
        return total


# ---------------------------------------------------------------------------
# Term parsing
# ---------------------------------------------------------------------------

# Transform prefixes recognised by parse_model_term.
# Maps Patsy prefix → (numpy function name used for factor extraction).
_TRANSFORM_PREFIXES: Tuple[str, ...] = (
    "np.log(",
    "np.sqrt(",
    "np.exp(",
    "I(1/",
)


def _extract_transform_factor(term: str) -> Optional[str]:
    """
    Return the raw factor name embedded in a transform term, or None.

    Handles the four recognised transform forms::

        np.log(A)   → 'A'
        np.sqrt(A)  → 'A'
        np.exp(A)   → 'A'
        I(1/A)      → 'A'

    The term must start with one of the known prefixes; the factor name is
    the content between the prefix and the matching closing parenthesis.

    Parameters
    ----------
    term : str
        A single Patsy fragment (no cross ``*`` operators).

    Returns
    -------
    str or None
        Raw factor name, or ``None`` if not a recognised transform.
    """
    for prefix in _TRANSFORM_PREFIXES:
        if term.startswith(prefix):
            # Strip prefix and trailing ')'
            inner = term[len(prefix):]
            if inner.endswith(')'):
                inner = inner[:-1]
            return inner.strip()
    return None


def _split_at_outer_star(term: str) -> Tuple[str, str]:
    """
    Split a term at the first ``*`` that sits outside all parentheses.

    Returns ``(left, right)`` where ``right`` is everything after the ``*``.
    If no outer ``*`` exists, returns ``(term, ''')``.

    Used to separate ``np.log(A)*B`` into ``('np.log(A)', 'B')``.
    """
    depth = 0
    for i, ch in enumerate(term):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '*' and depth == 0:
            return term[:i], term[i + 1:]
    return term, ''


def parse_model_term(term: str) -> Tuple[List[str], str]:
    """
    Parse a patsy-notation model term into its constituent factors and operator.

    Recognised term forms
    ---------------------
    - Main effect         : ``'A'``              → ``(['A'], '')``
    - Interaction         : ``'A*B'``            → ``(['A', 'B'], '*')``
    - Quadratic           : ``'I(A**2)'``        → ``(['A'], '**')``
    - Transform           : ``'np.log(A)'``      → ``(['A'], 'transform')``
    - Transform power     : ``'I(np.log(A)**2)'``→ ``(['A'], 'transform_power')``
    - Transform cross     : ``'np.log(A)*B'``    → ``(['A', 'B'], 'transform_cross')``
    - Transform pwr cross : ``'I(np.log(A)**2)*B'`` → ``(['A', 'B'], 'transform_power_cross')``

    The ``factor_list`` always contains raw design factor names (e.g.
    ``'Temperature'``), never the transform wrapper strings.

    Parameters
    ----------
    term : str
        A single model term in patsy notation.

    Returns
    -------
    factor_list : List[str]
        Names of the constituent factor(s).
    operator : str
        One of ``''``, ``'*'``, ``'**'``, ``'transform'``,
        ``'transform_power'``, ``'transform_cross'``,
        ``'transform_power_cross'``.

    Examples
    --------
    >>> parse_model_term('Temperature')
    (['Temperature'], '')
    >>> parse_model_term('Temperature*Pressure')
    (['Temperature', 'Pressure'], '*')
    >>> parse_model_term('I(Temperature**2)')
    (['Temperature'], '**')
    >>> parse_model_term('np.log(Temperature)')
    (['Temperature'], 'transform')
    >>> parse_model_term('np.log(Temperature)*Pressure')
    (['Temperature', 'Pressure'], 'transform_cross')
    >>> parse_model_term('I(np.log(Temperature)**2)*Pressure')
    (['Temperature', 'Pressure'], 'transform_power_cross')
    """
    # ------------------------------------------------------------------ #
    # I(...**n) — power wrapper; base may be a plain factor or transform  #
    # Examples: I(A**2), I(np.log(A)**2), I(np.log(A)**2)*B              #
    # ------------------------------------------------------------------ #
    if term.startswith('I(') and '**' in term:
        # Find the ')' that closes the outer I( by tracking paren depth.
        # term.index(')') would find the first ')' which may be inside a
        # nested transform such as I(np.log(A)**2).
        depth = 0
        close = -1
        for _i, _ch in enumerate(term):
            if _ch == '(':
                depth += 1
            elif _ch == ')':
                depth -= 1
                if depth == 0:
                    close = _i
                    break
        if close == -1:
            # Malformed term — fall through to main-effect
            return [term.strip()], ''
        power_content = term[2:close]           # e.g. 'A**2' or 'np.log(A)**2'
        remainder = term[close + 1:]            # e.g. '' or '*B'

        base_fragment, _ = power_content.rsplit('**', 1)
        base_fragment = base_fragment.strip()
        raw_factor = _extract_transform_factor(base_fragment)

        if remainder.startswith('*'):
            # Power × cross: I(A**2)*B  or  I(np.log(A)**2)*B
            cross_factor = remainder[1:].strip()
            if raw_factor is not None:
                return [raw_factor, cross_factor], 'transform_power_cross'
            return [base_fragment, cross_factor], '**'

        if raw_factor is not None:
            return [raw_factor], 'transform_power'
        return [base_fragment], '**'

    # ------------------------------------------------------------------ #
    # Transform terms (no I() power wrapper)                              #
    # Examples: np.log(A), np.log(A)*B                                   #
    # ------------------------------------------------------------------ #
    for prefix in _TRANSFORM_PREFIXES:
        if term.startswith(prefix):
            base_fragment, remainder = _split_at_outer_star(term)
            raw_factor = _extract_transform_factor(base_fragment)
            if raw_factor is None:
                break  # Malformed — fall through to main-effect
            if remainder:
                cross_factor = remainder.strip()
                return [raw_factor, cross_factor], 'transform_cross'
            return [raw_factor], 'transform'

    # ------------------------------------------------------------------ #
    # Plain interaction: A*B  (checked AFTER transforms to avoid          #
    # misclassifying np.log(A)*B as an interaction)                      #
    # ------------------------------------------------------------------ #
    if '*' in term:
        factor_list = [_unwrap_c(seg) for seg in term.split('*')]
        return factor_list, '*'

    # ------------------------------------------------------------------ #
    # Plain main effect: A (or C(A) for categorical factors)              #
    # ------------------------------------------------------------------ #
    return [_unwrap_c(term.strip())], ''


def _unwrap_c(segment: str) -> str:
    """
    Strip a patsy ``C(...)`` wrapper from a factor segment so its underlying
    factor name can be found in the factor registry.

    Examples
    --------
    >>> _unwrap_c('C(Egg_lot)')
    'Egg_lot'
    >>> _unwrap_c('Egg_lot')
    'Egg_lot'
    """
    segment = segment.strip()
    if segment.startswith('C(') and segment.endswith(')'):
        return segment[2:-1].strip()
    return segment


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
