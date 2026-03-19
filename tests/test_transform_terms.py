"""
Tests for transform model terms in ANOVA analysis.

Covers Patsy formula construction, model fitting, and coefficient
estimation for the four transform types introduced in the model builder:

    np.log(A)       — natural log
    np.sqrt(A)      — square root
    I(1/A)          — reciprocal
    np.exp(A)       — exponential

Also covers composed terms produced by the Advanced Terms expander:

    I(np.log(A)**2)*B   — powered transform crossed with another factor
    np.log(A)*B         — transform crossed with another factor (power=1 path)

parse_model_term and validate_model_terms fully support all transform term
forms. TestTransformTermValidation verifies acceptance and rejection cases.
"""

import warnings
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.core.analysis import ANOVAAnalysis, validate_model_terms
from src.core.analysis_base import parse_model_term, enforce_hierarchy
from src.core.factors import Factor, FactorType, ChangeabilityLevel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _continuous_factors() -> List[Factor]:
    """Two continuous factors with strictly positive ranges (safe for all transforms)."""
    return [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1.0, 5.0]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[2.0, 8.0]),
    ]


def _make_design(n_center: int = 4) -> pd.DataFrame:
    """
    Small full-factorial + centre points in natural units.
    A in [1, 5], B in [2, 8] — all values positive, safe for log/sqrt/recip.
    """
    corners = pd.DataFrame({
        "A": [1.0, 1.0, 5.0, 5.0],
        "B": [2.0, 8.0, 2.0, 8.0],
    })
    centres = pd.DataFrame({
        "A": [3.0] * n_center,
        "B": [5.0] * n_center,
    })
    return pd.concat([corners, centres], ignore_index=True)


# ---------------------------------------------------------------------------
# TestFormulaConstruction
# Verify that _build_formula produces valid Patsy strings that OLS can parse.
# These tests call analysis.fit() and assert it does not raise.
# ---------------------------------------------------------------------------

class TestFormulaConstruction:
    """Transform terms round-trip cleanly through Patsy formula evaluation."""

    def _analysis(self, design: pd.DataFrame, response: np.ndarray) -> ANOVAAnalysis:
        factors = _continuous_factors()
        return ANOVAAnalysis(
            design=design,
            response=response,
            factors=factors,
            response_name="Y",
        )

    def test_log_term_fits_without_error(self):
        """np.log(A) is accepted by Patsy and OLS fits cleanly."""
        design = _make_design()
        rng = np.random.default_rng(0)
        response = 5.0 + 2.0 * np.log(design["A"]) + rng.normal(0, 0.1, len(design))

        analysis = self._analysis(design, response)
        results = analysis.fit(["1", "np.log(A)"], enforce_hierarchy_flag=False)

        assert results is not None
        assert len(results.residuals) == len(design)

    def test_sqrt_term_fits_without_error(self):
        """np.sqrt(A) is accepted by Patsy and OLS fits cleanly."""
        design = _make_design()
        rng = np.random.default_rng(1)
        response = 3.0 + 1.5 * np.sqrt(design["A"]) + rng.normal(0, 0.1, len(design))

        analysis = self._analysis(design, response)
        results = analysis.fit(["1", "np.sqrt(A)"], enforce_hierarchy_flag=False)

        assert results is not None
        assert len(results.residuals) == len(design)

    def test_reciprocal_term_fits_without_error(self):
        """I(1/A) is accepted by Patsy and OLS fits cleanly."""
        design = _make_design()
        rng = np.random.default_rng(2)
        response = 10.0 + 4.0 * (1.0 / design["A"]) + rng.normal(0, 0.1, len(design))

        analysis = self._analysis(design, response)
        results = analysis.fit(["1", "I(1/A)"], enforce_hierarchy_flag=False)

        assert results is not None
        assert len(results.residuals) == len(design)

    def test_exp_term_fits_without_error(self):
        """
        np.exp(A) is accepted by Patsy and OLS fits cleanly.
        A is kept small (centred) to avoid numerical overflow.
        """
        design = _make_design()
        # Centre A so exp(A) stays manageable
        a_centred = design["A"] - design["A"].mean()
        rng = np.random.default_rng(3)
        response = 2.0 + 0.5 * np.exp(a_centred) + rng.normal(0, 0.1, len(design))
        design = design.copy()
        design["A"] = a_centred

        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-2.0, 2.0]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[2.0, 8.0]),
        ]
        analysis = ANOVAAnalysis(
            design=design, response=response, factors=factors, response_name="Y"
        )
        results = analysis.fit(["1", "np.exp(A)"], enforce_hierarchy_flag=False)

        assert results is not None
        assert len(results.residuals) == len(design)

    def test_log_squared_cross_fits_without_error(self):
        """I(np.log(A)**2)*B is accepted by Patsy and OLS fits cleanly."""
        design = _make_design(n_center=6)
        rng = np.random.default_rng(4)
        response = (
            4.0
            + 1.0 * (np.log(design["A"]) ** 2) * design["B"]
            + rng.normal(0, 0.1, len(design))
        )

        analysis = self._analysis(design, response)
        # enforce_hierarchy_flag=False: transform terms bypass hierarchy checks
        results = analysis.fit(
            ["1", "np.log(A)", "B", "I(np.log(A)**2)*B"],
            enforce_hierarchy_flag=False,
        )

        assert results is not None

    def test_log_cross_power1_fits_without_error(self):
        """np.log(A)*B (power=1 path) is accepted by Patsy and OLS fits cleanly."""
        design = _make_design(n_center=6)
        rng = np.random.default_rng(5)
        response = (
            3.0
            + 2.0 * np.log(design["A"]) * design["B"]
            + rng.normal(0, 0.1, len(design))
        )

        analysis = self._analysis(design, response)
        results = analysis.fit(
            ["1", "np.log(A)", "B", "np.log(A)*B"],
            enforce_hierarchy_flag=False,
        )

        assert results is not None


# ---------------------------------------------------------------------------
# TestCoefficientRecovery
# With a large-ish, clean dataset and known true coefficients, verify that
# the fitted coefficients are in the right ballpark.
# ---------------------------------------------------------------------------

class TestCoefficientRecovery:
    """Fitted coefficients approximate known true values for transform terms."""

    def _big_design(self, n: int = 80) -> pd.DataFrame:
        """Random design in natural units, all values positive."""
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "A": rng.uniform(1.0, 5.0, n),
            "B": rng.uniform(2.0, 8.0, n),
        })

    def test_log_coefficient_sign_correct(self):
        """
        When the true model is Y = 5 + 3*log(A), the fitted coefficient
        on np.log(A) should be positive and in the right direction.
        """
        design = self._big_design()
        response = 5.0 + 3.0 * np.log(design["A"])

        factors = _continuous_factors()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Y")
        results = analysis.fit(["1", "np.log(A)"], enforce_hierarchy_flag=False)

        # Patsy stores the coefficient under 'np.log(A)'
        coef_keys = [k for k in results.effect_estimates.index if "log" in k.lower()]
        assert coef_keys, "No log coefficient found in effect_estimates"
        coef = results.effect_estimates.loc[coef_keys[0], "Coefficient"]
        assert coef > 0, f"Expected positive log coefficient, got {coef:.4f}"
        assert abs(coef - 3.0) < 0.5, f"Log coefficient {coef:.4f} far from true 3.0"

    def test_sqrt_coefficient_sign_correct(self):
        """
        When the true model is Y = 2 + 4*sqrt(A), the fitted coefficient
        on np.sqrt(A) should be positive.
        """
        design = self._big_design()
        response = 2.0 + 4.0 * np.sqrt(design["A"])

        factors = _continuous_factors()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Y")
        results = analysis.fit(["1", "np.sqrt(A)"], enforce_hierarchy_flag=False)

        coef_keys = [k for k in results.effect_estimates.index if "sqrt" in k.lower()]
        assert coef_keys, "No sqrt coefficient found in effect_estimates"
        coef = results.effect_estimates.loc[coef_keys[0], "Coefficient"]
        assert coef > 0, f"Expected positive sqrt coefficient, got {coef:.4f}"

    def test_reciprocal_coefficient_sign_correct(self):
        """
        When the true model is Y = 8 - 2*(1/A), the fitted coefficient
        on I(1/A) should be negative.
        """
        design = self._big_design()
        response = 8.0 - 2.0 * (1.0 / design["A"])

        factors = _continuous_factors()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Y")
        results = analysis.fit(["1", "I(1/A)"], enforce_hierarchy_flag=False)

        coef_keys = [k for k in results.effect_estimates.index if "1 / A" in k or "1/A" in k]
        assert coef_keys, (
            f"No reciprocal coefficient found. Keys: {list(results.effect_estimates.index)}"
        )
        coef = results.effect_estimates.loc[coef_keys[0], "Coefficient"]
        assert coef < 0, f"Expected negative reciprocal coefficient, got {coef:.4f}"

    def test_r_squared_high_for_transform_model(self):
        """
        A model that exactly matches the DGP (no noise) should achieve R²≈1.
        """
        design = self._big_design()
        response = 5.0 + 3.0 * np.log(design["A"]) + 2.0 * design["B"]

        factors = _continuous_factors()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Y")
        results = analysis.fit(["1", "np.log(A)", "B"], enforce_hierarchy_flag=False)

        assert results.r_squared > 0.99, (
            f"Expected R²≈1 for exact transform model, got {results.r_squared:.4f}"
        )


# ---------------------------------------------------------------------------
# TestDomainErrors
# Verify that domain-violating values (log of zero/negative, sqrt of
# negative, division by zero) raise or produce NaN/Inf that surfaces clearly
# rather than silently corrupting results.
# ---------------------------------------------------------------------------

class TestDomainErrors:
    """Invalid input domains for transforms fail loudly, not silently."""

    def test_log_of_zero_raises_or_warns(self):
        """
        A design containing A=0 with a log(A) term should either raise a
        ValueError / produce a runtime warning — not silently return a
        numerically valid result.
        """
        design = pd.DataFrame({
            "A": [0.0, 1.0, 2.0, 3.0, 1.5, 2.5],
            "B": [1.0, 2.0, 3.0, 4.0, 2.5, 3.5],
        })
        response = np.array([1.0, 2.0, 3.0, 4.0, 2.5, 3.5])

        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0.0, 3.0]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1.0, 4.0]),
        ]
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Y")

        with pytest.warns((RuntimeWarning,)) as warning_info:
            try:
                results = analysis.fit(["1", "np.log(A)"], enforce_hierarchy_flag=False)
                # If it fits, residuals must not all be finite (log(0) = -inf)
                assert not np.all(np.isfinite(results.fitted_values)), (
                    "Expected non-finite fitted values when log(0) present"
                )
            except (ValueError, FloatingPointError):
                pass  # Raising is also acceptable

        # Confirm the warning relates to log domain
        warning_messages = [str(w.message) for w in warning_info.list]
        assert any(
            "divide" in m.lower() or "invalid" in m.lower() or "log" in m.lower()
            for m in warning_messages
        ), f"Expected a log-domain warning, got: {warning_messages}"

    def test_reciprocal_of_zero_raises_or_warns(self):
        """
        A design containing A=0 with an I(1/A) term should either raise an
        exception or produce non-finite values.  statsmodels raises
        MissingDataError when the Patsy matrix contains inf/NaN; numpy may
        emit a RuntimeWarning during evaluation.  Either outcome is acceptable.
        """
        design = pd.DataFrame({
            "A": [0.0, 1.0, 2.0, 3.0, 1.5, 2.5],
            "B": [1.0, 2.0, 3.0, 4.0, 2.5, 3.5],
        })
        response = np.array([1.0, 2.0, 3.0, 4.0, 2.5, 3.5])

        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0.0, 3.0]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1.0, 4.0]),
        ]
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Y")

        raised_or_bad = False
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                results = analysis.fit(["1", "I(1/A)"], enforce_hierarchy_flag=False)
                # If it fits, fitted values must not all be finite
                if not np.all(np.isfinite(results.fitted_values)):
                    raised_or_bad = True
            except Exception:
                raised_or_bad = True

        assert raised_or_bad, (
            "Expected an exception or non-finite fitted values when I(1/A) "
            "is evaluated at A=0, but fit succeeded with finite values."
        )


# ---------------------------------------------------------------------------
# TestTransformTermValidation
# Document current behaviour of validate_model_terms and parse_model_term
# with transform terms.  These are marked xfail where the functions do not
# yet handle transforms — they should be updated to pass once those
# functions are extended.
# ---------------------------------------------------------------------------

class TestTransformTermValidation:
    """
    validate_model_terms and parse_model_term correctly handle transform terms.

    parse_model_term now recognises all four transform prefixes and their
    composed forms (cross, power, power-cross).  validate_model_terms
    uses the parsed factor names to validate existence and continuity.
    """

    def _factors_and_design(self):
        factors = _continuous_factors()
        design = _make_design()
        return factors, design

    def test_validate_accepts_log_term(self):
        """validate_model_terms accepts np.log(A) for a known continuous factor."""
        factors, design = self._factors_and_design()
        validate_model_terms(["1", "np.log(A)"], factors, design)

    def test_validate_accepts_sqrt_term(self):
        """validate_model_terms accepts np.sqrt(A) for a known continuous factor."""
        factors, design = self._factors_and_design()
        validate_model_terms(["1", "np.sqrt(A)"], factors, design)

    def test_validate_accepts_reciprocal_term(self):
        """validate_model_terms accepts I(1/A) for a known continuous factor."""
        factors, design = self._factors_and_design()
        validate_model_terms(["1", "I(1/A)"], factors, design)

    def test_validate_accepts_exp_term(self):
        """validate_model_terms accepts np.exp(A) for a known continuous factor."""
        factors, design = self._factors_and_design()
        validate_model_terms(["1", "np.exp(A)"], factors, design)

    def test_validate_accepts_log_cross_term(self):
        """validate_model_terms accepts np.log(A)*B."""
        factors, design = self._factors_and_design()
        validate_model_terms(["1", "np.log(A)", "B", "np.log(A)*B"], factors, design)

    def test_validate_rejects_transform_on_categorical_factor(self):
        """validate_model_terms rejects a transform applied to a categorical factor."""
        factors = [
            Factor("A", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, levels=["Lo", "Hi"]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1.0, 5.0]),
        ]
        design = pd.DataFrame({"A": ["Lo", "Hi", "Lo", "Hi"], "B": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match="continuous factor"):
            validate_model_terms(["1", "np.log(A)"], factors, design)

    def test_parse_model_term_fallthrough_behaviour(self):
        """
        parse_model_term now correctly recognises transform terms.
        np.log(A) should return the raw factor name and 'transform' operator.
        """
        factor_list, operator = parse_model_term("np.log(A)")
        assert operator == "transform"
        assert factor_list == ["A"]

    def test_enforce_hierarchy_passthrough_for_transform_terms(self):
        """
        enforce_hierarchy should not crash on transform terms even if it
        cannot meaningfully classify them — it should pass them through.
        """
        terms = ["1", "A", "B", "np.log(A)", "np.log(A)*B"]
        factor_names = ["A", "B"]
        # Should not raise
        result_terms, added = enforce_hierarchy(terms, factor_names)
        # Transform terms should survive the round-trip
        assert "np.log(A)" in result_terms
        assert "np.log(A)*B" in result_terms


# ---------------------------------------------------------------------------
# TestResultsStructure
# Verify that results objects from transform-term models have the same
# structure as results from standard models.
# ---------------------------------------------------------------------------

class TestResultsStructure:
    """Results from transform-term models have the expected structure."""

    def _fit_log_model(self) -> tuple:
        design = pd.DataFrame({
            "A": [1.0, 1.0, 5.0, 5.0, 3.0, 3.0, 3.0, 3.0],
            "B": [2.0, 8.0, 2.0, 8.0, 5.0, 5.0, 5.0, 5.0],
        })
        rng = np.random.default_rng(99)
        response = 4.0 + 2.0 * np.log(design["A"]) + rng.normal(0, 0.05, len(design))
        factors = _continuous_factors()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Y")
        results = analysis.fit(["1", "np.log(A)"], enforce_hierarchy_flag=False)
        return results, design

    def test_residuals_length_matches_design(self):
        """Residuals have one entry per run."""
        results, design = self._fit_log_model()
        assert len(results.residuals) == len(design)

    def test_fitted_values_length_matches_design(self):
        """Fitted values have one entry per run."""
        results, design = self._fit_log_model()
        assert len(results.fitted_values) == len(design)

    def test_r_squared_in_valid_range(self):
        """R² is between 0 and 1."""
        results, _ = self._fit_log_model()
        assert 0.0 <= results.r_squared <= 1.0

    def test_effect_estimates_not_empty(self):
        """effect_estimates DataFrame is non-empty."""
        results, _ = self._fit_log_model()
        assert len(results.effect_estimates) > 0

    def test_logworth_computed(self):
        """LogWorth values are present and finite for non-intercept terms."""
        results, _ = self._fit_log_model()
        assert "LogWorth" in results.logworth.columns
        finite_lw = results.logworth["LogWorth"].dropna()
        assert len(finite_lw) > 0
        assert all(np.isfinite(v) for v in finite_lw)

    def test_model_terms_preserved_in_results(self):
        """model_terms on the results object matches what was passed to fit()."""
        results, _ = self._fit_log_model()
        assert "np.log(A)" in results.model_terms

    def test_diagnostics_shapiro_wilk_present(self):
        """Shapiro-Wilk diagnostic is computed for transform-term models."""
        results, _ = self._fit_log_model()
        assert "shapiro_wilk" in results.diagnostics
        assert "p_value" in results.diagnostics["shapiro_wilk"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
