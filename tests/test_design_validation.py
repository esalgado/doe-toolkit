"""
Unit tests for the shared design-validity service.

The central guarantee tested here is *parity*: ``validate_design`` must never
disagree with the design generators.  A disagreement in either direction is a
bug -- the UI either offers a design that cannot be built, or hides one that
can.  ``TestServiceMatchesGenerators`` asserts that directly by attempting real
generation for a matrix of factor sets.
"""

import pytest

from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.design_validation import (
    DESIGN_TYPES,
    DesignValidationResult,
    ValidationIssue,
    errors_for,
    is_valid_design,
    validate_design,
    warnings_for,
)

EASY = ChangeabilityLevel.EASY
HARD = ChangeabilityLevel.HARD
VERY_HARD = ChangeabilityLevel.VERY_HARD


def cont(name, levels=(0.0, 1.0), changeability=EASY, validate=True):
    return Factor(name, FactorType.CONTINUOUS, changeability,
                  levels=list(levels), _validate_on_init=validate)


def cont_raw(name, levels):
    """A continuous factor with >2 declared levels, bypassing Factor validation.

    ``Factor._validate_continuous`` forbids this at construction, but
    ``csv_parser`` builds factors with ``_validate_on_init=False``, so a
    hand-edited or externally produced CSV really can carry one.  The
    generators must reject it rather than silently truncate to the extremes.
    """
    return cont(name, levels, validate=False)


def disc(name, levels=(10, 20), changeability=EASY):
    return Factor(name, FactorType.DISCRETE_NUMERIC, changeability,
                  levels=list(levels))


def cat(name, levels=("lo", "hi"), changeability=EASY):
    return Factor(name, FactorType.CATEGORICAL, changeability,
                  levels=list(levels))


def codes(result):
    return {issue.code for issue in result.issues}


class TestFullFactorial:
    """Full Factorial: any factor type; continuous must be 2-level."""

    def test_two_level_mixed_factors_valid(self):
        factors = [cont("A"), cont("B"), disc("C"), cat("D")]
        assert is_valid_design(factors, "Full Factorial")

    def test_multi_level_categorical_allowed(self):
        """``itertools.product`` handles any level count, so this is legal."""
        assert is_valid_design(
            [cont("A"), cont("B"), cat("C", ("x", "y", "z"))],
            "Full Factorial",
        )

    def test_multi_level_discrete_allowed(self):
        assert is_valid_design(
            [cont("A"), cont("B"), disc("C", (1, 2, 3))], "Full Factorial"
        )

    def test_multi_level_continuous_rejected(self):
        """The generator lays continuous factors on a 2-level coded grid."""
        result = validate_design(
            [cont("A"), cont("B"), cont_raw("C", (0.0, 0.5, 1.0))],
            "Full Factorial",
        )
        assert not result.is_valid
        assert "full_factorial_continuous_two_levels" in codes(result)

    def test_large_run_count_warns_but_allows(self):
        factors = [cont(chr(65 + i)) for i in range(8)]
        result = validate_design(factors, "Full Factorial")
        assert result.is_valid
        assert "full_factorial_run_count" in codes(result)

    def test_empty_factors_rejected(self):
        assert not is_valid_design([], "Full Factorial")


class TestFractionalFactorial:
    """Fractional Factorial: 3+ factors, every factor exactly 2 levels."""

    def test_three_two_level_factors_valid(self):
        assert is_valid_design(
            [cont("A"), cont("B"), cont("C")], "Fractional Factorial"
        )

    def test_two_level_categorical_is_valid(self):
        """Regression for the Phase A fix: a 2-level categorical is a normal
        member of a 2^(k-p) design."""
        factors = [cont("A"), cont("B"), cont("C"), cat("D")]
        assert is_valid_design(factors, "Fractional Factorial")

    def test_two_level_discrete_is_valid(self):
        assert is_valid_design(
            [cont("A"), cont("B"), cont("C"), disc("D")], "Fractional Factorial"
        )

    def test_fewer_than_three_factors_rejected(self):
        result = validate_design([cont("A"), cont("B")], "Fractional Factorial")
        assert not result.is_valid
        assert "fractional_min_factors" in codes(result)

    def test_multi_level_categorical_rejected(self):
        result = validate_design(
            [cont("A"), cont("B"), cat("C", ("x", "y", "z"))],
            "Fractional Factorial",
        )
        assert not result.is_valid
        assert "fractional_requires_two_levels" in codes(result)

    def test_multi_level_discrete_rejected(self):
        result = validate_design(
            [cont("A"), cont("B"), disc("C", (1, 2, 3))], "Fractional Factorial"
        )
        assert not result.is_valid
        assert "fractional_requires_two_levels" in codes(result)

    def test_multi_level_continuous_rejected(self):
        """Every factor in a 2^(k-p) design must be 2-level, continuous
        included -- otherwise a declared level would be silently dropped."""
        result = validate_design(
            [cont("A"), cont("B"), cont_raw("C", (0.0, 0.5, 1.0))],
            "Fractional Factorial",
        )
        assert not result.is_valid
        assert "fractional_requires_two_levels" in codes(result)

    def test_error_names_the_offending_factor(self):
        result = validate_design(
            [cont("A"), cont("B"), cat("Catalyst", ("x", "y", "z"))],
            "Fractional Factorial",
        )
        offending = [name for issue in result.errors for name in issue.factor_names]
        assert "Catalyst" in offending
        assert any("Catalyst" in m for m in result.messages)


class TestResponseSurface:
    """CCD and Box-Behnken: all factors continuous."""

    @pytest.mark.parametrize(
        "design_type", ["Response Surface (CCD)", "Response Surface (Box-Behnken)"]
    )
    def test_continuous_factors_valid(self, design_type):
        assert is_valid_design(
            [cont("A"), cont("B"), cont("C")], design_type
        )

    @pytest.mark.parametrize(
        "design_type", ["Response Surface (CCD)", "Response Surface (Box-Behnken)"]
    )
    def test_categorical_rejected(self, design_type):
        result = validate_design([cont("A"), cont("B"), cat("C")], design_type)
        assert not result.is_valid
        assert "rsm_requires_continuous" in codes(result)

    def test_ccd_allows_two_factors(self):
        assert is_valid_design(
            [cont("A"), cont("B")], "Response Surface (CCD)"
        )

    def test_box_behnken_needs_three(self):
        result = validate_design(
            [cont("A"), cont("B")], "Response Surface (Box-Behnken)"
        )
        assert not result.is_valid
        assert "box_behnken_min_factors" in codes(result)

    def test_single_factor_rejected(self):
        result = validate_design([cont("A")], "Response Surface (CCD)")
        assert "rsm_min_factors" in codes(result)

    def test_too_many_factors_rejected(self):
        result = validate_design(
            [cont(chr(65 + i)) for i in range(11)], "Response Surface (CCD)"
        )
        assert "rsm_impractical" in codes(result)


class TestDOptimal:
    """D-Optimal: all factors continuous."""

    def test_continuous_valid(self):
        assert is_valid_design([cont("A"), cont("B")], "D-Optimal")

    def test_categorical_rejected(self):
        result = validate_design([cont("A"), cat("B")], "D-Optimal")
        assert not result.is_valid
        assert "d_optimal_requires_continuous" in codes(result)

    def test_discrete_rejected(self):
        result = validate_design([cont("A"), disc("B")], "D-Optimal")
        assert "d_optimal_requires_continuous" in codes(result)


class TestLatinHypercube:
    """Latin Hypercube: every factor type is buildable; categorical warns."""

    def test_mixed_factors_valid(self):
        """Regression: the Step 3 gate used to require all-continuous even
        though the generator stratifies categorical factors."""
        assert is_valid_design(
            [cont("A"), disc("B"), cat("C")], "Latin Hypercube"
        )

    def test_categorical_produces_warning_not_error(self):
        result = validate_design([cont("A"), cat("B")], "Latin Hypercube")
        assert result.is_valid
        assert "latin_hypercube_categorical_stratified" in codes(result)

    def test_all_continuous_has_no_warning(self):
        result = validate_design([cont("A"), cont("B")], "Latin Hypercube")
        assert result.is_valid
        assert not result.warnings


class TestSplitPlot:
    """Split-Plot: needs a whole-plot stratum."""

    def test_hard_factor_valid(self):
        assert is_valid_design(
            [cont("A", changeability=HARD), cont("B")], "Split-Plot"
        )

    def test_very_hard_factor_valid(self):
        assert is_valid_design(
            [cont("A", changeability=VERY_HARD), cont("B")], "Split-Plot"
        )

    def test_all_easy_rejected(self):
        result = validate_design([cont("A"), cont("B")], "Split-Plot")
        assert not result.is_valid
        assert "split_plot_requires_hard_factor" in codes(result)


class TestResultObject:
    """The result container's own contract."""

    def test_errors_sorted_before_warnings(self):
        factors = [cont("A"), cont("B"), cont("C"), cont("D"), cont("E"),
                   cont("F")]
        result = validate_design(factors, "Full Factorial")
        severities = [i.severity for i in result.issues]
        assert severities == sorted(severities, key=lambda s: s != "error")

    def test_valid_result_has_no_errors(self):
        result = validate_design([cont("A"), cont("B")], "D-Optimal")
        assert result.is_valid
        assert result.errors == ()
        assert "valid" in result.reason()

    def test_reason_joins_error_messages(self):
        result = validate_design([cont("A"), cont("B")], "Fractional Factorial")
        assert not result.is_valid
        assert result.reason() == " ".join(i.message for i in result.errors)

    def test_helpers_agree_with_result(self):
        factors = [cont("A"), cont("B"), cont("C"), cat("D", ("x", "y", "z"))]
        result = validate_design(factors, "Fractional Factorial")
        assert errors_for(factors, "Fractional Factorial") == [
            i.message for i in result.errors
        ]
        assert warnings_for(factors, "Fractional Factorial") == [
            i.message for i in result.warnings
        ]

    def test_model_terms_add_warning_not_error(self):
        factors = [cont("A"), cont("B"), cont("C")]
        result = validate_design(
            factors, "Fractional Factorial", model_terms=["1", "A", "I(B**2)"]
        )
        assert result.is_valid
        assert "fractional_no_quadratic" in codes(result)

    def test_unknown_design_type_raises(self):
        with pytest.raises(KeyError):
            validate_design([cont("A")], "Nonexistent Design")

    def test_every_advertised_design_type_is_routable(self):
        for design_type in DESIGN_TYPES:
            assert isinstance(
                validate_design([cont("A"), cont("B")], design_type),
                DesignValidationResult,
            )

    def test_validation_issue_is_immutable(self):
        issue = ValidationIssue("c", "error", "m")
        with pytest.raises(Exception):
            issue.code = "other"


class TestServiceMatchesGenerators:
    """Parity: the service must never disagree with real generation.

    This is the regression that keeps the original bug from returning.  Step 3
    gated Fractional Factorial on factor count while ``FractionalFactorial``
    also rejected categorical factors, so the UI offered a design the backend
    refused.  Any future drift in either direction fails here.
    """

    CASES = {
        "one_continuous": [cont("A")],
        "two_continuous": [cont("A"), cont("B")],
        "three_continuous": [cont("A"), cont("B"), cont("C")],
        "categorical_2lvl": [cont("A"), cont("B"), cont("C"), cat("D")],
        "categorical_3lvl": [cont("A"), cont("B"), cont("C"), cat("D", ("x", "y", "z"))],
        "discrete_2lvl": [cont("A"), cont("B"), cont("C"), disc("D")],
        "discrete_3lvl": [cont("A"), cont("B"), cont("C"), disc("D", (1, 2, 3))],
        "continuous_3lvl": [cont("A"), cont("B"), cont_raw("C", (0.0, 0.5, 1.0))],
        "all_categorical_2lvl": [cat("A"), cat("B"), cat("C")],
        "mixed_six_2lvl": [cont("A"), cont("B"), cont("C"), cont("D"),
                            cat("K"), disc("E")],
        "hard_factor": [cont("A", changeability=HARD), cont("B")],
        "very_hard_factor": [cont("A", changeability=VERY_HARD), cont("B"),
                             cont("C")],
    }

    @staticmethod
    def _build(factors, design_type):
        """Actually attempt to build the design.  Returns True on success."""
        from src.core.full_factorial import full_factorial
        from src.core.fractional_factorial import FractionalFactorial
        from src.core.latin_hypercube import generate_latin_hypercube
        from src.core.optimal.design_generation import generate_optimal_design
        from src.core.response_surface import (
            BoxBehnkenDesign,
            CentralCompositeDesign,
        )
        from src.core.split_plot import generate_split_plot_design

        try:
            if design_type == "Full Factorial":
                full_factorial(factors)
            elif design_type == "Fractional Factorial":
                FractionalFactorial(factors, fraction="1/2").generate(
                    randomize=False
                )
            elif design_type == "Response Surface (CCD)":
                CentralCompositeDesign(factors)
            elif design_type == "Response Surface (Box-Behnken)":
                BoxBehnkenDesign(factors)
            elif design_type == "D-Optimal":
                generate_optimal_design(
                    factors, model_type="linear", n_runs=len(factors) + 2
                )
            elif design_type == "Latin Hypercube":
                generate_latin_hypercube(factors, n_runs=8)
            elif design_type == "Split-Plot":
                generate_split_plot_design(factors)
        except Exception:
            return False
        return True

    @pytest.mark.parametrize("case", sorted(CASES))
    @pytest.mark.parametrize("design_type", DESIGN_TYPES)
    def test_service_agrees_with_generator(self, case, design_type):
        factors = self.CASES[case]
        declared = validate_design(factors, design_type).is_valid
        built = self._build(factors, design_type)
        assert declared == built, (
            f"service says valid={declared} but generator succeeded={built} "
            f"for {case} / {design_type}"
        )


class TestOneIssuePerRule:
    """A rule broken by several factors yields one issue, not one per factor.

    ``DesignValidationResult.reason`` joins every issue message into a single
    line, so per-factor issues turned the Step 2 blocked-design list into a wall
    of near-duplicate sentences -- the CCD row repeated the same requirement
    once per categorical factor.
    """

    @pytest.mark.parametrize(
        "design_type,code,expected,phrase",
        [
            # a 2-level categorical is legal in a fractional design, so only
            # the 3-level factors offend the level-count rule
            ("Fractional Factorial", "fractional_requires_two_levels", {"a", "b"},
             "must have exactly 2 levels"),
            # the response surface and D-optimal rules reject every
            # non-continuous factor, so all four offend
            ("Response Surface (CCD)", "rsm_requires_continuous",
             {"a", "b", "c", "d"}, "must be continuous"),
            ("Response Surface (Box-Behnken)", "rsm_requires_continuous",
             {"a", "b", "c", "d"}, "must be continuous"),
            ("D-Optimal", "d_optimal_requires_continuous", {"a", "b", "c", "d"},
             "must be continuous"),
        ],
    )
    def test_several_offending_factors_give_exactly_one_error(
        self, design_type, code, expected, phrase
    ):
        factors = [
            cont("Temp"), cont("Pressure"), cont("Speed"),
            cat("a", ("w", "x", "y")),
            cat("b", ("w", "x", "y")),
            cat("c", ("lo", "hi")),
            cat("d", ("lo", "hi")),
        ]
        result = validate_design(factors, design_type)
        assert not result.is_valid
        matching = [i for i in result.errors if i.code == code]
        assert len(matching) == 1, (
            f"expected a single {code} issue, got {len(matching)}: "
            f"{[i.message for i in matching]}"
        )
        assert set(matching[0].factor_names) == expected
        # the rule text appears exactly once, not once per offending factor
        assert matching[0].message.count(phrase) == 1
        # every offender is named exactly once
        for name in expected:
            assert matching[0].message.count(f"'{name}'") == 1

    def test_level_counts_are_shown_for_level_count_rules(self):
        result = validate_design(
            [cont("A"), cont("B"), cat("a", ("w", "x", "y"))],
            "Fractional Factorial",
        )
        issue = next(i for i in result.errors if i.code == "fractional_requires_two_levels")
        assert "'a' (3 levels)" in issue.message
        assert "must have exactly 2 levels" in issue.message

    def test_singular_wording_for_one_offender(self):
        result = validate_design(
            [cont("A"), cont("B"), cat("a", ("lo", "hi"))],
            "Response Surface (CCD)",
        )
        issue = next(i for i in result.errors if i.code == "rsm_requires_continuous")
        assert issue.message.startswith("Factor 'a' ")
        assert issue.factor_names == ("a",)
        assert issue.factor_name == "a"

    def test_whole_design_issues_carry_no_factor(self):
        result = validate_design(
            [cont("A"), cont("B")], "Response Surface (Box-Behnken)"
        )
        issue = next(
            i for i in result.errors if i.code == "box_behnken_min_factors"
        )
        assert issue.factor_names == ()
        assert issue.factor_name is None


class TestMessagesNeverNameAnotherDesign:
    """Issue messages must not recommend a design by name.

    Whether an alternative is usable depends on the entire factor set, and the
    rule that produced the message is often the very reason the alternative is
    blocked -- "Fractional Factorial requires 2 levels ... use full factorial
    or optimal designs" is self-contradictory when full factorial is itself
    blocked by a multi-level continuous factor.  The service states rules; the
    Step 2 compatibility panel is the one place that names alternatives,
    because it has already evaluated all of them.
    """

    FAMILIES = [
        ("Full Factorial",),
        ("Fractional Factorial",),
        ("Response Surface (CCD)", "Response Surface (Box-Behnken)"),
        ("D-Optimal",),
        ("Latin Hypercube",),
        ("Split-Plot",),
    ]

    POOL = [
        cont("T"),
        cont_raw("P", (1.0, 2.0, 3.0)),
        cont("S", changeability=VERY_HARD),
        cat("k3", ("w", "x", "y")),
        cat("k2", ("lo", "hi")),
        Factor("dn", FactorType.DISCRETE_NUMERIC, EASY, levels=[1.0, 2.0, 3.0]),
    ]

    def _all_factor_sets(self):
        import itertools

        for size in range(1, len(self.POOL) + 1):
            for combo in itertools.combinations(self.POOL, size):
                yield list(combo)

    def test_no_message_names_an_unavailable_design(self):
        offenders = []
        for factors in self._all_factor_sets():
            available = {
                family
                for family in self.FAMILIES
                if any(validate_design(factors, d).is_valid for d in family)
            }
            for design_type in DESIGN_TYPES:
                for issue in validate_design(factors, design_type).issues:
                    for family in self.FAMILIES:
                        if family in available:
                            continue
                        for other in family:
                            if other != design_type and other in issue.message:
                                offenders.append(
                                    (design_type, issue.code, other,
                                     [f.name for f in factors])
                                )
        assert not offenders, (
            "issue messages name a design that is itself unavailable: "
            + "\n".join(
                f"  [{code}] -> {other} (factors={names})"
                for _, code, other, names in offenders[:10]
            )
        )
