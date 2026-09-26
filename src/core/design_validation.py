"""
Shared design-validity authority for every design type.

This module is the single place that answers "can these factors produce a
valid design of this type?".  It exists because the rules were previously
duplicated across the Streamlit pages and the design generators, and the copies
had drifted apart: Step 3 gated Fractional Factorial on factor *count* only,
while ``FractionalFactorial`` also rejected categorical factors, so the UI
offered a design the backend would refuse.

The rules here are the authority for the *user-facing* gate.  The generators
keep their own constructor checks as a last line of defence -- if a rule here
is ever loosened without loosening the generator, generation still fails
safely rather than producing a malformed design.

The module is Streamlit-free and imports nothing from ``src.ui``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.core.factors import Factor

__all__ = [
    "ValidationIssue",
    "DesignValidationResult",
    "DESIGN_TYPES",
    "validate_design",
    "is_valid_design",
    "errors_for",
    "warnings_for",
]


# Design-type keys used by the Streamlit session state.  These are the exact
# strings Step 3 writes to ``st.session_state['design_type']`` so the service
# and the pages cannot disagree about spelling.
DESIGN_TYPES: Tuple[str, ...] = (
    "Full Factorial",
    "Fractional Factorial",
    "Response Surface (CCD)",
    "Response Surface (Box-Behnken)",
    "D-Optimal",
    "Latin Hypercube",
    "Split-Plot",
)

# Above this many runs a full factorial stops being practical to lay out, run,
# and analyse, so the service warns.  A warning, not an error.
MAX_RECOMMENDED_RUNS = 64


# Messages in this module deliberately never name another design type.  Whether
# an alternative is usable depends on the whole factor set -- a message like
# "use D-Optimal instead" is a guess that this module cannot make, and it is
# routinely wrong: the rule that produced the message is often the very reason
# the alternative is blocked.  Each issue therefore states only the rule, and
# the compatibility panel (Step 2) is the single place that names alternatives,
# because it already evaluates all of them.  ``test_design_validation.py``
# enforces this.


@dataclass(frozen=True)
class ValidationIssue:
    """A single reason a design is (or is not) valid.

    Attributes
    ----------
    code : str
        Stable machine-readable identifier, e.g.
        ``'fractional_requires_two_levels'``.  Safe to branch on.
    severity : str
        ``'error'`` blocks the design; ``'warning'`` is advisory and lets the
        user proceed.
    message : str
        Complete, user-facing sentence.  The UI shows this verbatim so the
        explanation on Step 3 is the same text a failure would have produced.
    factor_names : tuple of str, optional
        Every factor the issue concerns.  A rule that is violated by four
        factors yields one issue naming all four, rather than four issues
        repeating the same sentence -- the rules are reported per rule, not
        per factor.
    """

    code: str
    severity: str
    message: str
    factor_names: Tuple[str, ...] = ()

    @property
    def factor_name(self) -> Optional[str]:
        """First offending factor, or ``None`` for whole-design issues."""
        return self.factor_names[0] if self.factor_names else None

    @property
    def is_error(self) -> bool:
        """``True`` when this issue blocks the design."""
        return self.severity == "error"


@dataclass(frozen=True)
class DesignValidationResult:
    """Outcome of validating a factor set against one design type.

    Attributes
    ----------
    design_type : str
        The design type that was validated.
    issues : tuple of ValidationIssue
        All issues found, errors before warnings.
    """

    design_type: str
    issues: Tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> Tuple[ValidationIssue, ...]:
        """Issues that block the design."""
        return tuple(i for i in self.issues if i.is_error)

    @property
    def warnings(self) -> Tuple[ValidationIssue, ...]:
        """Advisory issues that do not block the design."""
        return tuple(i for i in self.issues if not i.is_error)

    @property
    def is_valid(self) -> bool:
        """``True`` when no blocking error was found."""
        return not self.errors

    @property
    def messages(self) -> List[str]:
        """User-facing messages for every issue, in order."""
        return [i.message for i in self.issues]

    def reason(self) -> str:
        """Single-line summary, suitable for an exception message."""
        if self.is_valid:
            return f"{self.design_type} is valid for these factors."
        return " ".join(i.message for i in self.errors)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _needs_two_levels(factor: Factor) -> bool:
    """``True`` when a factor is not a 2-level factor."""
    return len(factor.levels) != 2

def _cartesian_run_count(factors: Sequence[Factor]) -> int:
    """Number of runs a full factorial needs: the product of level counts."""
    total = 1
    for factor in factors:
        total *= max(len(factor.levels), 1)
    return total


def _rule_issue(
    code: str,
    offenders: Sequence[Factor],
    rule: str,
    remedy: str = "",
    with_levels: bool = False,
) -> ValidationIssue:
    """One :class:`ValidationIssue` naming every factor that broke one rule.

    A rule violated by four factors produces a single issue -- "Factors 'a',
    'b', 'c', 'd' must be continuous" -- instead of four issues repeating that
    sentence.  ``DesignValidationResult.reason`` joins the messages of every
    issue into one line, so per-factor issues turned the blocked-design list
    into a wall of near-duplicate text.

    Parameters
    ----------
    code :
        Stable issue code, shared by every factor set that trips the rule.
    offenders :
        Factors that violated the rule, in declaration order.
    rule :
        The requirement, phrased to follow "Factors 'a', 'b' must ...".
    remedy :
        Optional second sentence telling the user what to do instead.
    with_levels :
        Include each offender's level count, e.g. "'a' (3 levels)".  Only
        useful for level-count rules, where the count is the diagnosis.
    """
    listed = ", ".join(
        f"'{f.name}' ({len(f.levels)} levels)" if with_levels else f"'{f.name}'"
        for f in offenders
    )
    plural = "Factors" if len(offenders) > 1 else "Factor"
    message = f"{plural} {listed} {rule}"
    if remedy:
        message = f"{message} {remedy}"
    return ValidationIssue(
        code=code,
        severity="error",
        message=message,
        factor_names=tuple(f.name for f in offenders),
    )


# ---------------------------------------------------------------------------
# Per-design-type rules
# ---------------------------------------------------------------------------


def _validate_full_factorial(factors: Sequence[Factor]) -> List[ValidationIssue]:
    """Full Factorial: any factor type, but continuous factors must be 2-level.

    ``itertools.product`` over the declared levels handles discrete and
    categorical factors at any level count, so multi-level non-continuous
    factors are allowed here and only cost runs.  Continuous factors are
    different: the generator lays them out on the coded ``[-1, +1]`` grid and
    decodes to ``[min, max]``, which is a 2-level construction by definition.
    """
    issues: List[ValidationIssue] = []
    if not factors:
        issues.append(
            ValidationIssue(
                code="no_factors",
                severity="error",
                message="At least one factor must be provided.",
            )
        )
        return issues

    offenders = [
        f for f in factors if f.is_continuous() and _needs_two_levels(f)
    ]
    if offenders:
        issues.append(
            _rule_issue(
                "full_factorial_continuous_two_levels",
                offenders,
                "is continuous, but full factorial lays continuous factors out "
                "on a 2-level coded grid, so exactly 2 levels are required.",
                "Choose a design that supports multi-level continuous "
                "factors.",
                with_levels=True,
            )
        )

    n_runs = _cartesian_run_count(factors)
    if n_runs > MAX_RECOMMENDED_RUNS:
        issues.append(
            ValidationIssue(
                code="full_factorial_run_count",
                severity="warning",
                message=(
                    f"{len(factors)} factors at these level counts need "
                    f"{n_runs} runs, which exceeds the practical full-factorial "
                    f"limit of {MAX_RECOMMENDED_RUNS} runs."
                ),
            )
        )
    return issues


def _validate_fractional(factors: Sequence[Factor]) -> List[ValidationIssue]:
    """Fractional Factorial: at least 3 factors, every factor exactly 2 levels.

    A 2-level categorical factor is a legitimate member of a 2^(k-p) design --
    the coded [-1, +1] grid maps bijectively onto its two labels, and the
    aliasing engine works on algebraic symbols independent of factor type.
    """
    issues: List[ValidationIssue] = []
    k = len(factors)

    if k < 3:
        issues.append(
            ValidationIssue(
                code="fractional_min_factors",
                severity="error",
                message=(
                    "Fractional Factorial requires at least 3 factors, and a "
                    "half fraction only saves runs from 4 factors upward."
                ),
            )
        )

    offenders = [f for f in factors if _needs_two_levels(f)]
    if offenders:
        issues.append(
            _rule_issue(
                "fractional_requires_two_levels",
                offenders,
                "must have exactly 2 levels for a fractional factorial.",
                "Choose a design that supports multi-level factors.",
                with_levels=True,
            )
        )

    return issues


def _response_surface_issues(
    factors: Sequence[Factor], variant: str
) -> List[ValidationIssue]:
    """Shared CCD / Box-Behnken rules.

    Mirrors ``ResponseSurfaceDesign._validate_factors``: all factors continuous,
    between 2 and 10 of them.  Box-Behnken additionally needs at least 3.
    """
    issues: List[ValidationIssue] = []
    k = len(factors)

    if k < 2:
        issues.append(
            ValidationIssue(
                code="rsm_min_factors",
                severity="error",
                message=(
                    "Response surface design requires at least 2 factors "
                    "to estimate curvature."
                ),
            )
        )
    elif variant == "Box-Behnken" and k < 3:
        issues.append(
            ValidationIssue(
                code="box_behnken_min_factors",
                severity="error",
                message=(
                    "Box-Behnken design requires at least 3 factors, "
                    "since it never places runs at the corners of the factor "
                    "space."
                ),
            )
        )

    if k > 10:
        issues.append(
            ValidationIssue(
                code="rsm_impractical",
                severity="error",
                message=(
                    f"Response surface design with {k} factors is impractical. "
                    f"Consider screening or sequential approaches."
                ),
            )
        )

    offenders = [f for f in factors if not f.is_continuous()]
    if offenders:
        issues.append(
            _rule_issue(
                "rsm_requires_continuous",
                offenders,
                "must be continuous for response surface design.",
                "Choose a design that supports categorical/discrete factors.",
            )
        )

    return issues


def _validate_d_optimal(factors: Sequence[Factor]) -> List[ValidationIssue]:
    """D-Optimal: all factors continuous.

    Mirrors the check in ``optimal/design_generation.py``, which rejects
    categorical and discrete factors outright.
    """
    issues: List[ValidationIssue] = []
    if not factors:
        issues.append(
            ValidationIssue(
                code="no_factors",
                severity="error",
                message="At least one factor must be provided.",
            )
        )
    offenders = [f for f in factors if not f.is_continuous()]
    if offenders:
        issues.append(
            _rule_issue(
                "d_optimal_requires_continuous",
                offenders,
                "must be continuous. Categorical/discrete not yet supported.",
            )
        )
    return issues


def _validate_latin_hypercube(factors: Sequence[Factor]) -> List[ValidationIssue]:
    """Latin Hypercube: every factor type is supported.

    ``generate_latin_hypercube`` stratifies continuous and discrete-numeric
    factors and samples categorical factors by stratified random sampling, so
    a mixed factor set is genuinely buildable.  The screening guarantee is
    weaker for the non-continuous columns, which is worth a warning.
    """
    issues: List[ValidationIssue] = []
    if not factors:
        issues.append(
            ValidationIssue(
                code="no_factors",
                severity="error",
                message="At least one factor must be provided.",
            )
        )
        return issues

    categorical = [f.name for f in factors if f.is_categorical()]
    if categorical:
        issues.append(
            ValidationIssue(
                code="latin_hypercube_categorical_stratified",
                severity="warning",
                message=(
                    f"Categorical factors ({', '.join(categorical)}) are sampled "
                    f"by stratified random assignment rather than by Latin "
                    f"hypercube, so the space-filling guarantee does not apply "
                    f"to them."
                ),
            )
        )
    return issues


def _validate_split_plot(factors: Sequence[Factor]) -> List[ValidationIssue]:
    """Split-Plot: at least one HARD or VERY_HARD factor to change.

    With every factor EASY the structure has no whole-plot stratum, and
    ``generate_split_plot_design`` refuses it in favour of a plain factorial.
    """
    from src.core.factors import ChangeabilityLevel

    issues: List[ValidationIssue] = []
    if not factors:
        issues.append(
            ValidationIssue(
                code="no_factors",
                severity="error",
                message="At least one factor must be provided.",
            )
        )

    has_hard = any(
        f.changeability
        in (ChangeabilityLevel.HARD, ChangeabilityLevel.VERY_HARD)
        for f in factors
    )
    if factors and not has_hard:
        issues.append(
            ValidationIssue(
                code="split_plot_requires_hard_factor",
                severity="error",
                message=(
                    "All factors are EASY, so there is no whole-plot stratum to "
                    "randomise. Split-Plot requires at least one HARD or "
                    "VERY_HARD factor."
                ),
            )
        )
    return issues


_RULES = {
    "Full Factorial": _validate_full_factorial,
    "Fractional Factorial": _validate_fractional,
    "Response Surface (CCD)": lambda f: _response_surface_issues(f, "CCD"),
    "Response Surface (Box-Behnken)": lambda f: _response_surface_issues(
        f, "Box-Behnken"
    ),
    "D-Optimal": _validate_d_optimal,
    "Latin Hypercube": _validate_latin_hypercube,
    "Split-Plot": _validate_split_plot,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_design(
    factors: Sequence[Factor],
    design_type: str,
    model_terms: Optional[Sequence[str]] = None,
) -> DesignValidationResult:
    """Validate *factors* against *design_type*.

    Parameters
    ----------
    factors : sequence of Factor
        The project's factor definitions.
    design_type : str
        One of :data:`DESIGN_TYPES`.
    model_terms : sequence of str, optional
        The Step 2 model terms.  When supplied, model-driven mismatches are
        reported as warnings -- for example a quadratic term against a
        two-level screening design, which is unidentifiable rather than fatal.

    Returns
    -------
    DesignValidationResult
        Errors first, then warnings.

    Raises
    ------
    KeyError
        If *design_type* is not a known design type.  Callers rendering a
        selectbox should not hit this, and a hard failure is better than
        silently skipping validation for an unrecognised label.

    Examples
    --------
    >>> from src.core.factors import Factor, FactorType
    >>> f = [Factor("A", FactorType.CATEGORICAL, levels=["x", "y", "z"]),
    ...      Factor("B", FactorType.CATEGORICAL, levels=["x", "y"]),
    ...      Factor("C", FactorType.CATEGORICAL, levels=["x", "y"])]
    >>> validate_design(f, "Fractional Factorial").is_valid
    False
    """
    rule = _RULES[design_type]
    issues: List[ValidationIssue] = list(rule(factors))

    if model_terms:
        issues.extend(_model_issues(design_type, factors, model_terms))

    errors = [i for i in issues if i.is_error]
    warns = [i for i in issues if not i.is_error]
    return DesignValidationResult(
        design_type=design_type, issues=tuple(errors + warns)
    )


def _model_issues(
    design_type: str,
    factors: Sequence[Factor],
    model_terms: Sequence[str],
) -> List[ValidationIssue]:
    """Model-vs-design mismatches, reported as warnings.

    These mirror the "Compatibility Notes" Step 2 already shows.  They are
    warnings rather than errors because the design can still be built and the
    user may intend to drop the offending term.
    """
    issues: List[ValidationIssue] = []
    has_quadratic = any(
        t.startswith("I(") and "**2" in t for t in model_terms
    )
    has_interactions = any(
        "*" in t and not t.startswith("I(") for t in model_terms
    )

    if has_quadratic and design_type == "Fractional Factorial":
        issues.append(
            ValidationIssue(
                code="fractional_no_quadratic",
                severity="warning",
                message=(
                    "Fractional Factorial is a 2-level screening design, so "
                    "quadratic terms are not estimable. They will be "
                    "suppressed from the model."
                ),
            )
        )

    if has_quadratic and design_type in (
        "Full Factorial",
        "Fractional Factorial",
    ):
        two_level_only = all(
            f.is_continuous() or len(f.levels) == 2 for f in factors
        )
        if two_level_only:
            issues.append(
                ValidationIssue(
                    code="two_level_no_quadratic",
                    severity="warning",
                    message=(
                        "Every factor has exactly 2 levels, so a squared column "
                        "is constant and the quadratic term is unidentifiable. "
                        "Add levels or center points to identify the "
                        "curvature."
                    ),
                )
            )

    if (has_interactions or has_quadratic) and design_type == "Latin Hypercube":
        issues.append(
            ValidationIssue(
                code="latin_hypercube_screening_only",
                severity="warning",
                message=(
                    "Latin Hypercube is a space-filling screening design and "
                    "does not estimate interactions or curvature."
                ),
            )
        )

    return issues


def is_valid_design(
    factors: Sequence[Factor], design_type: str
) -> bool:
    """Convenience predicate.  See :func:`validate_design`."""
    return validate_design(factors, design_type).is_valid


def errors_for(
    factors: Sequence[Factor], design_type: str
) -> List[str]:
    """Blocking messages only.  Empty list means the design is allowed."""
    return [i.message for i in validate_design(factors, design_type).errors]


def warnings_for(
    factors: Sequence[Factor], design_type: str
) -> List[str]:
    """Advisory messages only."""
    return [i.message for i in validate_design(factors, design_type).warnings]
