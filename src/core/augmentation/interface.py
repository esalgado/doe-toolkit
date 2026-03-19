"""
Unified Augmentation Interface.

This module provides the single entry point for augmentation,
supporting both Mode A (diagnostics-driven) and Mode B (goal-driven).
"""

from typing import List, Dict, Optional, Literal
from dataclasses import dataclass

from src.core.diagnostics import DesignDiagnosticSummary
from src.core.augmentation.plan import AugmentationPlan
from src.core.augmentation.recommendations import recommend_from_diagnostics
from src.core.augmentation.goal_driven import (
    recommend_from_goal,
    recommend_from_type,
    GoalDrivenContext
)
from src.core.augmentation.goals import (
    AugmentationGoal,
    get_available_goals,
    GOAL_CATALOG
)


@dataclass
class AugmentationRequest:
    """
    Request for augmentation recommendations.

    Attributes
    ----------
    mode : {'fix_issues', 'enhance_design', 'select_type'}
        Augmentation mode.
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics.
    selected_goal : AugmentationGoal, optional
        User's goal (required for 'enhance_design' mode).
    selected_type : str, optional
        Augmentation type key (required for 'select_type' mode).
    budget_constraint : int, optional
        Maximum runs to add.
    user_adjustments : dict, optional
        User-requested parameter adjustments.
    """

    mode: Literal['fix_issues', 'enhance_design', 'select_type']
    diagnostics: DesignDiagnosticSummary
    selected_goal: Optional[AugmentationGoal] = None
    selected_type: Optional[str] = None
    budget_constraint: Optional[int] = None
    user_adjustments: Optional[Dict] = None


def recommend_augmentation(request: AugmentationRequest) -> List[AugmentationPlan]:
    """
    Generate augmentation recommendations.

    This is the unified entry point that routes to either:
    - Mode A: Diagnostics-driven (fix detected issues)
    - Mode B legacy: Goal-driven (user intent via AugmentationGoal enum)
    - Mode B v2: Type-driven (user picks augmentation type directly)

    Parameters
    ----------
    request : AugmentationRequest
        Augmentation request specifying mode and parameters.

    Returns
    -------
    List[AugmentationPlan]
        Ranked augmentation plans.

    Raises
    ------
    ValueError
        If mode is invalid or required parameters are missing.

    Examples
    --------
    Mode A (Fix Issues):
    >>> request = AugmentationRequest(
    ...     mode='fix_issues',
    ...     diagnostics=diagnostics,
    ...     budget_constraint=16
    ... )
    >>> plans = recommend_augmentation(request)

    Mode B v2 (Select Type):
    >>> request = AugmentationRequest(
    ...     mode='select_type',
    ...     diagnostics=diagnostics,
    ...     selected_type='foldover',
    ...     budget_constraint=20
    ... )
    >>> plans = recommend_augmentation(request)
    """

    if request.mode == 'fix_issues':
        # Mode A: Diagnostics-driven
        return recommend_from_diagnostics(
            diagnostics=request.diagnostics,
            budget_constraint=request.budget_constraint
        )

    elif request.mode == 'enhance_design':
        # Mode B (legacy goal-driven path)
        if request.selected_goal is None:
            raise ValueError(
                "selected_goal is required for 'enhance_design' mode"
            )

        context = GoalDrivenContext(
            selected_goal=request.selected_goal,
            design_diagnostics=request.diagnostics,
            budget_constraint=request.budget_constraint,
            user_adjustments=request.user_adjustments or {}
        )

        return recommend_from_goal(context)

    elif request.mode == 'select_type':
        # Mode B v2: user picks augmentation type directly
        if request.selected_type is None:
            raise ValueError(
                "selected_type is required for 'select_type' mode"
            )
        return recommend_from_type(
            augmentation_type=request.selected_type,
            diagnostics=request.diagnostics,
            budget_constraint=request.budget_constraint,
            user_adjustments=request.user_adjustments or {},
        )

    else:
        raise ValueError(f"Invalid mode: {request.mode}")


def get_mode_availability(
    diagnostics: DesignDiagnosticSummary
) -> Dict[str, bool]:
    """
    Determine which augmentation modes are available.

    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics.

    Returns
    -------
    Dict[str, bool]
        Mode availability:
        - 'fix_issues': bool (True if issues detected)
        - 'enhance_design': bool (always True)

    Examples
    --------
    >>> availability = get_mode_availability(diagnostics)
    >>> if availability['fix_issues']:
    ...     print("Mode A available: Issues detected")
    """

    has_issues = diagnostics.needs_any_augmentation()

    return {
        'fix_issues': has_issues,
        'enhance_design': True
    }


def get_mode_recommendations(
    diagnostics: DesignDiagnosticSummary
) -> Dict[str, str]:
    """
    Get user-facing recommendations for which mode to use.

    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics.

    Returns
    -------
    Dict[str, str]
        Mode recommendations with descriptions.

    Examples
    --------
    >>> recommendations = get_mode_recommendations(diagnostics)
    >>> print(recommendations['fix_issues'])
    "Critical aliasing detected - strongly recommend addressing first"
    """

    recommendations = {}

    has_critical = any(
        any(i.severity == 'critical' for i in diag.issues)
        for diag in diagnostics.response_diagnostics.values()
    )

    has_warnings = any(
        any(i.severity == 'warning' for i in diag.issues)
        for diag in diagnostics.response_diagnostics.values()
    )

    if has_critical:
        recommendations['fix_issues'] = (
            "⚠️ **Strongly Recommended**: Critical issues detected that should be addressed."
        )
    elif has_warnings:
        recommendations['fix_issues'] = (
            "⚡ **Suggested**: Warnings detected - consider addressing before proceeding."
        )
    else:
        recommendations['fix_issues'] = (
            "✅ **Not Needed**: Design quality is satisfactory."
        )

    if has_critical:
        recommendations['enhance_design'] = (
            "💡 **Available**: You can enhance the design, but fixing critical issues first is recommended."
        )
    else:
        recommendations['enhance_design'] = (
            "🎯 **Ready**: Design is healthy - choose a goal to enhance capabilities."
        )

    return recommendations


def get_available_enhancement_goals(
    diagnostics: DesignDiagnosticSummary
) -> List[Dict[str, str]]:
    """
    Get available enhancement goals for Mode B (legacy goal-driven path).

    Returns goals appropriate for the current design type,
    formatted for UI display.

    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics.

    Returns
    -------
    List[Dict[str, str]]
        Goal information for UI display.

    Examples
    --------
    >>> goals = get_available_enhancement_goals(diagnostics)
    >>> for goal in goals:
    ...     print(f"{goal['title']}: {goal['description']}")
    """

    available = get_available_goals(
        current_design_type=diagnostics.design_type,
        has_replicates=_check_has_replicates(diagnostics),
        has_center_points=diagnostics.has_center_points,
        is_fractional=(diagnostics.design_type == 'fractional')
    )

    goal_info = []
    for goal in available:
        desc = GOAL_CATALOG[goal]
        alignment = _check_goal_alignment(goal, diagnostics)

        goal_info.append({
            'goal': goal.value,
            'title': desc.title,
            'description': desc.description,
            'typical_strategies': ', '.join(desc.typical_strategies),
            'when_appropriate': desc.when_appropriate,
            'example_scenario': desc.example_scenario,
            'diagnostic_alignment': alignment
        })

    return goal_info


def _check_has_replicates(diagnostics: DesignDiagnosticSummary) -> bool:
    """Check if design has replicate runs."""

    design = diagnostics.original_design
    factor_cols = [f.name for f in diagnostics.factors]

    if not factor_cols:
        return False

    n_unique = design[factor_cols].drop_duplicates().shape[0]
    return n_unique < len(design)


def _check_goal_alignment(
    goal: AugmentationGoal,
    diagnostics: DesignDiagnosticSummary
) -> str:
    """
    Check if diagnostics suggest this goal is particularly appropriate.

    Returns
    -------
    str
        Alignment message, or empty string if no special alignment.
    """

    has_aliasing = any(
        diag.resolution and diag.resolution <= 4
        for diag in diagnostics.response_diagnostics.values()
    )

    if goal == AugmentationGoal.REDUCE_ALIASING and has_aliasing:
        return "✨ Diagnostics detected aliasing - this goal is highly relevant"

    has_lof = any(
        diag.lack_of_fit_p_value and diag.lack_of_fit_p_value < 0.05
        for diag in diagnostics.response_diagnostics.values()
    )

    if goal == AugmentationGoal.MODEL_CURVATURE and has_lof:
        return "✨ Diagnostics detected lack of fit - curvature modeling recommended"

    has_high_var = any(
        diag.prediction_variance_stats and
        diag.prediction_variance_stats.get('max', 0) > 3 * diag.prediction_variance_stats.get('mean', 1)
        for diag in diagnostics.response_diagnostics.values()
    )

    if goal == AugmentationGoal.IMPROVE_PREDICTION and has_high_var:
        return "✨ Diagnostics show uneven prediction variance - this goal is highly relevant"

    has_low_r2 = any(
        diag.r_squared < 0.70
        for diag in diagnostics.response_diagnostics.values()
    )

    if goal == AugmentationGoal.INCREASE_CONFIDENCE and has_low_r2:
        return "💡 Model fit is marginal - additional runs may help"

    return ""


# ---------------------------------------------------------------------------
# Type catalogue for Mode B v2 (direct type selection)
# ---------------------------------------------------------------------------

_AUGMENTATION_TYPE_CATALOGUE: List[Dict] = [
    {
        'type': 'foldover',
        'label': 'Foldover',
        'description': (
            'Creates a mirror-image copy of the design by flipping all factor signs. '
            'Breaks alias chains in fractional factorials and increases resolution by 1.'
        ),
        'when_to_use': 'Use when main effects or interactions are aliased in a fractional factorial.',
        'typical_runs': 'Doubles the current run count.',
    },
    {
        'type': 'ccd',
        'label': 'Central Composite (CCD)',
        'description': (
            'Adds axial (star) points and center points to estimate quadratic curvature. '
            'Upgrades a 2-level factorial into a full response surface design.'
        ),
        'when_to_use': 'Use when you suspect curvature or want to find an optimum.',
        'typical_runs': '2k + 3 to 2k + 6 additional runs (k = number of factors).',
    },
    {
        'type': 'center_points',
        'label': 'Center Points',
        'description': (
            'Adds runs at the midpoint of all factor ranges. '
            'Enables a lack-of-fit test for curvature and improves precision at the center.'
        ),
        'when_to_use': (
            'Use when you want to cheaply test for curvature before committing to a full RSM.'
        ),
        'typical_runs': '3 to 5 additional runs.',
    },
    {
        'type': 'replicates',
        'label': 'Replicates',
        'description': (
            'Repeats a subset of existing design points. '
            'Provides a pure error estimate for formal lack-of-fit testing and '
            'increases statistical power for detecting effects.'
        ),
        'when_to_use': (
            'Use when you need a pure error estimate or want to confirm borderline effects.'
        ),
        'typical_runs': '25% of the current run count (minimum 2 runs).',
    },
    {
        'type': 'd_optimal',
        'label': 'D-Optimal',
        'description': (
            'Adds runs that maximise the determinant of the information matrix (D-criterion). '
            'Best for improving the precision of parameter estimates in the current model.'
        ),
        'when_to_use': (
            'Use when you want to add runs that provide maximum support '
            'for your current model with no specific structural goal.'
        ),
        'typical_runs': 'k + 1 to 2k additional runs (k = number of factors).',
    },
    {
        'type': 'i_optimal',
        'label': 'I-Optimal',
        'description': (
            'Adds runs that minimise average prediction variance across the design space '
            '(I-criterion). Best for models used to predict at many untested conditions.'
        ),
        'when_to_use': (
            'Use when prediction accuracy across the whole space matters more than '
            'precise coefficient estimates.'
        ),
        'typical_runs': 'k + 1 to 2k additional runs (k = number of factors).',
    },
]


def get_available_augmentation_types(
    diagnostics: DesignDiagnosticSummary,
) -> List[Dict]:
    """
    Return augmentation type catalogue with eligibility for the current design.

    Each entry is a dict with display and eligibility fields.  Ineligible types
    are returned with ``'eligible': False`` and a ``'lock_reason'`` string;
    eligible types have ``'eligible': True`` and ``'lock_reason': None``.

    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics used to determine eligibility.

    Returns
    -------
    List[Dict]
        All six types, each containing:

        - ``type`` (str): internal key
        - ``label`` (str): display name
        - ``description`` (str): what it does
        - ``when_to_use`` (str): guidance
        - ``typical_runs`` (str): run-count estimate
        - ``eligible`` (bool)
        - ``lock_reason`` (str or None): shown in UI when not eligible

    Examples
    --------
    >>> types = get_available_augmentation_types(diagnostics)
    >>> [t['label'] for t in types if t['eligible']]
    ['Replicates', 'D-Optimal', 'I-Optimal']
    """
    design_type = diagnostics.design_type
    is_fractional = design_type == 'fractional'
    is_rsm = design_type in ('response_surface', 'ccd', 'box_behnken')
    is_full_factorial = design_type == 'full_factorial'
    has_center_points = diagnostics.has_center_points

    # Determine resolution from response diagnostics (None if unavailable)
    resolution: Optional[int] = None
    for diag in diagnostics.response_diagnostics.values():
        if diag.resolution is not None:
            resolution = diag.resolution
            break
    is_high_resolution = resolution is not None and resolution >= 5

    results: List[Dict] = []
    for entry in _AUGMENTATION_TYPE_CATALOGUE:
        aug_type = entry['type']
        eligible = True
        lock_reason: Optional[str] = None

        if aug_type == 'foldover':
            if is_full_factorial:
                eligible = False
                lock_reason = 'Full factorial designs have no aliasing - foldover is not needed.'
            elif is_rsm:
                eligible = False
                lock_reason = 'Response surface designs have no aliasing - foldover is not applicable.'
            elif not is_fractional:
                eligible = False
                lock_reason = 'Foldover only applies to fractional factorial designs.'
            elif is_high_resolution:
                eligible = False
                lock_reason = (
                    f'Design is already Resolution {resolution} - '
                    'all main effects are clear of 2FI aliasing.'
                )

        elif aug_type == 'ccd':
            if is_rsm:
                eligible = False
                lock_reason = 'Design is already a response surface - CCD augmentation is not needed.'

        elif aug_type == 'center_points':
            if has_center_points:
                eligible = False
                lock_reason = 'Design already includes center points.'
            elif is_rsm:
                eligible = False
                lock_reason = 'Response surface designs already include center points.'

        results.append({
            **entry,
            'eligible': eligible,
            'lock_reason': lock_reason,
        })

    return results


def create_plan_comparison_table(
    plans: List[AugmentationPlan]
) -> List[Dict]:
    """
    Create comparison table data for UI display.

    Parameters
    ----------
    plans : List[AugmentationPlan]
        Plans to compare.

    Returns
    -------
    List[Dict]
        Table rows for display.

    Examples
    --------
    >>> table = create_plan_comparison_table(plans)
    >>> df = pd.DataFrame(table)
    >>> st.dataframe(df)
    """

    rows = []

    for plan in plans:
        row = {
            'Rank': plan.rank,
            'Plan': plan.plan_name,
            'Strategy': plan.strategy.replace('_', ' ').title(),
            'Runs to Add': plan.n_runs_to_add,
            'Total After': plan.total_runs_after,
            'Utility': f"{plan.utility_score:.0f}/100",
            'Mode': plan.metadata.get('mode', 'unknown').replace('_', ' ').title()
        }

        rows.append(row)

    return rows
