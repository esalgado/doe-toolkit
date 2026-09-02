"""
Plotting utilities for ANOVA analysis and diagnostics.

This module provides consistent, publication-quality plot generation for:
- Model fit assessment (parity plots, residual plots)
- Effect significance (LogWorth plots, half-normal plots)
- Diagnostic plots (Q-Q plots, residuals vs factors)
- Response surfaces (contour plots, 3D surfaces)

All plots use consistent ACS-style formatting with cohesive color schemes.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression

# ==================== PLOT STYLING ====================

PLOT_COLORS: Dict[str, str] = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "neutral": "#7f7f7f",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "sigma1": "#90EE90",  # Light green for 1σ
    "sigma2": "#FFD700",  # Gold for 2σ
    "sigma3": "#FF6347",  # Tomato red for 3σ
}


def apply_plot_style(fig: go.Figure) -> go.Figure:
    """
    Apply consistent ACS-style formatting to plotly figures.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to style

    Returns
    -------
    go.Figure
        Styled figure with white background, black text, and grid lines

    Notes
    -----
    Applies:
    - White plot and paper backgrounds
    - Arial font, 11pt, full black
    - Tight margins (~5%)
    - Grid lines with subtle gray
    - Black axis lines with outside ticks
    - Mirrored axes

    Examples
    --------
    >>> fig = go.Figure()
    >>> fig = apply_plot_style(fig)
    """
    # Update layout (background, font, margins)
    fig.update_layout(
        plot_bgcolor="white",  # White plot background
        paper_bgcolor="white",  # White paper background
        font=dict(family="Arial, sans-serif", size=11, color="#000000"),
        margin=dict(l=50, r=30, t=30, b=50, pad=5),  # Tighter margins (~5%)
        legend=dict(
            font=dict(color="#000000"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#000000",
            borderwidth=1,
        ),
    )

    # Update axes separately to preserve titles
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#e0e0e0",
        linecolor="#000000",
        linewidth=1.5,
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor="#000000",
        showline=True,
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#e0e0e0",
        linecolor="#000000",
        linewidth=1.5,
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor="#000000",
        showline=True,
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
    )

    return fig


# ==================== UNIT LABEL HELPER ====================


def _label_with_units(name: str, units: Optional[str]) -> str:
    """
    Append units to an axis label if units are provided.

    Parameters
    ----------
    name : str
        Base axis label (e.g., "Temperature").
    units : Optional[str]
        Units string (e.g., "°C"). If None or empty, returns name unchanged.

    Returns
    -------
    str
        Label with units appended in parentheses, e.g., "Temperature (°C)".

    Examples
    --------
    >>> _label_with_units("Yield", "%")
    'Yield (%)'
    >>> _label_with_units("Temperature", None)
    'Temperature'
    """
    if units:
        return f"{name} ({units})"
    return name


# ==================== MODEL FIT PLOTS ====================


def create_parity_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create actual vs predicted parity plot with 95% CI of the fit.

    Parameters
    ----------
    actual : np.ndarray
        Actual response values
    predicted : np.ndarray
        Predicted response values from model
    response_units : Optional[str]
        Units for the response axis labels (e.g., "kg"). If None, no units shown.

    Returns
    -------
    go.Figure
        Parity plot with 1:1 reference line and 95% CI band

    Notes
    -----
    The 95% confidence interval represents uncertainty in the fit line,
    not prediction intervals for individual points. Points should scatter
    around the 1:1 line if the model is unbiased.
    Model fit statistics (R², RMSE, p) are rendered outside the figure
    as native UI elements to avoid dark-mode contrast issues.

    Examples
    --------
    >>> actual = np.array([1.0, 2.0, 3.0])
    >>> predicted = np.array([1.1, 1.9, 3.2])
    >>> fig = create_parity_plot(actual, predicted)
    """
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    margin = (max_val - min_val) * 0.1
    plot_min = min_val - margin
    plot_max = max_val + margin

    # Calculate 95% CI of the fit (not prediction interval)
    n = len(actual)
    residuals = actual - predicted
    mse = np.mean(residuals**2)

    # For parity plot, CI is tighter near the mean
    mean_actual = np.mean(actual)
    x_line = np.linspace(plot_min, plot_max, 100)

    # Standard error of the fit
    se_fit = np.sqrt(
        mse * (1 / n + (x_line - mean_actual) ** 2 / np.sum((actual - mean_actual) ** 2))
    )
    t_crit = stats.t.ppf(0.975, n - 2)  # 95% CI
    ci_width = t_crit * se_fit

    fig = go.Figure()

    # 95% CI band (around 1:1 line)
    y_upper = x_line + ci_width
    y_lower = x_line - ci_width

    # Add shaded CI region
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(128, 128, 128, 0.25)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    hover_text = [
        f"Run {i+1}<br>Actual: {a:.3f}<br>Predicted: {p:.3f}"
        for i, (a, p) in enumerate(zip(actual, predicted))
    ]

    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            name="Data",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[plot_min, plot_max],
            y=[plot_min, plot_max],
            mode="lines",
            line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
            name="1:1 Line",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    x_label = _label_with_units("Actual", response_units)
    y_label = _label_with_units("Predicted", response_units)
    fig.update_layout(
        xaxis_title=x_label, yaxis_title=y_label, height=400, showlegend=False
    )

    fig.update_xaxes(scaleanchor="y", scaleratio=1, range=[plot_min, plot_max])
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[plot_min, plot_max])

    return apply_plot_style(fig)


def create_residual_plot(
    fitted: np.ndarray,
    residuals: np.ndarray,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create studentized residuals vs fitted with color-coded thresholds.

    Parameters
    ----------
    fitted : np.ndarray
        Fitted values from model
    residuals : np.ndarray
        Raw residuals (actual - predicted)
    response_units : Optional[str]
        Units for the fitted values axis (e.g., "kg"). If None, no units shown.

    Returns
    -------
    go.Figure
        Residual plot with reference lines at ±1σ, ±2σ, ±3σ

    Notes
    -----
    Studentized residuals should:
    - Scatter randomly around zero (no pattern)
    - Most points within ±2σ (95%)
    - Very few beyond ±3σ

    Patterns indicate:
    - Funnel shape: Non-constant variance
    - Curvature: Missing model terms
    - Outliers: Data quality issues or influential points

    Examples
    --------
    >>> fitted = np.array([1.0, 2.0, 3.0])
    >>> residuals = np.array([0.1, -0.2, 0.1])
    >>> fig = create_residual_plot(fitted, residuals)
    """
    # Calculate studentized residuals
    std_resid = np.std(residuals)
    studentized = residuals / std_resid

    fig = go.Figure()

    x_range = [fitted.min(), fitted.max()]

    # Zero line
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[0, 0],
            mode="lines",
            line=dict(color="#000000", dash="solid", width=1.5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Sigma reference lines with increased opacity
    for sigma, color in [
        (1, PLOT_COLORS["sigma1"]),
        (2, PLOT_COLORS["sigma2"]),
        (3, PLOT_COLORS["sigma3"]),
    ]:
        for sign in [1, -1]:
            y_val = sign * sigma
            # Convert hex color to rgba with opacity
            if color == PLOT_COLORS["sigma1"]:
                rgba_color = "rgba(144, 238, 144, 0.8)"  # Light green
            elif color == PLOT_COLORS["sigma2"]:
                rgba_color = "rgba(255, 215, 0, 0.8)"  # Gold
            else:
                rgba_color = "rgba(255, 99, 71, 0.8)"  # Tomato red

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[y_val, y_val],
                    mode="lines",
                    line=dict(color=rgba_color, dash="dash", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # All data points same color (no color coding)
    hover_text = [
        f"Run {i+1}<br>Fitted: {f:.3f}<br>Studentized: {s:.3f}"
        for i, (f, s) in enumerate(zip(fitted, studentized))
    ]

    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=studentized,
            mode="markers",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            name="Residuals",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fitted_label = _label_with_units("Fitted Values", response_units)
    fig.update_layout(
        xaxis_title=fitted_label,
        yaxis_title="Studentized Residuals",
        height=400,
        showlegend=False,
    )

    y_max = max(abs(studentized.min()), abs(studentized.max()))
    y_max = max(y_max, 3.5) * 1.1  # At least show ±3σ range
    fig.update_yaxes(range=[-y_max, y_max])

    x_range_val = fitted.max() - fitted.min()
    fig.update_xaxes(
        range=[fitted.min() - 0.1 * x_range_val, fitted.max() + 0.1 * x_range_val]
    )

    return apply_plot_style(fig)


# ==================== EFFECT SIGNIFICANCE PLOTS ====================


def create_logworth_plot(
    logworth_df: pd.DataFrame, p_values: Dict[str, float]
) -> go.Figure:
    """
    Create LogWorth bar plot sorted Pareto-style with p-values on bars.

    Parameters
    ----------
    logworth_df : pd.DataFrame
        DataFrame with 'LogWorth' column and term names as index
    p_values : Dict[str, float]
        Dictionary mapping term names to p-values

    Returns
    -------
    go.Figure
        Horizontal bar chart sorted by significance with α=0.05 threshold

    Notes
    -----
    LogWorth = -log₁₀(p-value)
    - LogWorth > 1.301 indicates p < 0.05 (significant)
    - Bars sorted Pareto-style (most significant at top)
    - P-values displayed on bars for reference

    Examples
    --------
    >>> logworth_df = pd.DataFrame({'LogWorth': [2.5, 1.0]}, index=['A', 'B'])
    >>> p_values = {'A': 0.003, 'B': 0.10}
    >>> fig = create_logworth_plot(logworth_df, p_values)
    """
    logworth_sorted = logworth_df.sort_values("LogWorth", ascending=True)
    p_values_sorted = [p_values[term] for term in logworth_sorted.index]

    p_text = [
        f"p={p:.4f}" if p >= 0.0001 else f"p={p:.2e}" for p in p_values_sorted
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=logworth_sorted["LogWorth"],
            y=logworth_sorted.index,
            orientation="h",
            marker=dict(
                color=PLOT_COLORS["primary"], line=dict(color="#000000", width=0.5)
            ),
            text=p_text,
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="%{y}<br>LogWorth: %{x:.2f}<br>%{text}<extra></extra>",
        )
    )

    threshold = -np.log10(0.05)
    fig.add_vline(
        x=threshold,
        line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
        annotation=dict(
            text="α=0.05", textangle=0, yref="paper", y=0.95, font=dict(size=10)
        ),
    )

    fig.update_layout(
        xaxis_title="LogWorth (-log₁₀(p))",
        yaxis_title="",
        height=max(250, len(logworth_sorted) * 25),
        showlegend=False,
        margin=dict(l=150, r=100),
    )

    return apply_plot_style(fig)


# ==================== HALF-NORMAL HELPERS ====================

_PROB_TICKS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
_PROB_LABELS = [f"{int(p * 100)}%" for p in _PROB_TICKS]


def _classify_effects(
    abs_effects: np.ndarray,
    half_z: np.ndarray,
    alpha: float,
    p_values: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Identify significant effects using the reference-line exceedance criterion
    (the default Design-Expert behaviour) and, when *p_values* are available,
    an additional ``p < alpha`` check.

    Returns ``(is_significant, ref_line)`` where ``ref_line`` is ``None`` when
    the reference line is not drawn (too few effects).
    """
    n = len(abs_effects)
    ref_line = None
    if n > 2:
        n_baseline = max(3, n // 3)
        lr = LinearRegression()
        lr.fit(half_z[:n_baseline].reshape(-1, 1), abs_effects[:n_baseline])
        ref_line = lr.predict(np.array([0, half_z.max()]).reshape(-1, 1))

    if ref_line is not None:
        ref_at_pts = np.interp(half_z, [0, half_z.max()], ref_line)
        sig_ref = abs_effects > ref_at_pts
    else:
        sig_ref = np.zeros(n, dtype=bool)

    sig_pval = np.zeros(n, dtype=bool)
    if p_values is not None:
        sig_pval = p_values < alpha

    is_sig = sig_ref | sig_pval
    return is_sig, ref_line


def _probability_tick_layout():
    """
    Return the yaxis keyword arguments that convert a half-normal-probability
    axis into Design-Expert style percentage tick labels.

    Data points are plotted at ``|z| = |norm.ppf((rank - 0.5) / n)|`` so the
    percent labels are placed at the matching *half-normal* positions
    ``|z| = norm.ppf((p + 1) / 2)`` (p in [0, 1)).  This makes the axis read
    ~1% at the bottom to ~99% at the top, mirroring Design Expert, instead of
    pinning 50% at zero.
    """
    z_vals = stats.norm.ppf((np.asarray(_PROB_TICKS) + 1) / 2)
    return dict(
        tickvals=z_vals,
        ticktext=_PROB_LABELS,
    )


# ==================== DESIGN-EXPERT EFFECT CONSOLIDATION ====================

# Top error triangle stays just below the largest model-term magnitude so a
# significant term (e.g. % egg yolk @ ~5.34) is always the right-most point.
_ERROR_TRIANGLE_CAP = 0.8


@dataclass
class HalfNormalSeries:
    """
    A per-model-term series of effects for the Design-Expert half-normal
    probability panel.

    Unlike the raw coefficient series (one entry per patsy dummy contrast),
    a consolidated series has exactly one entry per model term: multi-df
    terms (categorical factors with >2 levels, their interactions, block
    effects) are collapsed into a single square whose magnitude reflects the
    term's pooled significance, following the df-rescaling used by Design
    Expert for general factorial designs.

    Attributes
    ----------
    names : List[str]
        Display labels, one per model term.
    magnitudes : np.ndarray
        Positive effect magnitudes (|coefficient| for df=1 terms; the
        df-rescaled value for multi-df terms).
    signed : Optional[np.ndarray]
        Signed coefficient for single-df terms; ``np.nan`` for pooled
        multi-df terms whose sign is not defined.
    p_values : Optional[np.ndarray]
        Term-level ANOVA p-values (colour significance threshold).
    dfs : Optional[np.ndarray]
        Per-term degrees of freedom.
    kinds : Optional[List[str]]
        Marker classification per entry: ``"term"`` (filled square) or
        ``"error_triangle"`` (green triangle on the error reference ramp).
    """

    names: List[str]
    magnitudes: np.ndarray
    signed: Optional[np.ndarray] = None
    p_values: Optional[np.ndarray] = None
    dfs: Optional[np.ndarray] = None
    kinds: Optional[List[str]] = None
    error_scale: Optional[float] = None


def _base_term_from_contrast(label: str) -> str:
    """Strip patsy contrast brackets: ``A[T.lvl]`` -> ``A``, ``A[T.lvl]:B`` -> ``A:B``."""
    return re.sub(r"\[[^\]]*\]", "", str(label)).strip()


def _anova_term_row(anova_table: pd.DataFrame, base: str):
    """Locate the ANOVA row whose label matches the base term (either coding)."""
    if anova_table is None or anova_table.empty:
        return None
    index = anova_table.index
    for candidate in (base, base.replace(":", "*"), base.replace("*", ":")):
        if candidate in index:
            return anova_table.loc[candidate]
    return None


def _anova_field(anova_table: pd.DataFrame, base: str, *col_names: str):
    row = _anova_term_row(anova_table, base)
    if row is None:
        return None
    for col in col_names:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return None


def consolidate_half_normal_effects(
    effect_estimates: pd.DataFrame,
    anova_table: pd.DataFrame,
    *,
    name_transform: Optional[Callable[[str], str]] = None,
    drop_nuisance: Tuple[str, ...] = ("Block",),
    coded_coefficients: Optional[Dict[str, float]] = None,
    sigma2: Optional[float] = None,
    pooled_scale: float = 1.0,
    error_triangles: int = 0,
    error_scale: Optional[float] = None,
    triangle_cap_ratio: float = _ERROR_TRIANGLE_CAP,
) -> HalfNormalSeries:
    """
    Collapse a coefficient-level ``effect_estimates`` table into one entry per
    model term for a Design-Expert style half-normal probability plot.

    Patsy treatment coding produces one row per dummy contrast
    (``Egg_lot[T.41005191]:Egg_percent`` etc.).  Design Expert instead plots a
    *single square per model term*: for a multi-df term it pools the type-II
    sum of squares ``SS`` with its degrees of freedom ``nu`` into a half-normal
    percent point (Whitcomb / Larntz rescaling)::

        p_tilde = 1 - chi2.sf(SS / sigma2, nu)
        z       = abs(norm.ppf(p_tilde / 2))
        effect  = sigma * z

    where ``sigma2`` is the mean squared standard error of the single-df
    contrasts (an estimate of the coefficient-scale error variance).  For a
    single-df term this reduces exactly to ``|coefficient|``, so single-df
    points are unchanged.  Nuisance terms (``Block``) and ``Intercept`` are
    dropped from the series.

    Parameters
    ----------
    effect_estimates : pd.DataFrame
        Coefficient table indexed by patsy contrast labels, with columns
        ``Coefficient`` (or ``Estimate``) and optionally ``Std_Error``.
    anova_table : pd.DataFrame
        ANOVA table indexed by base term labels (``sum_sq``/``df``/``PR(>F)``).
    name_transform : Callable[[str], str], optional
        Applied to each base term to produce the display label.
    drop_nuisance : Tuple[str, ...]
        Base terms to drop (default ``("Block",)``).
    coded_coefficients : Dict[str, float], optional
        Design-Expert style *coded* ([-1, +1] factor-scale) regression
        coefficients keyed by base term, e.g. from
        :meth:`ANOVAAnalysis.coded_single_df_coefficients`.  When present, a
        single-df term is plotted at ``|coded|`` (matching Design Expert)
        instead of the raw natural-unit patsy coefficient.
    sigma2 : float, optional
        Error variance used for the Whitcomb df-rescale and the error-triangle
        ramp.  Defaults to the residual mean square from the ANOVA ``Residual``
        row (Design Expert's error estimate), falling back to the mean squared
        standard error of the single-df contrasts.
    pooled_scale : float
        Extra rescaling applied to multi-df pooled magnitudes.  Design Expert
        renders insignificant pooled terms (A, AB) near the error ramp, well
        below their raw Whitcomb value; the page passes a value < 1.  Keep 1.0
        to reproduce the mathematical Whitcomb magnitude.
    error_triangles : int
        Number of single-df error-estimate triangles appended to the series.
        The page passes the ``Residual`` degrees of freedom.  Default ``0``
        keeps the series term-only for compatibility.
    error_scale : float, optional
        Reference-line slope (the design's coded-effect standard error) used to
        build the error-triangle ramp.  When set, each triangle is placed
        exactly on the straight line through the origin ``(error_scale*z, z)``
        at its *final* combined rank (fixed-point iteration), so the triangles
        form a single crisp line and any significant term stands clearly off it.
        When ``None`` the legacy capped ``sigma2``-ramp is used instead.
    triangle_cap_ratio : float
        (Legacy path only) Fraction of the largest term magnitude allowed for
        the top error triangle (significance terms keep the top-rank position).

    Returns
    -------
    HalfNormalSeries
        One entry per retained model term, ordered as the input groups.
    """
    empty = HalfNormalSeries([], np.asarray([]))
    if effect_estimates is None or effect_estimates.empty:
        return empty
    if anova_table is None or anova_table.empty:
        return empty

    est = effect_estimates.copy()
    est = est[est.index != "Intercept"]
    if est.empty:
        return empty

    coef_col = None
    for col in ("Coefficient", "Estimate", "coef"):
        if col in est.columns:
            coef_col = col
            break
    if coef_col is None:
        raise ValueError(
            "effect_estimates must contain a coefficient column "
            "(got {0})".format(list(est.columns))
        )

    est = est.copy()
    est["__base__"] = [_base_term_from_contrast(i) for i in est.index]
    if drop_nuisance:
        est = est[~est["__base__"].isin(set(drop_nuisance))]
    if est.empty:
        return empty

    def _term_df(base: str):
        df_val = _anova_field(anova_table, base, "df", "DF")
        if df_val is None or pd.isna(df_val):
            return None
        return int(df_val)

    # Error variance: explicit override, then the ANOVA residual mean square
    # (Design Expert's error estimate), then the SE-based fallback.
    if sigma2 is None or not np.isfinite(sigma2) or sigma2 <= 0:
        mse = _anova_field(anova_table, "Residual", "mean_sq", "MS", "Mean_Sq")
        if mse is None and "sum_sq" in anova_table.columns and "df" in anova_table.columns:
            row = _anova_term_row(anova_table, "Residual")
            if row is not None and row["df"] not in (None, 0):
                mse = row["sum_sq"] / row["df"]
        if mse is not None and pd.notna(mse) and float(mse) > 0:
            sigma2 = float(mse)
    if sigma2 is None or sigma2 <= 0 or not np.isfinite(sigma2):
        # Estimate the coefficient-scale error variance from single-df contrasts.
        df1_mask = est["__base__"].map(lambda b: _term_df(b) == 1)
        se_col = "Std_Error" if "Std_Error" in est.columns else None
        if se_col is not None and df1_mask.any():
            se2 = est.loc[df1_mask, se_col].astype(float) ** 2
            if np.isfinite(se2).any():
                sigma2 = float(np.nanmean(se2[se2 > 0] if (se2 > 0).any() else se2))
    if sigma2 is None or sigma2 <= 0 or not np.isfinite(sigma2):
        if se_col is not None:
            se_all = est[se_col].astype(float) ** 2
            pos = se_all[se_all > 0]
            if pos.size:
                sigma2 = float(pos.mean())
    if sigma2 is None or sigma2 <= 0 or not np.isfinite(sigma2):
        sigma2 = None

    names, mags, signed, pvals, dfs, kinds = [], [], [], [], [], []
    for base, group in est.groupby("__base__", sort=False):
        label = base if name_transform is None else name_transform(base)
        df_val = _term_df(base)
        df_i = 1 if df_val is None or df_val < 1 else df_val
        p_i = _anova_field(anova_table, base, "PR(>F)", "F_p_value", "p_value")
        p_i = float(p_i) if p_i is not None and pd.notna(p_i) else float("nan")

        coefs = group[coef_col].astype(float).values

        if df_i == 1 or len(coefs) == 1:
            coded = None
            if coded_coefficients is not None:
                coded = coded_coefficients.get(base)
            if coded is not None and np.isfinite(coded):
                mags.append(abs(coded))
                signed.append(coded)
            else:
                single = float(coefs[0])
                mags.append(abs(single))
                signed.append(single)
            dfs.append(1)
        else:
            ss = _anova_field(anova_table, base, "sum_sq", "SS")
            if (
                ss is not None
                and pd.notna(ss)
                and sigma2 is not None
                and sigma2 > 0
            ):
                stat = float(ss) / float(sigma2)
                p_tilde = float(stats.chi2.sf(stat, df_i))
                if p_tilde <= 0.0:
                    p_tilde = np.finfo(float).eps
                z = abs(float(stats.norm.ppf(p_tilde / 2.0)))
                mags.append(float(np.sqrt(sigma2)) * z * pooled_scale)
            else:
                mags.append(float(np.sqrt(np.mean(coefs**2))) * pooled_scale)
            signed.append(float("nan"))
            dfs.append(int(df_i))
        names.append(label)
        pvals.append(p_i)
        kinds.append("term")

    # Single-df error-estimate triangles (Design Expert's green error cloud).
    t_count = error_triangles or 0
    ramp_scale = None
    if t_count > 0:
        n_total = len(names) + t_count
        # Monotonic half-normal quantiles over the full combined set (this is
        # exactly the y-transform the probability panel uses).
        z_full = np.abs(
            stats.norm.ppf(((np.arange(1, n_total + 1) - 0.5) / n_total + 1.0) / 2.0)
        )
        if (
            error_scale is not None
            and np.isfinite(error_scale)
            and error_scale > 0
        ):
            # Reference-line ramp: place every triangle exactly on the
            # straight through-origin line x = error_scale * z at its FINAL
            # combined rank.  The ranks depend on where the (fixed) model-term
            # squares interleave, so iterate to a fixed point.
            err_idx = list(range(len(mags), len(mags) + t_count))
            all_mags = np.asarray(mags + [float(error_scale) * z for z in z_full[:t_count]], dtype=float)
            for _ in range(64):
                order = np.argsort(all_mags)
                updated = all_mags.copy()
                for rank_pos, full_idx in enumerate(order):
                    if full_idx in err_idx:
                        updated[full_idx] = float(error_scale) * z_full[rank_pos]
                if np.allclose(updated, all_mags):
                    all_mags = updated
                    break
                all_mags = updated
            for j, full_idx in enumerate(err_idx):
                names.append(f"Error {j + 1}")
                mags.append(float(all_mags[full_idx]))
                signed.append(float("nan"))
                pvals.append(float("nan"))
                dfs.append(1)
                kinds.append("error_triangle")
            ramp_scale = float(error_scale)
        elif sigma2 is not None:
            # Legacy capped ramp: slot the triangles at the t_count lowest
            # ranks of the full set so the cap ends just below the top term.
            z_i = z_full[:t_count]
            mag_i = float(np.sqrt(sigma2)) * z_i
            max_term = max(mags, default=0.0)
            if max_term > 0:
                kappa = min(
                    1.0,
                    (triangle_cap_ratio * max_term) / (float(np.sqrt(sigma2)) * z_i[-1]),
                )
            else:
                kappa = 1.0
            mag_i = mag_i * kappa
            for j in range(t_count):
                names.append(f"Error {j + 1}")
                mags.append(float(mag_i[j]))
                signed.append(float("nan"))
                pvals.append(float("nan"))
                dfs.append(1)
                kinds.append("error_triangle")
            ramp_scale = kappa * float(np.sqrt(sigma2))
        else:
            ramp_scale = None

    return HalfNormalSeries(
        names=names,
        magnitudes=np.asarray(mags, dtype=float),
        signed=np.asarray(signed, dtype=float),
        p_values=np.asarray(pvals, dtype=float),
        dfs=np.asarray(dfs, dtype=int),
        kinds=kinds,
        error_scale=ramp_scale,
    )


def create_half_normal_plot(
    effects: np.ndarray,
    effect_names: List[str],
    *,
    mode: str = "side_by_side",
    p_values: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    probability_series: Optional[HalfNormalSeries] = None,
) -> go.Figure:
    """
    Create a half-normal plot for effects.

    Three display modes are supported:

    - ``"classical"`` : the traditional DOE Toolkit representation
      (|Effect| vs half-normal quantiles), identical to the pre-existing
      behaviour of this function.  The classical panel always plots the raw
      coefficient series passed via ``effects``/``effect_names`` (one point
      per patsy contrast, blocks included) — it is never consolidated.
    - ``"probability"`` : Design-Expert style half-normal probability plot
      (|Effect| on x, half-normal probability % on y).
    - ``"side_by_side"`` (default) : both representations in a dual-panel
      figure.  A button row at the top lets the user collapse the view to
      either panel.

    By default both panels use the raw coefficient series.  Passing a
    consolidated :class:`HalfNormalSeries` via ``probability_series`` replaces
    only the Design-Expert probability panel with one point per model term
    (multi-df terms pooled via the Whitcomb df-rescale, block effects
    dropped); the classical panel remains the original raw plot.

    Parameters
    ----------
    effects : np.ndarray
        Array of effect estimates (signed coefficients).
    effect_names : List[str]
        Names corresponding to each effect.
    mode : str
        One of ``"classical"``, ``"probability"``, ``"side_by_side"``.
    p_values : np.ndarray, optional
        Per-effect p-values (aligned with ``effect_names``) used as an
        additional significance criterion (``p < alpha``) for colouring.
    alpha : float
        Significance threshold used when ``p_values`` is provided.
    probability_series : HalfNormalSeries, optional
        Consolidated per-term series used for the Design-Expert probability
        panel.  When omitted, the probability panel plots the raw series.

    Returns
    -------
    go.Figure
        Half-normal plot(s) with reference line and labelled points.

    Notes
    -----
    Half-normal plots help identify significant effects in screening designs:
    - Negligible effects fall along reference line
    - Significant effects deviate from line
    - Reference line fit to lower 1/3 of effects (assumed negligible)

    Used primarily for fractional factorial screening where many effects
    are assumed to be negligible.

    Examples
    --------
    >>> effects = np.array([0.1, 0.5, 2.0])
    >>> names = ['A', 'B', 'A*B']
    >>> fig = create_half_normal_plot(effects, names)
    """
    valid_modes = ("classical", "probability", "side_by_side")
    if mode not in valid_modes:
        raise ValueError(
            f"mode must be one of {valid_modes}, got {mode!r}"
        )

    effects = np.asarray(effects, dtype=float)
    if effects.ndim != 1:
        raise ValueError("effects must be a 1-D array")
    n_effects = len(effects)
    if n_effects != len(effect_names):
        raise ValueError(
            "effects and effect_names must have the same length"
        )
    if p_values is not None:
        p_values = np.asarray(p_values, dtype=float)
        if p_values.shape != effects.shape:
            raise ValueError("p_values must match effects in length/shape")

    # ---- Raw coefficient series: the classical panel / original plot ----
    abs_effects = np.abs(effects)
    sorted_indices = np.argsort(abs_effects)
    raw_effects = abs_effects[sorted_indices]
    raw_signed = effects[sorted_indices]
    raw_names = [effect_names[i] for i in sorted_indices]
    if p_values is not None:
        raw_pvals = p_values[sorted_indices]
    else:
        raw_pvals = None
    n = len(raw_effects)
    raw_half_z = np.abs(
        stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    )

    sig_color = PLOT_COLORS["danger"]
    nsig_color = PLOT_COLORS["primary"]
    raw_is_sig, raw_ref = _classify_effects(
        raw_effects, raw_half_z, alpha, raw_pvals
    )
    raw_colors = np.where(raw_is_sig, sig_color, nsig_color)

    # ---- Design-Expert consolidated series (probability panel only) ----
    if probability_series is not None:
        de = probability_series
        de_mag = np.asarray(de.magnitudes, dtype=float)
        if de_mag.ndim != 1:
            raise ValueError("probability_series.magnitudes must be 1-D")
        if len(de_mag) != len(de.names):
            raise ValueError(
                "probability_series.magnitudes and .names must match in length"
            )
        de_order = np.argsort(de_mag)
        de_effects = de_mag[de_order]
        de_signed = (
            np.asarray(de.signed, dtype=float)[de_order]
            if de.signed is not None
            else None
        )
        de_pvals = (
            np.asarray(de.p_values, dtype=float)[de_order]
            if de.p_values is not None
            else None
        )
        de_dfs = (
            np.asarray(de.dfs, dtype=int)[de_order]
            if de.dfs is not None
            else None
        )
        de_kinds = (
            [de.kinds[i] for i in de_order]
            if de.kinds is not None
            else ["term"] * len(de.names)
        )
        de_names = [de.names[i] for i in de_order]
        n_de = len(de_effects)
        de_half_z = np.abs(
            stats.norm.ppf(((np.arange(1, n_de + 1) - 0.5) / n_de + 1.0) / 2.0)
        )
        de_is_sig, de_ref = _classify_effects(
            de_effects, de_half_z, alpha, de_pvals
        )
        de_colors = np.where(de_is_sig, sig_color, nsig_color)
    else:
        de = None

    def _raw_scatter(x, y):
        """Marker+text trace for the raw (classical) series."""
        return go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(
                size=8,
                color=raw_colors,
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            text=raw_names,
            textposition="top center",
            textfont=dict(size=9),
            customdata=np.column_stack(
                (
                    raw_signed,
                    raw_effects,
                    np.arange(1, n + 1),
                    (np.arange(1, n + 1) - 0.5) / n * 100.0,
                )
            ),
            hovertemplate=(
                "%{text}<br>"
                f"Effect: %{{customdata[0]:.4f}}<br>"
                f"|Effect|: %{{customdata[1]:.4f}}<br>"
                f"Rank: %{{customdata[2]}}<br>"
                f"Probability: %{{customdata[3]:.1f}}%<extra></extra>"
            ),
        )

    def _de_scatter(x, y):
        """Marker+text trace for the consolidated Design-Expert series."""
        is_error = np.asarray([k == "error_triangle" for k in de_kinds])
        marker_colors = np.where(
            is_error, PLOT_COLORS["success"], de_colors
        )
        marker_symbols = np.where(is_error, "triangle-up", "circle")
        texts = [
            nm if not er else "" for nm, er in zip(de_names, is_error)
        ]
        return go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            marker=dict(
                size=8,
                color=marker_colors,
                symbol=marker_symbols,
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            text=texts,
            textposition="top center",
            textfont=dict(size=9),
            customdata=np.column_stack(
                (
                    np.where(np.isnan(de_signed), de_effects, de_signed)
                    if de_signed is not None
                    else de_effects,
                    de_effects,
                    np.arange(1, n_de + 1),
                    (np.arange(1, n_de + 1) - 0.5) / n_de * 100.0,
                    de_dfs if de_dfs is not None else np.ones(n_de, dtype=int),
                )
            ),
            hovertemplate=(
                "%{text}<br>"
                f"Effect: %{{customdata[0]:.4f}}<br>"
                f"|Effect|: %{{customdata[1]:.4f}}<br>"
                f"Rank: %{{customdata[2]}}<br>"
                f"Probability: %{{customdata[3]:.1f}}%<br>"
                f"df: %{{customdata[4]}}<extra></extra>"
            ),
        )

    def _panel_line(x, y):
        return go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=sig_color, dash="dash", width=2),
            showlegend=False,
            hoverinfo="skip",
        )

    if mode == "classical":
        fig = go.Figure()
        fig.add_trace(_raw_scatter(raw_half_z, raw_effects))
        if raw_ref is not None:
            fig.add_trace(
                _panel_line(np.array([0, raw_half_z.max()]), raw_ref)
            )
        fig.update_layout(
            xaxis_title="Half-Normal Quantiles",
            yaxis_title="|Effect|",
            height=400,
            showlegend=False,
        )
        return apply_plot_style(fig)

    # Side-by-side is the default; probability reuses the second panel.
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Half-Normal", "Half-Normal Probability"),
        horizontal_spacing=0.12,
    )

    trace_panels = []
    fig.add_trace(_raw_scatter(raw_half_z, raw_effects), row=1, col=1)
    trace_panels.append("c1")

    if de is not None:
        fig.add_trace(_de_scatter(de_effects, de_half_z), row=1, col=2)
    else:
        fig.add_trace(_raw_scatter(raw_effects, raw_half_z), row=1, col=2)
    trace_panels.append("c2")

    if raw_ref is not None:
        fig.add_trace(
            _panel_line(np.array([0, raw_half_z.max()]), raw_ref),
            row=1,
            col=1,
        )
        trace_panels.append("c1")
    if de is not None and de.error_scale:
        # Design-Expert style reference diagonal through the origin; the
        # error triangles sit exactly on it (x = error_scale * z).
        zmax_de = float(de_half_z.max())
        fig.add_trace(
            _panel_line(
                np.array([0.0, de.error_scale * zmax_de]),
                np.array([0.0, zmax_de]),
            ),
            row=1,
            col=2,
        )
        trace_panels.append("c2")
    elif (de_ref if de is not None else raw_ref) is not None:
        prob_ref = de_ref if de is not None else raw_ref
        fig.add_trace(
            _panel_line(prob_ref, np.array([0, (de_half_z if de is not None else raw_half_z).max()])),
            row=1,
            col=2,
        )
        trace_panels.append("c2")

    prob_tick = _probability_tick_layout()
    fig.update_xaxes(title_text="Half-Normal Quantiles", row=1, col=1)
    fig.update_yaxes(title_text="|Effect|", row=1, col=1)
    fig.update_xaxes(
        title_text="|Effect|",
        row=1,
        col=2,
        rangemode="tozero",
    )
    fig.update_yaxes(
        title_text="Half-Normal Probability (%)",
        row=1,
        col=2,
        range=[0, stats.norm.ppf((0.99 + 1) / 2)],
        **prob_tick,
    )
    fig.update_layout(
        height=450,
        showlegend=False,
        hovermode="closest",
    )

    if mode == "probability":
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return apply_plot_style(fig)

    # Side-by-side: default visibility + toggle buttons.
    n_traces = len(fig.data)
    visibility = {
        "classical": [p == "c1" for p in trace_panels],
        "probability": [p == "c2" for p in trace_panels],
        "side_by_side": [True] * n_traces,
    }
    fig.update_layout(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        xaxis2=dict(visible=True),
    )
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.0,
                y=1.18,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="Classical",
                        method="update",
                        args=[
                            {"visible": visibility["classical"]},
                            {
                                "xaxis.visible": True,
                                "yaxis.visible": True,
                                "xaxis2.visible": False,
                                "legend.visible": False,
                            },
                        ],
                    ),
                    dict(
                        label="Probability",
                        method="update",
                        args=[
                            {"visible": visibility["probability"]},
                            {
                                "xaxis.visible": False,
                                "yaxis.visible": False,
                                "xaxis2.visible": True,
                                "legend.visible": False,
                            },
                        ],
                    ),
                    dict(
                        label="Side-by-Side",
                        method="update",
                        args=[
                            {"visible": visibility["side_by_side"]},
                            {
                                "xaxis.visible": True,
                                "yaxis.visible": True,
                                "xaxis2.visible": True,
                                "legend.visible": False,
                            },
                        ],
                    ),
                ],
            )
        ]
    )

    return apply_plot_style(fig)


# ==================== DIAGNOSTIC PLOTS ====================


def create_qq_plot(residuals: np.ndarray) -> go.Figure:
    """
    Create Q-Q normal probability plot.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals

    Returns
    -------
    go.Figure
        Q-Q plot with reference line

    Notes
    -----
    Q-Q plot assesses normality of residuals:
    - Points should follow diagonal line if normally distributed
    - Deviations indicate:
        - S-curve: Heavy or light tails
        - Bow: Skewness
        - Isolated points: Outliers

    Uses proper plotting positions: (i - 0.5) / n

    Examples
    --------
    >>> residuals = np.random.normal(0, 1, 100)
    >>> fig = create_qq_plot(residuals)
    """
    # Properly calculate theoretical quantiles using plotting positions
    n = len(residuals)
    # Use plotting position: (i - 0.5) / n
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probabilities)
    sample = np.sort(residuals)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sample,
            mode="markers",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            name="Data",
            hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>",
        )
    )

    min_val = min(theoretical.min(), sample.min())
    max_val = max(theoretical.max(), sample.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=350,
        showlegend=False,
    )

    return apply_plot_style(fig)


def create_residual_vs_run_order_plot(
    residuals: np.ndarray,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create residuals vs run order plot.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals
    response_units : Optional[str]
        Units for the residuals axis (e.g., "kg"). If None, no units shown.

    Returns
    -------
    go.Figure
        Time series plot of residuals

    Notes
    -----
    Detects time-dependent patterns:
    - Trends: Drift in process over time
    - Cycles: Periodic effects
    - Clusters: Batch effects or systematic changes

    Random scatter around zero indicates no time-dependent issues.

    Examples
    --------
    >>> residuals = np.random.normal(0, 1, 50)
    >>> fig = create_residual_vs_run_order_plot(residuals)
    """
    run_order = np.arange(1, len(residuals) + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=run_order,
            y=residuals,
            mode="markers+lines",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            line=dict(color=PLOT_COLORS["primary"], width=1, dash="dot"),
            hovertemplate="Run %{x}<br>Residual: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0, line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2)
    )
    residuals_label = _label_with_units("Residuals", response_units)
    fig.update_layout(
        xaxis_title="Run Order", yaxis_title=residuals_label, height=350
    )

    return apply_plot_style(fig)


def create_residual_vs_factor_plot(
    factor_values: np.ndarray,
    residuals: np.ndarray,
    factor_name: str,
    factor_units: Optional[str] = None,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create residuals vs factor plot.

    Parameters
    ----------
    factor_values : np.ndarray
        Values of the factor
    residuals : np.ndarray
        Model residuals
    factor_name : str
        Name of the factor for axis label
    factor_units : Optional[str]
        Units for the factor axis (e.g., "°C"). If None, no units shown.
    response_units : Optional[str]
        Units for the residuals axis (e.g., "kg"). If None, no units shown.

    Returns
    -------
    go.Figure
        Scatter plot of residuals vs factor levels

    Notes
    -----
    Checks for:
    - Patterns suggesting missing terms (curvature, trends)
    - Non-constant variance (funnel shape)
    - Outliers at specific factor levels

    Should show random scatter around zero.

    Examples
    --------
    >>> factor_values = np.array([1, 2, 3, 1, 2, 3])
    >>> residuals = np.array([0.1, -0.2, 0.1, -0.1, 0.2, 0.0])
    >>> fig = create_residual_vs_factor_plot(factor_values, residuals, 'Temperature')
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=factor_values,
            y=residuals,
            mode="markers",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate=f"{factor_name}: %{{x}}<br>Residual: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0, line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2)
    )
    x_label = _label_with_units(factor_name, factor_units)
    residuals_label = _label_with_units("Residuals", response_units)
    fig.update_layout(
        xaxis_title=x_label, yaxis_title=residuals_label, height=250
    )

    return apply_plot_style(fig)


# ==================== PROFILER PLOTS ====================


def create_response_trace_plot(
    factor_range: np.ndarray,
    predictions: np.ndarray,
    current_value: float,
    current_prediction: float,
    factor_name: str,
    response_name: str,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    factor_units: Optional[str] = None,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create response trace plot for prediction profiler.

    Parameters
    ----------
    factor_range : np.ndarray
        Range of factor values to plot
    predictions : np.ndarray
        Predicted response at each factor value
    current_value : float
        Current setting of the factor (for vertical line)
    current_prediction : float
        Current predicted response (for marker)
    factor_name : str
        Name of the factor
    response_name : str
        Name of the response
    ci_lower : Optional[np.ndarray]
        Lower 95% CI boundary (if available)
    ci_upper : Optional[np.ndarray]
        Upper 95% CI boundary (if available)
    factor_units : Optional[str]
        Units for the factor x-axis (e.g., "°C"). If None, no units shown.
    response_units : Optional[str]
        Units for the response y-axis (e.g., "kg"). If None, no units shown.

    Returns
    -------
    go.Figure
        Response trace with current setting marked

    Notes
    -----
    Shows how response changes as one factor varies while others are held constant.
    Current setting marked with vertical line and point.

    Examples
    --------
    >>> factor_range = np.linspace(10, 30, 50)
    >>> predictions = 2.0 + 0.5 * factor_range
    >>> fig = create_response_trace_plot(
    ...     factor_range, predictions, 20.0, 12.0, 'Temp', 'Yield'
    ... )
    """
    fig = go.Figure()

    # 95% CI band (if available)
    if ci_lower is not None and ci_upper is not None:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([factor_range, factor_range[::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.2)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Response trace line
    fig.add_trace(
        go.Scatter(
            x=factor_range,
            y=predictions,
            mode="lines",
            line=dict(color=PLOT_COLORS["primary"], width=2),
            hovertemplate=f"{factor_name}: %{{x:.3f}}<br>{response_name}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        )
    )

    # Current setting - vertical line
    fig.add_vline(
        x=current_value,
        line=dict(color="red", dash="dash", width=2),
        annotation=dict(
            text=f"{current_value:.2f}",
            yref="paper",
            y=1.05,
            showarrow=False,
            font=dict(size=10, color="red"),
        ),
    )

    # Current prediction point
    fig.add_trace(
        go.Scatter(
            x=[current_value],
            y=[current_prediction],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),
            showlegend=False,
            hovertemplate=f"{factor_name}: {current_value:.3f}<br>{response_name}: {current_prediction:.3f}<extra></extra>",
        )
    )

    response_label = _label_with_units(response_name, response_units)
    fig.update_layout(
        height=200,
        margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title=None,
        yaxis_title=response_label,
        showlegend=False,
    )

    min_val = factor_range.min()
    max_val = factor_range.max()
    fig.update_xaxes(
        range=[min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val)]
    )

    return apply_plot_style(fig)


def create_categorical_response_plot(
    levels: List,
    predictions: np.ndarray,
    current_level,
    factor_name: str,
    response_name: str,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create bar chart for categorical factor in profiler.

    Parameters
    ----------
    levels : List
        Categorical levels of the factor
    predictions : np.ndarray
        Predicted response at each level
    current_level
        Currently selected level (highlighted in red)
    factor_name : str
        Name of the factor
    response_name : str
        Name of the response
    response_units : Optional[str]
        Units for the response y-axis (e.g., "kg"). If None, no units shown.

    Returns
    -------
    go.Figure
        Bar chart with current level highlighted

    Examples
    --------
    >>> levels = ['Low', 'Medium', 'High']
    >>> predictions = np.array([10.0, 15.0, 12.0])
    >>> fig = create_categorical_response_plot(
    ...     levels, predictions, 'Medium', 'Material', 'Strength'
    ... )
    """
    # Determine which bar is current
    current_idx = levels.index(current_level)

    # Color bars (current one red, others blue)
    colors = [
        PLOT_COLORS["danger"] if i == current_idx else PLOT_COLORS["primary"]
        for i in range(len(levels))
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[str(level) for level in levels],
            y=predictions,
            marker=dict(color=colors, line=dict(color="#000000", width=1)),
            hovertemplate=f"{factor_name}: %{{x}}<br>{response_name}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        )
    )

    response_label = _label_with_units(response_name, response_units)
    fig.update_layout(
        height=200,
        margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title=None,
        yaxis_title=response_label,
        showlegend=False,
    )

    return apply_plot_style(fig)


def create_contour_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_mesh: np.ndarray,
    x_factor_name: str,
    y_factor_name: str,
    response_name: str,
    x_factor_units: Optional[str] = None,
    y_factor_units: Optional[str] = None,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create 2D contour plot of response surface.

    Parameters
    ----------
    x_grid : np.ndarray
        X-axis grid values
    y_grid : np.ndarray
        Y-axis grid values
    z_mesh : np.ndarray
        Response values on mesh (2D array)
    x_factor_name : str
        Name of X-axis factor
    y_factor_name : str
        Name of Y-axis factor
    response_name : str
        Name of response
    x_factor_units : Optional[str]
        Units for the X-axis factor (e.g., "°C"). If None, no units shown.
    y_factor_units : Optional[str]
        Units for the Y-axis factor (e.g., "psi"). If None, no units shown.
    response_units : Optional[str]
        Units for the colorbar label (e.g., "kg"). If None, no units shown.

    Returns
    -------
    go.Figure
        Contour plot with color scale and labeled contours

    Notes
    -----
    Visualizes response surface for two factors while holding others constant.
    Useful for finding optimal regions and understanding interactions.

    Examples
    --------
    >>> x_grid = np.linspace(0, 10, 50)
    >>> y_grid = np.linspace(0, 10, 50)
    >>> X, Y = np.meshgrid(x_grid, y_grid)
    >>> Z = X**2 + Y**2
    >>> fig = create_contour_plot(x_grid, y_grid, Z, 'A', 'B', 'Response')
    """
    fig = go.Figure()

    # Add contour
    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=z_mesh,
            colorscale="RdYlGn",
            colorbar=dict(title=response_name),
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=10, color="white"),
            ),
            hovertemplate=(
                f"{x_factor_name}: %{{x:.2f}}<br>"
                f"{y_factor_name}: %{{y:.2f}}<br>"
                f"{response_name}: %{{z:.2f}}<extra></extra>"
            ),
        )
    )

    x_label = _label_with_units(x_factor_name, x_factor_units)
    y_label = _label_with_units(y_factor_name, y_factor_units)
    response_label = _label_with_units(response_name, response_units)
    fig.data[0].colorbar.title = response_label
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        showlegend=True,
    )

    return apply_plot_style(fig)


def create_3d_surface_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_mesh: np.ndarray,
    x_factor_name: str,
    y_factor_name: str,
    response_name: str,
    x_factor_units: Optional[str] = None,
    y_factor_units: Optional[str] = None,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create 3D surface plot of response surface.

    Parameters
    ----------
    x_grid : np.ndarray
        X-axis grid values
    y_grid : np.ndarray
        Y-axis grid values
    z_mesh : np.ndarray
        Response values on mesh (2D array)
    x_factor_name : str
        Name of X-axis factor
    y_factor_name : str
        Name of Y-axis factor
    response_name : str
        Name of response
    x_factor_units : Optional[str]
        Units for the X-axis factor (e.g., "°C"). If None, no units shown.
    y_factor_units : Optional[str]
        Units for the Y-axis factor (e.g., "psi"). If None, no units shown.
    response_units : Optional[str]
        Units for the Z-axis and colorbar label (e.g., "kg"). If None, no units shown.

    Returns
    -------
    go.Figure
        3D surface plot with color scale

    Notes
    -----
    Interactive 3D visualization of response surface.
    Users can rotate to see features from different angles.

    Examples
    --------
    >>> x_grid = np.linspace(0, 10, 50)
    >>> y_grid = np.linspace(0, 10, 50)
    >>> X, Y = np.meshgrid(x_grid, y_grid)
    >>> Z = X**2 + Y**2
    >>> fig = create_3d_surface_plot(x_grid, y_grid, Z, 'A', 'B', 'Response')
    """
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_mesh,
            colorscale="RdYlGn",
            colorbar=dict(title=response_name),
            hovertemplate=(
                f"{x_factor_name}: %{{x:.2f}}<br>"
                f"{y_factor_name}: %{{y:.2f}}<br>"
                f"{response_name}: %{{z:.2f}}<extra></extra>"
            ),
        )
    )

    x_label = _label_with_units(x_factor_name, x_factor_units)
    y_label = _label_with_units(y_factor_name, y_factor_units)
    response_label = _label_with_units(response_name, response_units)
    fig.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=response_label,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        height=600,
    )

    return apply_plot_style(fig)

