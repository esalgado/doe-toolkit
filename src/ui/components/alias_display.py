"""
Alias and correlation display component.

Provides two display modes:

1. **Fractional factorial** — exact aliasing from the defining relation,
   displayed as a heatmap of |correlation| between model-matrix columns
   plus a table of aliased-effect pairs.

2. **All other designs** (D-optimal, RSM, full factorial, LHS) — partial
   aliasing shown as the absolute column-correlation matrix of the model
   matrix, with pairs above the 0.5 threshold flagged in the table.

The heatmap uses a continuous blue (|r|=0) to red (|r|=1) colour scale.
Diagonal cells (self-correlation = 1) are excluded from the flagged-pair
table to avoid noise.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.factors import Factor
from src.core.diagnostics.variance import build_model_matrix
from src.ui.components.model_builder import format_term_for_display

# Threshold above which a pair is flagged as partially aliased
_ALIAS_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _correlation_matrix(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the absolute column-correlation matrix of the model matrix.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms to include (intercept excluded from display).

    Returns
    -------
    corr_abs : np.ndarray
        Shape (p, p) absolute correlation matrix (diagonal = 1).
    col_names : List[str]
        Column labels corresponding to rows/cols of corr_abs.
    """
    display_terms = [t for t in model_terms if t != "1"]
    if not display_terms:
        return np.array([[]]), []

    # Build model matrix including intercept so categorical expansion works,
    # then drop the intercept column before correlating.
    X, raw_names = build_model_matrix(design, factors, model_terms)

    non_intercept_idx = [i for i, n in enumerate(raw_names) if n != "Intercept"]
    X = X[:, non_intercept_idx]
    col_names = [raw_names[i] for i in non_intercept_idx]

    if X.shape[1] == 0:
        return np.array([[]]), []

    # Standardise columns before correlating
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # avoid divide-by-zero for constant columns
    X_std = (X - X.mean(axis=0)) / stds

    corr = (X_std.T @ X_std) / X.shape[0]
    corr_abs = np.abs(corr)

    return corr_abs, col_names


def _make_heatmap(
    corr_abs: np.ndarray,
    col_names: List[str],
    title: str = "Term Correlation (|r|)",
) -> go.Figure:
    """
    Build a Plotly heatmap with blue to red continuous colour scale.

    Parameters
    ----------
    corr_abs : np.ndarray
        Absolute correlation matrix.
    col_names : List[str]
        Axis labels.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    display_names = [format_term_for_display(n) for n in col_names]
    z_rounded = np.round(corr_abs, 3)

    fig = go.Figure(
        go.Heatmap(
            z=z_rounded,
            x=display_names,
            y=display_names,
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#1a6faf"],   # blue  — no correlation
                [0.5, "#d4d4d4"],   # grey  — moderate
                [1.0, "#c0392b"],   # red   — full aliasing
            ],
            colorbar=dict(
                title="|r|",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0", "0.25", "0.5*", "0.75", "1.0"],
                len=0.8,
            ),
            hovertemplate="%{y} \u00d7 %{x}<br>|r| = %{z}<extra></extra>",
            text=z_rounded,
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )

    n = len(col_names)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        height=max(300, 60 + n * 35),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
    )

    return fig


def _flagged_pairs_table(
    corr_abs: np.ndarray,
    col_names: List[str],
    threshold: float = _ALIAS_THRESHOLD,
) -> pd.DataFrame:
    """
    Build a table of term pairs with |r| >= threshold, excluding diagonal.

    Parameters
    ----------
    corr_abs : np.ndarray
        Absolute correlation matrix.
    col_names : List[str]
        Column labels.
    threshold : float
        Minimum |r| to include.

    Returns
    -------
    pd.DataFrame
        Columns: Term A, Term B, |r|, Flag.
        Sorted descending by |r|.
    """
    rows = []
    n = len(col_names)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_abs[i, j]
            if r >= threshold:
                flag = "\u26a0\ufe0f High" if r >= 0.8 else "* Moderate"
                rows.append(
                    {
                        "Term A": format_term_for_display(col_names[i]),
                        "Term B": format_term_for_display(col_names[j]),
                        "|r|": round(float(r), 4),
                        "Flag": flag,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["Term A", "Term B", "|r|", "Flag"])

    return (
        pd.DataFrame(rows)
        .sort_values("|r|", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def display_alias_correlation(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    design_type: Optional[str] = None,
    alias_structure: Optional[Dict] = None,
    resolution: Optional[int] = None,
    section_title: str = "\ud83d\udd17 Alias / Correlation Structure",
    expanded: bool = False,
) -> None:
    """
    Render alias / partial-aliasing information inside a Streamlit expander.

    For every design type the absolute column-correlation matrix of the
    model matrix is computed and shown as a heatmap plus a flagged-pairs
    table.  For fractional factorial designs an additional exact-aliasing
    section is shown above the heatmap using the stored alias structure.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Model terms to analyse (intercept ``'1'`` is handled internally).
    design_type : str, optional
        Design type string (e.g. ``'Fractional Factorial'``).
    alias_structure : dict, optional
        Pre-computed alias structure (algebraic) from ``AliasingEngine`` or
        ``FractionalFactorial.alias_structure``.  Only relevant when
        ``design_type == 'Fractional Factorial'``.
    resolution : int, optional
        Design resolution.  Only relevant for fractional factorial.
    section_title : str
        Title shown on the expander.
    expanded : bool
        Whether the expander starts open.

    Returns
    -------
    None
        All output is rendered directly in Streamlit.
    """
    display_terms = [t for t in model_terms if t != "1"]
    if not display_terms:
        return

    with st.expander(section_title, expanded=expanded):

        # --- Fractional factorial: exact aliasing section -----------------
        is_fractional = (design_type or "").lower().startswith("fractional")
        if is_fractional and alias_structure:
            _render_exact_aliasing(alias_structure, factors, resolution)
            st.divider()

        # --- Correlation heatmap (all design types) -----------------------
        try:
            corr_abs, col_names = _correlation_matrix(design, factors, model_terms)
        except Exception as e:
            st.warning(f"Could not compute correlation matrix: {e}")
            return

        if corr_abs.size == 0 or len(col_names) == 0:
            st.info("No non-intercept terms to correlate.")
            return

        subheader = (
            "Exact + Partial Aliasing (Correlation Matrix)"
            if is_fractional
            else "Term Correlation Matrix"
        )
        st.markdown(f"**{subheader}**")
        st.caption(
            "Colour encodes |r| between model-matrix columns: "
            "blue = 0 (orthogonal), red = 1 (fully aliased). "
            "Pairs with |r| \u2265 0.5 are flagged below."
        )

        fig = _make_heatmap(corr_abs, col_names)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # --- Flagged pairs table ------------------------------------------
        flagged = _flagged_pairs_table(corr_abs, col_names)

        if flagged.empty:
            st.success(
                "\u2705 No term pairs exceed |r| = 0.5 \u2014 "
                "design is well orthogonalised."
            )
        else:
            st.markdown(
                f"**Flagged Pairs (|r| \u2265 0.5)** \u2014 {len(flagged)} pair(s)"
            )
            st.dataframe(flagged, hide_index=True, use_container_width=True)

            high_count = int((flagged["|r|"] >= 0.8).sum())
            if high_count:
                st.warning(
                    f"\u26a0\ufe0f {high_count} pair(s) with |r| \u2265 0.8 \u2014 "
                    "these terms are highly correlated. Consider removing "
                    "redundant terms or augmenting the design."
                )


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------


def _render_exact_aliasing(
    alias_structure: Dict,
    factors: List[Factor],
    resolution: Optional[int],
) -> None:
    """
    Render exact alias chains for a fractional factorial design.

    Uses real factor names when the alias structure is in algebraic form
    (single uppercase letters) and a FactorMapper is constructable.

    Parameters
    ----------
    alias_structure : dict
        Mapping effect -> list of aliased effects (algebraic symbols).
    factors : List[Factor]
        Factor definitions (used to translate algebraic to real names).
    resolution : int, optional
        Design resolution.
    """
    from src.core.aliasing import FactorMapper, _translate_algebraic_term

    st.markdown("**Exact Aliasing (Defining Relation)**")

    res_label = f"Resolution {resolution}" if resolution else "Unknown Resolution"
    if resolution and resolution <= 3:
        res_color = "#5c1a1a"
        res_icon = "\U0001f534"
    elif resolution and resolution == 4:
        res_color = "#4a3800"
        res_icon = "\U0001f7e1"
    else:
        res_color = "#0d3320"
        res_icon = "\U0001f7e2"

    st.markdown(
        f'<div style="background:{res_color};padding:6px 12px;border-radius:4px;'
        f'font-size:0.88rem;margin-bottom:8px;">'
        f"{res_icon} <b>{res_label}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Try to translate algebraic symbols to real names
    try:
        mapper = FactorMapper(factors)
        use_real = True
    except Exception:
        use_real = False

    def _fmt(term: str) -> str:
        if not use_real:
            return term
        try:
            return _translate_algebraic_term(term, mapper)
        except Exception:
            return term

    # Separate critical (ME aliased with 2FI) from others
    critical_rows = []
    other_rows = []

    for effect, aliases in sorted(
        alias_structure.items(), key=lambda x: (len(x[0]), x[0])
    ):
        if not aliases:
            continue
        effect_display = _fmt(effect)
        aliases_display = " = ".join(_fmt(a) for a in aliases)
        row_str = f"**{effect_display}** = {aliases_display}"

        is_critical = len(effect) == 1 and any(len(a) == 2 for a in aliases)
        if is_critical:
            critical_rows.append(row_str)
        else:
            other_rows.append(row_str)

    if critical_rows:
        st.error("**Critical: Main effects aliased with 2-factor interactions**")
        for row in critical_rows:
            st.markdown(f"- {row}")

    if other_rows:
        with st.expander("Other alias chains", expanded=False):
            for row in other_rows:
                st.markdown(f"- {row}")

    if not critical_rows and not other_rows:
        st.success("\u2705 No aliasing detected at this resolution.")
