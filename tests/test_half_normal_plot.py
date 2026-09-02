"""
Tests for the half-normal plot component.

The half-normal plot core lives in ``src.ui.utils.plotting`` as a pure,
testable function (:func:`create_half_normal_plot`) so these tests exercise
the figure-building logic directly without a Streamlit harness, matching the
convention used by the other pure plotting helpers.

The function supports three display modes:
- ``classical``    : |Effect| vs half-normal quantiles (legacy behaviour)
- ``probability``  : |Effect| vs half-normal probability % (Design-Expert style)
- ``side_by_side`` (default): both panels in a dual subplot figure with a
  button row to collapse the view.

Edge cases covered:
- classical mode matches the pre-existing single-panel figure
- side-by-side layout with two subplots and shared colouring
- probability-mode axis, tick labels and panel layout
- reference-line exceedance significance colouring
- p-value based significance colouring (``p < alpha``)
- hover tooltip fields (name, signed effect, absolute effect, rank, probability)
- view toggle buttons / visibility
- degenerate inputs: 1, 2 and 0 effects; invalid modes; length mismatch
- large effect sets
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.ui.utils.plotting import (
    create_half_normal_plot,
    PLOT_COLORS,
    HalfNormalSeries,
    consolidate_half_normal_effects,
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.full_factorial import full_factorial
from src.core.analysis import ANOVAAnalysis

EFFECTS = np.array([0.1, -0.5, 2.0, 0.3, -0.8, 1.5])
NAMES = ["A", "B", "C", "D", "E", "F"]


class TestClassicalMode:
    """The classical mode must reproduce the legacy single-panel figure."""

    def test_two_traces_for_three_plus_effects(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="classical")
        assert len(fig.data) == 2  # markers + reference line

    def test_scatter_then_ref_line_order(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="classical")
        assert fig.data[0].type == "scatter"
        assert fig.data[0].mode == "markers+text"
        assert fig.data[1].mode == "lines"
        assert fig.data[1].line.dash == "dash"

    def test_single_panel_axes(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="classical")
        assert fig.layout.xaxis is not None
        assert fig.layout.yaxis is not None
        assert fig.layout.height == 400
        assert fig.layout.xaxis.title.text == "Half-Normal Quantiles"
        assert fig.layout.yaxis.title.text == "|Effect|"

    def test_three_effects_no_ref_line(self):
        """Reference line requires >2 effects (as in the legacy code)."""
        fig = create_half_normal_plot(
            np.array([0.1, 0.5, 2.0]), ["A", "B", "C"], mode="classical"
        )
        assert len(fig.data) == 2  # still drawn (n>2)

    def test_two_effects_no_ref_line(self):
        fig = create_half_normal_plot(
            np.array([0.5, 1.0]), ["x", "y"], mode="classical"
        )
        assert len(fig.data) == 1  # markers only

    def test_one_effect(self):
        fig = create_half_normal_plot(
            np.array([-3.0]), ["x"], mode="classical"
        )
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 1


class TestSideBySideMode:
    def test_two_subplots_four_traces(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="side_by_side")
        assert len(fig.data) == 4  # scatter, scatter, line, line
        assert fig.data[0].xaxis == "x"
        assert fig.data[0].yaxis == "y"
        assert fig.data[1].xaxis == "x2"
        assert fig.data[1].yaxis == "y2"
        assert fig.data[2].xaxis == "x"
        assert fig.data[3].xaxis == "x2"

    def test_panel_titles(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="side_by_side")
        assert fig.layout.annotations[0].text == "Half-Normal"
        assert fig.layout.annotations[1].text == "Half-Normal Probability"

    def test_both_panels_same_marker_colours(self):
        """Shared significance colouring: both panels carry identical points."""
        fig = create_half_normal_plot(
            EFFECTS,
            NAMES,
            mode="side_by_side",
            p_values=np.array([0.1, 0.02, 0.001, 0.5, 0.9, 0.05]),
        )
        left = np.asarray(fig.data[0].marker.color).tolist()
        right = np.asarray(fig.data[1].marker.color).tolist()
        assert left == right

    def test_height_450(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="side_by_side")
        assert fig.layout.height == 450

    def test_many_effects_sorted(self):
        rng = np.random.default_rng(42)
        effects = rng.normal(size=20)
        names = [f"E{i}" for i in range(20)]
        fig = create_half_normal_plot(effects, names, mode="side_by_side")
        assert len(fig.data) == 4
        abs_points = np.abs(np.asarray(fig.data[0].y))
        assert all(np.diff(abs_points) >= 0)  # sorted ascending


class TestProbabilityMode:
    def test_single_panel_single_axis(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="probability")
        # right panel shown, left panel hidden
        assert fig.layout.xaxis.visible is False
        assert fig.layout.yaxis.visible is False
        assert fig.data[0].xaxis == "x"  # left panel, hidden
        assert fig.data[1].xaxis == "x2"  # right panel, shown
        assert fig.data[1].yaxis == "y2"

    def test_probability_tick_labels(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="probability")
        expected = ["1%", "5%", "10%", "20%", "30%", "50%",
                    "70%", "80%", "90%", "95%", "99%"]
        assert list(fig.layout.yaxis2.ticktext) == expected

    def test_probability_tick_positions_half_normal(self):
        """Ticks sit at half-normal positions (positive |z|), not signed z."""
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="probability")
        ticks = np.asarray(fig.layout.yaxis2.tickvals, dtype=float)
        probs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50,
                 0.70, 0.80, 0.90, 0.95, 0.99]
        expected = stats.norm.ppf((np.asarray(probs) + 1) / 2)
        assert np.all(ticks >= 0)
        assert np.allclose(ticks, expected)

    def test_axis_titles(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="probability")
        assert fig.layout.xaxis2.title.text == "|Effect|"
        assert fig.layout.yaxis2.title.text == "Half-Normal Probability (%)"

    def test_x_axis_is_absolute_effect(self):
        """|Effect| (0..max) on x, never a signed/symmetric range."""
        fig = create_half_normal_plot(
            np.array([0.31, -0.48, 1.02, -2.15, 0.67, -1.44]),
            ["A", "B", "C", "D", "E", "F"],
            mode="probability",
        )
        x = np.asarray(fig.data[1].x, dtype=float)
        assert np.all(x >= 0)
        assert np.isclose(x.max(), 2.15)
        assert np.array_equal(x, np.sort(np.abs([0.31, -0.48, 1.02, -2.15, 0.67, -1.44])))

    def test_y_axis_is_half_normal_score(self):
        """y positions are |z| (half-normal scores), all non-negative."""
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="probability")
        y = np.asarray(fig.data[1].y, dtype=float)
        n = len(EFFECTS)
        expected = np.abs(stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n))
        assert np.all(y >= 0)
        assert np.allclose(y, expected)

    def test_y2_range_starts_at_zero_ends_at_99pct(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="probability")
        assert fig.layout.yaxis2.range[0] == 0
        assert np.isclose(fig.layout.yaxis2.range[1], stats.norm.ppf(0.995))

    def test_x2_rangemode_tozero(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="side_by_side")
        assert fig.layout.xaxis2.rangemode == "tozero"


class TestSignificanceColouring:
    def test_above_ref_line_is_significant(self):
        """A single dominating effect deviates from the reference line."""
        fig = create_half_normal_plot(
            np.array([0.1, 0.1, 0.1, 5.0]),
            ["a", "b", "c", "d"],
            mode="classical",
        )
        colors = np.asarray(fig.data[0].marker.color).tolist()
        assert colors[-1] == PLOT_COLORS["danger"]
        assert set(colors[:-1]) == {PLOT_COLORS["primary"]}

    def test_all_small_effects_insignificant(self):
        fig = create_half_normal_plot(
            np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            ["a", "b", "c", "d", "e"],
            mode="classical",
        )
        colors = np.asarray(fig.data[0].marker.color).tolist()
        assert list(set(colors)) == [PLOT_COLORS["primary"]]

    def test_p_value_colouring(self):
        fig = create_half_normal_plot(
            np.array([0.05, 0.05, 0.05]),
            ["a", "b", "c"],
            mode="classical",
            p_values=np.array([0.001, 0.01, 0.5]),
            alpha=0.05,
        )
        colors = np.asarray(fig.data[0].marker.color).tolist()
        assert colors[0] == PLOT_COLORS["danger"]
        assert colors[1] == PLOT_COLORS["danger"]
        assert colors[2] == PLOT_COLORS["primary"]

    def test_p_value_colouring_not_enabled_without_p_values(self):
        """Without p-values the reference-line criterion alone is used."""
        fig = create_half_normal_plot(
            np.array([0.05, 0.05, 0.05, 0.05]),
            ["a", "b", "c", "d"],
            mode="classical",
        )
        colors = np.asarray(fig.data[0].marker.color).tolist()
        assert list(set(colors)) == [PLOT_COLORS["primary"]]


class TestHoverInfo:
    def test_effect_names_in_hover(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="classical")
        assert "%{text}" in fig.data[0].hovertemplate

    def test_signed_effect_rank_probability_in_hover(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="side_by_side")
        cd = fig.data[0].customdata
        assert cd.shape == (len(EFFECTS), 4)
        # signed effects retain their original signs (sorted by |effect|)
        assert np.allclose(cd[:, 0], EFFECTS[np.argsort(np.abs(EFFECTS))])
        # absolute effects column matches sorted |effect|
        assert np.allclose(cd[:, 1], np.sort(np.abs(EFFECTS)))
        # ranks are 1..n
        assert np.allclose(cd[:, 2], np.arange(1, len(EFFECTS) + 1))
        # probabilities are percentages
        assert np.allclose(
            cd[:, 3], ((np.arange(1, 7) - 0.5) / 6) * 100
        )

    def test_hover_template_fields(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="classical")
        tpl = fig.data[0].hovertemplate
        for frag in ("Effect:", "|Effect|:", "Rank:", "Probability:"):
            assert frag in tpl


class TestToggleButtons:
    def test_three_buttons_present(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="side_by_side")
        assert len(fig.layout.updatemenus) == 1
        buttons = fig.layout.updatemenus[0].buttons
        assert [b.label for b in buttons] == [
            "Classical", "Probability", "Side-by-Side"
        ]

    def test_button_visibility_vectors(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="side_by_side")
        vis = [b.args[0]["visible"] for b in fig.layout.updatemenus[0].buttons]
        # trace order: [classical scatter, probability scatter,
        #               classical line, probability line]
        assert vis == [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, True, True],
        ]

    def test_default_all_visible(self):
        fig = create_half_normal_plot(EFFECTS, NAMES, mode="side_by_side")
        assert fig.layout.xaxis.visible is not False
        assert fig.layout.xaxis2.visible is not False


class TestInputValidation:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            create_half_normal_plot(EFFECTS, NAMES, mode="bogus")

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            create_half_normal_plot(np.array([0.1, 0.2]), ["a"])

    def test_p_value_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="p_values"):
            create_half_normal_plot(
                np.array([0.1, 0.2, 0.3]),
                ["a", "b", "c"],
                p_values=np.array([0.1, 0.2]),
            )

    def test_empty_effects(self):
        fig = create_half_normal_plot(
            np.array([]), [], mode="side_by_side"
        )
        assert len(fig.data) == 2  # one (empty) scatter per panel, no ref line
        assert fig.data[0].xaxis == "x"
        assert fig.data[1].xaxis == "x2"

    def test_default_mode_is_side_by_side(self):
        fig = create_half_normal_plot(EFFECTS, NAMES)
        assert fig.layout.xaxis2 is not None


class TestConsolidation:
    """Design-Expert style consolidation of multi-df terms."""

    @staticmethod
    def dummy_effects():
        return pd.DataFrame(
            {
                "Coefficient": [1.0, -1.2, 2.4, -0.6, 0.5, -3.0, 0.9, -1.4],
                "Std_Error": [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4],
            },
            index=[
                "Egg_lot[T.41007587]",
                "Egg_lot[T.41005191]",
                "Egg_lot[T.41007741]",
                "Block[T.2]",
                "Egg_percent",
                "Egg_lot[T.41007587]:Egg_percent",
                "Egg_lot[T.41005191]:Egg_percent",
                "Egg_lot[T.41007741]:Egg_percent",
            ],
        )

    @staticmethod
    def dummy_anova():
        return pd.DataFrame(
            {
                "df": [3.0, 1.0, 1.0, 3.0],
                "sum_sq": [4.5, 0.3, 2.0, 9.0],
                "PR(>F)": [0.02, 0.9, 0.001, 0.05],
            },
            index=["Egg_lot", "Block", "Egg_percent", "Egg_lot:Egg_percent"],
        )

    def test_groups_multi_df_term(self):
        series = consolidate_half_normal_effects(
            self.dummy_effects(), self.dummy_anova()
        )
        assert series.names == ["Egg_lot", "Egg_percent", "Egg_lot:Egg_percent"]
        assert list(series.dfs) == [3, 1, 3]

    def test_drops_nuisance_block(self):
        series = consolidate_half_normal_effects(
            self.dummy_effects(), self.dummy_anova()
        )
        assert "Block" not in series.names

    def test_single_df_magnitude_is_abs_coef(self):
        series = consolidate_half_normal_effects(
            self.dummy_effects(), self.dummy_anova()
        )
        i = series.names.index("Egg_percent")
        assert np.isclose(series.magnitudes[i], 0.5)
        assert np.isclose(series.signed[i], 0.5)

    def test_pooled_magnitude_whitcomb_rescale(self):
        series = consolidate_half_normal_effects(
            self.dummy_effects(), self.dummy_anova()
        )
        # sigma2 = mean(SE^2) over df=1 rows = 0.4^2
        sigma2 = 0.4 ** 2
        for base, df, ss, p_expected_name in [
            ("Egg_lot", 3, 4.5, "Egg_lot"),
            ("Egg_lot:Egg_percent", 3, 9.0, "Egg_lot:Egg_percent"),
        ]:
            i = series.names.index(p_expected_name)
            p_tilde = stats.chi2.sf(float(ss) / sigma2, df)
            expected = np.sqrt(sigma2) * abs(stats.norm.ppf(p_tilde / 2))
            assert np.isclose(series.magnitudes[i], expected)
            assert np.isnan(series.signed[i])

    def test_term_p_values_from_anova(self):
        series = consolidate_half_normal_effects(
            self.dummy_effects(), self.dummy_anova()
        )
        assert np.isclose(series.p_values[0], 0.02)
        assert np.isclose(series.p_values[1], 0.001)
        assert np.isclose(series.p_values[2], 0.05)

    def test_star_form_anova_lookup(self):
        anova = self.dummy_anova()
        anova = anova.rename(index={"Egg_lot:Egg_percent": "Egg_lot*Egg_percent"})
        series = consolidate_half_normal_effects(self.dummy_effects(), anova)
        assert series.names == ["Egg_lot", "Egg_percent", "Egg_lot:Egg_percent"]
        assert np.isclose(series.p_values[2], 0.05)
        assert np.all(np.isfinite(series.magnitudes))

    def test_name_transform_applied(self):
        series = consolidate_half_normal_effects(
            self.dummy_effects(),
            self.dummy_anova(),
            name_transform=lambda t: t.replace(":", "x"),
        )
        assert series.names == ["Egg_lot", "Egg_percent", "Egg_lotxEgg_percent"]


class TestConsolidatedFigure:
    """The probability panel honours a consolidated series."""

    @staticmethod
    def de_series():
        return HalfNormalSeries(
            names=["Egg_lot", "Egg_percent", "Egg_lot:Egg_percent"],
            magnitudes=np.array([1.2, 0.5, 0.7]),
            signed=np.array([np.nan, 0.5, np.nan]),
            p_values=np.array([0.02, 0.001, 0.05]),
            dfs=np.array([3, 1, 3]),
        )

    RAW_EFFECTS = np.array([1.0, -1.2, 2.4, -0.6, 0.5, -3.0])
    RAW_NAMES = ["A", "B", "C", "D", "Egg_percent", "E"]

    def test_probability_panel_uses_consolidated_series(self):
        fig = create_half_normal_plot(
            self.RAW_EFFECTS,
            self.RAW_NAMES,
            mode="side_by_side",
            probability_series=self.de_series(),
        )
        # Classical panel keeps the raw coefficient series (6 contrasts).
        assert len(fig.data[0].x) == 6
        # Probability panel plots one point per model term.
        assert len(fig.data[1].x) == 3
        assert np.allclose(np.sort(fig.data[1].x), [0.5, 0.7, 1.2])

    def test_probability_panel_hover_includes_df(self):
        fig = create_half_normal_plot(
            self.RAW_EFFECTS,
            self.RAW_NAMES,
            mode="side_by_side",
            probability_series=self.de_series(),
        )
        assert "df:" in fig.data[1].hovertemplate
        cd = fig.data[1].customdata
        assert cd.shape == (3, 5)
        # sorted by magnitude: Egg_percent(1), Egg_lot:Egg_percent(3), Egg_lot(3)
        assert list(cd[:, 4]) == [1, 3, 3]

    def test_standalone_probability_mode_consolidates(self):
        fig = create_half_normal_plot(
            self.RAW_EFFECTS,
            self.RAW_NAMES,
            mode="probability",
            probability_series=self.de_series(),
        )
        assert len(fig.data[1].x) == 3
        assert fig.layout.xaxis.visible is False

    def test_classical_panel_unchanged_by_consolidation(self):
        base = create_half_normal_plot(
            self.RAW_EFFECTS, self.RAW_NAMES, mode="side_by_side"
        )
        fig = create_half_normal_plot(
            self.RAW_EFFECTS,
            self.RAW_NAMES,
            mode="side_by_side",
            probability_series=self.de_series(),
        )
        assert np.array_equal(np.asarray(fig.data[0].x), np.asarray(base.data[0].x))
        assert np.array_equal(np.asarray(fig.data[0].y), np.asarray(base.data[0].y))

    def test_empty_consolidated_series_renders(self):
        empty = HalfNormalSeries(names=[], magnitudes=np.asarray([]))
        fig = create_half_normal_plot(
            self.RAW_EFFECTS,
            self.RAW_NAMES,
            mode="side_by_side",
            probability_series=empty,
        )
        assert len(fig.data[1].x) == 0


class TestConsolidationIntegration:
    """End-to-end with a blocked full factorial and a real ANOVA fit."""

    LOT_LEVELS = ["41007587", "41005191", "41007741", "41007302"]

    def test_blocked_design_end_to_end(self):
        factors = [
            Factor(
                "Egg_lot",
                FactorType.CATEGORICAL,
                ChangeabilityLevel.EASY,
                levels=self.LOT_LEVELS,
                _validate_on_init=False,
            ),
            Factor(
                "Egg_percent",
                FactorType.DISCRETE_NUMERIC,
                ChangeabilityLevel.EASY,
                levels=[1.2, 1.8],
            ),
        ]
        design = full_factorial(
            factors, n_replicates=2, n_blocks=2, randomize=False
        )
        rng = np.random.default_rng(1)
        response = (
            design["Egg_lot"]
            .map({lot: m for lot, m in zip(self.LOT_LEVELS, [100, 102, 98, 105])})
            .astype(float)
            + 1.5 * design["Egg_percent"]
            + rng.normal(0, 0.1, len(design))
        )

        results = ANOVAAnalysis(design, response, factors).fit(
            ["Egg_lot", "Egg_percent", "Egg_lot*Egg_percent"]
        )
        effects_data = results.effect_estimates[
            results.effect_estimates.index != "Intercept"
        ]
        coef = (
            effects_data["Coefficient"]
            if "Coefficient" in effects_data
            else effects_data["Estimate"]
        )
        names = [t for t in effects_data.index]

        series = consolidate_half_normal_effects(
            effects_data,
            results.anova_table,
            name_transform=lambda t: t.replace(":", "*"),
        )
        assert sorted(series.names) == sorted(
            ["Egg_lot", "Egg_percent", "Egg_lot*Egg_percent"]
        )
        assert "Block" not in series.names
        assert set(series.dfs.tolist()) == {1, 3}

        fig = create_half_normal_plot(
            coef.values,
            names,
            mode="side_by_side",
            probability_series=series,
        )
        # Classical panel keeps every raw contrast including the Block rows.
        assert len(fig.data[0].x) == len(effects_data)
        assert any("Block" in n for n in fig.data[0].text)
        # Probability panel shows only the three model terms.
        assert len(fig.data[1].x) == 3
        assert np.allclose(np.sort(fig.data[1].x), np.sort(series.magnitudes))


class TestCodedCoefficients:
    """coded_single_df_coefficients reproduces Design Expert's values."""

    # The 24-run 4-lot x 2-% blocked egg experiment the user analysed in
    # Design Expert (response: Day_0_Max_Force).  Design Expert shows
    # % egg yolk = 5.34 @ >95% in the default (no term selected) view.
    LOT_LEVELS = ["L41007587", "L41005191", "L41007741", "L41007666"]
    DATA = (
        "StdOrder\tRunOrder\tBlock\tEgg_lot\tEgg_percent\tReplicate\tDay_0_Max_Force\n"
        "3\t1\t1\tL41007587\t1.8\t1\t44.30320509\n"
        "19\t2\t1\tL41007666\t1.8\t3\t52.6851969\n"
        "20\t3\t1\tL41007587\t1.2\t3\t32.29610375\n"
        "21\t4\t1\tL41005191\t1.8\t3\t43.17182065\n"
        "18\t5\t1\tL41007741\t1.2\t3\t34.25238725\n"
        "17\t6\t2\tL41007666\t1.8\t3\t43.60381135\n"
        "6\t7\t2\tL41007741\t1.8\t1\t45.99586221\n"
        "5\t8\t2\tL41005191\t1.2\t1\t29.712\n"
        "7\t9\t2\tL41007587\t1.8\t1\t45.92887158\n"
        "24\t10\t2\tL41007587\t1.2\t3\t32.05880188\n"
        "22\t11\t3\tL41005191\t1.8\t3\t52.43439243\n"
        "11\t12\t3\tL41005191\t1.2\t2\t31.64070038\n"
        "10\t13\t3\tL41007666\t1.8\t2\t41.0026392\n"
        "12\t14\t3\tL41007666\t1.2\t2\t36.12542777\n"
        "23\t15\t3\tL41007741\t1.2\t3\t35.02021911\n"
        "9\t16\t4\tL41005191\t1.2\t2\t28.99199957\n"
        "16\t17\t4\tL41007741\t1.8\t2\t44.14363832\n"
        "13\t18\t4\tL41007741\t1.2\t2\t39.63008883\n"
        "14\t19\t4\tL41007666\t1.2\t2\t40.90340513\n"
        "1\t20\t4\tL41007587\t1.2\t1\t33.08700033\n"
        "15\t21\t5\tL41007741\t1.8\t2\t41.61058443\n"
        "8\t22\t5\tL41007587\t1.8\t1\t45.57535572\n"
        "2\t23\t5\tL41005191\t1.8\t1\t38.15793833\n"
        "4\t24\t5\tL41007666\t1.2\t1\t41.51233406"
    )

    @pytest.fixture
    def fitted(self):
        import io

        df = pd.read_csv(io.StringIO(self.DATA), sep="\t")
        for col in ["Block", "Egg_lot", "Replicate"]:
            df[col] = df[col].astype(str)
        df["Egg_percent"] = df["Egg_percent"].astype(float)
        response = df["Day_0_Max_Force"].astype(float)
        factors = [
            Factor("Egg_lot", FactorType.CATEGORICAL,
                   ChangeabilityLevel.EASY, levels=self.LOT_LEVELS,
                   _validate_on_init=False),
            Factor("Egg_percent", FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[1.2, 1.8]),
        ]
        analysis = ANOVAAnalysis(df, response, factors,
                                 response_name="Day_0_Max_Force")
        results = analysis.fit(["Egg_lot", "Egg_percent", "Egg_lot*Egg_percent"])
        return analysis, results

    def test_egg_percent_coded_coefficient(self, fitted):
        analysis, _ = fitted
        coded = analysis.coded_single_df_coefficients()
        # Design Expert displays % egg yolk at 5.34 in this design.
        assert np.isclose(coded["Egg_percent"], 5.33, atol=0.02)

    def test_insignificant_interaction_is_dropped(self, fitted):
        analysis, results = fitted
        assert results.anova_table.loc["Egg_lot:Egg_percent", "PR(>F)"] > 0.05
        coded = analysis.coded_single_df_coefficients()
        assert "Egg_lot:Egg_percent" not in coded

    def test_no_model_returns_empty(self):
        import io

        df = pd.read_csv(io.StringIO(self.DATA), sep="\t")
        for col in ["Block", "Egg_lot", "Replicate"]:
            df[col] = df[col].astype(str)
        df["Egg_percent"] = df["Egg_percent"].astype(float)
        factors = [
            Factor("Egg_lot", FactorType.CATEGORICAL,
                   ChangeabilityLevel.EASY, levels=self.LOT_LEVELS,
                   _validate_on_init=False),
            Factor("Egg_percent", FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[1.2, 1.8]),
        ]
        analysis = ANOVAAnalysis(df, df["Day_0_Max_Force"], factors)
        assert analysis.coded_single_df_coefficients() == {}


class TestErrorTriangles:
    """Error-estimate triangles and the Design-Expert default readout."""

    @pytest.fixture
    def fitted(self):
        import io

        df = pd.read_csv(io.StringIO(TestCodedCoefficients.DATA), sep="\t")
        for col in ["Block", "Egg_lot", "Replicate"]:
            df[col] = df[col].astype(str)
        df["Egg_percent"] = df["Egg_percent"].astype(float)
        response = df["Day_0_Max_Force"].astype(float)
        factors = [
            Factor("Egg_lot", FactorType.CATEGORICAL,
                   ChangeabilityLevel.EASY, levels=TestCodedCoefficients.LOT_LEVELS,
                   _validate_on_init=False),
            Factor("Egg_percent", FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[1.2, 1.8]),
        ]
        analysis = ANOVAAnalysis(df, response, factors,
                                 response_name="Day_0_Max_Force")
        results = analysis.fit(["Egg_lot", "Egg_percent", "Egg_lot*Egg_percent"])
        effects_data = results.effect_estimates[
            results.effect_estimates.index != "Intercept"
        ]
        resid = results.anova_table.loc["Residual"]
        return consolidate_half_normal_effects(
            effects_data,
            results.anova_table,
            coded_coefficients=analysis.coded_single_df_coefficients(),
            sigma2=float(resid["sum_sq"] / resid["df"]),
            pooled_scale=0.12,
            error_triangles=int(resid["df"]),
            error_scale=analysis.coded_effect_se(),
        )

    def test_fifteen_points_including_twelve_triangles(self, fitted):
        de_readout = fitted
        kinds = de_readout.kinds
        assert len(de_readout.names) == 15
        assert kinds.count("error_triangle") == 12
        assert kinds.count("term") == 3
        assert all(de_readout.dfs[k] == 1 for k in range(15)
                   if kinds[k] == "error_triangle")

    def test_design_expert_readout(self, fitted):
        de_readout = fitted
        scores = {
            name: mag
            for name, mag in zip(de_readout.names, de_readout.magnitudes)
        }
        # % egg yolk ~5.34, A and AB pooled far below it.
        assert np.isclose(scores["Egg_percent"], 5.33, atol=0.02)
        assert scores["Egg_lot"] < 1.0
        assert scores["Egg_lot:Egg_percent"] < 1.0
        # % egg yolk is the top point => 96.7% probability.
        top = max(de_readout.magnitudes)
        assert np.isclose(top, scores["Egg_percent"])

    def test_panel_sorting_gives_b_top_probability(self, fitted):
        de_readout = fitted
        fig = create_half_normal_plot(
            np.zeros(len(de_readout.magnitudes)),
            de_readout.names,
            mode="side_by_side",
            probability_series=de_readout,
        )
        trace = fig.data[1]
        names = list(trace.text)
        mags = np.asarray(trace.x)
        probs = trace.customdata[:, 3]
        i_b = names.index("Egg_percent")
        assert i_b == int(np.argmax(mags))
        assert np.isclose(probs[i_b], 96.7, atol=0.1)

    def test_triangles_render_green_triangle_up(self, fitted):
        de_readout = fitted
        fig = create_half_normal_plot(
            np.zeros(len(de_readout.magnitudes)),
            de_readout.names,
            mode="side_by_side",
            probability_series=de_readout,
        )
        trace = fig.data[1]
        order = np.argsort(np.asarray(de_readout.magnitudes))
        kinds = np.asarray(de_readout.kinds)[order]
        symbols = np.asarray(trace.marker.symbol)
        colors = np.asarray(trace.marker.color)
        assert set(symbols[kinds == "error_triangle"]) == {"triangle-up"}
        assert set(colors[kinds == "error_triangle"]) == {
            PLOT_COLORS["success"]
        }

    def test_coded_override_used_when_provided(self):
        effects = TestConsolidation.dummy_effects()
        anova = TestConsolidation.dummy_anova()
        series = consolidate_half_normal_effects(
            effects, anova, coded_coefficients={"Egg_percent": 5.33},
            sigma2=0.16,
        )
        i = series.names.index("Egg_percent")
        assert np.isclose(series.magnitudes[i], 5.33)
        assert np.isclose(series.signed[i], 5.33)

    def test_triangles_capped_below_top_term(self):
        effects = TestConsolidation.dummy_effects()
        anova = TestConsolidation.dummy_anova()
        series = consolidate_half_normal_effects(
            effects, anova, coded_coefficients={"Egg_percent": 5.33},
            sigma2=0.16, error_triangles=4,
        )
        assert series.kinds.count("error_triangle") == 4
        max_term = max(
            m for m, k in zip(series.magnitudes, series.kinds) if k == "term"
        )
        max_tri = max(
            m for m, k in zip(series.magnitudes, series.kinds)
            if k == "error_triangle"
        )
        assert max_tri < max_term

    def test_error_scale_makes_triangles_collinear(self, fitted):
        de_readout = fitted
        se = de_readout.error_scale
        assert se is not None and se > 0
        n = len(de_readout.names)
        z = np.abs(
            stats.norm.ppf(((np.arange(1, n + 1) - 0.5) / n + 1.0) / 2.0)
        )
        order = np.argsort(de_readout.magnitudes)
        diffs = []
        for pos, idx in enumerate(order):
            if de_readout.kinds[idx] == "error_triangle":
                diffs.append(
                    abs(de_readout.magnitudes[idx] - se * z[pos])
                )
        assert diffs and all(np.isfinite(d) for d in diffs)
        assert max(diffs) < 1e-3

    def test_probability_axis_is_monotonic(self, fitted):
        de_readout = fitted
        fig = create_half_normal_plot(
            np.zeros(len(de_readout.magnitudes)),
            de_readout.names,
            mode="side_by_side",
            probability_series=de_readout,
        )
        trace = fig.data[1]
        y = np.asarray(trace.y)
        assert np.all(np.diff(y) > 0)

    def test_b_well_above_reference_diagonal(self, fitted):
        de_readout = fitted
        se = de_readout.error_scale
        order = np.argsort(de_readout.magnitudes)
        pos_b = order.tolist().index(
            list(de_readout.names).index("Egg_percent")
        )
        n = len(de_readout.names)
        z_b = np.abs(
            stats.norm.ppf(((pos_b + 1 - 0.5) / n + 1.0) / 2.0)
        )
        line_top = se * z_b
        b_mag = de_readout.magnitudes[
            list(de_readout.names).index("Egg_percent")
        ]
        assert b_mag > line_top * 1.5

    def test_reference_diagonal_trace_drawn(self, fitted):
        de_readout = fitted
        fig = create_half_normal_plot(
            np.zeros(len(de_readout.magnitudes)),
            de_readout.names,
            mode="side_by_side",
            probability_series=de_readout,
        )
        line_traces = [
            t for t in fig.data if t.mode == "lines"
            and hasattr(t.x, "__len__") and len(t.x) == 2
        ]
        assert len(line_traces) == 2
        x0, x1 = line_traces[1].x
        y0, y1 = line_traces[1].y
        assert np.isclose(x0, 0.0) and np.isclose(y0, 0.0)
        assert np.isclose(x1 / y1, de_readout.error_scale)

    def test_error_scale_is_data_driven_coded_se(self):
        import io

        df = pd.read_csv(io.StringIO(TestCodedCoefficients.DATA), sep="\t")
        for col in ["Block", "Egg_lot", "Replicate"]:
            df[col] = df[col].astype(str)
        df["Egg_percent"] = df["Egg_percent"].astype(float)
        response = df["Day_0_Max_Force"].astype(float)
        factors = [
            Factor("Egg_lot", FactorType.CATEGORICAL,
                   ChangeabilityLevel.EASY, levels=TestCodedCoefficients.LOT_LEVELS,
                   _validate_on_init=False),
            Factor("Egg_percent", FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[1.2, 1.8]),
        ]
        analysis = ANOVAAnalysis(df, response, factors,
                                 response_name="Day_0_Max_Force")
        analysis.fit(["Egg_lot", "Egg_percent", "Egg_lot*Egg_percent"])
        se = analysis.coded_effect_se()
        # sqrt(2 * MSE_refit / N) for the coded reference refit on 24 runs;
        # fully data-driven (no fitted constants), magnitude ~1.26 here.
        assert np.isclose(se, 1.26, atol=0.05)
        coded = analysis.coded_single_df_coefficients()
        assert np.isclose(coded["Egg_percent"], 5.33, atol=0.02)