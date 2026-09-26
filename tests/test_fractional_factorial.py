"""
Unit tests for fractional factorial designs.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.fractional_factorial import FractionalFactorial


def _choose_design_app(factor_count):
    factors = [
        Factor(
            f"Factor_{chr(65 + i)}",
            FactorType.CONTINUOUS,
            ChangeabilityLevel.EASY,
            levels=[-1, 1],
        )
        for i in range(factor_count)
    ]
    app = AppTest.from_file(
        Path(__file__).parents[1] / "src/ui/pages/3_choose_design.py",
        default_timeout=10,
    )
    app.session_state["factors"] = factors
    app.session_state["model_terms"] = ["Intercept"]
    app.session_state["design_type"] = "Fractional Factorial"
    return app


def _resolution_snapshot(app):
    badges = [item.value for item in app.markdown if "Resolution " in item.value]
    captions = [item.value for item in app.caption]
    markdown = [item.value for item in app.markdown]
    return {
        "badge": badges[0],
        "caption": next(item for item in captions if "effects" in item),
        "quality": next(item for item in markdown if item.startswith("**Quality:**")),
        "runs": next(item for item in markdown if item.startswith("**Estimated runs:**")),
    }


class TestFractionalFactorialCreation:
    """Test fractional factorial design creation."""
    
    def test_2_5_1_design(self):
        """Test 2^(5-1) fractional factorial (Resolution V)."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=5)
        
        assert ff.k == 5
        assert ff.p == 1
        assert ff.resolution == 5
        
        design = ff.generate(randomize=False)
        
        # Should have 2^(5-1) = 16 runs
        assert len(design) == 16
        
        # Should have all 5 factors
        for factor in factors:
            assert factor.name in design.columns
    
    def test_2_4_1_design(self):
        """Test 2^(4-1) fractional factorial (Resolution IV)."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=4)
        
        assert ff.k == 4
        assert ff.p == 1
        assert ff.resolution == 4
        
        design = ff.generate(randomize=False)
        
        # Should have 2^(4-1) = 8 runs
        assert len(design) == 8
    
    @pytest.mark.parametrize(
        ("factor_count", "fraction", "expected_resolution", "expected_runs"),
        [
            (3, "1/2", 3, 4),
            (4, "1/2", 4, 8),
            (5, "1/2", 5, 16),
            (5, "1/4", 3, 8),
            (6, "1/2", 6, 32),
            (6, "1/8", 3, 8),
            (7, "1/2", 7, 64),
            (9, "1/16", 4, 32),
            (10, "1/16", 4, 64),
        ],
    )
    def test_standard_design_quality_inputs(
        self, factor_count, fraction, expected_resolution, expected_runs
    ):
        factors = [
            Factor(
                f"Factor_{chr(65 + i)}",
                FactorType.CONTINUOUS,
                ChangeabilityLevel.EASY,
                levels=[-1, 1],
            )
            for i in range(factor_count)
        ]

        ff = FractionalFactorial(factors=factors, fraction=fraction)

        assert ff.resolution == expected_resolution
        assert len(ff.generate(randomize=False)) == expected_runs

    def test_2_7_3_design(self):
        """Test 2^(7-3) fractional factorial (Resolution IV)."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(7)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/8", resolution=4)
        
        assert ff.k == 7
        assert ff.p == 3
        
        design = ff.generate(randomize=False)
        
        # Should have 2^(7-3) = 16 runs
        assert len(design) == 16
    
    def test_custom_generators(self):
        """Test fractional factorial with custom generators."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(
            factors,
            fraction="1/2",
            generators=["E=ABCD"]
        )
        
        # Check generator was parsed correctly (now stored in algebraic form)
        assert ff.generators_algebraic == [("E", "ABCD")]
        
        # Check defining relation exists
        assert hasattr(ff, 'defining_relation')
        assert "ABCDE" in ff.defining_relation
    
    def test_with_blocking(self):
        """Test fractional factorial with blocking."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2")
        design = ff.generate(randomize=False, n_blocks=2)
        
        # Should have Block column
        assert 'Block' in design.columns
        
        # Should have 2 blocks
        assert design['Block'].nunique() == 2
        
        # Each block should have 4 runs (8 runs / 2 blocks)
        assert all(design['Block'].value_counts() == 4)
    
    def test_randomization_reproducibility(self):
        """Test that same seed gives same design."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff1 = FractionalFactorial(factors, fraction="1/2")
        design1 = ff1.generate(randomize=True, random_seed=42)
        
        ff2 = FractionalFactorial(factors, fraction="1/2")
        design2 = ff2.generate(randomize=True, random_seed=42)
        
        pd.testing.assert_frame_equal(design1, design2)


class TestFractionalFactorialUx:
    @pytest.mark.parametrize(
        ("factor_count", "roman", "quality", "background", "runs"),
        [
            (4, "IV", "Good", "#FFF4CC", 8),
            (5, "V", "Excellent", "#EAF7EE", 16),
            (6, "VI", "Excellent", "#EAF7EE", 32),
            (7, "VII", "Excellent", "#EAF7EE", 64),
        ],
    )
    def test_standard_resolution_badge(
        self, factor_count, roman, quality, background, runs
    ):
        app = _choose_design_app(factor_count)
        with patch("src.ui.components.sidebar.build_standard_sidebar"):
            app.run()

        assert not app.exception
        snapshot = _resolution_snapshot(app)
        assert f"Resolution {roman} ({quality})" in snapshot["badge"]
        assert f"background:{background}" in snapshot["badge"]
        assert snapshot["quality"] == f"**Quality:** {quality}"
        assert snapshot["runs"] == f"**Estimated runs:** {runs}"

    def test_three_factor_design_has_savings_caution(self):
        app = _choose_design_app(3)
        with patch("src.ui.components.sidebar.build_standard_sidebar"):
            app.run()

        assert not app.exception
        assert app.selectbox[0].options == ["1/2"]
        snapshot = _resolution_snapshot(app)
        assert "Resolution III (Screening Only)" in snapshot["badge"]
        assert snapshot["runs"] == "**Estimated runs:** 4"
        assert any(
            "Only 4 runs are saved compared with the full factorial. "
            "Consider a full factorial unless experimental cost is extremely "
            "constrained." in item.value
            for item in app.warning
        )

    def test_fraction_change_updates_badge_quality_text_and_runs(self):
        app = _choose_design_app(5)
        with patch("src.ui.components.sidebar.build_standard_sidebar"):
            app.run()
            initial = _resolution_snapshot(app)
            app.selectbox[0].select("1/4")
            app.run()
            updated = _resolution_snapshot(app)

        assert not app.exception
        assert "Resolution V (Excellent)" in initial["badge"]
        assert initial["runs"] == "**Estimated runs:** 16"
        assert "Resolution III (Screening Only)" in updated["badge"]
        assert updated["quality"] == "**Quality:** Screening Only"
        assert updated["runs"] == "**Estimated runs:** 8"
        assert "Main effects may be confounded" in updated["caption"]


class TestDefiningRelation:
    """Test defining relation calculation."""
    
    def test_2_5_1_defining_relation(self):
        """Test defining relation for 2^(5-1) design."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=5)
        
        # For E=ABCD, defining relation should be I, ABCDE (sorted alphabetically)
        assert "I" in ff.defining_relation
        assert "ABCDE" in ff.defining_relation
    
    def test_2_4_1_defining_relation(self):
        """Test defining relation for 2^(4-1) design."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=4)
        
        # For D=ABC, defining relation should be I, ABCD (sorted alphabetically)
        assert "I" in ff.defining_relation
        assert "ABCD" in ff.defining_relation


class TestAliasStructure:
    """Test alias structure calculation."""
    
    def test_resolution_v_main_effects_clear(self):
        """Test that Resolution V has clear main effects and 2FI."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=5)
        
        # Main effects should be clear (aliased with 4FI or higher)
        for letter in "ABCDE":
            if letter in ff.alias_structure:
                aliases = ff.alias_structure[letter]
                # Should not be aliased with any main effects or 2FI
                for alias in aliases:
                    assert len(alias) >= 4, f"{letter} aliased with {alias}"
    
    def test_resolution_iv_2fi_aliased(self):
        """Test that Resolution IV has 2FI aliased with other 2FI."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=4)
        
        # Main effects should be clear
        for letter in "ABCD":
            if letter in ff.alias_structure:
                aliases = ff.alias_structure[letter]
                for alias in aliases:
                    assert len(alias) >= 3, f"Main effect {letter} aliased with {alias}"
    
    def test_alias_summary_dataframe(self):
        """Test that alias summary returns valid DataFrame."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2")
        summary = ff.get_alias_summary()
        
        # Should be a DataFrame
        assert isinstance(summary, pd.DataFrame)
        
        # Should have required columns
        assert 'Effect' in summary.columns
        assert 'Aliased_With' in summary.columns
        
        # Should have entries
        assert len(summary) > 0


class TestValidation:
    """Test input validation."""
    
    def test_too_few_factors(self):
        """Test that < 3 factors raises error."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        with pytest.raises(ValueError, match="at least 3 factors"):
            FractionalFactorial(factors, fraction="1/2")
    
    def test_multi_level_categorical_rejected(self):
        """Test that categorical factors with more than 2 levels are rejected."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("C", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, levels=["X", "Y", "Z"])
        ]
        
        with pytest.raises(ValueError, match="must have exactly 2 levels"):
            FractionalFactorial(factors, fraction="1/2")
    
    def test_two_level_categorical_accepted(self):
        """Test that 2-level categorical factors are accepted: they are
        ordinary 2-level factors, so the 2^(k-p) math is unchanged."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("D", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, levels=["X", "Y"])
        ]

        ff = FractionalFactorial(factors, fraction="1/2")

        assert ff.k == 4
        assert ff.p == 1
        assert len(ff.generate(randomize=False)) == 8

    def test_non_2_level_discrete_rejected(self):
        """Test that non-2-level discrete factors are rejected."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("C", FactorType.DISCRETE_NUMERIC, ChangeabilityLevel.EASY, 
                  levels=[100, 150, 200])
        ]
        
        with pytest.raises(ValueError, match="must have exactly 2 levels"):
            FractionalFactorial(factors, fraction="1/2")
    
    def test_invalid_fraction_format(self):
        """Test that invalid fraction format raises error."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        with pytest.raises(ValueError, match="must be power of 2"):
            FractionalFactorial(factors, fraction="1/3")
    
    def test_too_large_fraction(self):
        """Test that fraction >= k raises error."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                   ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        with pytest.raises(ValueError, match="Cannot create"):
            FractionalFactorial(factors, fraction="1/16")
    
    def test_base_factor_on_generator_lhs_rejected(self):
        """Regression (#47): a generator whose LHS is a base factor must fail
        validation at construction, not crash later with an opaque KeyError."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                   ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        with pytest.raises(ValueError, match="must define a generated factor"):
            FractionalFactorial(factors, fraction="1/2", generators=["A=BCD"])
    
    def test_custom_multi_generator_design(self):
        """Regression (follow-up to #47): a valid 2^(6-2) custom design with
        generators on the second generated factor must build (doc example
        E=ACD, F=ABD)."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                   ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(6)
        ]
        
        ff = FractionalFactorial(
            factors,
            fraction="1/4",
            generators=["E=ACD", "F=ABD"]
        )
        
        assert ff.generators_algebraic == [("E", "ACD"), ("F", "ABD")]
        assert ff.resolution == 4
        
        design = ff.generate(randomize=False)
        assert len(design) == 16


class TestDesignGeneration:
    """Test the actual design matrix generation."""
    
    def test_generator_multiplication(self):
        """Test that generated factors follow generator rule."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        # D = ABC
        ff = FractionalFactorial(factors, fraction="1/2", generators=["D=ABC"])
        design = ff.generate(randomize=False)
        
        # Verify D = A * B * C for all rows
        factor_names = [f.name for f in factors]
        for idx, row in design.iterrows():
            expected_d = row[factor_names[0]] * row[factor_names[1]] * row[factor_names[2]]
            actual_d = row[factor_names[3]]
            assert expected_d == actual_d, f"Row {idx}: D should equal A*B*C"
    
    def test_all_runs_unique(self):
        """Test that all runs are unique."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                   ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2")
        design = ff.generate(randomize=False)
        
        # Get factor columns only
        factor_cols = [f.name for f in factors]
        
        # Check for duplicates
        n_unique = design[factor_cols].drop_duplicates().shape[0]
        assert n_unique == len(design), "Design contains duplicate runs"
    
    def test_natural_range_factors_not_double_decoded(self):
        """Regression (#45): designs with real-world factor ranges must not
        double-decode. Every output column must stay within its declared
        levels ([-1, 1] is idempotent under decode, so it hides this bug)."""
        levels = [(10, 14), (11, 15), (12, 16), (13, 17), (14, 18)]
        factors = [
            Factor(f"F{i}", FactorType.CONTINUOUS,
                   ChangeabilityLevel.EASY, levels=[lo, hi])
            for i, (lo, hi) in enumerate(levels)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2")
        design = ff.generate(randomize=False)
        
        for factor, (lo, hi) in zip(factors, levels):
            values = set(design[factor.name].unique())
            assert values <= {float(lo), float(hi)}, (
                f"{factor.name} values {values} outside declared "
                f"levels [{lo}, {hi}]"
            )
    
    def test_half_fraction_resolution_6_and_7_reachable(self):
        """Regression (#48): 2^(6-1) Res VI and 2^(7-1) Res VII must be
        reachable through the default (auto-resolution) path."""
        for k, expected_res, expected_generator, n_runs in [
            (6, 6, ("F", "ABCDE"), 32),
            (7, 7, ("G", "ABCDEF"), 64),
        ]:
            factors = [
                Factor(f"F{i}", FactorType.CONTINUOUS,
                       ChangeabilityLevel.EASY, levels=[-1, 1])
                for i in range(k)
            ]
            
            ff = FractionalFactorial(factors, fraction="1/2")
            
            assert ff.resolution == expected_res
            assert ff.generators_algebraic == [expected_generator]
            assert len(ff.generate(randomize=False)) == n_runs


class TestCategoricalFactors:
    """Test 2-level categorical support in fractional factorials.

    A 2-level categorical factor is an ordinary 2-level factor: the coded
    [-1, +1] grid maps bijectively onto its two declared labels, and the
    alias structure is computed on algebraic symbols independent of type.
    """

    @staticmethod
    def _mixed_factors() -> list:
        """Three continuous + one 2-level categorical, natural ranges.

        k=4 with a 1/2 fraction (2^(4-1) = 8 runs, Resolution IV) keeps these
        tests independent of the 3-factor generator work on another branch.
        """
        return [
            Factor("Temp", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[150.0, 200.0]),
            Factor("Pressure", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[50.0, 100.0]),
            Factor("Time", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[10.0, 20.0]),
            Factor("Catalyst", FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                   levels=["X", "Y"]),
        ]

    def test_categorical_column_carries_labels_not_coded_values(self):
        """The categorical column must carry declared labels, never +/-1.0."""
        ff = FractionalFactorial(self._mixed_factors(), fraction="1/2")
        design = ff.generate(randomize=False)

        values = set(design["Catalyst"].unique())
        assert values == {"X", "Y"}, (
            f"Expected declared labels, got {values} "
            f"(dtypes: {design['Catalyst'].dtype})"
        )
        assert not values & {-1.0, 1.0, -1, 1}

    def test_continuous_factors_still_decoded(self):
        """Adding a categorical factor must not disturb continuous decoding."""
        design = FractionalFactorial(
            self._mixed_factors(), fraction="1/2"
        ).generate(randomize=False)

        assert set(design["Temp"].unique()) == {150.0, 200.0}
        assert set(design["Pressure"].unique()) == {50.0, 100.0}
        assert set(design["Time"].unique()) == {10.0, 20.0}

    def test_categorical_as_generated_factor_follows_generator_rule(self):
        """A categorical factor built by a generator must be labelled
        consistently with the product of its generators."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[-1.0, 1.0]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[-1.0, 1.0]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[-1.0, 1.0]),
            Factor("D", FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                   levels=["low", "high"]),
        ]

        design = FractionalFactorial(
            factors, fraction="1/2", generators=["D=ABC"]
        ).generate(randomize=False)

        assert set(design["D"].unique()) == {"low", "high"}
        # D == A*B*C on the coded grid, so +1 -> second declared label.
        for _, row in design.iterrows():
            product = row["A"] * row["B"] * row["C"]
            assert row["D"] == ("high" if product > 0 else "low")

    def test_alias_structure_and_resolution_independent_of_type(self):
        """Categorical factors must not perturb resolution or aliasing."""
        continuous = [
            Factor(f"F{i}", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[-1.0, 1.0])
            for i in range(5)
        ]
        with_categorical = [
            Factor(f"F{i}",
                   FactorType.CATEGORICAL if i == 3 else FactorType.CONTINUOUS,
                   ChangeabilityLevel.EASY,
                   levels=["lo", "hi"] if i == 3 else [-1.0, 1.0])
            for i in range(5)
        ]

        ff_cont = FractionalFactorial(continuous, fraction="1/2", resolution=5)
        ff_cat = FractionalFactorial(with_categorical, fraction="1/2", resolution=5)

        assert ff_cat.resolution == ff_cont.resolution
        assert ff_cat.generators_algebraic == ff_cont.generators_algebraic
        assert ff_cat.defining_relation == ff_cont.defining_relation
        assert ff_cat.alias_structure == ff_cont.alias_structure

    def test_coded_pattern_matches_equivalent_continuous_design(self):
        """Relabelling the categorical column back to +/-1 must reproduce the
        all-continuous design for the same k and fraction."""
        with_categorical = self._mixed_factors()
        all_continuous = [
            Factor(f.name, FactorType.CONTINUOUS, f.changeability,
                   levels=[-1.0, 1.0])
            for f in with_categorical
        ]

        cat_design = FractionalFactorial(
            with_categorical, fraction="1/2"
        ).generate(randomize=False)
        cont_design = FractionalFactorial(
            all_continuous, fraction="1/2"
        ).generate(randomize=False)

        recoded = cat_design.copy()
        recoded["Catalyst"] = cat_design["Catalyst"].map({"X": -1.0, "Y": 1.0})
        centers = {"Temp": (175.0, 25.0), "Pressure": (75.0, 25.0),
                   "Time": (15.0, 5.0)}
        for col, (center, half_range) in centers.items():
            recoded[col] = (recoded[col] - center) / half_range

        cols = ["StdOrder", "RunOrder", "Temp", "Pressure", "Time", "Catalyst"]
        pd.testing.assert_frame_equal(recoded[cols], cont_design[cols])

    def test_csv_round_trip_and_analysis_fit(self):
        """A fractional design with a categorical factor must survive the CSV
        round-trip and fit as a 1-df effect, exactly like a 2-level numeric."""
        from src.core.analysis import ANOVAAnalysis
        from src.ui.utils.csv_parser import generate_doe_csv, parse_doe_csv

        factors = self._mixed_factors()
        design = FractionalFactorial(factors, fraction="1/2").generate(
            randomize=False
        )
        design["Yield"] = np.arange(len(design), dtype=float)

        csv = generate_doe_csv(
            design=design,
            factors=factors,
            response_definitions=[{"name": "Yield", "units": "%"}],
            design_type="fractional_factorial",
        )
        parsed = parse_doe_csv(csv)

        assert parsed.is_valid, parsed.error
        catalyst = next(f for f in parsed.factors if f.name == "Catalyst")
        assert catalyst.is_categorical()
        assert sorted(catalyst.levels) == ["X", "Y"]
        assert set(parsed.design_data["Catalyst"].unique()) == {"X", "Y"}

        analysis = ANOVAAnalysis(
            design=parsed.design_data[design.columns],
            response=parsed.design_data["Yield"].to_numpy(),
            factors=parsed.factors,
            response_name="Yield",
        )
        results = analysis.fit(["Temp", "Pressure", "Time", "Catalyst"])

        # 8 runs, intercept + 3 continuous + 1 categorical (k-1 = 1 df) = 5 params
        assert results.fitted_model.df_resid == 3
        assert any("Catalyst" in str(idx) for idx in results.anova_table.index)

    def test_discrete_numeric_levels_unchanged(self):
        """Widening the level restore must not alter discrete-numeric output."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[-1.0, 1.0]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[-1.0, 1.0]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[-1.0, 1.0]),
            Factor("D", FactorType.DISCRETE_NUMERIC, ChangeabilityLevel.EASY,
                   levels=[100, 200]),
        ]

        design = FractionalFactorial(factors, fraction="1/2").generate(
            randomize=False
        )

        assert set(design["D"].unique()) == {100, 200}
