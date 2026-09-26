"""
Rendering tests for the design-availability gates on Steps 2, 3 and 4.

The shared validation service in ``src/core/design_validation.py`` is covered by
``test_design_validation.py``, and the load/save messaging by
``test_project_design_mismatch.py``.  Neither drives the pages themselves, and
that gap shipped a crash: ``2_select_model.py`` accumulated its "unavailable
designs" list twice, and the second accumulation replaced the first with
``(name, result)`` pairs that the render loop then re-indexed by name, raising

    KeyError: DesignValidationResult(design_type='Fractional Factorial', ...)

on any factor set that blocked at least one design -- the common case of a
multi-level categorical factor.  The service was correct the whole time; only
the page that consumed it was wrong.

These tests drive the real pages through ``streamlit.testing.v1.AppTest`` and
assert on rendered output, so this class of bug fails loudly here.

Note: only ``AppTest``'s own ``session_state`` is used.  Assigning to the real
``streamlit.session_state`` replaces Streamlit's session-state proxy with a
plain dict, which leaks into later tests in the same process and breaks every
subsequent ``AppTest`` run with ``KeyError: 'url_pathname'``.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from src.core.factors import ChangeabilityLevel, Factor, FactorType

PAGES = Path(__file__).parents[1] / "src/ui/pages"


def cont(name):
    return Factor(
        name, FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
        levels=[10.0, 20.0],
    )


def cat(name, levels):
    return Factor(
        name, FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
        levels=list(levels),
    )


def four_categoricals_two_multi_level():
    """The reported case: 4 categorical factors, 2 of them 3-level."""
    return [
        cont("Temp"),
        cont("Pressure"),
        cont("Speed"),
        cat("hola", ["a", "b", "c"]),
        cat("good_bye", ["x", "y", "z"]),
        cat("plain", ["lo", "hi"]),
        cat("plain2", ["lo", "hi"]),
    ]


def all_two_level():
    return [
        cont("Temp"),
        cont("Pressure"),
        cont("Speed"),
        cat("hola", ["a", "b"]),
        cat("good_bye", ["x", "y"]),
    ]


def fully_feasible():
    """A factor set every design type can build, so nothing is blocked.

    All continuous (required by the response surfaces and D-Optimal), at least
    three (required by fractional and Box-Behnken), and one hard-to-change
    factor (required by Split-Plot).  Two-level, so fractional accepts it.
    """
    return [
        Factor("Temp", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
               levels=[10.0, 20.0]),
        Factor("Pressure", FactorType.CONTINUOUS, ChangeabilityLevel.HARD,
               levels=[10.0, 20.0]),
        Factor("Speed", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
               levels=[10.0, 20.0]),
    ]


def _terms_for(factors):
    """Main-effects model, using names that are not Patsy-reserved."""
    return ["1"] + [f.name for f in factors]


def _run(page_name, factors, design_type=None, extra=None):
    app = AppTest.from_file(PAGES / page_name, default_timeout=15)
    app.session_state["factors"] = factors
    app.session_state["model_terms"] = _terms_for(factors)
    if design_type is not None:
        app.session_state["design_type"] = design_type
    for key, value in (extra or {}).items():
        app.session_state[key] = value
    # The sidebar's quick navigation calls st.page_link, which raises
    # KeyError: 'url_pathname' when a page file is driven directly by AppTest
    # (Streamlit's default page registry has no url_pathname).  Patching the
    # sidebar out is the pattern already used by test_fractional_factorial.py;
    # none of these tests assert on sidebar content.
    with patch("src.ui.components.sidebar.build_standard_sidebar"):
        app.run()
    return app


def _texts(items):
    return " ".join(item.value for item in items)


class TestStep2RecommendationsRender:
    """Step 2 lists which designs the current factors can build."""

    def test_multi_level_categoricals_do_not_crash(self):
        """Regression: the reported ``KeyError`` on the blocked-design list."""
        app = _run("2_select_model.py", four_categoricals_two_multi_level())
        assert not app.exception, [e.value for e in app.exception]

    def test_blocked_section_names_the_offending_factors(self):
        app = _run("2_select_model.py", four_categoricals_two_multi_level())
        body = _texts(app.markdown)
        assert "Unavailable for the current factors" in body
        assert "hola" in body
        assert "good_bye" in body

    def test_blocked_section_names_the_blocked_designs(self):
        app = _run("2_select_model.py", four_categoricals_two_multi_level())
        body = _texts(app.markdown)
        assert "Fractional Factorial" in body
        assert "D-Optimal" in body

    def test_full_factorial_is_offered_for_multi_level_categoricals(self):
        """The one design that does build, so it must not be buried."""
        app = _run("2_select_model.py", four_categoricals_two_multi_level())
        body = _texts(app.markdown)
        assert "Compatible Design Types" in body
        assert "**Full Factorial**" in body

    def test_nothing_blocked_has_no_blocked_section(self):
        """Only factors every design type accepts leave the list empty."""
        app = _run("2_select_model.py", fully_feasible())
        assert not app.exception, [e.value for e in app.exception]
        assert "Unavailable for the current factors" not in _texts(app.markdown)

    def test_all_two_level_still_blocks_the_continuous_only_designs(self):
        """Two-level categoricals unblock fractional but not the RSMs."""
        app = _run("2_select_model.py", all_two_level())
        body = _texts(app.markdown)
        assert "**Fractional Factorial**" in body
        assert "D-Optimal" in body

    def test_all_two_level_offers_fractional(self):
        app = _run("2_select_model.py", all_two_level())
        assert "**Fractional Factorial**" in _texts(app.markdown)

    def test_two_factors_reports_box_behnken_and_ccd_separately(self):
        """CCD works with 2 factors, Box-Behnken does not."""
        app = _run("2_select_model.py", [cont("Temp"), cont("Pressure")])
        assert not app.exception, [e.value for e in app.exception]
        body = _texts(app.markdown)
        assert "Response Surface (CCD)" in body


class TestStep3LockedDesignRender:
    """Step 3 locks designs its factors cannot build."""

    def test_selected_but_invalid_design_does_not_crash(self):
        app = _run(
            "3_choose_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        assert not app.exception, [e.value for e in app.exception]

    def test_saved_design_mismatch_is_called_out(self):
        app = _run(
            "3_choose_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        body = _texts(app.error)
        assert "design saved in this project" in body

    def test_mismatch_message_explains_it_was_not_rewritten(self):
        app = _run(
            "3_choose_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        body = _texts(app.error) + _texts(app.info)
        assert "kept as saved" in body or "preserved" in body

    def test_no_misleading_generator_set_guidance(self):
        """The fraction is not the problem; the 3-level factors are.

        ``FractionalFactorial.__init__`` raises for the factor reason, and the
        handler used to restate it as "no standard generator set exists",
        pointing the user at a fraction change that can never help.
        """
        app = _run(
            "3_choose_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        body = _texts(app.warning)
        assert "No standard generator set exists" not in body

    def test_preview_paused_notice_is_shown(self):
        app = _run(
            "3_choose_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        assert "Design preview paused" in _texts(app.warning)

    def test_valid_design_still_shows_its_resolution_badge(self):
        """The gate must not suppress previews for buildable designs."""
        app = _run(
            "3_choose_design.py",
            all_two_level(),
            design_type="Fractional Factorial",
        )
        assert not app.exception, [e.value for e in app.exception]
        assert "Resolution " in _texts(app.markdown)

    def test_valid_fractional_shows_no_mismatch_error(self):
        app = _run(
            "3_choose_design.py",
            all_two_level(),
            design_type="Fractional Factorial",
        )
        assert "design saved in this project" not in _texts(app.error)


class TestStep4GenerationGateRender:
    """Step 4 refuses to generate, and says why."""

    def test_invalid_design_blocks_generation_without_crashing(self):
        app = _run(
            "4_preview_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        assert not app.exception, [e.value for e in app.exception]

    def test_invalid_design_states_it_cannot_be_generated(self):
        app = _run(
            "4_preview_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        assert "cannot be generated from the current factors" in _texts(app.error)

    def test_invalid_design_names_the_offending_factor(self):
        app = _run(
            "4_preview_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        assert "hola" in _texts(app.error)

    def test_generate_button_is_absent_when_blocked(self):
        """The gate must actually prevent generation, not just warn."""
        app = _run(
            "4_preview_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        labels = [b.label for b in app.button]
        assert not any("Generate Design" in str(label) for label in labels)

    def test_pointers_to_the_fixing_steps_are_shown(self):
        app = _run(
            "4_preview_design.py",
            four_categoricals_two_multi_level(),
            design_type="Fractional Factorial",
        )
        body = _texts(app.caption)
        assert "Step 1" in body and "Step 3" in body

    def test_valid_design_offers_the_generate_button(self):
        app = _run(
            "4_preview_design.py",
            all_two_level(),
            design_type="Fractional Factorial",
        )
        assert not app.exception, [e.value for e in app.exception]
        labels = [b.label for b in app.button]
        assert any("Generate Design" in str(label) for label in labels)

    def test_valid_design_has_no_blocking_error(self):
        app = _run(
            "4_preview_design.py",
            all_two_level(),
            design_type="Fractional Factorial",
        )
        assert "cannot be generated" not in _texts(app.error)


class TestStep2NeverRecommendsABlockedDesign:
    """The panel must not advise a design it also reports as unavailable.

    The reported bug family here is a message that points at an alternative the
    same panel has just rejected.  The 4-categorical case produced
    "Use D-Optimal for a model-based design" in the Latin Hypercube warning
    while D-Optimal sat in the blocked list one section below, because D-Optimal
    requires all-continuous factors and the warning only fires when a
    categorical factor is present.

    Rather than pin each message, these tests check the panel's own output: no
    design named in the blocked list may be named as a recommendation.
    """

    BLOCK_MARK = "Unavailable for the current factors"

    def _blocked_names(self, app):
        """Design labels rendered under the blocked heading, and everything below."""
        blocks = list(app.markdown)
        start = next(
            (i for i, b in enumerate(blocks) if self.BLOCK_MARK in b.value), None
        )
        assert start is not None, "expected a blocked-design section"
        names = []
        for block in blocks[start + 1:]:
            value = block.value
            if value.startswith("---") or value.startswith("📝"):
                break
            head = value.lstrip("❌ ").split(" - ")[0]
            names.append(head.replace("**", ""))
        return names

    def _recommendations(self, app):
        """The compatible + notes sections, which is where advice lives."""
        blocks = list(app.markdown)
        end = next(
            (i for i, b in enumerate(blocks) if self.BLOCK_MARK in b.value),
            len(blocks),
        )
        return " ".join(b.value for b in blocks[:end])

    @pytest.mark.parametrize(
        "factors_fn",
        [
            four_categoricals_two_multi_level,
            all_two_level,
            lambda: [cont("Temp"), cont("Pressure"), cont("Speed")],
            lambda: [
                cont("Temp"),
                Factor("Catalyst", FactorType.CATEGORICAL,
                       ChangeabilityLevel.EASY, levels=["lo", "hi"]),
            ],
            lambda: [cont("A"), cont("B"), cont("C"), cont("D"), cont("E"),
                     cont("F"), cont("G"), cont("H")],
        ],
    )
    def test_no_recommendation_names_a_blocked_design(self, factors_fn):
        app = _run("2_select_model.py", factors_fn())
        assert not app.exception, [e.value for e in app.exception]
        blocked = self._blocked_names(app)
        advice = self._recommendations(app)
        for name in blocked:
            assert name not in advice, (
                f"the panel recommends {name!r} in its compatible/notes "
                f"section while also listing it as blocked; blocked={blocked}"
            )


class TestStep2BlockedListNamesEveryDesign:
    """A blocked design must be named, never silently dropped.

    The two response surface designs are presented as one merged row.  When the
    whole group is blocked the row has to name both members, otherwise
    Box-Behnken vanishes from the panel without ever being reported.
    """

    def test_box_behnken_is_named_when_both_rsm_designs_are_blocked(self):
        app = _run("2_select_model.py", four_categoricals_two_multi_level())
        assert not app.exception, [e.value for e in app.exception]
        body = _texts(app.markdown)
        assert "Unavailable for the current factors" in body
        assert "CCD" in body
        assert "Box-Behnken" in body, (
            "Box-Behnken is blocked (a categorical factor is not continuous) but "
            "the merged response surface row did not name it"
        )

    def test_blocked_reason_names_each_rule_once(self):
        """The requirement text must not repeat once per offending factor."""
        app = _run("2_select_model.py", four_categoricals_two_multi_level())
        body = _texts(app.markdown)
        assert body.count("must be continuous for response surface design") == 1
        assert body.count("must have exactly 2 levels") == 1


def four_categoricals_one_continuous():
    """The 5-factor case from the report: 4 categorical, 1 continuous."""
    return [
        cat("C1", ["lo", "hi"]),
        cat("C2", ["lo", "hi"]),
        cat("C3", ["lo", "hi"]),
        cat("C4", ["lo", "hi"]),
        cont("Temp"),
    ]


def _frac_config(fraction):
    return {
        "fraction": fraction,
        "resolution": None,
        "generator_mode": "Standard (Recommended)",
        "custom_generators": None,
        "n_blocks": 1,
        "randomize": True,
    }


class _StaleSession:
    """A Step 4 session reused across reruns, so state changes are visible.

    ``AppTest`` keeps its own ``session_state`` between ``run()`` calls, which
    is what the real page does: Step 4 never regenerates by itself, so a
    configuration edited on Step 3 leaves the already-generated design sitting
    in session state.
    """

    def __init__(self, factors, design_config, design_type="Fractional Factorial"):
        self._app = AppTest.from_file(
            PAGES / "4_preview_design.py", default_timeout=20
        )
        self._app.session_state["factors"] = factors
        self._app.session_state["model_terms"] = _terms_for(factors)
        self._app.session_state["design_type"] = design_type
        self._app.session_state["design_config"] = dict(design_config)
        self.run()

    def run(self):
        with patch("src.ui.components.sidebar.build_standard_sidebar"):
            self._app.run()
        return self._app

    def set_config(self, **changes):
        self._app.session_state["design_config"] = {
            **self._app.session_state["design_config"], **changes
        }
        return self.run()

    def generate(self):
        button = next(
            b for b in self._app.button if "Generate" in b.label
        )
        with patch("src.ui.components.sidebar.build_standard_sidebar"):
            button.click().run()
        return self._app

    @property
    def app(self):
        return self._app

    @property
    def design(self):
        return self._app.session_state["design"]

    @property
    def metadata(self):
        return self._app.session_state["design_metadata"]

    @property
    def warnings(self):
        return " ".join(w.value for w in self._app.warning)

    @property
    def infos(self):
        return " ".join(i.value for i in self._app.info)


class TestFractionMatchesTheGeneratedDesign:
    """Step 4 must not show a design that predates the current configuration.

    Reported case: 5 factors, a 1/2 fraction selected, Step 3 showing
    Resolution V / 16 runs, and the generated design showing 8 runs at
    Resolution III.  Both readings were individually correct -- the run count
    follows ``design_config['fraction']``, and Step 4 renders the design
    already in session state without regenerating.  Nothing invalidated the
    design when only the *configuration* changed, so the 1/4 design stayed on
    screen under a 1/2 configuration with nothing saying so.
    """

    def test_half_fraction_generates_sixteen_runs_at_resolution_five(self):
        """The configuration the user selected is the one that is built."""
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/2"))
        s.generate()
        assert len(s.design) == 16
        assert s.metadata["resolution"] == 5
        assert s.metadata["fraction"] == "1/2"

    def test_quarter_fraction_generates_eight_runs_at_resolution_three(self):
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/4"))
        s.generate()
        assert len(s.design) == 8
        assert s.metadata["resolution"] == 3

    def test_changing_the_fraction_marks_the_design_out_of_date(self):
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/4"))
        s.generate()
        assert len(s.design) == 8
        s.set_config(fraction="1/2")
        assert "out of date" in s.warnings
        assert "`fraction`: generated with `1/4`, now `1/2`" in s.warnings
        # the stale design is still on screen, and the warning says so
        assert len(s.design) == 8
        assert "8 runs at Resolution 3" in s.infos
        assert "Generate New Design" in s.infos

    def test_no_staleness_warning_when_the_configuration_matches(self):
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/2"))
        s.generate()
        s.run()
        assert "out of date" not in s.warnings
        assert "generated with" not in s.warnings

    def test_regenerating_adopts_the_new_configuration_and_clears_the_warning(self):
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/4"))
        s.generate()
        s.set_config(fraction="1/2")
        assert "out of date" in s.warnings
        s.generate()
        assert len(s.design) == 16
        assert s.metadata["resolution"] == 5
        assert s.metadata["fraction"] == "1/2"
        assert "out of date" not in s.warnings

    def test_other_configuration_keys_are_detected_too(self):
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/2"))
        s.generate()
        s.set_config(n_blocks=2)
        assert "out of date" in s.warnings
        assert "`n_blocks`: generated with `1`, now `2`" in s.warnings

    def test_design_without_a_config_snapshot_does_not_crash(self):
        """Projects saved before the snapshot existed must still render."""
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/2"))
        s.generate()
        s.metadata.pop("config_snapshot", None)
        app = s.run()
        assert not app.exception, [e.value for e in app.exception]
        assert "out of date" not in s.warnings


class TestFractionWidgetAdoptsStoredConfig:
    """Step 3's Fraction Size widget must open on the configured fraction.

    ``design_config`` is restored verbatim on project load and is never cleared
    by ``invalidate_downstream_state``, but a ``selectbox`` with no ``index``
    always opens at 0 ('1/2').  The widget therefore showed one fraction while
    Step 4 generated from another, and the first Step 3 render silently
    overwrote the loaded value.
    """

    def test_widget_opens_on_the_stored_fraction(self):
        app = _run("3_choose_design.py", four_categoricals_one_continuous(),
                   design_type="Fractional Factorial",
                   extra={"design_config": _frac_config("1/4")})
        assert not app.exception, [e.value for e in app.exception]
        box = next(s for s in app.selectbox if s.label == "Fraction Size")
        assert box.value == "1/4"
        body = _texts(app.markdown)
        assert "**Estimated runs:** 8" in body
        assert app.session_state["design_config"]["fraction"] == "1/4"

    def test_stored_fraction_invalid_for_k_falls_back_to_the_first_option(self):
        """1/16 needs k>=7; with 5 factors it must not be offered or kept."""
        app = _run("3_choose_design.py", four_categoricals_one_continuous(),
                   design_type="Fractional Factorial",
                   extra={"design_config": _frac_config("1/16")})
        assert not app.exception, [e.value for e in app.exception]
        box = next(s for s in app.selectbox if s.label == "Fraction Size")
        assert box.value == "1/2"
        assert app.session_state["design_config"]["fraction"] == "1/2"

    def test_no_stored_config_opens_on_the_first_option(self):
        app = _run("3_choose_design.py", four_categoricals_one_continuous(),
                   design_type="Fractional Factorial")
        assert not app.exception, [e.value for e in app.exception]
        box = next(s for s in app.selectbox if s.label == "Fraction Size")
        assert box.value == "1/2"
        assert "**Estimated runs:** 16" in _texts(app.markdown)


class TestDesignMatrixRepaintsOnRegeneration:
    """The design matrix table must repaint when a new design is generated.

    An unkeyed ``st.dataframe`` keeps its existing frontend component instance
    across the regenerate -> ``st.rerun()`` chain, so after "Generate New
    Design" the browser kept showing the previous table and the only way to see
    the new one was to toggle the Display radio. The server was already sending
    the new data -- the dataframe element carried the new row count and
    RunOrder immediately -- so this is a client-side repaint, invisible to
    ``AppTest`` (which has no browser) and invisible to the dataframe's own
    data-diffing.

    The fix keys the table off a generation counter bumped on every successful
    generation, forcing Streamlit to recreate the component. ``AppTest`` does not
    expose an element's ``key``, so these tests assert the counter -- the thing
    the key is derived from -- plus the server-side content invariant.
    """

    def _generation(self, session):
        # AppTest's session_state has no .get(); probe with `in` instead.
        if "_design_generation" in session:
            return session["_design_generation"]
        return 0

    def _preview_runs(self, app):
        """RunOrder values of the rendered 'First 10 rows' preview table."""
        for element in app.dataframe:
            frame = element.value
            if frame is None or "RunOrder" not in frame.columns:
                continue
            return list(frame["RunOrder"])
        return None

    def test_counter_starts_absent_and_increments_per_generation(self):
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/2"))
        assert self._generation(s.app.session_state) == 0
        s.generate()
        assert self._generation(s.app.session_state) == 1
        s.generate()
        assert self._generation(s.app.session_state) == 2

    def test_preview_table_reflects_the_regenerated_design(self):
        """Server-side guard: the table carries the new design's runs."""
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/4"))
        s.generate()
        assert self._preview_runs(s.app) == [1, 2, 3, 4, 5, 6, 7, 8]
        s.set_config(fraction="1/2")
        s.generate()
        # the regenerated design has 16 runs, so "First 10 rows" shows 10
        assert self._preview_runs(s.app) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert len(s.design) == 16

    def test_identical_regeneration_still_bumps_the_counter(self):
        """A byte-identical table must still remount, so a counter is required.

        A key derived from the table contents would be unchanged here and the
        component would not be recreated, leaving the user with no visible
        feedback that anything happened.
        """
        config = {**_frac_config("1/2"), "randomize": False}
        s = _StaleSession(four_categoricals_one_continuous(), config)
        s.generate()
        first = self._preview_runs(s.app)
        assert self._generation(s.app.session_state) == 1
        s.generate()
        assert self._preview_runs(s.app) == first, (
            "expected a byte-identical table for an unrandomised re-generation"
        )
        assert self._generation(s.app.session_state) == 2, (
            "the counter must advance even when the regenerated table is "
            "identical, or the keyed component is not recreated"
        )

    def test_loaded_design_without_a_generation_renders(self):
        """A design arriving from a project file has no counter; it must render."""
        s = _StaleSession(four_categoricals_one_continuous(), _frac_config("1/2"))
        s.generate()
        del s.app.session_state["_design_generation"]
        app = s.run()
        assert not app.exception, [e.value for e in app.exception]
        assert self._preview_runs(app) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class TestPreviewTableKeyWiring:
    """Structural guard: the preview table's key must come from the counter.

    ``AppTest`` does not expose an element's ``key``, and the generation counter
    advances regardless of how the key is built, so no behavioural test can
    catch the key being derived from the table contents instead. Doing that
    would reintroduce the original bug for an unchanged configuration: the
    regenerated table is byte-identical, the key would not change, and the
    component would not be recreated.

    This is therefore a source-level assertion. It is deliberately narrow --
    it checks that the key expression reads the generation counter, nothing
    about how the table is built.
    """

    def test_preview_table_key_is_derived_from_the_generation_counter(self):
        # The page module is named "4_preview_design.py" and cannot be imported
        # by name, so the source is read from disk. 4_preview_design.py itself
        # is already exercised by AppTest elsewhere in this file.
        source = (PAGES / "4_preview_design.py").read_text(encoding="utf-8")
        marker = "st.dataframe(\n        preview_df,"
        assert marker in source, "preview table call site moved; update this guard"
        call = source.split(marker, 1)[1].split(")", 1)[0]
        assert "_design_generation" in call, (
            "the design preview table must be keyed off the generation counter "
            "so a regeneration recreates the component; got: " + call.strip()
        )
