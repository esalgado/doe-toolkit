"""
Tests for loading and saving projects whose design no longer matches its factors.

A ``.doeproject`` can be saved in a state the current generators cannot build --
most commonly when a factor is given an extra level after the design type was
chosen.  Loading used to restore that state silently, so the mismatch only
surfaced several steps later as a traceback from a generator.

The contract these tests lock in:

* loading reports the mismatch, with the specific reason;
* loading never rewrites ``design_type`` or ``design_config``;
* saving an unbuildable project is allowed, with a warning.

The Streamlit session is faked, following ``test_export_profiler.py``, so these
run without a live app.
"""

import json
import types

import pytest

import src.ui.utils.state_management as sm
import src.ui.components.sidebar as sidebar
from src.core.factors import ChangeabilityLevel, Factor, FactorType


class FakeSessionState(dict):
    """Stand-in for ``st.session_state``."""


class StRecorder:
    """Collects the messages a helper emits, in order."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.infos = []
        self.captions = []

    def error(self, msg, *a, **k):
        self.errors.append(str(msg))

    def warning(self, msg, *a, **k):
        self.warnings.append(str(msg))

    def info(self, msg, *a, **k):
        self.infos.append(str(msg))

    def success(self, msg, *a, **k):
        pass

    def caption(self, msg, *a, **k):
        self.captions.append(str(msg))

    def dataframe(self, *a, **k):
        pass

    def markdown(self, *a, **k):
        pass

    @property
    def sidebar(self):
        return self


@pytest.fixture
def recorder(monkeypatch):
    """Fake Streamlit for both modules under test.

    Both ``sm`` and ``sidebar`` must be patched.  Assigning to the *real*
    ``streamlit.session_state`` replaces Streamlit's session-state proxy with a
    plain dict, and that survives into later tests in the same process -- it
    breaks every ``streamlit.testing.v1.AppTest`` run that follows, since
    AppTest relies on that proxy.  Patching both module-level ``st`` references
    keeps the real module untouched.
    """
    rec = StRecorder()
    monkeypatch.setattr(sm, "st", rec, raising=False)
    monkeypatch.setattr(sidebar, "st", rec, raising=False)
    return rec


def _factor(name, ftype, levels, changeability=ChangeabilityLevel.EASY):
    return Factor(name, ftype, changeability, levels=list(levels))


def cont2(name):
    # Names avoid the reserved tokens in factor_naming.PATSY_RESERVED, which
    # include "C" and "I" (Patsy operators).  Those are renamed on load by
    # design; see TestReservedNamesBelow.
    return _factor(name, FactorType.CONTINUOUS, [10.0, 20.0])


def disc3(name):
    return _factor(name, FactorType.DISCRETE_NUMERIC, [10.0, 20.0, 30.0])


def cat2(name):
    return _factor(name, FactorType.CATEGORICAL, ["lo", "hi"])


def project_json(factors, design_type, design_config=None):
    return json.dumps({
        "version": "0.3.0",
        "factors": [
            {
                "name": f.name,
                "type": f.factor_type.value,
                "changeability": f.changeability.value,
                "levels": f.levels,
                "units": f.units,
            }
            for f in factors
        ],
        "design_type": design_type,
        "design_config": design_config or {},
    })


class TestLoadReportsMismatch:
    """The real-world case: Fractional Factorial + a 3-level factor."""

    def test_three_level_factor_under_fractional_is_reported(self, recorder):
        factors = [cont2("A"), cont2("B"), cont2("C"), disc3("niveles")]
        original_config = {"fraction": "1/8", "randomize": True}

        sm.st.session_state = FakeSessionState()
        sm.load_project_file(
            project_json(factors, "Fractional Factorial", original_config)
        )

        assert recorder.errors, "expected an error describing the mismatch"
        joined = " ".join(recorder.errors)
        assert "niveles" in joined
        assert "3 levels" in joined

    def test_design_type_and_config_are_preserved(self, recorder):
        """The user's decision: never auto-switch, never rewrite the config."""
        factors = [cont2("A"), cont2("B"), cont2("C"), disc3("niveles")]
        original_config = {"fraction": "1/8", "randomize": True}

        sm.st.session_state = FakeSessionState()
        sm.load_project_file(
            project_json(factors, "Fractional Factorial", original_config)
        )

        assert sm.st.session_state["design_type"] == "Fractional Factorial"
        assert sm.st.session_state["design_config"] == original_config

    def test_two_level_categorical_project_loads_without_error(self, recorder):
        """The Phase A fix makes this a legitimate design."""
        factors = [cont2("A"), cont2("B"), cont2("C"), cat2("Lotes")]

        sm.st.session_state = FakeSessionState()
        sm.load_project_file(
            project_json(factors, "Fractional Factorial", {"fraction": "1/2"})
        )

        assert not recorder.errors
        assert sm.st.session_state["design_type"] == "Fractional Factorial"

    def test_too_few_factors_reported(self, recorder):
        sm.st.session_state = FakeSessionState()
        sm.load_project_file(
            project_json([cont2("A"), cont2("B")], "Fractional Factorial")
        )
        assert any("at least 3 factors" in e for e in recorder.errors)

    def test_non_continuous_blocks_response_surface(self, recorder):
        factors = [cont2("A"), cont2("B"), cat2("C")]
        sm.st.session_state = FakeSessionState()
        sm.load_project_file(
            project_json(factors, "Response Surface (CCD)")
        )
        assert any("continuous" in e for e in recorder.errors)

    def test_warning_only_project_is_not_an_error(self, recorder):
        """A large Full Factorial warns but still loads cleanly."""
        factors = [cont2(chr(65 + i)) for i in range(8)]
        sm.st.session_state = FakeSessionState()
        sm.load_project_file(project_json(factors, "Full Factorial"))

        assert not recorder.errors
        assert any("runs" in w for w in recorder.warnings)

    def test_unknown_design_type_is_reported_not_raised(self, recorder):
        factors = [cont2("A"), cont2("B"), cont2("C")]
        sm.st.session_state = FakeSessionState()
        sm.load_project_file(project_json(factors, "Plackett-Burman"))

        assert sm.st.session_state["design_type"] == "Plackett-Burman"
        assert any("does not recognize" in w for w in recorder.warnings)

    def test_project_without_design_type_is_silent(self, recorder):
        """Nothing to validate when no design type was saved."""
        factors = [cont2("A"), cont2("B"), cont2("Temp"), disc3("niveles")]
        sm.st.session_state = FakeSessionState()
        sm.load_project_file(project_json(factors, None))

        assert not recorder.errors
        assert not recorder.warnings

    def test_user_is_told_how_to_fix_it(self, recorder):
        factors = [cont2("A"), cont2("B"), cont2("C"), disc3("niveles")]
        sm.st.session_state = FakeSessionState()
        sm.load_project_file(
            project_json(factors, "Fractional Factorial")
        )
        assert any("Step 3" in i or "Step 1" in i for i in recorder.infos)


class TestReservedNamesBelow:
    """Factor names that collide with Patsy operators are renamed on load.

    ``factor_naming`` reserves ``C``, ``I``, ``Q`` and friends because Patsy
    reads them as operators, so a project storing a factor literally named
    ``C`` comes back as ``f_C``.  That is the pre-existing sanitization
    contract and is independent of design/factor fitness, but the two interact:
    a design whose factor set contains a reserved name should still be
    validated against the names as loaded.
    """

    def test_reserved_name_is_reported_as_renamed(self, recorder):
        sm.st.session_state = FakeSessionState()
        sm.load_project_file(
            project_json([cont2("A"), cont2("B"), cont2("C")],
                         "Fractional Factorial")
        )
        assert any("updated for compatibility" in w for w in recorder.warnings)

    def test_renamed_project_still_validates_cleanly(self, recorder):
        """The rename must not be mistaken for a design/factor mismatch."""
        sm.st.session_state = FakeSessionState()
        sm.load_project_file(
            project_json([cont2("A"), cont2("B"), cont2("C")],
                         "Fractional Factorial")
        )
        assert not recorder.errors


class TestSaveAllowsUnbuildableProject:
    """Saving stays available; it only warns."""

    def test_warns_for_unbuildable_design(self, recorder):
        sidebar.st.session_state = FakeSessionState({
            "factors": [cont2("A"), cont2("B"), cont2("C"), disc3("niveles")],
            "design_type": "Fractional Factorial",
        })
        sidebar._warn_if_design_unbuildable()

        assert recorder.warnings
        assert "niveles" in " ".join(recorder.captions)

    def test_silent_for_buildable_design(self, recorder):
        sidebar.st.session_state = FakeSessionState({
            "factors": [cont2("A"), cont2("B"), cont2("C")],
            "design_type": "Fractional Factorial",
        })
        sidebar._warn_if_design_unbuildable()
        assert not recorder.warnings

    def test_silent_without_design_type(self, recorder):
        sidebar.st.session_state = FakeSessionState({"factors": [cont2("A")]})
        sidebar._warn_if_design_unbuildable()
        assert not recorder.warnings

    def test_silent_for_unknown_design_type(self, recorder):
        sidebar.st.session_state = FakeSessionState({
            "factors": [cont2("A")],
            "design_type": "Plackett-Burman",
        })
        sidebar._warn_if_design_unbuildable()
        assert not recorder.warnings

    def test_project_file_still_serializes_everything(self, recorder):
        """A warned-about project must still save the design as-is."""
        state = FakeSessionState({
            "factors": [cont2("A"), cont2("B"), cont2("C"), disc3("niveles")],
            "design_type": "Fractional Factorial",
            "design_config": {"fraction": "1/8"},
        })
        sm.st.session_state = state

        saved = json.loads(sm.create_project_file())
        assert saved["design_type"] == "Fractional Factorial"
        assert saved["design_config"] == {"fraction": "1/8"}
        assert len(saved["factors"]) == 4
