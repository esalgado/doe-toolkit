"""Application state for the NiceGUI prototype.

Streamlit is untouched; this module mirrors the session-state keys that
``src/ui/utils/state_management.initialize_session_state`` establishes, so the two
apps stay parallel until the full cutover. Each browser client gets its own
isolated state via NiceGUI's per-client ``app.storage.user``.

Persistence contract (full port):
------------------------------
``app.storage.user`` must contain **only JSON-serializable** values
(lists/dicts/str/num/None). Non-serializable objects (``Factor``,
``LinearConstraint``, ``DesignSpace``, pandas DataFrames, numpy arrays) are
materialized on read from their serialized forms by ``workspace.py``; keep that
layer the single place that converts to/from objects. This differs from the
Streamlit app where raw objects are stored in memory per session, but it keeps
the NiceGUI client state out of the storage-file pitfalls described in
AGENTS.md.
"""

import copy
from typing import Any, Dict

from nicegui import app

DEFAULTS: Dict[str, Any] = {
    # Step 1: Define Factors (stored as JSON-safe factor ROWS, not Factor objects)
    'factors': [],

    # Step 2: Select Model
    'model_terms': None,

    # Step 3: Choose Design
    'design_type': None,
    'design_config': {},
    'constraints': [],  # JSON-safe: {coefficients:{name:float}, bound:float, type:'le'|'ge'|'eq'}

    # Step 4: Preview Design
    # 'design'/'design_natural' are stored as records (list of row dicts) or None
    'design': None,
    'design_natural': None,
    'design_space': None,
    'design_metadata': {},
    'random_seed': None,
    'random_seed_used': True,
    'preview_display_mode': 'first',

    # Step 5: Import Results ('responses' = {name: [float|None, ...]})
    'responses': {},
    'response_names': [],
    'excluded_rows': [],
    'response_definitions': [],  # [{name, units}]

    # Step 6: Analyze (non-serializable artifacts are recomputed per render)
    'fitted_models': {},
    'model_terms_per_response': {},
    'diagnostics_summary': None,
    'quality_report': None,
    'step_2_complete': False,

    # Step 7: Augmentation
    'show_augmentation': False,
    'augmentation_plans': [],
    'selected_plan': None,
    'augmented_design': None,

    # Step 8: Optimize
    'optimization_results': None,

    # UI state
    'current_step': 1,
    'show_save_project': False,
    'show_generate_report': False,
}


def get_state() -> Dict[str, Any]:
    """Return this client's state dict, seeding any missing keys."""
    state = app.storage.user
    for key, default in DEFAULTS.items():
        if key not in state:
            state[key] = copy.deepcopy(default)
    return state


def reset_state(state: Dict[str, Any]) -> None:
    """Reset a client state dict to the pristine ``DEFAULTS``.

    Used by the 'Start Fresh' actions on Home and in the sidebar shell.
    ``DEFAULTS`` holds shared mutable containers, so each value is deep-copied to
    keep clients (and later resets) independent.
    """
    for key, default in DEFAULTS.items():
        state[key] = copy.deepcopy(default)


_PROJECT_KEYS = (
    'factors', 'model_terms', 'design_type', 'design', 'responses',
    'response_definitions', 'augmented_design', 'optimization_results',
)


def any_project_state(state: Dict[str, Any]) -> bool:
    """True if the user has any (potentially loss-worthy) project work in state."""
    return any(bool(state.get(key)) for key in _PROJECT_KEYS)


def is_step_complete(state: Dict[str, Any], step: int) -> bool:
    """Check whether a workflow step is complete and valid (dict-based port)."""
    if step == 1:  # Define Factors
        return len(state.get('factors') or []) > 0

    elif step == 2:  # Select Model (optional — complete if user proceeded or terms set)
        return (
            state.get('model_terms') is not None or
            state.get('design_type') is not None
        )

    elif step == 3:  # Choose Design
        return state.get('design_type') is not None

    elif step == 4:  # Preview Design
        return state.get('design') is not None

    elif step == 5:  # Import Results
        responses = state.get('responses', {})
        return (len(responses) > 0) and (state.get('design') is not None)

    return False


def can_access_step(state: Dict[str, Any], step: int) -> bool:
    """Check whether the user can access a step (all prerequisites complete)."""
    if step == 1:
        return True
    if step == 5:
        # Auto-detect CSV import mode can be the entry point too.
        return True
    if step == 6:
        return is_step_complete(state, 5)
    if step == 7:
        return state.get('design') is not None
    if step == 8:
        return is_step_complete(state, 6)
    for i in range(1, step):
        if not is_step_complete(state, i):
            return False
    return True


def get_workflow_progress(state: Dict[str, Any]) -> int:
    """Return the furthest workflow step that has been 'entered' so far."""
    progress = 1
    for step in range(1, 9):
        if state.get('current_step', 1) >= step:
            progress = step
    return progress


def _reset_downstream(state: Dict[str, Any], keys: list) -> None:
    for key in keys:
        if key in DEFAULTS:
            state[key] = DEFAULTS[key]


def invalidate_downstream_state(state: Dict[str, Any], from_step: int) -> None:
    """Invalidate state for steps after the given step (dict-based port).

    Called when the user modifies an earlier step (e.g. changes factors).
    """
    cleared = [
        'design', 'design_natural', 'design_space', 'design_metadata',
        'responses', 'response_names', 'excluded_rows', 'fitted_models',
        'model_terms_per_response', 'diagnostics_summary', 'quality_report',
        'augmentation_plans', 'selected_plan', 'augmented_design',
        'optimization_results',
    ]
    if from_step <= 1:
        # Factors changed — clear everything downstream.
        state['design_type'] = None
        state['design_config'] = DEFAULTS['design_config']
        _reset_downstream(state, cleared)
    elif from_step <= 2:
        # Design type changed — clear design and downstream.
        _reset_downstream(state, cleared)
    elif from_step <= 3:
        # Design config regenerated — clear results and downstream.
        _reset_downstream(state, cleared)
    elif from_step <= 4:
        # New data imported — clear analysis and downstream.
        _reset_downstream(state, cleared[5:])
    elif from_step <= 5:
        # Model refit — clear diagnostics and downstream.
        _reset_downstream(state, cleared[6:])