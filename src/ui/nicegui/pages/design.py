"""Step 3: Choose & Configure Design (NiceGUI).

Mirrors ``3_choose_design.py`` but persists the configuration into
``state['design_config']`` as the user adjusts widgets (JSON-safe), and — unlike
the Streamlit app — that same dict is the ONLY input ``design_generation``
consumes, closing the Step 3 -> Step 4 handoff gap.
"""

from typing import Any, Callable, Dict, List, Optional

from nicegui import ui

from src.core.factors import Factor
from src.core.optimal.constraints import LinearConstraint
from src.ui.components.model_builder import format_full_equation
from src.ui.nicegui.design_config import (
    CCD_ALPHA_CHOICES,
    alpha_for_label,
    design_options,
    ensure_ccd_config,
    ensure_d_optimal_config,
    ensure_fractional_config,
    ensure_full_factorial_config,
    ensure_lhs_config,
    estimate_runs,
    fraction_to_p,
    split_plot_factors,
    valid_fractions,
    validate_config,
)
from src.ui.nicegui.layout import FIELD_CLASSES, FIELD_PROPS, create_shell
from src.ui.nicegui.state import can_access_step, get_state, invalidate_downstream_state
from src.ui.nicegui.workspace import get_constraints, get_factors, store_constraints


def _cfg(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg = state.get('design_config')
    if not isinstance(cfg, dict):
        cfg = {}
        state['design_config'] = cfg
    return cfg


@ui.page('/design')
def design_page(state: Dict[str, Any] | None = None) -> None:
    app_state = state if state is not None else get_state()
    create_shell(app_state, active='/design')
    cfg = _cfg(app_state)

    with ui.column().classes('p-6 w-full max-w-4xl'):
        ui.label('Step 3: Choose Design Type').classes('text-2xl font-bold')

        if not can_access_step(app_state, 3):
            ui.label('Please complete Steps 1-2 first.').classes('text-gray-600')
            ui.button('2. Select Model', on_click=lambda: ui.navigate.to('/model')).props('color=primary')
            return

        factors = get_factors(app_state)
        model_terms = list(app_state.get('model_terms') or [])

        _show_factor_summary(factors)
        _show_selected_model(model_terms)

        ui.separator().classes('my-2')
        ui.label('Select design type').classes('text-lg font-bold')
        ui.label(f'{len(factors)} defined factors').classes('text-sm text-gray-600')

        design_type = app_state.get('design_type')
        options = design_options(factors)
        with ui.grid(columns=2).classes('w-full gap-4 mt-3'):
            for option in options:
                with ui.card().classes('w-full'):
                    selected = option['name'] == design_type
                    title = f'✓ {option["name"]}' if selected else option['name']
                    ui.label(title).classes(f'text-base font-bold {"text-blue-600" if selected else ""}')
                    ui.label(option['description']).classes('text-sm text-gray-600')
                    ui.label(f'When to use: {option["when_to_use"]}').classes('text-xs text-gray-500')
                    ui.label(f'Typical runs: {option["runs"]}').classes('text-xs text-gray-500')
                    if option['enabled']:
                        ui.button(
                            'Selected' if selected else 'Select',
                            on_click=lambda name=option['name']: _select_design(app_state, name),
                        ).props('color=primary' if not selected else 'disable' if selected else '')
                    else:
                        ui.label(f'🔒 {option["disabled_reason"]}').classes('text-xs text-orange-500')

        if design_type:
            ui.separator().classes('my-4')
            ui.label(f'⚙️ Configure {design_type}').classes('text-lg font-bold')
            _render_config_form(cfg, app_state, design_type, factors, model_terms)

            ui.separator().classes('my-4')
            with ui.row().classes('gap-2'):
                ui.button('← Back to Model', on_click=lambda: ui.navigate.to('/model')).props('outline')
                ui.button(
                    'Generate Design →',
                    on_click=lambda: _go_to_preview(app_state, design_type, factors, model_terms),
                ).props('color=primary')


def _select_design(state: Dict[str, Any], name: str) -> None:
    state['design_type'] = name
    invalidate_downstream_state(state, from_step=2)
    ui.navigate.reload()


def _show_factor_summary(factors: List[Factor]) -> None:
    if not factors:
        return
    import pandas as pd

    rows = [{
        'Factor': f.name,
        'Type': f.factor_type.value.replace('_', ' ').title(),
        'Levels': f'[{f.levels[0]}, {f.levels[1]}]' if f.is_continuous() else f'{len(f.levels)} levels',
        'Changeability': f.changeability.value.title(),
    } for f in factors]
    with ui.expansion('Defined factors').classes('mt-2'):
        ui.table.from_pandas(pd.DataFrame(rows), row_key='Factor').props('flat bordered dense')


def _show_selected_model(model_terms: List[str]) -> None:
    if not model_terms:
        ui.label(
            '⚠️ No analysis model selected yet. Some design types (especially D-Optimal) '
            'require knowing the model upfront.'
        ).classes('text-orange-600 text-sm')
        return
    with ui.expansion('Selected model (from Step 2)').classes('mt-1').props('icon=model_training'):
        ui.markdown(f'**{format_full_equation(model_terms, "Y")}**')
        ui.label(f'{len(model_terms)} terms to estimate').classes('text-sm text-gray-600')


# --- Config forms -----------------------------------------------------------


def _render_config_form(
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    design_type: str,
    factors: List[Factor],
    model_terms: List[str],
) -> None:
    if design_type == 'Full Factorial':
        _form_full_factorial(cfg, factors)
    elif design_type == 'Fractional Factorial':
        _form_fractional(cfg, factors, model_terms)
    elif design_type == 'Response Surface (CCD)':
        _form_ccd(cfg, factors)
    elif design_type == 'Response Surface (Box-Behnken)':
        _form_bbd(cfg, factors)
    elif design_type == 'D-Optimal':
        _form_d_optimal(cfg, state, factors, model_terms)
    elif design_type == 'Latin Hypercube':
        _form_lhs(cfg, factors)
    elif design_type == 'Split-Plot':
        _form_split_plot(cfg, factors)


def _number(
    cfg: Dict[str, Any], label: str, key: str, value: Any, min_value: int, max_value: int,
    help_text: str = '', refresh=None,
) -> None:
    cfg.setdefault(key, value)
    ui.number(
        label, min=min_value, max=max_value, precision=0,
        value=int(cfg.get(key, value)),
        on_change=lambda e: (_store_int(cfg, key, e), refresh() if refresh else None),
    ).props(FIELD_PROPS).classes(FIELD_CLASSES).tooltip(help_text if help_text else None)


def _store_int(cfg: Dict[str, Any], key: str, event: Any) -> None:
    cfg.__setitem__(key, int(event.value))


def _center_points(cfg: Dict[str, Any], default: int, refresh=None) -> None:
    cfg.setdefault('n_center_points', default)
    ui.number(
        'Number of Center Points', min=0, max=10, precision=0,
        value=int(cfg.get('n_center_points', default)),
        on_change=lambda e: (_store_int(cfg, 'n_center_points', e), refresh() if refresh else None),
    ).props(FIELD_PROPS).classes(FIELD_CLASSES).tooltip('Center points estimate pure error')


def _randomize_switch(cfg: Dict[str, Any], text: str, key: str, refresh=None) -> None:
    cfg.setdefault(key, True)

    def on_value_change(event: Any) -> None:
        cfg.__setitem__(key, bool(event.value))
        if refresh:
            refresh()

    ui.switch(text, value=bool(cfg.get(key, True))).on_value_change(on_value_change)


def _form_full_factorial(cfg: Dict[str, Any], factors: List[Factor]) -> None:
    k = len(factors)
    ensure_full_factorial_config(cfg)
    est = _estimated(cfg, 'Full Factorial', k)
    with ui.column().classes('gap-1'):
        ui.label('Full Factorial Configuration — 2-level only (core limitation; '
                 '3-level option is preserved for parity but not yet in the core engine).').classes(
            'text-sm text-gray-500')
        ui.radio(
            {2: '2 levels', 3: '3 levels'},
            value=int(cfg.get('n_levels', 2)),
            on_change=lambda e: (cfg.__setitem__('n_levels', int(e.value)), est.refresh()),
        ).props('inline')
        _center_points(cfg, default=3, refresh=est.refresh)
        _number(cfg, 'Number of Replicates', 'n_replicates', 1, 1, 10,
                'Full repetitions of the entire design', refresh=est.refresh)
        _number(cfg, 'Number of Blocks', 'n_blocks', 1, 1, 10,
                'Divide runs into blocks to account for nuisance variation', refresh=est.refresh)
        _randomize_switch(cfg, 'Randomize Run Order', 'randomize', refresh=est.refresh)


def _form_fractional(cfg: Dict[str, Any], factors: List[Factor], model_terms: List[str]) -> None:
    k = len(factors)
    ensure_fractional_config(cfg, k)
    valid = valid_fractions(k)
    fraction = cfg.get('fraction', valid[0] if valid else '1/2')
    resolution = cfg.get('resolution', 5)
    if any(t.startswith('I(') and '**2' in t for t in model_terms):
        ui.label('⚠️ Your model includes quadratic terms — fractional factorial designs '
                 'cannot estimate quadratic effects. Prefer a Response Surface design.').classes(
            'text-orange-600 text-sm')

    est = _estimated(cfg, 'Fractional Factorial', k)
    with ui.column().classes('gap-1'):
        if not valid:
            ui.label('No valid fraction for this many factors.').classes('text-red-600 text-sm')
            return
        ui.select(
            {f: f for f in ['1/2', '1/4', '1/8', '1/16'] if f in valid},
            label='Fraction size', value=fraction,
            on_change=lambda e: (cfg.__setitem__('fraction', e.value), est.refresh()),
        ).props(FIELD_PROPS).classes(FIELD_CLASSES)
        ui.select(
            {3: 'III (minimum)', 4: 'IV', 5: 'V (least aliasing)'},
            label='Resolution', value=resolution,
            on_change=lambda e: (cfg.__setitem__('resolution', int(e.value)), est.refresh()),
        ).props(FIELD_PROPS).classes(FIELD_CLASSES)
        ui.radio(
            {'Standard (Recommended)': 'Standard (Recommended)', 'Custom': 'Custom'},
            value=cfg.get('generator_mode', 'Standard (Recommended)'),
            on_change=lambda e: (cfg.__setitem__('generator_mode', e.value), est.refresh()),
        ).props('inline')
        if cfg.get('generator_mode') == 'Custom':
            p = fraction_to_p(fraction)
            ui.textarea(
                f'Generators (one per line, need {p})',
                value='\n'.join(cfg.get('custom_generators') or []),
                on_change=lambda e: cfg.__setitem__(
                    'custom_generators',
                    [g.strip() for g in e.value.splitlines() if g.strip()],
                ),
            ).props(FIELD_PROPS).classes(FIELD_CLASSES)
        _center_points(cfg, default=3, refresh=est.refresh)
        _number(cfg, 'Number of Blocks', 'n_blocks', 1, 1, 10,
                'Divide runs into blocks to account for nuisance variation', refresh=est.refresh)
        _randomize_switch(cfg, 'Randomize Run Order', 'randomize', refresh=est.refresh)


def _form_ccd(cfg: Dict[str, Any], factors: List[Factor]) -> None:
    k = len(factors)
    ensure_ccd_config(cfg)
    label = cfg.get('alpha_label') or 'Rotatable'
    est = _estimated(cfg, 'Response Surface (CCD)', k)
    with ui.column().classes('gap-1'):
        ui.select(
            {c: c for c in CCD_ALPHA_CHOICES}, label='Axial distance (α)', value=label,
            on_change=lambda e: (cfg.__setitem__('alpha_label', e.value),
                                 cfg.__setitem__('alpha', alpha_for_label(e.value)),
                                 est.refresh()),
        ).props(FIELD_PROPS).classes(FIELD_CLASSES)
        _center_points(cfg, default=5, refresh=est.refresh)
        _randomize_switch(cfg, 'Randomize Run Order', 'randomize', refresh=est.refresh)


def _form_bbd(cfg: Dict[str, Any], factors: List[Factor]) -> None:
    k = len(factors)
    est = _estimated(cfg, 'Response Surface (Box-Behnken)', k)
    with ui.column().classes('gap-1'):
        if k < 3:
            ui.label('Box-Behnken requires at least 3 factors.').classes('text-red-600 text-sm')
            return
        if k > 7:
            ui.label(f'Box-Behnken with {k} factors is very large — consider CCD instead.').classes(
                'text-orange-600 text-sm')
        _center_points(cfg, default=5, refresh=est.refresh)
        _randomize_switch(cfg, 'Randomize Run Order', 'randomize', refresh=est.refresh)


def _form_d_optimal(
    cfg: Dict[str, Any], state: Dict[str, Any], factors: List[Factor], model_terms: List[str]
) -> None:
    if not model_terms:
        ui.label('⚠️ No model defined. Return to Step 2 to select an analysis model first.').classes(
            'text-orange-600 text-sm')
        ui.button('← Go to Step 2: Select Model', on_click=lambda: ui.navigate.to('/model')).props('outline')
        return

    min_runs = len(model_terms)
    ensure_d_optimal_config(cfg, min_runs)
    ui.markdown(f'**Selected model:** `{format_full_equation(model_terms, "Y")}`')
    ui.label(f'{len(model_terms)} terms to estimate — D-optimal design is optimized for these.').classes(
        'text-sm text-gray-600')

    n_runs = int(cfg.get('n_runs', min_runs * 2))
    ui.number(
        'Number of Runs (minimum: {})'.format(min_runs),
        min=min_runs, max=min_runs * 5, precision=0, value=n_runs,
        on_change=lambda e: cfg.__setitem__('n_runs', int(e.value)),
    ).props(FIELD_PROPS).classes(FIELD_CLASSES)

    ui.separator().classes('my-3')
    ui.label('Constraints').classes('text-base font-bold')
    _constraint_builder(state, factors)


def _form_lhs(cfg: Dict[str, Any], factors: List[Factor]) -> None:
    k = len(factors)
    ensure_lhs_config(cfg, k)
    n_runs = int(cfg.get('n_runs', k * 10))
    criterion = cfg.get('criterion', 'Maximin')
    est = _estimated(cfg, 'Latin Hypercube', k)
    with ui.column().classes('gap-1'):
        ui.number(
            'Number of Runs', min=k + 1, max=1000, precision=0, value=n_runs,
            on_change=lambda e: (cfg.__setitem__('n_runs', int(e.value)), est.refresh()),
        ).props(FIELD_PROPS).classes(FIELD_CLASSES).tooltip('Typically 5-10× the number of factors')
        ui.select(
            {c: c for c in ['None', 'Maximin', 'Correlation']},
            label='Optimization criterion', value=criterion,
            on_change=lambda e: (cfg.__setitem__('criterion', e.value), est.refresh()),
        ).props(FIELD_PROPS).classes(FIELD_CLASSES)


def _form_split_plot(cfg: Dict[str, Any], factors: List[Factor]) -> None:
    groups = split_plot_factors(factors)
    ui.markdown(
        f'**Hard-to-change factors:** {", ".join(groups["all_hard"])}\n\n'
        f'**Easy-to-change factors:** {", ".join(groups["easy"])}'
    )
    est = _estimated(cfg, 'Split-Plot', len(factors), factor_levels=[len(f.levels) for f in factors])
    with ui.column().classes('gap-1'):
        _number(cfg, 'Number of Replicates', 'n_replicates', 1, 1, 10,
                'Complete replicates of the design', refresh=est.refresh)
        _center_points(cfg, default=0, refresh=est.refresh)
        _number(cfg, 'Number of Blocks', 'n_blocks', 1, 1, 10,
                'Divide whole-plots into blocks', refresh=est.refresh)
        _randomize_switch(cfg, 'Randomize Whole-Plot Order', 'randomize_whole_plots', refresh=est.refresh)
        _randomize_switch(cfg, 'Randomize Sub-Plot Order', 'randomize_subplots', refresh=est.refresh)


def _estimated(
    cfg: Dict[str, Any],
    design_type: str,
    k: int,
    factor_levels: Optional[List[int]] = None,
) -> Callable[[], None]:
    """Render the 'Estimated runs' label as a per-form refreshable.

    Returns the refreshable callable so each form wires ``refresh=...`` into every
    widget's ``on_change`` and the estimate updates reactively.
    """
    @ui.refreshable
    def _est() -> None:
        total = estimate_runs(design_type, cfg, k, factor_levels)
        if total:
            ui.label(f'**Estimated runs:** {total}').classes('text-info mt-1 text-sm')
    _est()
    return _est


# --- Constraints (D-Optimal) -------------------------------------------------


def _constraint_builder(state: Dict[str, Any], factors: List[Factor]) -> None:
    constraints = get_constraints(state)
    if constraints:
        ui.label(f'{len(constraints)} constraint(s) defined').classes('text-sm text-gray-600')
        for i, c in enumerate(constraints):
            with ui.row().classes('items-center gap-2'):
                ui.label(_fmt_constraint(c)).classes('text-sm')
                ui.button(icon='delete', on_click=lambda idx=i: _delete_constraint(state, idx)).props('flat dense')

    with ui.card().classes('w-full'):
        ui.label('Add a constraint').classes('text-sm font-bold')
        type_select = ui.select(
            {'le': '≤ (at most)', 'ge': '≥ (at least)', 'eq': '= (exactly)'},
            value='le', label='Constraint type',
        ).props(FIELD_PROPS).classes(FIELD_CLASSES)
        coeffs: Dict[str, Any] = {}
        with ui.column().classes('gap-1 mt-1'):
            for f in factors:
                ui.number(
                    f'Coefficient for {f.name}', min=-1e6, max=1e6, precision=3, value=1.0,
                    on_change=lambda e, name=f.name: coeffs.__setitem__(name, float(e.value)),
                ).props(FIELD_PROPS).classes(FIELD_CLASSES)
        bound = ui.number('Bound (RHS)', min=-1e6, max=1e6, precision=3, value=500.0).props(
            FIELD_PROPS).classes(FIELD_CLASSES)
        ui.button(
            'Add constraint',
            on_click=lambda: _add_constraint(state, coeffs, bound.value, type_select.value),
        ).props('color=primary').classes('mt-2')


def _fmt_constraint(c: LinearConstraint) -> str:
    lhs = ' + '.join(f'{v}·{k}' for k, v in c.coefficients.items())
    symbol = {'le': '≤', 'ge': '≥', 'eq': '='}.get(c.constraint_type, c.constraint_type)
    return f'{lhs} {symbol} {c.bound}'


def _add_constraint(
    state: Dict[str, Any], coeffs: Dict[str, float], bound: float, constraint_type: str
) -> None:
    if not coeffs or all(abs(v) <= 1e-12 for v in coeffs.values()):
        ui.notify('Set at least one non-zero coefficient.', type='negative')
        return
    constraints = get_constraints(state)
    constraints.append(LinearConstraint(
        coefficients={k: v for k, v in coeffs.items()},
        bound=float(bound),
        constraint_type=str(constraint_type),
    ))
    store_constraints(state, constraints)
    ui.notify('Constraint added.', type='positive')
    ui.navigate.reload()


def _delete_constraint(state: Dict[str, Any], index: int) -> None:
    constraints = get_constraints(state)
    if 0 <= index < len(constraints):
        constraints.pop(index)
        store_constraints(state, constraints)
        ui.notify('Constraint removed.', type='positive')
        ui.navigate.reload()


# --- Next step ----------------------------------------------------------------


def _go_to_preview(
    state: Dict[str, Any], design_type: str, factors: List[Factor], model_terms: List[str]
) -> None:
    state['design_config'] = dict(_cfg(state))
    if design_type == 'D-Optimal':
        state['design_config']['model_terms'] = list(model_terms)

    ok, errors = validate_config(design_type, state['design_config'], len(factors), model_terms)
    if not ok:
        ui.notify('\n'.join(errors[:6]), type='negative', close_button=True, multi_line=True)
        return

    state['current_step'] = 4
    ui.navigate.to('/preview')