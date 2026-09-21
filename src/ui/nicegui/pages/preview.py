"""Step 4: Preview & Generate Design (NiceGUI).

Mirrors ``4_preview_design.py``. Generation goes through
``design_generation.generate_design`` with the exact ``design_config`` from
Step 3 (the Streamlit Step 3->4 keys gap is closed here); Box-Behnken generates
a real Box-Behnken design; the random seed is honored for every design type.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from nicegui import ui

from src.core.factors import Factor
from src.ui.components.constraint_builder import format_constraint_preview
from src.ui.nicegui.design_generation import compute_d_efficiency, generate_design
from src.ui.nicegui.layout import FIELD_CLASSES, FIELD_PROPS, create_shell
from src.ui.nicegui.responses_ui import response_grid
from src.ui.nicegui.state import can_access_step, get_state, invalidate_downstream_state
from src.ui.nicegui.workspace import (
    get_constraints,
    get_design,
    get_design_space,
    get_factors,
)


@ui.page('/preview')
def preview(state: Dict[str, Any] | None = None) -> None:
    app_state = state if state is not None else get_state()
    create_shell(app_state, active='/preview')

    with ui.column().classes('p-6 w-full max-w-5xl'):
        ui.label('Step 4: Preview & Generate Design').classes('text-2xl font-bold')

        if not can_access_step(app_state, 4):
            ui.label('Please complete Steps 1-3 first.').classes('text-gray-600')
            if len(app_state.get('factors') or []) == 0:
                ui.button('1. Define Factors', on_click=lambda: ui.navigate.to('/define')).props('color=primary')
            else:
                ui.button('3. Choose Design', on_click=lambda: ui.navigate.to('/design')).props('color=primary')
            return

        factors = get_factors(app_state)
        design_type = app_state.get('design_type')
        design = get_design(app_state)

        ui.label(f'**Design type:** {design_type} · **Factors:** {len(factors)}').classes('text-sm text-gray-600')

        if design is None:
            _render_generation(app_state, factors, design_type)
        else:
            _render_preview(app_state, factors, design)


# --- Generation ---------------------------------------------------------------


def _render_generation(
    state: Dict[str, Any], factors: List[Factor], design_type: str
) -> None:
    ui.separator().classes('my-3')

    ui.label('Define responses to measure').classes('text-lg font-bold')
    ui.label(
        'Declare what you will measure so the design CSV template includes these columns. '
        'Names with spaces/parentheses are auto-cleaned to be formula-safe.'
    ).classes('text-sm text-gray-600')
    response_grid(state)

    ui.separator().classes('my-3')

    with ui.expansion('Configuration summary').classes('w-full'):
        ui.label(f'Design type: {design_type}').classes('text-sm')
        ui.label(f'Number of factors: {len(factors)}').classes('text-sm')
        cfg = state.get('design_config') or {}
        if cfg:
            for key, value in cfg.items():
                ui.label(f'{key}: {value}').classes('text-xs text-gray-600')

    _render_constraint_warnings(state, factors, design_type)

    with ui.row().classes('items-center gap-4 mt-2'):
        use_seed = ui.checkbox(
            'Use Random Seed (for reproducibility)',
            value=state.get('random_seed_used', True),
        )
        seed_input = ui.number(
            'Seed', min=0, max=2**31 - 1, precision=0, value=int(state.get('random_seed') or 42),
        ).props(FIELD_PROPS).classes(FIELD_CLASSES)
        use_seed.on_value_change(lambda e: seed_input.set_enabled(bool(e.value)))
        seed_input.set_enabled(use_seed.value)

    ui.button(
        'Generate Design 🔬',
        on_click=lambda: _generate(state, factors, design_type, seed_input.value if use_seed.value else None),
    ).props('color=primary').classes('mt-4')


def _render_constraint_warnings(state: Dict[str, Any], factors: List[Factor], design_type: str) -> None:
    if design_type != 'D-Optimal':
        return
    constraints = get_constraints(state)
    if not constraints:
        return
    ui.label(f'ℹ️ Design will respect {len(constraints)} constraint(s)').classes('text-sm text-blue-600')
    with ui.expansion(f'View {len(constraints)} constraint(s)'):
        for i, c in enumerate(constraints):
            ui.code(f'{i + 1}. {format_constraint_preview(c.coefficients, c.bound, c.constraint_type)}')

    from src.ui.components.constraint_builder import validate_constraints

    is_valid, warnings = validate_constraints(constraints, factors)
    if not is_valid:
        ui.label('❌ Invalid constraints:').classes('text-red-600 font-bold')
        for warning in warnings:
            ui.label(f'• {warning}').classes('text-red-600 text-sm')
        _mark_blocked = True  # generation will fail validation anyway


def _generate(
    state: Dict[str, Any], factors: List[Factor], design_type: str, seed: Optional[int]
) -> None:
    try:
        state['random_seed'] = seed
        state['random_seed_used'] = seed is not None
        config = dict(state.get('design_config') or {})
        constraints = get_constraints(state)
        model_terms = list(state.get('model_terms') or [])
        design, metadata = generate_design(
            design_type=design_type, factors=factors, config=config,
            constraints=constraints, seed=seed, model_term_list=model_terms,
        )

        # First generation after prior design: stale analysis/responses must go.
        if state.get('design') is not None:
            invalidate_downstream_state(state, from_step=3)

        state['design_type'] = design_type
        state['current_step'] = 4
        from src.ui.nicegui.workspace import store_design
        store_design(state, design)
        state['design_metadata'] = metadata
        ui.notify(f'✓ Design generated successfully! ({len(design)} runs)', type='positive')
        ui.navigate.reload()
    except Exception as e:
        ui.notify(f'Design generation failed: {e}', type='negative', close_button=True, multi_line=True)


# --- Preview ------------------------------------------------------------------


def _render_preview(state: Dict[str, Any], factors: List[Factor], design: pd.DataFrame) -> None:
    metadata = state.get('design_metadata') or {}
    ui.markdown(f'**✓ Design generated** — `{len(design)}` runs').classes('mt-2')

    ui.label('Design Summary').classes('text-lg font-bold mt-3')
    n_factor_cols = len([f.name for f in factors if f.name in design.columns])
    with ui.grid(columns=4).classes('w-full gap-2 mt-1'):
        _metric_card('Total Runs', str(len(design)))
        if metadata.get('is_split_plot'):
            n_wp = design['WholePlot'].nunique() if 'WholePlot' in design.columns else 0
            _metric_card('Whole-Plots', str(n_wp))
        else:
            _metric_card('Factors', str(n_factor_cols))
        if 'Block' in design.columns:
            _metric_card('Blocks', str(design['Block'].nunique()))
        elif metadata.get('resolution'):
            _metric_card('Resolution', str(metadata['resolution']))
        else:
            _metric_card('Design Points', str(len(design)))
        de = compute_d_efficiency(design, factors, metadata)
        _metric_card('D-Efficiency', f'{de:.1f}%' if de is not None else '—')

    ui.separator().classes('my-3')
    ui.label('Design Matrix Preview').classes('text-lg font-bold')

    mode_value = state.get('preview_display_mode', 'first')
    mode = ui.radio(
        {'first': 'First 10', 'last': 'Last 10', 'random': 'Random 10', 'full': 'Full design'},
        value=mode_value,
        on_change=lambda e: _set_display_mode(state, e.value),
    ).props('inline').classes('mb-2')

    display = _slice_design(design, mode_value).round(4)
    ui.table.from_pandas(display, row_key='StdOrder').props('flat bordered dense').classes('w-full')

    _render_details(state, design, factors, metadata)

    ui.separator().classes('my-3')
    ui.label('Export design').classes('text-lg font-bold')

    with ui.row().classes('gap-4'):
        with ui.card().classes('min-w-[320px]'):
            ui.label('Design CSV (for experiments)').classes('font-bold text-sm')
            ui.label('Download this, run your experiments, then add response data.').classes('text-xs text-gray-600')
            ui.button(
                '📥 Download Design CSV',
                on_click=_download_design_csv,
            ).props('color=primary')
        with ui.card().classes('min-w-[320px]'):
            ui.label('Project file (save session)').classes('font-bold text-sm')
            ui.label('Save your entire project to resume later.').classes('text-xs text-gray-600')
            ui.button(
                '💾 Download Project',
                on_click=_download_project,
            ).props('outline')

    n_responses = len(state.get('response_definitions') or [])
    if n_responses:
        names = ', '.join(r['name'] for r in state['response_definitions'])
        ui.markdown(
            f'**{n_responses} response(s) defined.** Empty columns in the CSV are ready for data: '
            f'`{names}`. Download the CSV, run experiments, fill in values, then import in Step 5.'
        ).classes('mt-3')
    else:
        ui.markdown(
            'No responses defined yet — you can still generate a design, but the CSV '
            'template will not include response columns.'
        ).classes('text-gray-600 text-sm mt-3')

    with ui.row().classes('gap-2 mt-4'):
        ui.button('Regenerate Design', on_click=lambda: _reset_design(state)).props('outline')
        ui.button('Back to Design Config', on_click=lambda: ui.navigate.to('/design')).props('flat')


def _metric_card(label: str, value: str) -> None:
    with ui.card().classes('w-full text-center'):
        ui.label(value).classes('text-2xl font-bold text-primary')
        ui.label(label).classes('text-xs text-gray-500')


def _set_display_mode(state: Dict[str, Any], mode: str) -> None:
    state['preview_display_mode'] = mode
    ui.navigate.reload()


def _slice_design(design: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == 'last':
        return design.tail(10)
    if mode == 'random':
        return design.sample(n=min(10, len(design)), random_state=42)
    if mode == 'full':
        return design
    return design.head(10)


def _render_details(
    state: Dict[str, Any],
    design: pd.DataFrame,
    factors: List[Factor],
    metadata: Dict[str, Any],
) -> None:
    generators = metadata.get('generators')
    if generators:
        with ui.expansion('Design details — generators').classes('w-full'):
            for gen in generators:
                if isinstance(gen, tuple):
                    ui.code(f'{gen[0]} = {gen[1]}')
                else:
                    ui.code(str(gen))

    if metadata.get('alias_structure'):
        with ui.expansion('Alias structure'):
            aliases = metadata['alias_structure']
            if isinstance(aliases, dict):
                for effect, chain in list(aliases.items())[:20]:
                    ui.label(f'{effect} = {" = ".join(chain if isinstance(chain, list) else [str(chain)])}') \
                        .classes('text-xs')
            else:
                ui.code(str(aliases))

    _render_alias_correlation(design, factors, state)

    if metadata.get('is_split_plot') and 'WholePlot' in design.columns:
        with ui.expansion('Split-plot structure'):
            counts = design['WholePlot'].value_counts().sort_index().to_frame('Runs')
            ui.table.from_pandas(counts, row_key=None).props('flat bordered dense')


def _render_alias_correlation(design: pd.DataFrame, factors: List[Factor], state: Dict[str, Any]) -> None:
    model_terms = list(state.get('model_terms') or ['1'] + [f.name for f in factors])
    try:
        from src.ui.components.alias_display import (
            _build_heatmap,
            _build_model_columns,
            _compute_correlation_matrix,
            _find_flagged_pairs,
        )

        matrix, labels = _build_model_columns(design, factors, model_terms)
        if matrix is None or matrix.shape[1] < 2:
            return
        corr = _compute_correlation_matrix(matrix)
        fig = _build_heatmap(corr, labels)
        with ui.expansion('Term correlation heatmap (|r|)'):
            ui.plotly(fig).classes('w-full')
            flagged = _find_flagged_pairs(corr, labels)
            if flagged:
                flagged_df = pd.DataFrame(flagged, columns=['Term A', 'Term B', '|r|']) \
                    .sort_values('|r|', ascending=False)
                ui.label('Flagged pairs (|r| ≥ 0.5):').classes('font-bold mt-2')
                ui.table.from_pandas(flagged_df, row_key=None).props('flat bordered dense')
            else:
                ui.label('No term pairs with |r| ≥ 0.5 — model terms are well-separated.') \
                    .classes('text-green-700 text-sm')
    except Exception:
        pass  # diagnostic-only; never block preview


def _reset_design(state: Dict[str, Any]) -> None:
    state['design'] = None
    state['design_natural'] = None
    state['design_metadata'] = {}
    ui.navigate.reload()


def _download_design_csv() -> None:
    ui.navigate.to('api/design.csv')


def _download_project() -> None:
    ui.navigate.to('api/project.doeproject')