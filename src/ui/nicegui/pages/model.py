"""Step 2: Select Analysis Model (NiceGUI).

Mirrors ``2_select_model.py`` using the shared (Streamlit-free) model-builder
helpers for term presets / display formatting.
"""

from typing import Any, Dict, List

from nicegui import ui

from src.ui.components.model_builder import (
    format_full_equation,
    format_term_for_display,
    get_preset_terms,
)
from src.ui.nicegui.layout import FIELD_CLASSES, FIELD_PROPS, create_shell
from src.ui.nicegui.state import can_access_step, get_state, invalidate_downstream_state
from src.ui.nicegui.workspace import get_factors

PRESETS = ['Linear', 'Quadratic', 'RSM', 'Full Interaction']

_TRANSFORM_STARTS = ('np.log(', 'np.sqrt(', 'np.exp(', 'I(1/')


def _is_nonlinear(t: str) -> bool:
    return t.startswith('I(') or any(t.startswith(p) for p in _TRANSFORM_STARTS)


@ui.page('/model')
def model_page(state: Dict[str, Any] | None = None) -> None:
    app_state = state if state is not None else get_state()
    create_shell(app_state, active='/model')

    with ui.column().classes('p-6 w-full max-w-4xl'):
        ui.label('Step 2: Select Analysis Model').classes('text-2xl font-bold')

        if not can_access_step(app_state, 2):
            ui.label('Complete Step 1 (Define Factors) first.').classes('text-gray-600')
            ui.button('1. Define Factors', on_click=lambda: ui.navigate.to('/define')).props('color=primary')
            return

        factors = get_factors(app_state)
        if not factors:
            ui.label('No factors defined. Return to Step 1.').classes('text-gray-600')
            return

        current_terms = _ensure_terms(app_state, factors)
        selected: List[str] = list(current_terms)

        ui.markdown(
            '**Why define the model first?** D-optimal designs optimize for exactly '
            'these terms, and fractional factorials may alias interactions you care about. '
            'Determining the anticipated model now enables design types that support your analysis goals.'
        )

        with ui.expansion('Defined factors', icon='list'):
            for f in factors:
                levels = (
                    f'[{f.levels[0]}, {f.levels[1]}]' if f.is_continuous()
                    else f'{len(f.levels)} levels'
                )
                ui.label(f'{f.name} — {f.factor_type.value} — {levels} — {f.changeability.value}')

        ui.separator().classes('my-4')

        ui.label('Model presets').classes('text-lg font-bold')
        ui.label('Apply a preset, then fine-tune individual terms below.').classes('text-sm text-gray-600')

        preset = ui.select(PRESETS, value='Linear').props(FIELD_PROPS).classes(FIELD_CLASSES)
        ui.button(
            'Apply preset',
            on_click=lambda: _apply_preset(app_state, preset.value),
        ).props('color=primary')

        ui.separator().classes('my-4')

        ui.label('Selected model terms').classes('text-lg font-bold')
        term_ui: Dict[str, Dict[str, Any]] = {}
        with ui.column().classes('gap-1'):
            for term in _all_candidate_terms(factors):
                check = ui.checkbox(format_term_for_display(term), value=(term in selected))
                term_ui[term] = {'checkbox': check, 'label': format_term_for_display(term)}

        with ui.row().classes('gap-2 mt-2'):
            ui.button(
                'Save model',
                on_click=lambda: _save_model(app_state, term_ui, factors),
            ).props('color=primary')
            ui.button(
                'Simplify model',
                on_click=lambda: _simplify_model(app_state, term_ui, factors),
            ).props('outline')

        ui.separator().classes('my-4')

        equation = format_full_equation(current_terms, 'Y')
        ui.markdown(f'**Current model:** `{equation}`')

        _render_recommendations(app_state, factors, current_terms)

        with ui.row().classes('gap-2 mt-6'):
            ui.button('← Back to Factors', on_click=lambda: ui.navigate.to('/define')).props('outline')
            ui.button(
                'Next: Choose Design →',
                on_click=lambda: (app_state.__setitem__('step_2_complete', True), ui.navigate.to('/design')),
            ).props('color=primary' if current_terms else 'disable')


def _ensure_terms(state: Dict[str, Any], factors: List[Any]) -> List[str]:
    if state.get('model_terms'):
        return list(state['model_terms'])
    default = ['1'] + [f.name for f in factors]
    state['model_terms'] = default
    return default


def _all_candidate_terms(factors: List[Any]) -> List[str]:
    terms, _ = get_preset_terms('RSM', factors, include_intercept=True)
    return terms


def _apply_preset(state: Dict[str, Any], preset: str) -> None:
    factors = get_factors(state)
    terms, message = get_preset_terms(preset, factors, include_intercept=True)
    state['model_terms'] = terms
    invalidate_downstream_state(state, from_step=2)
    if message:
        ui.notify(message, type='warning', close_button=True)
    else:
        ui.notify(f'Applied {preset} preset ({len(terms)} terms).', type='positive')
    ui.navigate.reload()


def _save_model(state: Dict[str, Any], term_ui: Dict[str, Dict[str, Any]], factors: List[Any]) -> None:
    selected = [term for term, ui_state in term_ui.items() if ui_state['checkbox'].value]
    if not selected:
        ui.notify('Select at least one term.', type='negative')
        return
    ordered = _order_terms(selected)
    state['model_terms'] = ordered
    state['step_2_complete'] = True
    invalidate_downstream_state(state, from_step=2)
    ui.notify(f'Model saved with {len(ordered)} terms.', type='positive')
    ui.navigate.reload()


def _simplify_model(state: Dict[str, Any], term_ui: Dict[str, Dict[str, Any]], factors: List[Any]) -> None:
    terms, _ = get_preset_terms('Linear', factors, include_intercept=True)
    state['model_terms'] = terms
    state['step_2_complete'] = True
    invalidate_downstream_state(state, from_step=2)
    ui.notify('Model simplified to main effects.', type='positive')
    ui.navigate.reload()


def _order_terms(raw: List[str]) -> List[str]:
    intercept = [t for t in raw if t == '1']
    rest = [t for t in raw if t != '1']
    return intercept + rest


def _render_recommendations(state: Dict[str, Any], factors: List[Any], terms: List[str]) -> None:
    has_intercept = '1' in terms
    has_quadratic = any(t.startswith('I(') and '**2' in t for t in terms)
    has_interactions = any('*' in t and not t.startswith('I(') for t in terms)
    has_cubic_or_higher = any(t.startswith('I(') and '**' in t and '**2' not in t for t in terms)

    if has_cubic_or_higher:
        complexity = 'Very High (Cubic+ terms)'
    elif has_quadratic:
        complexity = 'High (Quadratic)'
    elif has_interactions:
        complexity = 'Medium (Interactions)'
    else:
        complexity = 'Low (Main effects only)'

    n_factors = len(factors)
    n_continuous = len([f for f in factors if f.is_continuous()])

    with ui.expansion(f'Compatibility — {complexity} ({len(terms)} terms)').classes('mt-4'):
        compatible: List[str] = []
        warnings: List[str] = []
        if n_factors <= 5:
            compatible.append('Full Factorial — can fit any model')
        else:
            warnings.append(f'Full Factorial — may need many runs (2^{n_factors})')
        if not has_quadratic and n_factors >= 4:
            compatible.append('Fractional Factorial — check resolution for interactions')
        elif has_quadratic:
            warnings.append('Fractional Factorial — cannot fit quadratic terms')
        if has_quadratic and n_continuous >= 2:
            compatible.append('Response Surface (CCD, Box-Behnken) — designed for quadratic models')
        elif has_quadratic and n_continuous < 2:
            warnings.append('Response Surface — requires 2+ continuous factors')
        elif not has_quadratic:
            warnings.append('Response Surface — overqualified (no quadratic terms in model)')
        compatible.append('D-Optimal — can fit this exact model efficiently')
        if not has_interactions and not has_quadratic:
            compatible.append('Latin Hypercube — good for main effects screening')
        else:
            warnings.append('Latin Hypercube — better for screening, not interaction/RSM models')

        ui.label('Compatible design types:').classes('font-bold')
        for item in compatible:
            ui.label(f'✓ {item}')
        if warnings:
            ui.label('Compatibility notes:').classes('font-bold mt-2')
            for item in warnings:
                ui.label(f'• {item}')