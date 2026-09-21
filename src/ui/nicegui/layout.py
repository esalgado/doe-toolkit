"""Shared header / left-drawer shell for the NiceGUI app.

Mirrors the Streamlit sidebar (``src/ui/components/sidebar.py``) so both apps offer
the same workflow navigation. Steps 5-8 render as locked entries until the port
reaches them (clicking does nothing), so the checkout stays stable.
"""

from typing import Dict, Any

from nicegui import ui

from src.ui.nicegui.state import (
    any_project_state,
    can_access_step,
    is_step_complete,
    reset_state,
)

# Shared field styling for config inputs (width-in-box, no `dense`: dense clips
# the floating outlined label). w-80 (20rem) keeps inputs readable while still
# fitting a form column.
FIELD_PROPS = 'outlined'
FIELD_CLASSES = 'w-80'

NAV_ITEMS = [
    ('/', 'Home', 0),
    ('/define', '1. Define Factors', 1),
    ('/model', '2. Select Model', 2),
    ('/design', '3. Choose Design', 3),
    ('/preview', '4. Preview & Generate', 4),
]

FUTURE_ITEMS = [
    ('5. Import Results', 5),
    ('6. Analyze', 6),
    ('7. Augmentation', 7),
    ('8. Optimize', 8),
]


def create_shell(state: Dict[str, Any], active: str) -> None:
    """Build the page header + workflow drawer (call first in each page)."""
    ui.colors(primary='#4a90e2')

    with ui.header().classes('bg-primary'):
        with ui.row().classes('items-center gap-3 px-4'):
            ui.label('DOE Toolkit').classes('text-lg font-bold text-white')
            ui.label('NiceGUI prototype').classes('text-xs text-white/70')

    with ui.left_drawer(value=True).classes('bg-gray-50'):
        with ui.column().classes('w-full gap-1 px-2 mt-4 p-4'):
            ui.label('Workflow').classes('text-xs font-bold text-gray-500')

            for route, label, step in NAV_ITEMS:
                active_class = 'bg-primary text-white' if route == active else ''
                enabled = (step == 0) or can_access_step(state, step)
                if step and is_step_complete(state, step):
                    label = f'✓ {label}'
                elif enabled:
                    label = f'○ {label}'
                else:
                    label = f'🔒 {label}'
                ui.button(
                    label,
                    on_click=lambda r=route: ui.navigate.to(r),
                ).props(f'flat {"disable" if step and not enabled else ""}').classes(f'w-full justify-start {active_class}')

            ui.separator().classes('mt-3')
            ui.label('Coming in the full port').classes('text-xs font-bold text-gray-500')
            for label, step in FUTURE_ITEMS:
                enabled = can_access_step(state, step)
                prefix = '✓' if is_step_complete(state, step) else '○' if enabled else '🔒'
                ui.button(f'{prefix} {label}').props('flat disable').classes(
                    'w-full justify-start text-gray-400'
                )

            ui.separator().classes('mt-3')
            n_factors = len(state.get('factors') or [])
            n_responses = len(state.get('responses') or {})
            design = state.get('design')
            ui.label(f'Factors defined: {n_factors}').classes('text-sm text-gray-600')
            ui.label(f'Responses: {n_responses}').classes('text-sm text-gray-600')
            ui.label(f'Design: {len(design)} runs' if design else 'Design: none').classes(
                'text-sm text-gray-600'
            )
            ui.label('All computation is local.').classes('text-xs text-gray-400')

            if any_project_state(state):
                ui.separator().classes('mt-3')
                start_fresh_button(state)


def start_fresh_button(state: Dict[str, Any]) -> None:
    """Render the 'Start Fresh' action with a destructive confirmation dialog."""
    ui.button('Start over', on_click=lambda: _confirm_start_fresh(state)) \
        .props('outline color=negative').classes('w-full mt-2')


def _confirm_start_fresh(state: Dict[str, Any]) -> None:
    with ui.dialog() as dialog, ui.card():
        ui.label('Start over from scratch?').classes('text-base font-bold')
        ui.label(
            'This clears all factors, model, design, responses and results. '
            'Use Export on Home to keep a copy first.'
        ).classes('text-sm text-gray-600')
        with ui.row().classes('justify-between w-full mt-3'):
            ui.button('Cancel', on_click=dialog.close).props('flat')
            ui.button(
                'Start over',
                on_click=lambda: (_reset_and_start(state), dialog.close()),
            ).props('color=negative')
    dialog.open()


def _reset_and_start(state: Dict[str, Any]) -> None:
    reset_state(state)
    ui.navigate.to('/define')