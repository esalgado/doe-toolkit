"""Step 1: Define Factors (NiceGUI)."""

from typing import Any, Dict

from nicegui import ui

from src.ui.nicegui.factors_ui import (
    create_factor_grid,
    empty_factor_row,
    factors_to_rows,
    rows_from_state,
    rows_to_factors,
)
from src.ui.nicegui.layout import create_shell
from src.ui.nicegui.state import get_state, invalidate_downstream_state


@ui.page('/define')
def define_factors(state: Dict[str, Any] | None = None) -> None:
    app_state = state if state is not None else get_state()
    create_shell(app_state, active='/define')

    with ui.column().classes('p-6 w-full'):
        ui.label('Step 1: Define Experimental Factors').classes('text-2xl font-bold')
        ui.label(
            'Edit cells directly, add/remove rows, then Save. '
            'Continuous factors use Min/Max; discrete/categorical factors use Levels.'
        ).classes('text-gray-600')

        rows = rows_from_state(app_state.get('factors') or [])
        if not rows:
            rows = [empty_factor_row()]
        grid = create_factor_grid(rows)

        with ui.row().classes('gap-2 mt-2'):
            ui.button(
                'Add factor',
                on_click=lambda: grid.run_grid_method('applyTransaction', {'add': [empty_factor_row()]}),
            )
            ui.button(
                'Remove selected',
                on_click=lambda: _remove_selected(grid),
            )
            ui.button(
                'Save factors',
                on_click=lambda: _save_factors(app_state, grid),
            ).props('color=primary')

        with ui.row().classes('gap-2 mt-6'):
            ui.button(
                'Continue → Choose Model',
                on_click=lambda: (app_state.__setitem__('current_step', 2), ui.navigate.to('/model')),
            ).props('color=primary' if len(app_state.get('factors') or []) > 0 else 'disable')


async def _remove_selected(grid: ui.aggrid) -> None:
    selected = await grid.get_selected_row()
    if selected is None:
        ui.notify('Select a row to remove first', type='warning')
        return
    grid.run_grid_method('applyTransaction', {'remove': [selected]})


async def _save_factors(state: Dict[str, Any], grid: ui.aggrid) -> None:
    rows = await grid.get_client_data()
    factors, errors = rows_to_factors(rows)

    if errors:
        message = '\n'.join(errors[:6])
        if len(errors) > 6:
            message += f'\n…and {len(errors) - 6} more'
        ui.notify(message, type='negative', close_button=True, multi_line=True)
        return

    names = [factor.name for factor in factors]
    if len(names) != len(set(names)):
        ui.notify('Duplicate factor names detected. Each factor must be unique.', type='negative')
        return

    new_rows = factors_to_rows(factors)
    if new_rows != rows_from_state(state.get('factors') or []):
        invalidate_downstream_state(state, from_step=1)
    state['factors'] = new_rows
    ui.notify(f'Saved {len(factors)} factor(s).', type='positive')
    ui.navigate.reload()