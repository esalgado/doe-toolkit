"""AG-Grid response-definition editor for the NiceGUI app.

Mirrors the JMP-style ``st.data_editor`` response grid from
``src/ui/pages/4_preview_design.py``, backed by the shared (Streamlit-free)
``response_rows_to_definitions`` / ``validate_response_name`` helpers from
``src/ui/utils/response_definitions.py`` so sanitization behaviour is identical.
"""

from typing import Any, Dict, List

import pandas as pd
from nicegui import ui

from src.ui.utils.response_definitions import response_rows_to_definitions

COLUMN_DEFS = [
    {
        'field': 'Name',
        'headerName': 'Response Name',
        'editable': True,
        'minWidth': 220,
        'tooltip': 'Descriptive name (e.g. Max Force). Spaces/parentheses are auto-cleaned.',
    },
    {'field': 'Units', 'headerName': 'Units', 'editable': True, 'width': 140},
]

GRID_OPTIONS: Dict[str, Any] = {
    'columnDefs': COLUMN_DEFS,
    'rowSelection': 'single',
    'stopEditingWhenCellsLoseFocus': True,
    'animateRows': True,
}


def empty_response_row() -> Dict[str, Any]:
    return {'Name': '', 'Units': ''}


def _rows_from_definitions(response_definitions: List[Dict]) -> List[Dict]:
    if not response_definitions:
        return []
    return [
        {'Name': str(r.get('name', '')), 'Units': str(r.get('units', '') or '')}
        for r in response_definitions
    ]


def response_grid(state: Dict[str, Any], key_prefix: str = 'responses') -> None:
    """Draw the response editor; persists edits into ``state['response_definitions']``."""
    defs = list(state.get('response_definitions') or [])
    rows = _rows_from_definitions(defs)
    if not rows:
        rows = [empty_response_row()]

    grid = ui.aggrid({**GRID_OPTIONS, 'rowData': rows}).classes('w-full mt-2')

    with ui.row().classes('gap-2 mt-2'):
        ui.button(
            'Add response',
            on_click=lambda: grid.run_grid_method(
                'applyTransaction', {'add': [empty_response_row()]}
            ),
        )
        ui.button('Remove selected', on_click=lambda: _remove_selected(grid))
        if defs:
            ui.button('Clear all', on_click=lambda: _clear_all(state, grid))
        ui.button(
            'Save responses',
            on_click=lambda: _save_responses(state, grid),
        ).props('color=primary')


async def _remove_selected(grid: ui.aggrid) -> None:
    selected = await grid.get_selected_row()
    if selected is None:
        ui.notify('Select a row to remove first', type='warning')
        return
    grid.run_grid_method('applyTransaction', {'remove': [selected]})


def _clear_all(state: Dict[str, Any], grid: ui.aggrid) -> None:
    state['response_definitions'] = []
    grid.run_grid_method('applyTransaction', {'remove': grid.options['rowData'][:]})
    ui.notify('Response definitions cleared.', type='positive')


async def _save_responses(state: Dict[str, Any], grid: ui.aggrid) -> None:
    rows = await grid.get_client_data()
    defs, errors, warnings = response_rows_to_definitions(pd.DataFrame(rows))

    if errors:
        ui.notify(
            '\n'.join(f'• {e}' for e in errors[:6]),
            type='negative',
            close_button=True,
            multi_line=True,
        )
        return

    state['response_definitions'] = defs

    if warnings:
        lines = [f"{len(warnings)} response name(s) were modified for compatibility:"]
        for warning in warnings[:8]:
            lines.append(
                f'Row {warning["row"]}: {warning["original"]} → {warning["sanitized"]}'
            )
        ui.notify(
            '\n'.join(lines),
            type='warning',
            close_button=True,
            multi_line=True,
        )
    else:
        ui.notify(f'Saved {len(defs)} response(s).', type='positive')