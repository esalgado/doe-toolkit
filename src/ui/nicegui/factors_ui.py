"""JMP-style factor grid for the NiceGUI prototype.

Editable AG-Grid replacing ``st.data_editor`` from ``1_define_factors.py``.
Row <-> Factor conversion mirrors ``dataframe_to_factors`` using the same core
``src.core.factors`` API so both apps produce identical ``Factor`` objects.
"""

from typing import Any, Dict, List, Tuple

from nicegui import ui

from src.core.factors import (
    Factor,
    FactorType,
    ChangeabilityLevel,
    sanitize_factor_name,
)

COLUMN_DEFS = [
    {'field': 'Name', 'headerName': 'Factor Name', 'editable': True},
    {
        'field': 'Type',
        'headerName': 'Type',
        'editable': True,
        'width': 160,
        'cellEditor': 'agSelectCellEditor',
        'cellEditorParams': {'values': ['continuous', 'discrete_numeric', 'categorical']},
    },
    {
        'field': 'Min',
        'headerName': 'Min',
        'editable': True,
        'width': 110,
        'cellDataType': 'number',
        'cellEditor': 'agNumberCellEditor',
    },
    {
        'field': 'Max',
        'headerName': 'Max',
        'editable': True,
        'width': 110,
        'cellDataType': 'number',
        'cellEditor': 'agNumberCellEditor',
    },
    {'field': 'Levels', 'headerName': 'Levels', 'editable': True, 'width': 200},
    {'field': 'Units', 'headerName': 'Units', 'editable': True, 'width': 100},
    {
        'field': 'Changeability',
        'headerName': 'Changeability',
        'editable': True,
        'width': 150,
        'cellEditor': 'agSelectCellEditor',
        'cellEditorParams': {'values': ['easy', 'hard', 'very_hard']},
    },
]

GRID_OPTIONS = {
    'columnDefs': COLUMN_DEFS,
    'rowSelection': 'single',
    'stopEditingWhenCellsLoseFocus': True,
    'animateRows': True,
}


def empty_factor_row() -> Dict:
    return {
        'Name': '',
        'Type': 'continuous',
        'Min': None,
        'Max': None,
        'Levels': '',
        'Units': '',
        'Changeability': 'easy',
    }


def factors_to_rows(factors: List[Factor]) -> List[Dict]:
    rows = []
    for factor in factors:
        if factor.is_continuous():
            rows.append({
                'Name': factor.name,
                'Type': factor.factor_type.value,
                'Min': float(factor.levels[0]),
                'Max': float(factor.levels[1]),
                'Levels': '',
                'Units': factor.units or '',
                'Changeability': factor.changeability.value,
            })
        else:
            rows.append({
                'Name': factor.name,
                'Type': factor.factor_type.value,
                'Min': None,
                'Max': None,
                'Levels': ', '.join(str(level) for level in factor.levels),
                'Units': factor.units or '',
                'Changeability': factor.changeability.value,
            })
    return rows


def rows_to_factors(rows: List[Dict]) -> Tuple[List[Factor], List[str]]:
    """Convert editor rows back to validated Factor objects (mirrors dataframe_to_factors)."""
    factors: List[Factor] = []
    errors: List[str] = []

    for idx, row in enumerate(rows, start=1):
        name_raw = str(row.get('Name') or '').strip()
        if not name_raw or name_raw.lower() in ('nan', 'none'):
            errors.append(f'Row {idx}: Name cannot be empty')
            continue
        clean_name, _ = sanitize_factor_name(name_raw)

        type_str = str(row.get('Type') or '').strip().lower()
        if type_str == 'continuous':
            factor_type = FactorType.CONTINUOUS
        elif type_str == 'discrete_numeric':
            factor_type = FactorType.DISCRETE_NUMERIC
        elif type_str == 'categorical':
            factor_type = FactorType.CATEGORICAL
        else:
            errors.append(f"Row {idx} ({clean_name}): Invalid type '{type_str}'")
            continue

        if factor_type == FactorType.CONTINUOUS:
            try:
                min_val = float(row.get('Min'))
                max_val = float(row.get('Max'))
            except (TypeError, ValueError):
                errors.append(f'Row {idx} ({clean_name}): Min/Max must be numeric')
                continue
            if min_val >= max_val:
                errors.append(f'Row {idx} ({clean_name}): Max must be > Min')
                continue
            levels = [min_val, max_val]
        else:
            levels_str = str(row.get('Levels') or '').strip()
            levels = [level.strip() for level in levels_str.split(',') if level.strip()]
            if len(levels) < 2:
                errors.append(f'Row {idx} ({clean_name}): Levels required for {factor_type.value}')
                continue
            if factor_type == FactorType.DISCRETE_NUMERIC:
                try:
                    levels = [float(level) for level in levels]
                except ValueError:
                    errors.append(f'Row {idx} ({clean_name}): Levels must be numeric for discrete_numeric')
                    continue

        change_str = str(row.get('Changeability') or '').strip().lower()
        if change_str == 'hard':
            changeability = ChangeabilityLevel.HARD
        elif change_str == 'very_hard':
            changeability = ChangeabilityLevel.VERY_HARD
        else:
            changeability = ChangeabilityLevel.EASY

        units = str(row.get('Units') or '').strip()
        units = units if units and units.lower() not in ('nan', 'none') else None

        factors.append(Factor(
            name=clean_name,
            factor_type=factor_type,
            changeability=changeability,
            levels=levels,
            units=units,
        ))

    return factors, errors


def create_factor_grid(rows: List[Dict]) -> ui.aggrid:
    options = dict(GRID_OPTIONS)
    options['rowData'] = rows
    return ui.aggrid(options).classes('w-full mt-2')


# State-shape handling -----------------------------------------------------
# ``app.storage.user`` is persisted (per session cookie) as JSON. ``Factor``
# objects must never be stored there directly: orjson serializes the dataclass
# to a plain attribute dict, and on reload NiceGUI re-wraps nested data as
# ObservableDict/ObservableList (so ``is_continuous()`` would fail on reload).
# Store editor-row dicts instead; the helpers below normalize whatever shape is
# found on read (live Factor objects, legacy persisted Factor-dicts, rows).


def _dict_to_factor(data: Dict[str, Any]) -> Factor:
    """Rebuild a Factor from a persisted legacy shape (orjson-dumped dataclass)."""
    return Factor(
        name=data['name'],
        factor_type=FactorType(data['factor_type']),
        changeability=ChangeabilityLevel(data['changeability']),
        levels=data['levels'],
        units=data.get('units') or None,
    )


def rows_from_state(factors_state: List[Any]) -> List[Dict]:
    """Normalize persisted ``state['factors']`` into editor-row dicts.

    Accepts live ``Factor`` objects (pre-row-shaping sessions), legacy
    persisted Factor-dicts (``{'name': ...}``) and already-row-shaped dicts.
    """
    rows: List[Dict] = []
    for item in factors_state:
        if isinstance(item, Factor):
            rows.extend(factors_to_rows([item]))
        elif isinstance(item, dict) and 'name' in item:
            rows.extend(factors_to_rows([_dict_to_factor(item)]))
        elif isinstance(item, dict):
            rows.append(item)
    return rows


def factors_from_state(factors_state: List[Any]) -> List[Factor]:
    """Return validated Factor objects from persisted ``state['factors']``."""
    factors, _errors = rows_to_factors(rows_from_state(factors_state))
    return factors