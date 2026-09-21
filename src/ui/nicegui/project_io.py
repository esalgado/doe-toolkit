"""Project (.doeproject) serialization for the NiceGUI app (pure, Streamlit-free).

Dict-based port of ``src/ui/utils/state_management.create_project_file`` /
``load_project_file`` that stays byte-compatible with the Streamlit project
format (``version``/``factors``/``design``/``responses``/... keys using the same
factor schema) so ``.doeproject`` files remain cross-compatible.

This port additionally round-trips keys the Streamlit version does not persist:
``response_definitions``, the design-level ``model_terms`` and ``constraints``
(see AGENTS.md parked items 2-3). Extra keys are ignored by the Streamlit
loader, so files written here keep loading there.
"""

import copy
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.factors import ChangeabilityLevel, Factor, FactorType, sanitize_factor_name
from src.ui.nicegui.factors_ui import factors_to_rows
from src.ui.nicegui.state import DEFAULTS
from src.ui.nicegui.workspace import (
    constraints_from_persisted,
    constraints_to_persisted,
    design_from_records,
    design_to_records,
    get_constraints,
    get_design,
    get_factors,
)


def create_project(state: Dict[str, Any]) -> str:
    """Serialize the current project state to a .doeproject JSON string."""
    factors = get_factors(state)
    factors_data = [{
        'name': f.name,
        'type': f.factor_type.value,
        'changeability': f.changeability.value,
        'levels': list(f.levels),
        'units': f.units,
    } for f in factors]

    project: Dict[str, Any] = {
        'version': '0.3.0',
        'created': datetime.now().isoformat(),
        'factors': factors_data,
        'design_type': state.get('design_type'),
        'design_config': state.get('design_config') or {},
        'design_metadata': state.get('design_metadata') or {},
    }

    design = get_design(state)
    if design is not None:
        project['design'] = design.to_dict(orient='records')

    responses = state.get('responses') or {}
    if responses:
        project['responses'] = {
            name: (list(data) if not isinstance(data, np.ndarray) else data.tolist())
            for name, data in responses.items()
        }
        project['response_names'] = state.get('response_names', [])

    if state.get('response_definitions'):
        project['response_definitions'] = state['response_definitions']

    if state.get('model_terms'):
        project['model_terms'] = state['model_terms']

    if state.get('model_terms_per_response'):
        project['model_terms_per_response'] = state['model_terms_per_response']

    if state.get('excluded_rows'):
        project['excluded_rows'] = state['excluded_rows']

    constraints = get_constraints(state)
    project['constraints'] = constraints_to_persisted(constraints)

    return json.dumps(project, indent=2)


class LoadReport:
    """Result of ``load_project`` — what changed and what to show the user."""

    def __init__(self) -> None:
        self.ok: bool = False
        self.error: Optional[str] = None
        self.n_factors: int = 0
        self.renamed_factors: List[Tuple[str, str]] = []
        self.design_runs: Optional[int] = None
        self.n_responses: int = 0
        self.design_type: Optional[str] = None
        self.current_step: int = 1
        self.messages: List[Tuple[str, str]] = []  # (kind, text) kind in info|warning|success|error

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__


def load_project(state: Dict[str, Any], file_content: str) -> LoadReport:
    """Restore project state from a .doeproject JSON string into ``state``.

    On success mutates ``state`` in place and returns a report with the load
    summary + user-facing messages (the caller renders them).
    """
    report = LoadReport()
    try:
        project = json.loads(file_content)
    except json.JSONDecodeError as e:
        report.ok, report.error = False, f"Invalid project file: {e}"
        return report

    factors: List[Factor] = []
    for f_data in project.get('factors', []):
        original_name = f_data.get('name', '')
        clean_name, was_modified = sanitize_factor_name(original_name)
        if was_modified:
            report.renamed_factors.append((original_name, clean_name))
        try:
            factors.append(Factor(
                name=clean_name,
                factor_type=FactorType(f_data['type']),
                changeability=ChangeabilityLevel(f_data['changeability']),
                levels=list(f_data['levels']),
                units=f_data.get('units') or None,
            ))
        except Exception as e:
            report.messages.append(('error', f"Failed to load factor '{original_name}': {e}"))
            continue

    if not factors:
        report.ok, report.error = False, "No valid factors found in project file"
        return report

    state.clear()
    state.update(copy.deepcopy(DEFAULTS))

    state['factors'] = factors_to_rows(factors)
    report.n_factors = len(factors)

    if report.renamed_factors:
        report.messages.append(('warning', (
            'Factor names were updated for compatibility: '
            + ', '.join(f"'{o}' -> '{s}'" for o, s in report.renamed_factors)
        )))

    # Design settings
    state['design_type'] = project.get('design_type')
    state['design_config'] = project.get('design_config') or {}
    state['design_metadata'] = project.get('design_metadata') or {}

    # Design (rename legacy columns to sanitized factor names)
    if 'design' in project:
        design_df = pd.DataFrame(project['design'])
        rename_map = {old: new for old, new in report.renamed_factors if old in design_df.columns}
        if rename_map:
            design_df = design_df.rename(columns=rename_map)
            report.messages.append(('info', (
                f"✓ Updated {len(rename_map)} design column(s) to sanitized factor names"
            )))
        state['design'] = design_to_records(design_df)
        report.design_runs = len(design_df)

    # Responses
    if 'responses' in project:
        responses: Dict[str, List[Any]] = {}
        for name, data in dict(project['responses']).items():
            responses[name] = [None if v is None else v for v in list(data)]
        state['responses'] = responses
        state['response_names'] = list(project.get('response_names') or responses.keys())
        report.n_responses = len(responses)

    # Response definitions (new in this port — Streamlit did not save these).
    if 'response_definitions' in project:
        state['response_definitions'] = list(project['response_definitions'])

    # Design-level model terms
    if 'model_terms' in project:
        state['model_terms'] = list(project['model_terms'])

    # Per-response model terms (with factor-rename handling)
    if 'model_terms_per_response' in project:
        model_terms = dict(project['model_terms_per_response'])
        if report.renamed_factors:
            rename_map = {old: new for old, new in report.renamed_factors}
            updated: Dict[str, List[str]] = {}
            for response_name, terms in model_terms.items():
                updated_terms: List[str] = []
                for term in list(terms):
                    updated_term = term
                    for old_name, new_name in rename_map.items():
                        if updated_term == old_name:
                            updated_term = new_name
                        updated_term = updated_term.replace(f"*{old_name}", f"*{new_name}")
                        updated_term = updated_term.replace(f"{old_name}*", f"{new_name}*")
                        updated_term = updated_term.replace(f"I({old_name}**", f"I({new_name}**")
                    updated_terms.append(updated_term)
                updated[response_name] = updated_terms
            state['model_terms_per_response'] = updated
            report.messages.append(('info', "✓ Updated model terms to sanitized factor names"))
        else:
            state['model_terms_per_response'] = model_terms

    if 'excluded_rows' in project:
        state['excluded_rows'] = list(project['excluded_rows'])

    if 'constraints' in project:
        state['constraints'] = list(project['constraints'])

    # Pick where to resume
    if report.n_responses:
        state['current_step'] = 5
    elif report.design_runs is not None:
        state['current_step'] = 4
    elif state.get('design_type'):
        state['current_step'] = 3
    else:
        state['current_step'] = 2 if factors else 1
    report.current_step = state['current_step']
    report.design_type = state.get('design_type')

    report.ok = True
    return report


def sanitize_design_columns(design: pd.DataFrame, factors: List[Factor]) -> pd.DataFrame:
    """Rename design columns that sanitize to one of the defined factor names."""
    factor_map: Dict[str, str] = {}
    for factor in factors:
        if factor.name not in design.columns:
            for col in design.columns:
                sanitized, _ = sanitize_factor_name(col)
                if sanitized == factor.name:
                    factor_map[col] = factor.name
                    break
    return design.rename(columns=factor_map) if factor_map else design