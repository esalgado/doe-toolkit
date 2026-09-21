"""Workspace layer: materialize live objects from the JSON-safe persisted state.

``app.storage.user`` only ever holds JSON-serializable data (see AGENTS.md /
the state module docstring). This module is the single place that converts
between the persisted forms and the live objects the pages work with:

- ``'factors'``     rows (list of dict)              <-> ``List[Factor]``
- ``'constraints'`` [{coefficients, bound, type}]    <-> ``List[LinearConstraint]``
- ``'design'``      records (list of dict) or None   <-> ``Optional[pd.DataFrame]``
- ``'responses'``   {name: [...float|None]}          <-> ``Dict[str, np.ndarray]``
- ``'design_space'`` unavailable (always derived)    <-> ``DesignSpace``
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from src.core.coding import DesignSpace
except ImportError:  # pragma: no cover - import-time guard only
    DesignSpace = Any  # type: ignore

from src.core.factors import Factor
from src.core.optimal.constraints import LinearConstraint
from src.ui.nicegui.factors_ui import factors_from_state, factors_to_rows


# --- Factors ----------------------------------------------------------------


def get_factors(state: Dict[str, Any]) -> List[Factor]:
    """Return validated Factor objects from persisted ``state['factors']`` rows."""
    return factors_from_state(state.get('factors') or [])


def store_factors(state: Dict[str, Any], factors: List[Factor]) -> None:
    """Persist Factor objects as JSON-safe editor rows."""
    state['factors'] = factors_to_rows(factors)


# --- Constraints ------------------------------------------------------------


def constraints_to_persisted(constraints: List[LinearConstraint]) -> List[Dict]:
    """Convert LinearConstraint objects to a JSON-safe dict form."""
    return [
        {
            'coefficients': {str(k): float(v) for k, v in c.coefficients.items()},
            'bound': float(c.bound),
            'type': c.constraint_type,
        }
        for c in constraints
    ]


def constraints_from_persisted(data: List[Any]) -> List[LinearConstraint]:
    """Rebuild LinearConstraint objects from the persisted dict form."""
    constraints: List[LinearConstraint] = []
    for item in data:
        if isinstance(item, LinearConstraint):
            constraints.append(item)
            continue
        if isinstance(item, dict):
            coefficients = {str(k): float(v) for k, v in item.get('coefficients', {}).items()}
            constraints.append(LinearConstraint(
                coefficients=coefficients,
                bound=float(item.get('bound', 0.0)),
                constraint_type=item.get('type', 'le'),
            ))
    return constraints


def get_constraints(state: Dict[str, Any]) -> List[LinearConstraint]:
    return constraints_from_persisted(state.get('constraints') or [])


def store_constraints(state: Dict[str, Any], constraints: List[LinearConstraint]) -> None:
    state['constraints'] = constraints_to_persisted(constraints)


# --- Design -----------------------------------------------------------------


def design_to_records(design: pd.DataFrame) -> List[Dict]:
    """Serialize a design DataFrame to JSON-safe records (keys are column names)."""
    records: List[Dict] = []
    for value in design.to_dict(orient='records'):
        records.append({
            str(key): (None if (val is None or (isinstance(val, float) and np.isnan(val))) else val)
            for key, val in value.items()
        })
    return records


def design_from_records(records: Optional[List[Dict]]) -> Optional[pd.DataFrame]:
    """Rebuild a design DataFrame from persisted records (or None)."""
    if not records:
        return None
    return pd.DataFrame(records)


def get_design(state: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Return the last generated design in natural units (records -> DataFrame)."""
    return design_from_records(state.get('design'))


def store_design(state: Dict[str, Any], design: pd.DataFrame) -> None:
    """Persist a generated design (natural units) as records."""
    records = design_to_records(design)
    state['design'] = records
    state['design_natural'] = records


def get_design_space(state: Dict[str, Any]) -> Optional[DesignSpace]:
    """Coding specs for the current factors (always derived, never persisted)."""
    factors = get_factors(state)
    if not factors:
        return None
    return DesignSpace.from_factors(factors)


# --- Responses --------------------------------------------------------------


def responses_to_persisted(responses: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Convert {name: np.ndarray} to a JSON-safe {name: [float|None]} mapping."""
    persisted: Dict[str, List[Any]] = {}
    for name, values in responses.items():
        if values is None:
            persisted[name] = []
            continue
        if isinstance(values, np.ndarray):
            persisted[name] = [None if v is None else float(v) for v in values.tolist()]
        elif isinstance(values, (list, tuple)):
            persisted[name] = [None if v is None else float(v) for v in values]
        elif isinstance(values, pd.Series):
            persisted[name] = [None if v is None else float(v) for v in values.tolist()]
    return persisted


def responses_from_persisted(data: Any) -> Dict[str, np.ndarray]:
    """Rebuild {name: np.ndarray} from the persisted mapping."""
    responses: Dict[str, np.ndarray] = {}
    if not data:
        return responses
    for name, values in dict(data).items():
        if values is None:
            responses[name] = np.array([], dtype=float)
            continue
        flat = list(values) if not isinstance(values, np.ndarray) else values.tolist()
        try:
            responses[name] = np.array(
                [np.nan if v is None else float(v) for v in flat], dtype=float
            )
        except (TypeError, ValueError):
            responses[name] = np.array([None if v is None else v for v in flat], dtype=object)
    return responses


def get_responses(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return responses_from_persisted(state.get('responses') or {})


def store_responses(state: Dict[str, Any], responses: Dict[str, Any]) -> None:
    state['responses'] = responses_to_persisted(responses)