"""Server-side download endpoints.

``ui.download(bytes, ...)`` sends the payload over the WebSocket and the browser
wraps it in a Blob URL opened in a new tab (``target=_blank`` + immediate
``revokeObjectURL``). That depends on the ``download`` attribute, which some
browsers ignore, so the file renders inline in the tab instead of downloading.

Serving the payload from a real HTTP endpoint with
``Content-Disposition: attachment`` forces a real download in every browser.

Routes are registered on the NiceGUI app, so they nest under any ``mount_path``
and survive the code-server prefix-stripping proxy (the trigger uses a relative
path so the browser resolves it against the current page URL).
"""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import Response

from src.ui.nicegui.workspace import get_design, get_factors


def build_design_csv(state: Dict[str, Any]) -> Optional[str]:
    """Rebuild the design CSV export from persisted JSON-safe state."""
    if not state.get('design'):
        return None
    from src.ui.utils.csv_parser import generate_doe_csv

    design = get_design(state)
    factors = get_factors(state)
    metadata = state.get('design_metadata') or {}
    design_terms: list = list(state.get('model_terms') or [])
    export_model_terms = {'__design__': design_terms} if design_terms else None
    return generate_doe_csv(
        design=design,
        factors=factors,
        response_definitions=state.get('response_definitions') or [],
        design_type=metadata.get('design_type', state.get('design_type', 'custom')),
        design_metadata={k: v for k, v in metadata.items() if v is not None},
        model_terms=export_model_terms,
    )


def _attachment_response(content: str, filename: str, media_type: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


def register_download_routes() -> None:
    """Register the download endpoints on the NiceGUI app (idempotent)."""
    from nicegui import app

    for route in getattr(app, 'routes', []):
        if getattr(route, 'path', '') == '/api/design.csv':
            return

    @app.get('/api/design.csv')
    def design_csv() -> Response:
        state: Dict[str, Any] = app.storage.user
        csv = build_design_csv(state)
        if csv is None:
            raise HTTPException(status_code=404, detail='No design has been generated yet.')
        filename = f'doe_design_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        return _attachment_response(csv, filename, 'text/csv')

    @app.get('/api/project.doeproject')
    def project_file() -> Response:
        from src.ui.nicegui.project_io import create_project

        state: Dict[str, Any] = app.storage.user
        try:
            project_json = create_project(state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Project export failed: {e}') from e
        filename = f'doe_project_{datetime.now().strftime("%Y%m%d_%H%M%S")}.doeproject'
        return _attachment_response(project_json, filename, 'application/json')