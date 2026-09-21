"""Home + project resume (NiceGUI)."""

from typing import Any, Dict

from nicegui import ui

from src.ui.nicegui.layout import create_shell, start_fresh_button
from src.ui.nicegui.project_io import load_project
from src.ui.nicegui.state import any_project_state, can_access_step, get_state, is_step_complete

_STEP_ROUTE = {1: '/define', 2: '/model', 3: '/design', 4: '/preview'}

_STEPS = [
    ('1. Define Factors', '/define', 1),
    ('2. Select Model', '/model', 2),
    ('3. Choose Design', '/design', 3),
    ('4. Preview & Generate', '/preview', 4),
]


def _route_for_step(step: int) -> str:
    return _STEP_ROUTE.get(step, '/preview')


@ui.page('/')
def home() -> None:
    state = get_state()
    create_shell(state, active='/')

    with ui.column().classes('p-6 w-full max-w-6xl'):
        ui.label('DOE Toolkit').classes('text-3xl font-bold')
        ui.label('Professional Design of Experiments for Everyone').classes('text-lg text-gray-600')
        ui.markdown(
            'Define factors, select an analysis model, choose and configure a design, '
            'then generate and export it — all locally in your browser.'
        )
        ui.label('Choose how you want to start:').classes('text-base font-bold mt-2')

        with ui.row().classes('gap-4 mt-4 w-full items-stretch flex-wrap'):
            # 1. Start over — begin a brand-new project.
            with ui.card().classes('flex-1 basis-1/3 min-w-[240px]'):
                ui.label('Start over').classes('text-base font-bold')
                ui.markdown('Clear the current session and begin a brand-new project from scratch.')
                start_fresh_button(state)

            # 2. Continue — resume from where you left off.
            with ui.card().classes('flex-1 basis-1/3 min-w-[240px]'):
                ui.label('Continue').classes('text-base font-bold')
                ui.markdown('Pick up where you left off.')
                current = 1
                for step in range(1, 5):
                    if can_access_step(state, step) and not is_step_complete(state, step):
                        current = step
                        break
                label, route = next(
                    (l, r) for l, r, s in _STEPS if s == current
                )
                done = '✓' if is_step_complete(state, current) else ''
                ui.label(f'{done} {label}').classes('text-base font-bold mt-2')
                ui.markdown(_HOME_BLURB[current])
                ui.button('Open', on_click=lambda r=route: ui.navigate.to(r)).props('color=primary')

            # 3. Upload Project — load a saved .doeproject.
            with ui.card().classes('flex-1 basis-1/3 min-w-[240px]'):
                ui.label('Upload Project (.doeproject)').classes('text-base font-bold')
                ui.markdown(
                    'Load a saved project to restore factors, design, responses, model terms, '
                    'response definitions and constraints. The file is **held** below — nothing '
                    'changes until you click **Load Project**.'
                )
                _render_project_summary(state)
                ui.separator().classes('my-2')
                pending: Dict[str, Any] = {}
                file_status = ui.label('No file selected yet.').classes('text-xs text-gray-500 mt-1')
                with ui.row().classes('gap-2 mt-2'):
                    load_button = ui.button(
                        'Load Project',
                        on_click=lambda: _load_selected_project(state, pending, load_button, file_status),
                    ).props('color=primary')
                    ui.button(
                        'Export current project (.doeproject)',
                        on_click=_export_project,
                    ).props('outline')
                load_button.disable()  # enabled once a file is actually captured
                ui.upload(
                    label='Choose a .doeproject file',
                    on_upload=lambda e: _capture_project_file(pending, e, file_status, load_button),
                    auto_upload=True,
                    max_files=1,
                ).props('accept=.doeproject').classes('max-w-xl')


def _render_project_summary(state: Dict[str, Any]) -> None:
    design = state.get('design')
    model_terms = state.get('model_terms')
    with ui.expansion('Current project (what Load Project will replace)', icon='folder_open'):
        ui.label(f'Factors defined: {len(state.get("factors") or [])}').classes('text-sm')
        ui.label(f'Model terms: {len(model_terms) if model_terms else 0}').classes('text-sm')
        ui.label(f'Design: {len(design)} runs' if design else 'Design: none yet').classes('text-sm')
        ui.label(f'Responses: {len(state.get("responses") or {})}').classes('text-sm')


_HOME_BLURB = {
    1: 'Edit factors in an Excel-style grid (continuous / discrete / categorical).',
    2: 'Pick the analysis model that will guide design generation.',
    3: 'Choose and configure a design (factorial, RSM, D-optimal, LHS, split-plot).',
    4: 'Define responses, generate the design, preview metrics and export CSV/project.',
}


async def _capture_project_file(
    pending: Dict[str, Any], event: Any, file_status: ui.label, load_button: ui.button
) -> None:
    pending['content'] = await event.file.read()
    pending['name'] = event.file.name
    file_status.text = f'Received "{pending["name"]}" — it is NOT applied yet. Click Load Project to replace the current session.'
    file_status.classes(replace='text-xs text-blue-600 mt-1')
    ui.notify(
        'Project file received. Nothing changed yet — click Load Project to apply.',
        type='info',
    )
    load_button.enable()


def _load_selected_project(
    state: Dict[str, Any],
    pending: Dict[str, Any],
    load_button: ui.button,
    file_status: ui.label,
) -> None:
    content = pending.get('content')
    if content is None:
        ui.notify('Choose a .doeproject file first.', type='negative')
        return
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        ui.notify('Could not read the project file (expected UTF-8 JSON).', type='negative')
        return

    report = load_project(state, text)
    if not report.ok:
        ui.notify(report.error or 'Failed to load project.', type='negative', close_button=True)
        return

    for kind, message in report.messages:
        ui.notify(message, type=kind, close_button=True, multi_line=True)
    ui.notify('✓ Project loaded successfully!', type='positive')
    pending.clear()
    load_button.disable()
    file_status.text = 'No file selected yet.'
    file_status.classes(replace='text-xs text-gray-500 mt-1')
    ui.navigate.to(_route_for_step(report.current_step))


def _export_project() -> None:
    ui.navigate.to('api/project.doeproject')
    ui.notify('Project exported.', type='positive')