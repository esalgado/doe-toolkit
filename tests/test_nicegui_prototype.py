"""Smoke tests for the NiceGUI prototype.

Pure round-trip tests (factor rows <-> Factor objects, design generation) run in
plain pytest. Render tests spawn the real app (``src/ui/nicegui/main.py``) on an
ephemeral port, probe it over HTTP like a browser, and tear it down — the same
"works on clean localhost" check the packaged app is validated against.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from src.core.full_factorial import full_factorial
from src.ui.nicegui.factors_ui import (
    empty_factor_row,
    factors_from_state,
    factors_to_rows,
    rows_from_state,
    rows_to_factors,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_FILE = PROJECT_ROOT / 'src' / 'ui' / 'nicegui' / 'main.py'


def _two_continuous_rows():
    return [
        {'Name': 'Temperature', 'Type': 'continuous', 'Min': 150, 'Max': 200,
         'Levels': '', 'Units': 'C', 'Changeability': 'easy'},
        {'Name': 'Pressure', 'Type': 'continuous', 'Min': 50, 'Max': 100,
         'Levels': '', 'Units': 'psi', 'Changeability': 'easy'},
    ]


def _launch_server(mount_path=None, external_prefix=None, storage_path=None):
    assert not (mount_path and external_prefix), 'proxy modes are mutually exclusive'
    import socket

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = str(sock.getsockname()[1])

    env = dict(os.environ)
    env.pop('PYTEST_CURRENT_TEST', None)  # keep the child out of NiceGUI's pytest mode
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['NICEGUI_PORT'] = port
    env['NICEGUI_HOST'] = '127.0.0.1'
    # Hermetic session storage: each test server persists to its own temp dir so
    # test runs never pollute the repo's .nicegui/ (and disk reloads are isolated).
    env['NICEGUI_STORAGE_PATH'] = str(
        storage_path if storage_path is not None else tempfile.mkdtemp(prefix='nicegui-storage-test-')
    )
    if mount_path:
        env['NICEGUI_MOUNT_PATH'] = mount_path
    if external_prefix:
        env['NICEGUI_EXTERNAL_PREFIX'] = external_prefix

    proc = subprocess.Popen(
        [sys.executable, str(MAIN_FILE)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    import urllib.request

    ready_paths = ['/']
    if mount_path:
        assert mount_path.startswith('/')
        ready_paths.append(mount_path + '/')
    try:
        deadline = time.monotonic() + 30
        ready = False
        while not ready:
            if proc.poll() is not None:
                out, err = proc.communicate()
                raise RuntimeError(
                    'NiceGUI server exited during startup\n'
                    f'STDOUT: {out.decode()[-1500:]}\n'
                    f'STDERR: {err.decode()[-3000:]}'
                )
            for ready_path in ready_paths:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}{ready_path}', timeout=1) as resp:
                        ready = resp.status == 200
                        if ready:
                            break
                except Exception:
                    continue
            if not ready:
                if time.monotonic() > deadline:
                    raise RuntimeError(f'NiceGUI server not ready on {ready_paths}')
                time.sleep(0.3)
        yield f'http://127.0.0.1:{port}'
    finally:
        proc.terminate()
        proc.wait(timeout=15)


@pytest.fixture(scope='module')
def live_server():
    yield from _launch_server()


@pytest.fixture(scope='module')
def live_server_proxy():
    yield from _launch_server(mount_path='/doe-toolkit')


@pytest.fixture(scope='module')
def live_server_external_prefix():
    yield from _launch_server(external_prefix='/proxy/8081')


@pytest.fixture(scope='module')
def live_server_autoconv():
    # code-server-style mount path misconfiguration must auto-correct to
    # external-prefix mode instead of failing with 404s
    yield from _launch_server(mount_path='/proxy/8081')


@pytest.mark.parametrize('path,expected', [
    ('/', 'DOE Toolkit'),
    ('/define', 'Step 1: Define Experimental Factors'),
    ('/model', 'Step 2: Select Analysis Model'),
    ('/design', 'Step 3: Choose Design Type'),
    ('/preview', 'Step 4: Preview'),
])
def test_pages_render(live_server, path, expected):
    import httpx

    with httpx.Client(base_url=live_server, timeout=10) as client:
        response = client.get(path)
    assert response.status_code == 200, path
    assert expected in response.text, path


def test_rows_to_factors_roundtrip():
    rows = [
        {'Name': 'Temperature', 'Type': 'continuous', 'Min': 150, 'Max': 200,
         'Levels': '', 'Units': 'C', 'Changeability': 'easy'},
        {'Name': 'Material', 'Type': 'categorical', 'Min': None, 'Max': None,
         'Levels': 'A, B, C', 'Units': '', 'Changeability': 'hard'},
    ]
    factors, errors = rows_to_factors(rows)
    assert errors == []
    assert factors[0].is_continuous()
    assert factors[0].levels == [150.0, 200.0]
    assert factors[1].is_categorical()
    assert factors[1].levels == ['A', 'B', 'C']
    assert factors[1].changeability.value == 'hard'


def test_factors_to_rows_roundtrip():
    factors, errors = rows_to_factors(_two_continuous_rows())
    assert not errors
    rows = factors_to_rows(factors)
    assert len(rows) == 2
    assert rows[0]['Min'] == 150.0
    restored, errors2 = rows_to_factors(rows)
    assert not errors2
    assert [f.name for f in restored] == ['Temperature', 'Pressure']


def test_rows_from_state_coerces_legacy_persisted_factors():
    """Sessions saved before the row-shaping change persisted orjson-dumped
    Factor dicts (factor_type/changeability as plain enum strings). Reading them
    must not crash and must yield the same rows as live Factor objects."""
    legacy = [
        {'name': 'afd', 'factor_type': 'categorical', 'changeability': 'easy',
         'levels': ['56', 'hola'], 'units': 'g'},
        {'name': 'Temp', 'factor_type': 'continuous', 'changeability': 'hard',
         'levels': [150.0, 200.0], 'units': 'C'},
        {'name': 'Speed', 'factor_type': 'continuous', 'changeability': 'very_hard',
         'levels': [1.0, 5.0], 'units': None},
    ]
    rows = rows_from_state(legacy)
    assert len(rows) == 3
    assert rows[0]['Name'] == 'afd'
    assert rows[0]['Type'] == 'categorical'
    assert rows[0]['Levels'] == '56, hola'
    assert rows[1] == {'Name': 'Temp', 'Type': 'continuous', 'Min': 150.0, 'Max': 200.0,
                       'Levels': '', 'Units': 'C', 'Changeability': 'hard'}
    assert rows[2]['Changeability'] == 'very_hard'

    factors = factors_from_state(legacy)
    assert [factor.name for factor in factors] == ['afd', 'Temp', 'Speed']
    assert factors[0].is_categorical()
    assert factors[0].levels == ['56', 'hola']
    assert factors[1].is_continuous()
    assert factors[1].changeability.value == 'hard'

    assert rows_from_state(rows) == rows  # already-row-shaped data passes through


def test_saved_factors_are_json_serializable_rows():
    """The stored value under state['factors'] must survive a JSON round-trip.

    Regression guard for the 500: Factor dataclasses were persisted and, on reload,
    re-wrapped as ObservableDicts, so is_continuous() crashed on the Define page.
    """
    factors, errors = rows_to_factors(_two_continuous_rows())
    assert not errors
    stored = factors_to_rows(factors)
    payload = json.loads(json.dumps(stored))  # simulate the on-disk round-trip
    assert all('Name' in row and 'Type' in row for row in payload)
    assert factors_from_state(payload)[0].name == 'Temperature'


def test_validation_errors_reported():
    rows = [
        {'Name': '', 'Type': 'continuous', 'Min': None, 'Max': None,
         'Levels': '', 'Units': '', 'Changeability': 'easy'},
        {'Name': 'Bad', 'Type': 'continuous', 'Min': 10, 'Max': 5,
         'Levels': '', 'Units': '', 'Changeability': 'easy'},
        {'Name': 'Cat', 'Type': 'categorical', 'Min': None, 'Max': None,
         'Levels': '', 'Units': '', 'Changeability': 'easy'},
    ]
    factors, errors = rows_to_factors(rows)
    assert factors == []
    assert len(errors) == 3


def test_empty_row_is_flagged_until_named():
    factors, errors = rows_to_factors([empty_factor_row()])
    assert factors == []
    assert len(errors) == 1


def test_design_roundtrip_through_core():
    factors, errors = rows_to_factors(_two_continuous_rows())
    assert not errors
    design = full_factorial(factors, randomize=False)
    assert len(design) == 4
    assert set(design['Temperature']) == {150.0, 200.0}
    assert set(design['Pressure']) == {50.0, 100.0}


def test_proxy_mount_path_serves_under_prefix(live_server_proxy):
    """Prefix-preserving proxy (nginx-style): app mounted at the prefix."""
    import httpx

    prefix = '/doe-toolkit'
    with httpx.Client(base_url=live_server_proxy, timeout=10) as client:
        home = client.get(f'{prefix}/')
        assert home.status_code == 200
        assert 'DOE Toolkit' in home.text
        assert f'{prefix}/_nicegui/' in home.text
        assert f'"vue":"{prefix}/_nicegui/' in home.text

        script = client.get(f'{prefix}/_nicegui/3.17.1/static/nicegui.js')
        assert script.status_code == 200
        assert 'javascript' in script.headers.get('content-type', '')

    # the app must NOT be reachable at the bare root when mounted under a prefix
    with httpx.Client(base_url=live_server_proxy, timeout=10) as client:
        bare = client.get('/')
    assert bare.status_code == 404


def test_external_prefix_stripping_proxy(live_server_external_prefix):
    """Prefix-stripping proxy (code-server /proxy/<port>/): app at '/', URLs prefixed.

    Models the code-server forwarding behaviour: the backend receives the path
    WITHOUT the /proxy/8081 prefix, but the served page must advertise prefixed
    URLs so the browser fetches them back through the proxy.
    """
    import httpx

    prefix = '/proxy/8081'
    with httpx.Client(base_url=live_server_external_prefix, timeout=10) as client:
        for path, expected in [('/', 'DOE Toolkit'), ('/define', 'Step 1: Define Experimental Factors')]:
            page = client.get(path)
            assert page.status_code == 200, path
            assert expected in page.text, path
            assert f'{prefix}/_nicegui/' in page.text, path
            assert f'"vue":"{prefix}/_nicegui/' in page.text, path
            assert f'prefix: "{prefix}"' in page.text, path

        # static assets live at the UNPREFIXED path on the backend (the proxy strips it)
        script = client.get('/_nicegui/3.17.1/static/nicegui.js')
        assert script.status_code == 200
        assert 'javascript' in script.headers.get('content-type', '')


def test_code_server_mount_misconfig_auto_corrects(live_server_autoconv):
    """NICEGUI_MOUNT_PATH=/proxy/<port> (a code-server forward) must behave like
    the external-prefix mode instead of mounting an app the proxy cannot reach."""
    import httpx

    prefix = '/proxy/8081'
    with httpx.Client(base_url=live_server_autoconv, timeout=10) as client:
        home = client.get('/')
        assert home.status_code == 200
        assert 'DOE Toolkit' in home.text
        assert f'"vue":"{prefix}/_nicegui/' in home.text
        assert client.get('/_nicegui/3.17.1/static/nicegui.js').status_code == 200


def _extract_page_assets(html: str):
    """Extract the client_id and the element tree embedded in a served NiceGUI page.

    The whole rendered element tree is inlined into the HTML as ``createApp(...)``
    arguments, so a test can find the grid and button element ids and their event
    listener ids without a browser.
    """
    client_id = re.search(r"client_id': '([0-9a-f-]+)'", html).group(1)
    match = re.search(r'createApp\(parseElements\(String\.raw`(.*?)`\)', html, re.S)
    raw = match.group(1)
    for old, new in [('&#36;', '$'), ('&#96;', '`'), ('&gt;', '>'), ('&lt;', '<'), ('&amp;', '&')]:
        raw = raw.replace(old, new)
    return client_id, json.loads(raw)


async def _click_factor_button(base_uri: str, client_id: str, button_id: int, listener_id: str) -> str:
    """Play a Real-browser click on a page via the socket.io wire protocol.

    Connects to the app like the browser's socket.io client does (implicit
    handshake queried at connect time), then emits the ``event`` message for the
    button. Returns the first ``run_javascript`` payload the server pushes back
    ('' if none arrived).
    """
    import urllib.parse as urlparse

    import socketio

    codes: list[str] = []
    sio = socketio.AsyncClient()
    sio.on('run_javascript', lambda data: codes.append(data.get('code', '')))
    query = urlparse.urlencode({
        'client_id': client_id,
        'next_message_id': '0',
        'implicit_handshake': 'true',
        'document_id': 'integration-test-doc',
        'tab_id': 'integration-test-tab',
    })
    try:
        await sio.connect(f'{base_uri}?{query}', socketio_path='_nicegui_ws/socket.io', transports=['polling'])
        await sio.emit('event', {
            'id': button_id,
            'client_id': client_id,
            'listener_id': listener_id,
            'args': [],
        })
        deadline = time.monotonic() + 10
        while not codes and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
    finally:
        await sio.disconnect()
    return codes[0] if codes else ''


def test_add_factor_dispatches_grid_api_method(live_server):
    """The Add-factor button must drive the AG-Grid API through run_grid_method.

    Regression guard for the NiceGUI 3.x AG-Grid gotcha: `grid.run_method(...)`
    dispatches to a nonexistent component method (`Method "applyTransaction" not
    found.` in the console) and silently does nothing. The correct dispatch is
    `grid.run_grid_method('applyTransaction', ...)`, which makes the server send
    the browser `return runMethod(<grid>, "run_grid_method", ["applyTransaction", ...])`.
    Injecting a real click over the socket.io protocol must produce that message.
    """
    import urllib.request

    with urllib.request.urlopen(f'{live_server}/define', timeout=5) as resp:
        html = resp.read().decode()
    client_id, elements = _extract_page_assets(html)

    grid_id = None
    add_button: tuple[int, str] | None = None
    for element_id, element in elements.items():
        tag = element.get('tag', '')
        if tag == 'nicegui-aggrid':
            grid_id = int(element_id)
        if tag == 'q-btn' and element.get('props', {}).get('label') == 'Add factor':
            add_button = (int(element_id), element['events'][0]['listener_id'])
    assert grid_id is not None, 'AG-Grid element not rendered on /define'
    assert add_button is not None, 'Add factor button not rendered on /define'

    button_id, listener_id = add_button
    code = asyncio.run(_click_factor_button(live_server, client_id, button_id, listener_id))
    expected = f'runMethod({grid_id}, "run_grid_method", ["applyTransaction"'
    assert expected in code, f'Add factor must dispatch run_grid_method, got: {code}'
    assert '"add"' in code, f'Add factor must send an applyTransaction add, got: {code}'


def test_define_page_never_dispatches_plain_run_method(live_server):
    """Both transaction buttons must use run_grid_method; run_method must not appear.

    Guards against reintroducing the silent-no-op dispatch on either the Add or
    the Remove-selected handler without needing the socket round-trip.
    """
    source = (PROJECT_ROOT / 'src' / 'ui' / 'nicegui' / 'pages' / 'define.py').read_text()
    assert source.count("grid.run_grid_method('applyTransaction'") >= 2, (
        'expected both Add factor and Remove selected to use run_grid_method'
    )
    assert "grid.run_method('applyTransaction'" not in source, (
        'run_method no-ops on AG-Grid API calls; use run_grid_method'
    )


async def _save_factors_via_socket(base_uri: str, client_id: str, button_id: int,
                                   listener_id: str, rows: list) -> None:
    """Click the Save-factors button over the socket.io protocol and let the save complete.

    The server saves the factor grid by calling ``grid.get_client_data()``, which
    awaits a JavaScript round-trip. A real browser would respond with the grid rows;
    this fake client answers the ``run_javascript`` request with the supplied rows
    and waits until the server reports the save via a ``notify`` event.
    """
    import urllib.parse as urlparse

    import socketio

    saved = asyncio.Event()
    sio = socketio.AsyncClient()

    @sio.on('run_javascript')
    async def on_run_javascript(data):
        request_id = data.get('request_id')
        if request_id:
            await sio.emit('javascript_response', {
                'request_id': request_id,
                'client_id': client_id,
                'result': rows,
            })

    @sio.on('notify')
    async def on_notify(data):
        if 'Saved' in str(data.get('message', '')):
            saved.set()

    query = urlparse.urlencode({
        'client_id': client_id,
        'next_message_id': '0',
        'implicit_handshake': 'true',
        'document_id': 'integration-test-doc',
        'tab_id': 'integration-test-tab',
    })
    try:
        await sio.connect(f'{base_uri}?{query}', socketio_path='_nicegui_ws/socket.io', transports=['polling'])
        await sio.emit('event', {
            'id': button_id,
            'client_id': client_id,
            'listener_id': listener_id,
            'args': [],
        })
        await asyncio.wait_for(saved.wait(), 15)
    finally:
        await sio.disconnect()


def test_persisted_factors_survive_restart_and_reload():
    """Regression guard for the define-page 500 after an app restart.

    Reproduces the reported crash exactly: save factors, restart the app with the
    same storage directory, and revisit /define with the same session cookie.
    Previously _save_factors stored Factor dataclasses; NiceGUI persisted them as
    orjson-dumped dicts and re-wrapped them as ObservableDicts on reload, so
    factors_to_rows crashed with
    ``AttributeError: 'ObservableDict' object has no attribute 'is_continuous'``.
    Now factors are stored as JSON-safe rows that reload cleanly, and the page
    renders the previously saved factor names after the restart.
    """
    import httpx

    storage_path = Path(tempfile.mkdtemp(prefix='nicegui-storage-restart-'))
    rows = _two_continuous_rows()

    server1 = _launch_server(storage_path=storage_path)
    try:
        base1 = next(server1)
        with httpx.Client(base_url=base1, timeout=10) as client:
            page = client.get('/define')
            assert page.status_code == 200
            client_id, elements = _extract_page_assets(page.text)
            save_button = next((
                (int(element_id), element['events'][0]['listener_id'])
                for element_id, element in elements.items()
                if element.get('tag') == 'q-btn' and element.get('props', {}).get('label') == 'Save factors'
            ), None)
            session_cookie = client.cookies.get('session')
        assert save_button is not None, 'Save factors button not rendered on /define'
        assert session_cookie, 'no session cookie issued'
        button_id, listener_id = save_button
        asyncio.run(_save_factors_via_socket(base1, client_id, button_id, listener_id, rows))

        # Same-session reload: the in-memory rows (already ObservableDict-wrapped)
        # must render without the is_continuous crash.
        with httpx.Client(base_url=base1, timeout=10, cookies={'session': session_cookie}) as client:
            page = client.get('/define')
            assert page.status_code == 200, page.text[:300]
            assert 'Temperature' in page.text and 'Pressure' in page.text

        # Wait for the async storage backup to land on disk.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if any('"Temperature"' in f.read_text() for f in storage_path.glob('storage-user-*.json')):
                break
            time.sleep(0.1)
        else:
            raise AssertionError('storage backup never written')
    finally:
        server1.close()

    # Fresh process reading the same storage dir + the same session cookie is the
    # exact reported-crash scenario: the file reload re-wraps rows as ObservableDicts.
    server2 = _launch_server(storage_path=storage_path)
    try:
        base2 = next(server2)
        with httpx.Client(base_url=base2, timeout=10, cookies={'session': session_cookie}) as client:
            deadline = time.monotonic() + 5
            for _ in range(25):
                page = client.get('/define')
                if page.status_code == 200 and 'Temperature' in page.text:
                    break
                time.sleep(0.2)
            assert page.status_code == 200, page.text[:300]
            assert 'Temperature' in page.text and 'Pressure' in page.text
    finally:
        server2.close()


# --- Checkpoint 1 (Steps 1-4 port): pure layer ----------------------------------


def _four_factors():
    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    return [
        Factor(name='Temp', factor_type=FactorType.CONTINUOUS, changeability=ChangeabilityLevel.EASY,
               levels=[30, 80]),
        Factor(name='Press', factor_type=FactorType.CONTINUOUS, changeability=ChangeabilityLevel.EASY,
               levels=[100, 300]),
        Factor(name='Time', factor_type=FactorType.CONTINUOUS, changeability=ChangeabilityLevel.EASY,
               levels=[1, 5]),
        Factor(name='Mix', factor_type=FactorType.CONTINUOUS, changeability=ChangeabilityLevel.HARD,
               levels=[0.1, 1.5]),
    ]


def test_generate_design_seven_types():
    from src.ui.nicegui import design_generation as dg

    factors = _four_factors()

    configs = [
        ('Full Factorial', {'n_center_points': 3, 'n_replicates': 1, 'n_blocks': 1,
                            'randomize': True, 'n_levels': 2}, 19),
        ('Fractional Factorial', {'fraction': '1/2', 'resolution': 4,
                                  'generator_mode': 'Standard (Recommended)',
                                  'n_center_points': 0, 'n_blocks': 1, 'randomize': True}, 8),
        ('Response Surface (CCD)', {'alpha': 'rotatable', 'alpha_label': 'Rotatable',
                                    'n_center_points': 3, 'randomize': True}, 27),
        ('Response Surface (Box-Behnken)', {'n_center_points': 3, 'randomize': True}, 27),
        ('Latin Hypercube', {'n_runs': 20, 'criterion': 'Maximin'}, 20),
        ('Split-Plot', {'n_replicates': 1, 'n_center_points': 2, 'n_blocks': 1,
                        'randomize_whole_plots': True, 'randomize_subplots': True}, 20),
    ]
    for design_type, config, runs in configs:
        design, metadata = dg.generate_design(factors, design_type, config, seed=42)
        assert len(design) == runs, f'{design_type}: {len(design)} runs'
        assert metadata['design_type'] == design_type
        # natural-unit factor columns present
        for f in factors:
            assert f.name in design.columns, f'{design_type}: missing {f.name}'

    # D-Optimal needs model terms + uses the design config for size.
    config = {'n_runs': 12}
    design, metadata = dg.generate_design(
        factors, 'D-Optimal', config,
        model_term_list=['1', 'Temp', 'Press', 'Time', 'Mix'],
        seed=42,
    )
    assert len(design) == 12
    assert metadata['d_efficiency'] is not None


def test_box_behnken_is_a_real_box_behnken_not_a_ccd():
    from src.ui.nicegui import design_generation as dg

    factors = _four_factors()
    design, metadata = dg.generate_design(
        factors, 'Response Surface (Box-Behnken)',
        {'n_center_points': 3, 'randomize': True}, seed=7,
    )
    assert metadata['variant'] == 'box_behnken'
    assert len(design) == 2 * 4 * 3 + 3  # 2k(k-1) + center points = 27

    # CCD preview code path would have produced axial (±alpha) points where one
    # factor sits at its center while others are at the extremes — verify BBD has
    # exactly the 2-level extreme grid for the "corner-avoiding" pairs.
    temp = design['Temp']
    assert not (temp == temp.max()).all()  # not a single-cube design


def test_generate_design_seed_determinism():
    from src.ui.nicegui import design_generation as dg

    factors = _four_factors()
    for design_type, config in [
        ('Full Factorial', {'n_center_points': 2, 'n_replicates': 1, 'n_blocks': 1,
                            'randomize': True, 'n_levels': 2}),
        ('Latin Hypercube', {'n_runs': 20, 'criterion': 'Maximin'}),
        ('Split-Plot', {'n_replicates': 1, 'n_center_points': 0, 'n_blocks': 1,
                        'randomize_whole_plots': True, 'randomize_subplots': True}),
    ]:
        d1, _ = dg.generate_design(factors, design_type, config, seed=123)
        d2, _ = dg.generate_design(factors, design_type, config, seed=123)
        assert d1.reset_index(drop=True).equals(d2.reset_index(drop=True)), design_type


def test_workflow_state_json_roundtrip_preserves_objects():
    """Every key of ``state['factors']``/``responses``/``design``/``constraints`` survives
    the JSON round-trip NiceGUI applies to app.storage.user (regression guard for the
    500 pattern applied to the new object-bearing keys)."""
    import json

    import numpy as np
    import pandas as pd

    from src.core.optimal.constraints import LinearConstraint
    from src.ui.nicegui import workspace

    state = {}
    workspace.store_factors(state, _four_factors()[:2])
    workspace.store_constraints(state, [
        LinearConstraint(coefficients={'Temp': 1.0, 'Press': 2.0}, bound=500.0, constraint_type='le'),
    ])
    workspace.store_design(state, pd.DataFrame({
        'StdOrder': [1, 2], 'RunOrder': [1, 2], 'Temp': [30.0, 80.0], 'Press': [100.0, 300.0],
    }))
    workspace.store_responses(state, {'Yield': np.array([1.5, 2.5])})

    persisted = json.loads(json.dumps(state))
    assert [f.name for f in workspace.get_factors(persisted)] == ['Temp', 'Press']
    assert workspace.get_design(persisted).shape == (2, 4)
    assert workspace.get_responses(persisted)['Yield'].tolist() == [1.5, 2.5]
    assert workspace.get_constraints(persisted)[0].bound == 500.0
    assert workspace.get_design_space(persisted) is not None


def test_project_roundtrip_with_new_keys():
    """The .doeproject round-trip preserves the newly-persisted keys: response
    definitions, design-level model terms and constraints (parked items 2-3)."""
    import json

    from src.core.factors import Factor, FactorType, ChangeabilityLevel
    from src.ui.nicegui.project_io import create_project, load_project
    from src.ui.nicegui.workspace import store_design, store_factors

    import numpy as np
    import pandas as pd

    state = {}
    store_factors(state, _four_factors()[:2])
    store_design(state, pd.DataFrame({
        'StdOrder': [1, 2], 'RunOrder': [1, 2], 'Temp': [30.0, 80.0], 'Press': [100.0, 300.0],
    }))
    state.update({
        'design_type': 'Full Factorial',
        'design_metadata': {'design_type': 'Full Factorial'},
        'response_definitions': [{'name': 'Yield', 'units': '%'}],
        'model_terms': ['1', 'Temp', 'Press', 'Temp*Press'],
        'responses': {'Yield': [1.0, 2.0]},
        'response_names': ['Yield'],
    })

    project_json = create_project(state)
    project = json.loads(project_json)
    for key in ('factors', 'design', 'responses', 'design_metadata', 'design_config',
                'response_definitions', 'model_terms', 'constraints'):
        assert key in project, key

    fresh = {}
    report = load_project(fresh, project_json)
    assert report.ok, report.error
    assert report.n_factors == 2
    assert report.n_responses == 1
    assert fresh['response_definitions'] == [{'name': 'Yield', 'units': '%'}]
    assert fresh['model_terms'] == ['1', 'Temp', 'Press', 'Temp*Press']


def test_load_project_sanitizes_factor_names():
    from src.ui.nicegui.project_io import create_project, load_project
    from src.ui.nicegui.workspace import store_factors

    from src.core.factors import Factor, FactorType, ChangeabilityLevel

    state = {}
    store_factors(state, [Factor(name='Day 0 (raw)', factor_type=FactorType.CONTINUOUS,
                                 changeability=ChangeabilityLevel.EASY, levels=[1, 10])])
    project_json = create_project(state)

    # Simulate a legacy project with an unsanitized factor name.
    import json
    project = json.loads(project_json)
    project['factors'][0]['name'] = 'A*B'
    project['factors'][0]['levels'] = [1, 10]
    fresh = {}
    report = load_project(fresh, json.dumps(project))
    assert report.ok
    assert len([m for m in report.messages if m[0] == 'warning']) >= 1
    assert fresh['factors'][0]['Name'].startswith('A_B') or fresh['factors'][0]['Name'] != 'A*B'


def test_preview_pages_render_new_routes(live_server):
    """The new checkpoint-1 routes render 200 with their expected headers."""
    import httpx

    with httpx.Client(base_url=live_server, timeout=10) as client:
        for path, marker in [
            ('/model', 'Step 2: Select Analysis Model'),
            ('/design', 'Step 3: Choose Design Type'),
            ('/preview', 'Step 4: Preview'),
        ]:
            page = client.get(path)
            assert page.status_code == 200, path
            assert marker in page.text, path


# --- Start Fresh & reactive estimate (pure layer) ------------------------------


def test_continue_button_enables_after_saving_factors(live_server):
    """Saving factors must re-enable the Define page's Continue button.

    Regression guard: define.py computes the button's disable prop at render time,
    and _save_factors mutates state without re-rendering (unlike model.py/design.py
    which reload() after committing) — leaving 'Continue → Choose Model' stuck
    disabled until the user navigates away and back.
    """
    import asyncio

    import httpx

    def _continue_state(elements):
        for element in elements.values():
            if (element.get('tag') == 'q-btn'
                    and 'Continue' in str(element.get('props', {}).get('label', ''))):
                return 'disable' in element.get('props', {})
        raise AssertionError('Continue button not rendered on /define')

    with httpx.Client(base_url=live_server, timeout=10) as client:
        page = client.get('/define')
        assert page.status_code == 200
        client_id, elements = _extract_page_assets(page.text)
        assert _continue_state(elements), 'Continue must start disabled on a fresh session'

        save_button = next((
            (int(element_id), element['events'][0]['listener_id'])
            for element_id, element in elements.items()
            if element.get('tag') == 'q-btn' and element.get('props', {}).get('label') == 'Save factors'
        ), None)
        assert save_button is not None, 'Save factors button not rendered on /define'
        button_id, listener_id = save_button
        asyncio.run(_save_factors_via_socket(live_server, client_id, button_id, listener_id, _two_continuous_rows()))

        page = client.get('/define')
        assert page.status_code == 200
        _, elements = _extract_page_assets(page.text)
        assert not _continue_state(elements), 'Continue must enable after saving factors'


def test_reset_state_clears_project_work_and_isolates_mutables():
    """reset_state() must wipe project work and deep-copy DEFAULTS so successive
    resets (and clients) never share mutable containers."""
    from src.ui.nicegui.state import DEFAULTS, reset_state

    a = {}
    reset_state(a)
    a['factors'].append({'Name': 'Temp', 'Type': 'continuous'})
    a['design_type'] = 'Full Factorial'
    a['model_terms'] = ['1', 'Temp']
    a['responses'] = {'Yield': [1.0]}
    a['response_definitions'] = [{'name': 'Yield', 'units': '%'}]

    # A second, independent reset starts from pristine DEFAULTS.
    b = {}
    reset_state(b)
    for key, default in DEFAULTS.items():
        assert b.get(key) == default, key
        if isinstance(default, list):
            assert b[key] == [] and b[key] is not default, key
    assert b['factors'] == []  # a's append did not leak into b

    # After clearing, the mutated state matches DEFAULTS again (nothing lingering).
    reset_state(a)
    for key, default in DEFAULTS.items():
        assert a.get(key) == default, key


def test_any_project_state_flags_real_work():
    from src.ui.nicegui.state import any_project_state, reset_state

    empty = {}
    reset_state(empty)
    assert not any_project_state(empty)

    cases = {
        'factors': [{'Name': 'Temp', 'Type': 'continuous'}],
        'model_terms': ['1', 'Temp'],
        'design_type': 'Full Factorial',
        'design': [{'Temp': 30.0}],
        'responses': {'Yield': [1.0]},
        'response_definitions': [{'name': 'Yield', 'units': '%'}],
        'optimization_results': {'best': []},
    }
    for key, value in cases.items():
        state = {}
        reset_state(state)
        state[key] = value
        assert any_project_state(state), key


def test_estimate_runs_reacts_to_config_changes():
    """The reactive 'Estimated runs' label recomputes when replicates, center
    points, fractions or blocks change (Step 3)."""
    from src.ui.nicegui.design_config import estimate_runs

    assert estimate_runs('Full Factorial', {'n_center_points': 3, 'n_replicates': 1}, 4) == 19
    assert estimate_runs('Full Factorial', {'n_center_points': 3, 'n_replicates': 2}, 4) == 38
    assert estimate_runs('Full Factorial', {'n_center_points': 0, 'n_replicates': 1}, 4) == 16

    assert estimate_runs('Fractional Factorial', {'fraction': '1/2', 'n_center_points': 0}, 4) == 8
    assert estimate_runs('Fractional Factorial', {'fraction': '1/2', 'n_center_points': 2}, 4) == 10
    assert estimate_runs('Fractional Factorial', {'fraction': '1/4', 'n_center_points': 0}, 4) == 4

    assert estimate_runs('Response Surface (CCD)', {'n_center_points': 3}, 3) == 17
    assert estimate_runs('Response Surface (CCD)', {'n_center_points': 5}, 3) == 19

    assert estimate_runs('Split-Plot',
                         {'n_replicates': 1, 'n_center_points': 0, 'n_blocks': 1}, 2, [2, 2]) == 4
    assert estimate_runs('Split-Plot',
                         {'n_replicates': 1, 'n_center_points': 0, 'n_blocks': 2}, 2, [2, 2]) == 8
    assert estimate_runs('Split-Plot',
                         {'n_replicates': 1, 'n_center_points': 0, 'n_blocks': 1}, 2, [3, 2]) == 6

    assert estimate_runs('D-Optimal', {'n_runs': 12}, 4) == 12
    assert estimate_runs('Latin Hypercube', {'n_runs': 20}, 3) == 20


def test_untouched_forms_validate_after_defaults_seeded():
    """Selecting a design and clicking Generate immediately must not fail because
    the values shown in the widgets were never persisted to design_config.

    Regression guard for the reported 'Latin Hypercube needs at least n_factors + 1
    runs' firing with 2 factors and 20 runs: the forms seed their defaults at
    render via ensure_*_config, and D-Optimal / Fractional / CCD had the same
    latent gap.
    """
    from src.ui.nicegui.design_config import (
        ensure_ccd_config,
        ensure_d_optimal_config,
        ensure_fractional_config,
        ensure_full_factorial_config,
        ensure_lhs_config,
        validate_config,
    )

    cfg = ensure_lhs_config({}, 2)
    ok, errors = validate_config('Latin Hypercube', cfg, 2)
    assert ok, errors
    assert cfg['n_runs'] == 20
    assert cfg['criterion'] == 'Maximin'

    model_terms = ['1', 'Temp', 'Press']
    cfg = ensure_d_optimal_config({}, len(model_terms))
    ok, errors = validate_config('D-Optimal', cfg, 2, model_terms)
    assert ok, errors
    assert cfg['n_runs'] == 6

    cfg = ensure_fractional_config({}, 4)
    ok, errors = validate_config('Fractional Factorial', cfg, 4)
    assert ok, errors
    assert cfg['fraction'] == '1/2'
    assert cfg['resolution'] == 5

    cfg = ensure_ccd_config({})
    ok, errors = validate_config('Response Surface (CCD)', cfg, 3)
    assert ok, errors
    assert cfg['alpha'] == 'rotatable'

    cfg = ensure_full_factorial_config({})
    ok, errors = validate_config('Full Factorial', cfg, 2)
    assert ok, errors
    assert cfg['n_levels'] == 2


def test_defaults_seeding_never_overwrites_user_values():
    from src.ui.nicegui.design_config import ensure_lhs_config

    cfg = {'n_runs': 15, 'criterion': 'Correlation'}
    ensure_lhs_config(cfg, 2)
    assert cfg['n_runs'] == 15
    assert cfg['criterion'] == 'Correlation'


def test_latin_hypercube_error_message_interpolates_minimum():
    from src.ui.nicegui.design_config import validate_config

    _, errors = validate_config('Latin Hypercube', {'n_runs': 2}, 2)
    assert len(errors) == 1
    assert '3' in errors[0]
    assert 'n_factors' not in errors[0]


def test_build_design_csv_from_state():
    from src.ui.nicegui.downloads import build_design_csv
    from src.ui.nicegui.factors_ui import rows_to_factors
    from src.ui.nicegui.workspace import design_to_records

    factors, errors = rows_to_factors(_two_continuous_rows())
    assert not errors
    design = full_factorial(factors, randomize=False)
    state = {
        'design': design_to_records(design),
        'factors': _two_continuous_rows(),
        'design_type': 'full_factorial',
        'design_metadata': {'design_type': 'full_factorial', 'resolution': 5},
        'model_terms': ['1', 'Temperature', 'Pressure'],
        'response_definitions': [{'name': 'Yield', 'units': 'g'}],
    }
    csv = build_design_csv(state)
    assert csv is not None
    assert '# DOE-TOOLKIT DESIGN' in csv
    assert '# Design Type: full_factorial' in csv
    assert '# MODEL TERMS' in csv
    assert '# RESPONSE DEFINITIONS' in csv
    assert 'Temperature' in csv
    assert 'Pressure' in csv
    assert 'Yield' in csv
    assert build_design_csv({'design': []}) is None
    assert build_design_csv({}) is None


def test_download_routes_registered():
    from nicegui import app

    from src.ui.nicegui.downloads import register_download_routes

    register_download_routes()
    paths = [getattr(route, 'path', '') for route in app.routes]
    assert '/api/design.csv' in paths
    assert '/api/project.doeproject' in paths
    register_download_routes()
    assert paths.count('/api/design.csv') == 1


def test_upload_enables_load_project_button(live_server):
    """Selecting a .doeproject file must enable the Load Project button.

    Regression guard for the reported stuck flow on the home page: auto_upload
    POSTs the selected file and _capture_project_file stores it, but the Load
    Project button was never re-enabled after its initial disable() — it stayed
    greyed out no matter what, and the Uploader's per-file actions couldn't
    help. The test reproduces the browser's auto-upload POST directly against
    the element's upload route and watches the socket for the enable update.
    """
    import asyncio
    import urllib.parse as urlparse

    import httpx
    import socketio

    with httpx.Client(base_url=live_server, timeout=10) as client:
        page = client.get('/')
        assert page.status_code == 200
        client_id, elements = _extract_page_assets(page.text)

        upload = next((
            (int(element_id), element)
            for element_id, element in elements.items()
            if '/upload/' in str(element.get('props', {}).get('url', ''))
        ), None)
        assert upload is not None, 'upload element not rendered on /'
        upload_id, upload_element = upload
        upload_url = upload_element['props']['url']

        load_id = next((
            int(element_id)
            for element_id, element in elements.items()
            if element.get('tag') == 'q-btn'
            and element.get('props', {}).get('label') == 'Load Project'
        ), None)
        assert load_id is not None, 'Load Project button not rendered on /'

        async def drive() -> list:
            messages: list = []
            sio = socketio.AsyncClient()
            sio.on('update', lambda data: messages.append(('update', data)))
            sio.on('notify', lambda data: messages.append(('notify', data)))
            query = urlparse.urlencode({
                'client_id': client_id,
                'next_message_id': '0',
                'implicit_handshake': 'true',
                'document_id': 'upload-test-doc',
                'tab_id': 'upload-test-tab',
            })
            await sio.connect(
                f'{live_server}?{query}', socketio_path='_nicegui_ws/socket.io', transports=['polling']
            )
            try:
                resp = client.post(
                    upload_url,
                    files={'file': ('doe_project_test.doeproject', b'{"version": 1}', 'application/json')},
                )
                assert resp.status_code == 200, resp.text[:300]
                await asyncio.sleep(2)
            finally:
                await sio.disconnect()
            return messages

        messages = asyncio.run(drive())

    assert any(
        kind == 'notify' and 'Project file received' in str(data)
        for kind, data in messages
    ), f'capture handler did not run: {messages}'
    updates = [
        data for kind, data in messages
        if kind == 'update' and str(load_id) in data
    ]
    assert updates, f'no enable update for Load button {load_id}: {[d for k, d in messages if k in ("update", "notify")]}'
    props = updates[-1][str(load_id)].get('props', {})
    assert not props.get('disable'), f'Load Project button still disabled after upload: {props}'