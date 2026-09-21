"""DOE Toolkit — NiceGUI prototype entry point.

Run locally (replaces `streamlit run src/ui/app.py` for the prototype):

    python src/ui/nicegui/main.py                 → http://127.0.0.1:8081

Behind a reverse proxy there are two flavours, distinguished by whether the
proxy *keeps* or *strips* the external path prefix:

* Prefix-KEEPING proxy (e.g. nginx `proxy_pass http://backend;` under
  `location /doe-toolkit/`, so the backend sees `/doe-toolkit/...`): mount the
  app at the prefix:

      NICEGUI_MOUNT_PATH=/doe-toolkit python src/ui/nicegui/main.py
      # open https://<host>/doe-toolkit/

* Prefix-STRIPPING proxy (e.g. code-server's port forward, which serves the
  backend at `https://<host>:8443/proxy/8081/` while forwarding the request to
  `http://127.0.0.1:8081/` *without* the prefix): keep the backend mounted at
  `/` but tell NiceGUI to emit the external prefix in every URL via an
  `X-Forwarded-Prefix` header (the compiled-in mechanism NiceGUI uses for a
  pass-through proxy):

      NICEGUI_EXTERNAL_PREFIX=/proxy/8081 python src/ui/nicegui/main.py
      # open https://<host>:8443/proxy/8081/

The two env vars are mutually exclusive. The Streamlit app
(`streamlit run src/ui/app.py`) is untouched and remains the shipped app until
the full NiceGUI port is accepted.
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.ui.nicegui.pages  # noqa: F401  (registers all @ui.page routes)
from nicegui import ui
from starlette.middleware.base import BaseHTTPMiddleware

from src.ui.nicegui.downloads import register_download_routes

register_download_routes()

_STORAGE_SECRET_ENV = 'NICEGUI_STORAGE_SECRET'
_DEFAULT_STORAGE_SECRET = 'doe-toolkit-dev-secret'
_MOUNT_PATH_ENV = 'NICEGUI_MOUNT_PATH'  # proxy keeps the prefix (app mounted at it)
_EXTERNAL_PREFIX_ENV = 'NICEGUI_EXTERNAL_PREFIX'  # proxy strips the prefix (URLs prefixed, app at '/')
_CODE_SERVER_PREFIX = re.compile(r'^/proxy/\d+$')  # code-server's /proxy/<port>/ forward


def _storage_secret() -> str:
    return os.environ.get(_STORAGE_SECRET_ENV, _DEFAULT_STORAGE_SECRET)


def _normalize_prefix(prefix: str | None) -> str | None:
    if not prefix:
        return None
    return '/' + prefix.strip('/')


class _ForwardedPrefixMiddleware(BaseHTTPMiddleware):
    """Stamp every request with the configured external prefix.

    Used for code-server-style proxies that send the request to the backend
    with the `/proxy/<port>/` prefix already stripped. Mounting NiceGUI at `/`
    plus stamping `X-Forwarded-Prefix` makes NiceGUI emit prefixed URLs (import
    map, script tags, JS `prefix`, redirects) that the stripping proxy can then
    serve back end-to-end.
    """

    def __init__(self, app: object, prefix: str):
        super().__init__(app)  # type: ignore[arg-type]
        self._prefix = prefix

    async def dispatch(self, request, call_next):
        from starlette.datastructures import MutableHeaders

        MutableHeaders(scope=request.scope)['X-Forwarded-Prefix'] = self._prefix
        return await call_next(request)


def _run() -> None:
    host = os.environ.get('NICEGUI_HOST', '127.0.0.1')
    port = int(os.environ.get('NICEGUI_PORT', '8081'))
    mount_path = _normalize_prefix(os.environ.get(_MOUNT_PATH_ENV))
    external_prefix = _normalize_prefix(os.environ.get(_EXTERNAL_PREFIX_ENV))
    if not external_prefix and mount_path and _CODE_SERVER_PREFIX.match(mount_path):
        # code-server (and VSCode tunnelling) serves forwards at /proxy/<port>/ but
        # STRIPS that prefix before handing the request to the backend, so it must
        # run in external-prefix mode, not mount mode. Correct it rather than fail.
        print(
            f'"{_MOUNT_PATH_ENV}={mount_path}" looks like a code-server forward. '
            f'Using {_EXTERNAL_PREFIX_ENV} mode instead (app stays mounted at "/"); '
            f'consider setting {_EXTERNAL_PREFIX_ENV}={mount_path} explicitly.'
        )
        external_prefix, mount_path = mount_path, None
    if mount_path and external_prefix:
        raise SystemExit(f'{_MOUNT_PATH_ENV} and {_EXTERNAL_PREFIX_ENV} are mutually exclusive')

    if external_prefix:
        import uvicorn
        from fastapi import FastAPI

        print(f'DOE Toolkit (NiceGUI)  →  http://{host}:{port}/  (proxy prefix: {external_prefix})')
        print('Press Ctrl+C to stop.')
        fastapi_app = FastAPI(title='DOE Toolkit (NiceGUI)')
        fastapi_app.add_middleware(_ForwardedPrefixMiddleware, prefix=external_prefix)
        ui.run_with(
            fastapi_app,
            mount_path='/',
            title='DOE Toolkit',
            favicon='🔬',
            dark=False,
            storage_secret=_storage_secret(),
        )
        uvicorn.run(fastapi_app, host=host, port=port, log_level='info')
    elif mount_path:
        import uvicorn
        from fastapi import FastAPI

        print(f'DOE Toolkit (NiceGUI)  →  http://{host}:{port}{mount_path}/')
        print('Press Ctrl+C to stop.')
        fastapi_app = FastAPI(title='DOE Toolkit (NiceGUI)')
        ui.run_with(
            fastapi_app,
            mount_path=mount_path,
            title='DOE Toolkit',
            favicon='🔬',
            dark=False,
            storage_secret=_storage_secret(),
        )
        uvicorn.run(fastapi_app, host=host, port=port, log_level='info')
    else:
        print(f'DOE Toolkit (NiceGUI)  →  http://{host}:{port}')
        ui.run(
            host=host,
            port=port,
            title='DOE Toolkit (NiceGUI prototype)',
            reload=False,
            show=False,
            storage_secret=_storage_secret(),
            favicon='🔬',
            dark=False,
        )


if __name__ in {'__main__', '__mp_main__'}:
    _run()