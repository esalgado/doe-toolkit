"""NiceGUI page routes (checkpoint 1: Steps 1-4 + home).

Converted from the prototype's single ``pages.py`` so each workflow step lives
in its own module. ``main.py`` imports this package to register every route.
"""

from src.ui.nicegui.pages import define, design, home, model, preview  # noqa: F401