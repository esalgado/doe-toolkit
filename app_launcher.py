"""
DOE Toolkit - Application Launcher

Main entry point for the packaged executable.
Launches the Streamlit application in a browser.
"""

import os
import sys
import subprocess
import socket
import webbrowser
import time
from pathlib import Path


def find_free_port(start_port=8501, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")


def launch_app():
    """Launch the DOE Toolkit Streamlit application."""
    # Get the application directory
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = Path(sys._MEIPASS)
    else:
        # Running as script
        app_dir = Path(__file__).parent

    # Path to the main Streamlit app
    app_path = app_dir / "src" / "ui" / "app.py"

    if not app_path.exists():
        print(f"ERROR: Could not find app.py at {app_path}")
        input("Press Enter to exit...")
        sys.exit(1)

    # Find available port
    try:
        port = find_free_port()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

    # Launch Streamlit
    print("=" * 60)
    print("DOE Toolkit - Design of Experiments Software")
    print("=" * 60)
    print(f"\nStarting application on http://localhost:{port}")
    print("\nThe application will open in your default web browser.")
    print("To stop the application, close this window or press Ctrl+C")
    print("=" * 60)
    print()

    # Start Streamlit process
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"

    streamlit_cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light",
    ]

    try:
        # Start Streamlit in subprocess
        process = subprocess.Popen(
            streamlit_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Wait for server to start
        time.sleep(3)

        # Open browser
        url = f"http://localhost:{port}"
        print(f"Opening browser at {url}...")
        webbrowser.open(url)

        # Stream output
        print("\nApplication logs:")
        print("-" * 60)
        for line in process.stdout:
            print(line, end='')

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        process.terminate()
        process.wait()
        print("Application stopped.")
    except Exception as e:
        print(f"\nERROR: Failed to start application: {e}")
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    launch_app()
