"""
Setup configuration for DOE-Toolkit.
"""

import re
from pathlib import Path
from setuptools import setup, find_packages


def get_version() -> str:
    """Read version from src/__version__.py without importing the package."""
    version_file = Path(__file__).parent / "src" / "__version__.py"
    content = version_file.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find __version__ in src/__version__.py")
    return match.group(1)


setup(
    name="doe-toolkit",
    version=get_version(),
    description="Free, open-source Design of Experiments software",
    author="DOE-Toolkit Contributors",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "cvxpy",
        "matplotlib",
        "plotly",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
)
