"""Configuration file for the Sphinx documentation builder."""

from __future__ import annotations

import importlib.metadata
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -- Project information -----------------------------------------------------

project = "aba_optimiser"
author = "Joshua Gray"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"  # noqa: A001

try:
    release = importlib.metadata.version("aba_optimiser")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
# Suppress duplicate object warnings from autosummary
suppress_warnings = ["autosummary", "ref.python"]
# Mock imports for optional dependencies
autodoc_mock_imports = ["omc3", "xtrack", "xobjects", "xpart", "xdeps", "nxcals", "psutil", "cpymad"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

todo_include_todos = True
nitpicky = False


def setup(app):  # type: ignore[override]
    """Inject custom configuration at Sphinx start-up."""
    app.add_css_file("custom.css")


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = "aba_optimiser documentation"
html_theme_options = {
    "sidebar_hide_name": True,
}

# -- Intersphinx configuration ----------------------------------------------

intersphinx_mapping: dict[str, tuple[str, str | None]] = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
