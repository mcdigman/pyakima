# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Sphinx configuration for the pyakima HTML documentation build.

Run locally with ``make -C docs html`` (or ``sphinx-build -b html docs
docs/_build/html``); CI builds the same target in .github/workflows/docs.yml.
"""

from importlib.metadata import version as _version

# -- Project information -----------------------------------------------------
project = 'pyakima'
author = 'Matthew C. Digman'
copyright = '2025-2026, Matthew C. Digman'  # noqa: A001 - Sphinx-mandated name

# The full version (e.g. 0.1.0rc1) is read from the installed distribution so
# it stays in lockstep with [project].version in pyproject.toml, the same
# single source of truth pyakima.__version__ uses.
release = _version('pyakima')
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'myst_parser',
]

# numpy-style docstrings (matching [tool.pydoclint] style = "numpy").
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Render type hints in the parameter descriptions rather than the signature.
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'

# Resolve cross-references to the standard library and numpy.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output -------------------------------------------------------------
html_theme = 'furo'
html_title = f'pyakima {release}'

# html_static_path directories have their *contents* copied into the build's
# _static/ directory. '_static' supplies custom.css; '../assets' pulls in the
# repo-root demo animations/figures (also written there by pyakima.demos and
# referenced by the GitHub README) so the docs can display them without moving
# the folder or duplicating the files under docs/. Docs pages therefore
# reference them as ``_static/akima_*.{gif,png}``.
html_static_path = ['_static', '../assets']

# custom.css drives the light/dark image switching. Unlike the README's
# <picture> (which keys off the OS prefers-color-scheme), it follows Furo's own
# theme toggle via the body[data-theme] attribute.
html_css_files = ['custom.css']
