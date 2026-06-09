"""Minimal Sphinx configuration.

Build with the optional ``docs`` dependency group::

    poetry install --with docs
    poetry run sphinx-build -b html docs docs/_build/html
"""
from __future__ import annotations

project = "taskwright"
author = "taskwright authors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

html_theme = "alabaster"
autodoc_typehints = "description"
