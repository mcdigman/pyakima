# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0
"""Resolve the installed pyakima distribution version."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version('pyakima')
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled source tree
    __version__ = '0.0.0+unknown'
