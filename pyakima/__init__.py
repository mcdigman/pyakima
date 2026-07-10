# SPDX-FileCopyrightText: Copyright 2026 Matthew C. Digman
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Provide numba-compatible Akima spline interpolation.

Expose the public building blocks of the :mod:`pyakima.pyakima` module: the
:class:`AkimaSpline` object for constructing and evaluating a spline, the
:class:`SplineCoeffs` named tuple that stores a spline's coefficients, and the
``njit``-compiled helpers that create and evaluate splines from within other
numba-compiled code.
"""

from pyakima._version import __version__
from pyakima.pyakima import (
    AkimaSpline,
    SplineCoeffs,
    akima_create_helper,
    cubic_call,
    cubic_call_scalar,
    cubic_call_vector,
    cubic_call_vector_linear,
    spline_single_knot_eval,
)

__all__ = [
    'AkimaSpline',
    'SplineCoeffs',
    '__version__',
    'akima_create_helper',
    'cubic_call',
    'cubic_call_scalar',
    'cubic_call_vector',
    'cubic_call_vector_linear',
    'spline_single_knot_eval',
]
