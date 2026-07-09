"""Demonstrate pyakima timing against optional alternate implementations.

Copyright 2026 Matthew C. Digman
"""

from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING

import numba
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyakima import (  # noqa: E402
    AkimaSpline,
    SplineCoeffs,
    akima_create_helper,
    cubic_call,
    cubic_call_scalar,
    cubic_call_vector,
    cubic_call_vector_linear,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

    import pygsl_lite.spline.akima

CONTROL_POINT_LENGTHS = (5, 64, 4096)
SCALAR_GRID_LENGTH = 257
VECTOR_CALL_LENGTHS = (8, 1000, 10000, 100000)
VECTOR_JIT_MIN_EVALS = 8192
EXT = 0
MIN_SECONDS = 0.01
REPEATS = 5


@dataclass(frozen=True)
class ModelCase:
    """Spline model used to pair pyakima with the closest optional backend."""

    label: str
    corner_model: int
    denom_small_cut: float
    alternate: str
    scipy_method: str | None = None


@dataclass(frozen=True)
class OptionalModule:
    """Optional import state for a backend used by this demo."""

    label: str
    module: ModuleType | None
    status: str


@dataclass(frozen=True)
class Timing:
    """Measured per-operation time or a reason the timing was skipped."""

    seconds: float | None
    loops: int = 0
    reason: str = ''


@dataclass(frozen=True)
class DemoOptions:
    """Command-line options for the timing demo."""

    show_overhead: bool
    show_all_call_models: bool


MODELS = (
    ModelCase(
        label='gsl-style',
        corner_model=0,
        denom_small_cut=0.0,
        alternate='gsl',
    ),
    ModelCase(
        label='scipy akima',
        corner_model=1,
        denom_small_cut=1.0e-9,
        alternate='scipy akima',
        scipy_method='akima',
    ),
    ModelCase(
        label='scipy makima',
        corner_model=2,
        denom_small_cut=0.0,
        alternate='scipy makima',
        scipy_method='makima',
    ),
)
DEFAULT_CALL_MODEL = ModelCase(
    label='scipy-style',
    corner_model=2,
    denom_small_cut=0.0,
    alternate='scipy-style',
    scipy_method='makima',
)
DEFAULT_CALL_MODELS = (MODELS[0], DEFAULT_CALL_MODEL)


def _installed_version(package_name: str) -> str | None:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return None


def _optional_import(module_name: str, label: str) -> OptionalModule:
    try:
        module = import_module(module_name)
    except ImportError as exc:
        return OptionalModule(label, None, f'skipped ({exc})')
    root_name = module_name.split('.', maxsplit=1)[0]
    version = getattr(module, '__version__', None)
    if version is None:
        package = import_module(root_name)
        version = getattr(package, '__version__', None)
    if version is None:
        version = _installed_version(root_name)
    detail = f'available {version}' if version else 'available'
    return OptionalModule(label, module, detail)


SCIPY = _optional_import('scipy.interpolate', 'scipy')
PYGSL = _optional_import('pygsl_lite.spline', 'pygsl_lite')


def _parse_args() -> DemoOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--show-overhead',
        action='store_true',
        help='include Python dispatch/class evaluation columns; constructor overhead is always shown',
    )
    parser.add_argument(
        '--show-all-call-models',
        action='store_true',
        help='include every call-model row; by default scalar/vector calls use gsl-style and scipy-style rows',
    )
    args = parser.parse_args()
    return DemoOptions(show_overhead=args.show_overhead, show_all_call_models=args.show_all_call_models)


def _control_points(n_control: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 10.0, n_control, dtype=np.float64)
    y = np.sin(2.0 * np.pi * x / 10.0) + 0.05 * np.cos(3.0 * x)
    return x, y


def _eval_points(n_eval: int) -> np.ndarray:
    return np.linspace(0.0, 10.0, n_eval, dtype=np.float64)


def _scalar_eval_points() -> np.ndarray:
    return np.linspace(0.0, 10.0, SCALAR_GRID_LENGTH, dtype=np.float64)


def _pyakima_spline(
    x: np.ndarray,
    y: np.ndarray,
    model: ModelCase,
    *,
    linear_vector_calls: int = 0,
) -> AkimaSpline:
    return AkimaSpline(
        x,
        y,
        ext=EXT,
        corner_model=model.corner_model,
        denom_small_cut=model.denom_small_cut,
        linear_vector_calls=linear_vector_calls,
    )


def _pyakima_helper(x: np.ndarray, y: np.ndarray, model: ModelCase) -> object:
    # Match AkimaSpline's ownership behavior so helper vs class isolates object overhead.
    return akima_create_helper(
        x,
        y,
        corner_model=model.corner_model,
        denom_small_cut=model.denom_small_cut,
    )


def _scipy_spline(x: np.ndarray, y: np.ndarray, method: str) -> object:
    if SCIPY.module is None:
        raise ImportError(SCIPY.status)
    interpolator = SCIPY.module.Akima1DInterpolator
    return interpolator(
        x,
        y,
        extrapolate=True,
        method=method,
    )


def _gsl_spline(x: np.ndarray, y: np.ndarray) -> object:
    if PYGSL.module is None:
        raise ImportError(PYGSL.status)
    akima = PYGSL.module.akima
    spline = akima(x.size)
    spline.init(x, y)
    return spline


def _alternate_spline(model: ModelCase, x: np.ndarray, y: np.ndarray) -> tuple[str, object]:
    if model.alternate == 'gsl':
        return 'gsl', _gsl_spline(x, y)
    if model.scipy_method is None:
        msg = f'{model.label} has no scipy method configured'
        raise ValueError(msg)
    return model.alternate, _scipy_spline(x, y, model.scipy_method)


def _call_gsl_scalar(spline: pygsl_lite.spline.akima, xint: float) -> object:  # pyrefly: ignore[not-a-type]
    if hasattr(spline, 'eval'):
        return spline.eval(xint)
    if hasattr(spline, 'eval_e'):
        return spline.eval_e(xint)
    msg = 'pygsl_lite spline has no scalar eval method'
    raise AttributeError(msg)


def _call_gsl_vector(spline: pygsl_lite.spline.akima, xint: np.ndarray) -> object:  # pyrefly: ignore[not-a-type]
    if not hasattr(spline, 'eval_vector'):
        msg = 'pygsl_lite spline has no eval_vector method'
        raise AttributeError(msg)
    return spline.eval_vector(xint)


def _call_python_spline(spline: Callable[[float | np.ndarray], object], xint: float | np.ndarray) -> object:
    return spline(xint)


def _call_alternate(spline: object, model: ModelCase, xint: float | np.ndarray) -> object:
    if model.alternate == 'gsl':
        if isinstance(xint, np.ndarray):
            return _call_gsl_vector(spline, xint)
        return _call_gsl_scalar(spline, xint)
    if not callable(spline):
        msg = f'{model.alternate} spline is not callable'
        raise TypeError(msg)
    return _call_python_spline(spline, xint)


@numba.njit()
def _cubic_call_scalar_grid_jit(x_scalars: np.ndarray, spline: SplineCoeffs) -> float:
    total = 0.0
    for x_scalar in x_scalars:
        total += cubic_call_scalar(float(x_scalar), spline, EXT)
    return total


@numba.njit()
def _cubic_call_vector_loop_jit(
    x_eval: np.ndarray,
    spline: SplineCoeffs,
    inner_loops: int,
) -> float:
    total = 0.0
    for _ in range(inner_loops):
        values = cubic_call(x_eval, spline, EXT)
        total += values[0]
        if values.size > 1:
            total += values[-1]
    return total


def _run_loop(callback: Callable[[], object], loops: int) -> float:
    start = perf_counter()
    for _ in range(loops):
        callback()
    return perf_counter() - start


def _time_required(callback: Callable[[], object]) -> Timing:
    callback()
    loops = 1
    elapsed = _run_loop(callback, loops)
    while elapsed < MIN_SECONDS:
        loops *= 2
        elapsed = _run_loop(callback, loops)

    samples = tuple(_run_loop(callback, loops) / loops for _ in range(REPEATS))
    return Timing(median(samples), loops)


def _time_optional(callback: Callable[[], object]) -> Timing:
    try:
        return _time_required(callback)
    except ImportError as exc:
        return Timing(None, reason=f'{type(exc).__name__}: {exc}')


def _time_scalar_grid_required(callback: Callable[[float], object], x_scalars: np.ndarray) -> Timing:
    def call_grid() -> None:
        for x_scalar in x_scalars:
            callback(float(x_scalar))

    timing = _time_required(call_grid)
    if timing.seconds is None:
        msg = 'required scalar-grid timing unexpectedly failed'
        raise RuntimeError(msg)
    return Timing(timing.seconds / x_scalars.size, timing.loops * x_scalars.size)


def _time_scalar_grid_optional(callback: Callable[[float], object], x_scalars: np.ndarray) -> Timing:
    try:
        return _time_scalar_grid_required(callback, x_scalars)
    except ImportError as exc:
        return Timing(None, reason=f'{type(exc).__name__}: {exc}')


def _time_scalar_jit_grid_required(x_scalars: np.ndarray, spline: SplineCoeffs) -> Timing:
    timing = _time_required(functools.partial(_cubic_call_scalar_grid_jit, x_scalars, spline))
    if timing.seconds is None:
        msg = 'required jitted scalar-grid timing unexpectedly failed'
        raise RuntimeError(msg)
    return Timing(timing.seconds / x_scalars.size, timing.loops * x_scalars.size)


def _vector_jit_inner_loops(n_eval: int) -> int:
    return max(1, VECTOR_JIT_MIN_EVALS // n_eval)


def _time_vector_jit_loop_required(x_eval: np.ndarray, spline: SplineCoeffs) -> Timing:
    inner_loops = _vector_jit_inner_loops(x_eval.size)
    timing = _time_required(functools.partial(_cubic_call_vector_loop_jit, x_eval, spline, inner_loops))
    if timing.seconds is None:
        msg = 'required jitted vector-loop timing unexpectedly failed'
        raise RuntimeError(msg)
    return Timing(timing.seconds / inner_loops, timing.loops * inner_loops)


def _format_time(timing: Timing) -> str:
    if timing.seconds is None:
        return 'skip'
    seconds = timing.seconds
    if seconds < 1.0e-6:
        return f'{seconds * 1.0e9:6.1f} ns'
    if seconds < 1.0e-3:
        return f'{seconds * 1.0e6:6.2f} us'
    if seconds < 1.0:
        return f'{seconds * 1.0e3:6.2f} ms'
    return f'{seconds:6.2f} s'


def _format_speedup(pyakima_timings: Sequence[Timing], alternate: Timing) -> str:
    pyakima_seconds = [timing.seconds for timing in pyakima_timings if timing.seconds is not None]
    if alternate.seconds is None or not pyakima_seconds:
        return 'skip'
    speedup = alternate.seconds / min(pyakima_seconds)
    return f'{speedup:6.2f}x'


def _print_table(title: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    print(f'\n{title}')
    if not rows:
        print('  no rows')
        return
    widths = [max(len(column), *(len(row[index]) for row in rows)) for index, column in enumerate(columns)]
    header = '  '.join(column.ljust(widths[index]) for index, column in enumerate(columns))
    rule = '  '.join('-' * width for width in widths)
    print(header)
    print(rule)
    for row in rows:
        print('  '.join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _print_availability() -> None:
    rows = (
        ('pyakima', 'available'),
        (SCIPY.label, SCIPY.status),
        (PYGSL.label, PYGSL.status),
    )
    _print_table('Optional backend availability', ('backend', 'status'), rows)


def _print_runtime_versions() -> None:
    print(f'python {sys.version.split()[0]} | numba {numba.__version__} | numpy {np.__version__}')


def _call_models(options: DemoOptions) -> tuple[ModelCase, ...]:
    if options.show_all_call_models:
        return MODELS
    return DEFAULT_CALL_MODELS


def _creation_rows() -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for model in MODELS:
        for n_control in CONTROL_POINT_LENGTHS:
            x, y = _control_points(n_control)
            class_time = _time_required(functools.partial(_pyakima_spline, x, y, model))
            helper_time = _time_required(functools.partial(_pyakima_helper, x, y, model))

            alternate_time = _time_optional(functools.partial(_alternate_spline, model, x, y))
            rows.append(
                (
                    model.label,
                    str(n_control),
                    _format_time(class_time),
                    _format_time(helper_time),
                    _format_time(alternate_time),
                    _format_speedup((class_time, helper_time), alternate_time),
                )
            )
    return rows


def _scalar_rows(options: DemoOptions) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    x_scalars = _scalar_eval_points()
    for model in _call_models(options):
        for n_control in CONTROL_POINT_LENGTHS:
            x, y = _control_points(n_control)
            py_spline = _pyakima_spline(x, y, model)
            coeffs = py_spline.spline

            def _call_scalar(coeffs: SplineCoeffs, xint: float) -> float:
                return cubic_call_scalar(xint, coeffs, EXT)

            def _call_cubic(coeffs: SplineCoeffs, xint: float) -> float:
                return cubic_call_scalar(xint, coeffs, EXT)

            scalar_time = _time_scalar_grid_required(
                functools.partial(_call_scalar, coeffs),
                x_scalars,
            )
            scalar_jit_time = _time_scalar_jit_grid_required(x_scalars, coeffs)
            pyakima_timings = [scalar_time]
            overhead_cells: tuple[str, ...] = ()
            if options.show_overhead:
                class_time = _time_scalar_grid_required(py_spline, x_scalars)
                dispatch_time = _time_scalar_grid_required(
                    functools.partial(_call_cubic, coeffs),
                    x_scalars,
                )
                pyakima_timings.extend((class_time, dispatch_time))
                overhead_cells = (_format_time(class_time), _format_time(dispatch_time))

            try:
                _, alternate = _alternate_spline(model, x, y)
            except ImportError as exc:
                alternate_time = Timing(None, reason=f'{type(exc).__name__}: {exc}')
            else:
                alternate_time = _time_scalar_grid_optional(
                    functools.partial(_call_alternate, alternate, model),
                    x_scalars,
                )

            rows.append(
                (
                    model.label,
                    str(n_control),
                    *overhead_cells,
                    _format_time(scalar_time),
                    _format_time(scalar_jit_time),
                    _format_time(alternate_time),
                    _format_speedup(pyakima_timings, alternate_time),
                    _format_speedup((scalar_jit_time,), alternate_time),
                )
            )
    return rows


def _vector_rows(options: DemoOptions) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for model in _call_models(options):
        for n_control in CONTROL_POINT_LENGTHS:
            x, y = _control_points(n_control)
            py_spline = _pyakima_spline(x, y, model)
            coeffs = py_spline.spline
            for n_eval in VECTOR_CALL_LENGTHS:
                x_eval = _eval_points(n_eval)

                vector_time = _time_required(functools.partial(cubic_call_vector, x_eval, coeffs, EXT))
                vector_linear_time = _time_required(functools.partial(cubic_call_vector_linear, x_eval, coeffs, EXT))
                vector_jit_time = _time_vector_jit_loop_required(x_eval, coeffs)
                pyakima_timings = [vector_time, vector_linear_time]
                overhead_cells: tuple[str, ...] = ()
                if options.show_overhead:
                    py_spline_linear = _pyakima_spline(x, y, model, linear_vector_calls=1)
                    class_time = _time_required(functools.partial(py_spline, x_eval))
                    class_linear_time = _time_required(functools.partial(py_spline_linear, x_eval))
                    dispatch_time = _time_required(functools.partial(cubic_call, x_eval, coeffs, EXT))
                    pyakima_timings.extend((class_time, class_linear_time, dispatch_time))
                    overhead_cells = (
                        _format_time(class_time),
                        _format_time(class_linear_time),
                        _format_time(dispatch_time),
                    )

                try:
                    _, alternate = _alternate_spline(model, x, y)
                except ImportError as exc:
                    alternate_time = Timing(None, reason=f'{type(exc).__name__}: {exc}')
                else:
                    alternate_time = _time_optional(functools.partial(_call_alternate, alternate, model, x_eval))

                rows.append(
                    (
                        model.label,
                        str(n_control),
                        str(n_eval),
                        *overhead_cells,
                        _format_time(vector_time),
                        _format_time(vector_linear_time),
                        _format_time(vector_jit_time),
                        _format_time(alternate_time),
                        _format_speedup(pyakima_timings, alternate_time),
                        _format_speedup((vector_jit_time,), alternate_time),
                    )
                )
    return rows


def main() -> None:
    """Run the timing demo."""
    options = _parse_args()
    scalar_overhead_columns = ('class', 'cubic_call') if options.show_overhead else ()
    vector_overhead_columns = ('class smart', 'class linear', 'cubic_call') if options.show_overhead else ()

    print('pyakima speed demo')
    _print_runtime_versions()
    print(f'median of {REPEATS} repeats; each repeat is adaptively looped to at least {MIN_SECONDS:.3f} s')
    print('all pyakima jitted call paths are warmed once with matching arguments')
    if not options.show_overhead:
        print('evaluation tables hide Python dispatch/class overhead; pass --show-overhead to include it')
    if not options.show_all_call_models:
        print(
            'evaluation tables use gsl-style and makima-backed scipy-style rows; '
            'pass --show-all-call-models for all models'
        )
    print(
        'scalar fn is one Python call per point; scalar jit loops over the same points inside njit '
        'and is excluded from py-call speedup'
    )
    print('fully-jitted speedup compares alternate backends with njit caller timings')

    _print_availability()
    _print_table(
        'Spline creation time per spline',
        (
            'model',
            'n_ctrl',
            'class',
            'helper',
            'alt time',
            'py speedup',
        ),
        _creation_rows(),
    )
    _print_table(
        f'Scalar call time per scalar ({SCALAR_GRID_LENGTH}-point grid average)',
        (
            'model',
            'n_ctrl',
            *scalar_overhead_columns,
            'scalar fn',
            'scalar jit',
            'alt time',
            'py-call speedup',
            'fully-jitted speedup',
        ),
        _scalar_rows(options),
    )
    _print_table(
        'Vector call time per call',
        (
            'model',
            'n_ctrl',
            'n_eval',
            *vector_overhead_columns,
            'vector fn',
            'linear fn',
            'cubic_call jit',
            'alt time',
            'py speedup',
            'fully-jitted speedup',
        ),
        _vector_rows(options),
    )


if __name__ == '__main__':
    main()
