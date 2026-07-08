"""Demonstrate pyakima timing against optional alternate implementations."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyakima import (  # noqa: E402
    AkimaSpline,
    akima_create_helper,
    cubic_call,
    cubic_call_scalar,
    cubic_call_vector,
    cubic_call_vector_linear,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import ModuleType

CONTROL_POINT_LENGTHS = (5, 64, 4096)
VECTOR_CALL_LENGTHS = (8, 10000, 100000)
EXT = 3
MIN_SECONDS = 0.01
REPEATS = 3


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

    @property
    def available(self) -> bool:
        """Whether the optional backend imported successfully."""
        return self.module is not None


@dataclass(frozen=True)
class Timing:
    """Measured per-operation time or a reason the timing was skipped."""

    seconds: float | None
    loops: int = 0
    reason: str = ''


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


def _optional_import(module_name: str, label: str) -> OptionalModule:
    try:
        module = import_module(module_name)
    except ImportError as exc:
        return OptionalModule(label, None, f'skipped ({exc})')
    version = getattr(module, '__version__', None)
    if version is None and '.' in module_name:
        package = import_module(module_name.split('.', maxsplit=1)[0])
        version = getattr(package, '__version__', None)
    detail = f'available {version}' if version else 'available'
    return OptionalModule(label, module, detail)


SCIPY = _optional_import('scipy.interpolate', 'scipy')
PYGSL = _optional_import('pygsl_lite.spline', 'pygsl_lite')


def _control_points(n_control: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 10.0, n_control, dtype=np.float64)
    y = np.sin(2.0 * np.pi * x / 10.0) + 0.05 * np.cos(3.0 * x)
    return x, y


def _eval_points(n_eval: int) -> np.ndarray:
    return np.linspace(0.0, 10.0, n_eval, dtype=np.float64)


def _scalar_eval_point() -> float:
    return 4.321


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
        x.copy(),
        y.copy(),
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


def _call_gsl_scalar(spline: object, xint: float) -> object:
    if hasattr(spline, 'eval'):
        return spline.eval(xint)
    if hasattr(spline, 'eval_e'):
        return spline.eval_e(xint)
    msg = 'pygsl_lite spline has no scalar eval method'
    raise AttributeError(msg)


def _call_gsl_vector(spline: object, xint: np.ndarray) -> object:
    if not hasattr(spline, 'eval_vector'):
        msg = 'pygsl_lite spline has no eval_vector method'
        raise AttributeError(msg)
    return spline.eval_vector(xint)


def _call_alternate(spline: object, model: ModelCase, xint: float | np.ndarray) -> object:
    if model.alternate == 'gsl':
        if isinstance(xint, np.ndarray):
            return _call_gsl_vector(spline, xint)
        return _call_gsl_scalar(spline, xint)
    return spline(xint)


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
    except Exception as exc:  # noqa: BLE001 - optional backends should skip cleanly.
        return Timing(None, reason=f'{type(exc).__name__}: {exc}')


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


def _creation_rows() -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for model in MODELS:
        for n_control in CONTROL_POINT_LENGTHS:
            x, y = _control_points(n_control)
            class_time = _time_required(lambda x=x, y=y, model=model: _pyakima_spline(x, y, model))
            helper_time = _time_required(lambda x=x, y=y, model=model: _pyakima_helper(x, y, model))

            alternate_name = model.alternate
            alternate_time = _time_optional(lambda model=model, x=x, y=y: _alternate_spline(model, x, y)[1])
            rows.append(
                (
                    model.label,
                    str(n_control),
                    _format_time(class_time),
                    _format_time(helper_time),
                    alternate_name,
                    _format_time(alternate_time),
                    _format_speedup((class_time, helper_time), alternate_time),
                )
            )
    return rows


def _scalar_rows() -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    x_scalar = _scalar_eval_point()
    for model in MODELS:
        for n_control in CONTROL_POINT_LENGTHS:
            x, y = _control_points(n_control)
            py_spline = _pyakima_spline(x, y, model)
            coeffs = py_spline.spline

            class_time = _time_required(lambda py_spline=py_spline: py_spline(x_scalar))
            dispatch_time = _time_required(lambda coeffs=coeffs: cubic_call(x_scalar, coeffs, EXT))
            scalar_time = _time_required(lambda coeffs=coeffs: cubic_call_scalar(x_scalar, coeffs, EXT))

            try:
                alternate_name, alternate = _alternate_spline(model, x, y)
            except Exception as exc:  # noqa: BLE001 - optional backends should skip cleanly.
                alternate_name = model.alternate
                alternate_time = Timing(None, reason=f'{type(exc).__name__}: {exc}')
            else:
                alternate_time = _time_optional(
                    lambda alternate=alternate, model=model: _call_alternate(
                        alternate,
                        model,
                        x_scalar,
                    )
                )

            rows.append(
                (
                    model.label,
                    str(n_control),
                    _format_time(class_time),
                    _format_time(dispatch_time),
                    _format_time(scalar_time),
                    alternate_name,
                    _format_time(alternate_time),
                    _format_speedup((class_time, dispatch_time, scalar_time), alternate_time),
                )
            )
    return rows


def _vector_rows() -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for model in MODELS:
        for n_control in CONTROL_POINT_LENGTHS:
            x, y = _control_points(n_control)
            py_spline = _pyakima_spline(x, y, model)
            py_spline_linear = _pyakima_spline(x, y, model, linear_vector_calls=1)
            coeffs = py_spline.spline
            for n_eval in VECTOR_CALL_LENGTHS:
                x_eval = _eval_points(n_eval)

                class_time = _time_required(lambda py_spline=py_spline, x_eval=x_eval: py_spline(x_eval))
                class_linear_time = _time_required(
                    lambda py_spline_linear=py_spline_linear, x_eval=x_eval: py_spline_linear(x_eval)
                )
                dispatch_time = _time_required(
                    lambda x_eval=x_eval, coeffs=coeffs: cubic_call(
                        x_eval,
                        coeffs,
                        EXT,
                    )
                )
                vector_time = _time_required(
                    lambda x_eval=x_eval, coeffs=coeffs: cubic_call_vector(
                        x_eval,
                        coeffs,
                        EXT,
                    )
                )
                vector_linear_time = _time_required(
                    lambda x_eval=x_eval, coeffs=coeffs: cubic_call_vector_linear(
                        x_eval,
                        coeffs,
                        EXT,
                    )
                )

                try:
                    alternate_name, alternate = _alternate_spline(model, x, y)
                except Exception as exc:  # noqa: BLE001 - optional backends should skip cleanly.
                    alternate_name = model.alternate
                    alternate_time = Timing(None, reason=f'{type(exc).__name__}: {exc}')
                else:
                    alternate_time = _time_optional(
                        lambda alternate=alternate, model=model, x_eval=x_eval: _call_alternate(
                            alternate, model, x_eval
                        )
                    )

                pyakima_timings = (
                    class_time,
                    class_linear_time,
                    dispatch_time,
                    vector_time,
                    vector_linear_time,
                )
                rows.append(
                    (
                        model.label,
                        str(n_control),
                        str(n_eval),
                        _format_time(class_time),
                        _format_time(class_linear_time),
                        _format_time(dispatch_time),
                        _format_time(vector_time),
                        _format_time(vector_linear_time),
                        alternate_name,
                        _format_time(alternate_time),
                        _format_speedup(pyakima_timings, alternate_time),
                    )
                )
    return rows


def main() -> None:
    """Run the timing demo."""
    print('pyakima speed demo')
    print(f'median of {REPEATS} repeats; each repeat is adaptively looped to at least {MIN_SECONDS:.3f} s')
    print('all pyakima jitted call paths are warmed once with matching arguments')

    _print_availability()
    _print_table(
        'Spline creation time per spline',
        (
            'model',
            'n_ctrl',
            'class',
            'helper+copy',
            'alt',
            'alt time',
            'best py speedup',
        ),
        _creation_rows(),
    )
    _print_table(
        'Scalar call time per call',
        (
            'model',
            'n_ctrl',
            'class',
            'cubic_call',
            'scalar fn',
            'alt',
            'alt time',
            'best py speedup',
        ),
        _scalar_rows(),
    )
    _print_table(
        'Vector call time per call',
        (
            'model',
            'n_ctrl',
            'n_eval',
            'class smart',
            'class linear',
            'cubic_call',
            'vector fn',
            'linear fn',
            'alt',
            'alt time',
            'best py speedup',
        ),
        _vector_rows(),
    )


if __name__ == '__main__':
    main()
