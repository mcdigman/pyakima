# pyakima

[![Build](https://github.com/mcdigman/pyakima/actions/workflows/build.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/build.yml)
[![Test](https://github.com/mcdigman/pyakima/actions/workflows/test.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/test.yml)
[![Coverage](https://github.com/mcdigman/pyakima/actions/workflows/coverage.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/coverage.yml)
[![Type Check](https://github.com/mcdigman/pyakima/actions/workflows/typecheck.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/typecheck.yml)
[![Lint](https://github.com/mcdigman/pyakima/actions/workflows/lint.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/lint.yml)
[![Dead Code](https://github.com/mcdigman/pyakima/actions/workflows/deadcode.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/deadcode.yml)
[![Docstrings](https://github.com/mcdigman/pyakima/actions/workflows/docstrings.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/docstrings.yml)
[![Pylint](https://github.com/mcdigman/pyakima/actions/workflows/pylint.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/pylint.yml)

`pyakima` is a fast, JIT-compatible Akima spline implementation written in
pure Python. It ships a small object-oriented Python API for ordinary use and
Numba-friendly helper functions for building and evaluating splines inside
fully jitted code.

The implementation is fully typed (`py.typed`) and keeps the public surface
small: `AkimaSpline` for normal Python callers, `SplineCoeffs` for stored
coefficients, and `akima_create_helper`/`cubic_call*` helpers for jitted
workloads.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/akima_demo_dark.gif">
  <img alt="Two-panel animation. Top: as one control point slides up and down, the pyakima makima curve hugs the data while a natural cubic spline rings above and below the spike. Bottom: a zoom on a sharp kink where the non-rounded, akima, and makima corner models round the corner by differing amounts." src="assets/akima_demo_light.gif">
</picture>

The top panel slides one control point up and down: the pyakima `makima` fit
stays local and flat on either side of the spike, while a natural cubic spline
rings above and below it. The bottom panel zooms into a sharp kink to show the
three corner models `pyakima` exports — `non-rounded` (GSL), `akima` (SciPy),
and `makima` — which round the corner by differing amounts.

## Installation

```bash
pip install pyakima
```

`pyakima` requires Python 3.10 or newer, NumPy, and Numba. The optional demo
dependencies are available with:

```bash
pip install "pyakima[demos]"
```

## Quick Start

```python
import numpy as np

from pyakima import AkimaSpline

x = np.linspace(0.0, 10.0, 16)
y = np.sin(x)

spline = AkimaSpline(x, y, corner_model="makima", ext=3)

print(spline(2.5))
print(spline(np.linspace(-1.0, 11.0, 1000)))
```

`AkimaSpline` is the ergonomic Python interface. The class stores a compiled
coefficient bundle and dispatches scalar or vector evaluations to the fast
helper functions.

## Jitted Use

Use `akima_create_helper` and `cubic_call` when the spline should be created or
evaluated from inside an `njit` function.

```python
import numpy as np
from numba import njit

from pyakima import akima_create_helper, cubic_call


@njit
def build_and_evaluate(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    coeffs = akima_create_helper(x, y, corner_model=2)
    return cubic_call(x_eval, coeffs, 0)
```

For lower-level control, call `cubic_call_scalar`,
`cubic_call_vector`, or `cubic_call_vector_linear` directly.

## Spline Options

`pyakima` supports several Akima corner models:

| `AkimaSpline` option | helper option | Behavior |
| --- | ---: | --- |
| `"non-rounded"` | `0` | Wodicka/GSL-style non-rounded sharp corners. |
| `"akima"` | `1` | Classic Akima behavior, close to `scipy` `method="akima"`. |
| `"makima"` | `2` | Modified Akima weights, close to `scipy` `method="makima"`. |

Boundary handling uses SciPy-like `ext` values:

| `ext` | Out-of-bounds behavior |
| ---: | --- |
| `0` | Extrapolate. |
| `1` | Return zero. |
| `3` | Return the nearest boundary value. |
| `4` | Return `nan`. |

`ext=2` (raise on out-of-bounds) is not implemented. `ext=4` is added for
NaN boundary handling. `AkimaSpline` defaults to `ext=3`.

## Regenerating the Demo

`pyakima.demos` ships as an example subpackage; run it from a source checkout so
it can write the README assets:

```bash
pip install -e '.[demos]'               # scipy, matplotlib, pygsl_lite
python -m pyakima.demos.animate_demo    # writes assets/akima_demo_{light,dark}.gif
```

## Performance Snapshot

Run `python -m pyakima.demos.speed_demo` to compare `pyakima` with the optional
SciPy and `pygsl_lite` backends available in your environment.

The current release-candidate snapshot was measured on a single M3 core with
Python 3.14.6, Numba 0.66.0, NumPy 2.4.6, SciPy 1.17.1, and `pygsl_lite`
0.1.8. The demo used 50 repeats with each repeat adaptively looped to at least
0.100 s.

Highlights from that run:

- SciPy-style spline creation was about 5.7-32.6x faster than SciPy's
  `Akima1DInterpolator`.
- Python-call scalar evaluation was about 2.3x faster than SciPy, while the
  fully jitted scalar path was about 109-361x faster than SciPy and 18-52x
  faster than `pygsl_lite`.
- Python-call vector evaluation was faster than SciPy in every tested case
  (about 1.7-4.8x in the SciPy-style rows).
- Against `pygsl_lite`, Python-call vector evaluation was faster once the call
  did enough work (for example, 1,000 or more evaluation points in the sampled
  cases), while scalar and tiny-vector cases can be dominated by dispatch
  overhead.
- Fully jitted vector evaluation was faster than SciPy in every tested case
  (about 1.7-29.0x). It was also faster than `pygsl_lite` for most non-tiny
  vector workloads in the sample (about 2.3-11.2x for 1,000 or more evaluation
  points).

Benchmark results depend on hardware, Python/NumPy/Numba versions, and whether
the call is made through Python or entirely inside jitted code.

## Quality Gates

The CI suite checks the package with:

- `pytest`, plus `coverage.py` branch coverage; pull requests targeting `main`
  are gated at 100% total coverage, while other targets use the development
  threshold in the coverage workflow.
- strict `mypy`, plus Pyrefly and Pyright type checking.
- Ruff with `select = ["ALL"]`, run through `prek` pre-commit hooks.
- Skylos dead-code detection.
- Pylint and pydoclint docstring/signature checks.
- source distribution and wheel builds, including install/import checks across
  the supported dependency range.

Useful local checks:

```bash
python -m pytest
NUMBA_DISABLE_JIT=1 python -m coverage run --branch --source=pyakima --omit='*/demos/*' -m pytest
python -m coverage report -m --fail-under=100
uvx prek run --all-files --show-diff-on-failure --color=always
uvx skylos pyakima tests
```

## License

`pyakima` is distributed under the Apache License 2.0. See `LICENSE` for the
full license text.
