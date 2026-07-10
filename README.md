# pyakima

[![DOI](https://zenodo.org/badge/972418978.svg)](https://zenodo.org/badge/latestdoi/972418978)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)<br>
[![Test](https://github.com/mcdigman/pyakima/actions/workflows/test.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/test.yml)
[![Coverage](https://github.com/mcdigman/pyakima/actions/workflows/coverage.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/coverage.yml)
[![Typed](https://github.com/mcdigman/pyakima/actions/workflows/typecheck.yml/badge.svg)](https://github.com/mcdigman/pyakima/actions/workflows/typecheck.yml)
[![Documentation Status](https://readthedocs.org/projects/pyakima/badge/?version=latest)](https://pyakima.readthedocs.io/en/latest/)


<!-- doc:intro:start -->
`pyakima` is a fast, JIT-compatible Akima spline implementation written in
pure Python.

Akima splines are a type of cubic spline that can guarantee continuous differentiability and local behavior while minimizing overshoot on both regular and irregular interpolation grids.

`pyakima` ships a small object-oriented Python API for ordinary use and
Numba-friendly helper functions for building and evaluating splines inside
fully jitted code.

The implementation is fully typed (`py.typed`) and keeps the public surface
small:
`AkimaSpline` is an object-oriented interface which simplifies calls for non-jitted Python callers; it has only a constructor and a `__call__` method.
For jitted workloads,
1. The spline interpolation itself is represented as a set of coefficients stored in a `SplineCoeffs` object, computed once via `akima_create_helper`.
2. The spline is called from `cubic_call`, which selects the appropriate coefficients and evaluates the polynomial with `spline_single_knot_eval`.
<!-- doc:intro:end -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mcdigman/pyakima/main/assets/akima_demo_dark.gif">
  <img alt="Two-panel animation. Top: as one control point slides up and down, the pyakima makima curve hugs the data while a natural cubic spline rings above and below the spike. Bottom: a zoom on a sharp kink where the non-rounded, akima, and makima corner models round the corner by differing amounts." src="https://raw.githubusercontent.com/mcdigman/pyakima/main/assets/akima_demo_light.gif">
</picture>

<!-- doc:corners:start -->
The top panel slides one control point up and down: the pyakima `makima` fit
stays local and flat on either side of the spike, while a natural cubic spline
rings above and below it. The bottom panel zooms into a sharp kink to show the
three corner models `pyakima` exports:
1. `non-rounded`: Algorithm based on <a href="#ref-1">[1]</a>, comparable numerical behavior to GSL; note the unstable behavior is because
     the algorithm is non-differentiable at corners, _not_ a peculiar limitation of this implementation <a href="#ref-2">[2]</a>.
2. `akima` ([SciPy parity](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html)) <a href="#ref-3">[3]</a>.
   Discontinuous behavior is less severe than `non-rounded`; slightly more prone to overshoot, and still has special edge-case handling.
3. `makima` [Modified Akima Algorithm](https://www.mathworks.com/help/matlab/ref/makima.html) <a href="#ref-4">[4]</a>; recommended default
   Less overshoot than `akima`, while mathematically guaranteed to preserve differentiability/continuous behavior at corners without special edge-case handling.
   Similar performance to `akima` in most cases.
<!-- doc:corners:end -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mcdigman/pyakima/main/assets/akima_grid_dark.gif">
  <img alt="Single panel animation. As the control points slide between a regular and irregular grid, the pyakima makima curve smoothly hugs the data, while the scipy default cubic spline oscillates so strongly it extends off the plotted y axis." src="https://raw.githubusercontent.com/mcdigman/pyakima/main/assets/akima_grid_light.gif">
</picture>

<!-- doc:grid:start -->
The control points oscillate smoothly between regular uniform-grid spacing and an inverse-CDF-based spacing.
Such a spacing is similar to what might be used when using Akima splines for a PSD estimation task, or in approximating a function
with sharp features with as few control points as possible. Such uses with irregular grids are a key modern application of Akima splines,
and are of central importance to their utility in gravitational-wave detection applications, such as using trans-dimensional MCMC
to adaptively fit Akima splines to un-modeled gravitational-wave sources <a href="#ref-5">[5]</a>, <a href="#ref-6">[6]</a>, <a href="#ref-7">[7]</a>.
Cubic splines, such as `scipy`'s default `CubicSpline` plotted above, oscillate wildly on the same irregularly-spaced grid, which typically makes them unsuitable for such analysis tasks.
<!-- doc:grid:end -->

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Jitted Use](#jitted-use)
- [Spline Options](#spline-options)
- [Regenerating the Demo](#regenerating-the-demo)
- [Performance Snapshot](#performance-snapshot)
- [Quality Gates](#quality-gates)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

<!-- doc:install:start -->
## Installation

```bash
pip install pyakima
```

`pyakima` requires Python 3.10 or newer, NumPy, and Numba. The optional demo
dependencies are available with:

```bash
pip install "pyakima[demos]"
```

Note that `pygsl_lite` from the `demos` dependencies may not always succeed on a `pip install`; any demo that uses it recovers gracefully if it cannot be imported.
`pyakima.demos.speed_demo` can recover if _none_ of the `demos` dependencies are present.
`pyakima.demos.step_demo` runs if only matplotlib is present.
<!-- doc:install:end -->

<!-- doc:quickstart:start -->
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
<!-- doc:quickstart:end -->

<!-- doc:jitted:start -->
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
<!-- doc:jitted:end -->

<!-- doc:options:start -->
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
<!-- doc:options:end -->

<!-- doc:regen:start -->
## Regenerating the Demo

`pyakima.demos` ships as an example subpackage; run it from a source checkout so
it can write the README assets:

```bash
pip install -e '.[demos]'                  # scipy, matplotlib, pygsl_lite
python -m pyakima.demos.animate_demo       # writes assets/akima_demo_{light,dark}.gif
python -m pyakima.demos.animate_grid_demo  # writes assets/akima_grid_{light,dark}.gif
python -m pyakima.demos.step_demo          # writes assets/akima_step_{light,dark}.png
```
<!-- doc:regen:end -->

## Performance Snapshot
<!-- doc:perf:start -->

Run `python -m pyakima.demos.speed_demo` to compare `pyakima` with the optional
SciPy and `pygsl_lite` backends available in your environment.

The current release-candidate snapshot was measured on a single Apple Silicon M3 core with
Python 3.14.6, Numba 0.66.0, NumPy 2.4.6, SciPy 1.17.1, and `pygsl_lite`
0.1.8. The demo used 50 repeats, with each repeat adaptively looped to at least
0.100 s, and a representative range of spline and caller sizes.
The full benchmark is available in
[`docs/benchmarks/m3_0_1_0_speeds.txt`](https://github.com/mcdigman/pyakima/blob/main/docs/benchmarks/m3_0_1_0_speeds.txt).

Highlights from that run:

- `pyakima` was minimum 1.7x faster than SciPy `Akima1DInterpolator` across all benchmarks.
- Spline creation was about 5.7-32.6x faster than SciPy.
- With Python-call overhead, scalar evaluation was about 2.3x faster than SciPy but 0.3-0.4x slower than `pygsl_lite`. When called fully jitted (no Python-call overhead),
scalar evaluation was about 109-361x faster than SciPy and 18-52x faster than `pygsl_lite`.
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
<!-- doc:perf:end -->

## Quality Gates

The CI suite checks the package with:

- `pytest` unit test suite; pull requests to `dev` only test modern versions,
   while pull requests targeting `main` run for all supported versions.
- `coverage.py` branch coverage; pull requests targeting `main`
  are gated at 100% total coverage, while other targets use the development
  threshold in the coverage workflow.
- strict `mypy`, plus Pyrefly and Pyright type checking.
- Ruff with `select = ["ALL"]` and `ruff format`, run through `prek` pre-commit hooks.
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

## Contributing
Thank you for your interest in helping improve the repository!
Feature requests and bug reports can be made through [GitHub Issues](https://github.com/mcdigman/pyakima/issues). Bug reports should be accompanied by a minimal reproducing example and description of the desired behavior.
Suggested contributions or fixes should be made through pull requests to `dev`.
All new code must be fully type annotated, and pass the static type-checking and linting rules; to reduce churn, ensure, at minimum, `prek run --all-files` passes before attempting to commit.
New core/utility functions should have full `numpy`-style docstrings (verify with `pydoclint --style=numpy`), and full unit test coverage, verified with:
```
NUMBA_DISABLE_JIT=1 python -m coverage run --branch --source=pyakima --omit='*/demos/*' -m pytest
python -m coverage report -m --fail-under=100
```


## License

`pyakima` is distributed under the Apache License 2.0. See
[`LICENSE`](https://github.com/mcdigman/pyakima/blob/main/LICENSE) for the
full license text.


<!-- doc:footnotes:start -->
## References

1. <a id="ref-1"></a>G. Engeln-Müllges & F. Uhlig, *Numerical Algorithms with C*, Springer, 1996, ch. 13 "Akima and Renner Subsplines," Algorithm 13.1. ISBN 978-3-642-64682-9.
2. <a id="ref-2"></a>Note: the internals of the GSL implementation have never been viewed by the repository author. However, calls to GSL exhibit near-identical behavior, supporting that the issue is algorithmic rather than due to implementation error.
3. <a id="ref-3"></a>Akima, Hiroshi. "A new method of interpolation and smooth curve fitting based on local procedures." Journal of the ACM (JACM) , 17.4, 1970, pp. 589–602.
4. <a id="ref-4"></a>C. Moler, [*Makima Piecewise Cubic Interpolation*](https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/),
   Cleve's Corner (MathWorks blog), 2019.
5. <a id="ref-5"></a>Detecting gravitational wave signals using a flexible model for the amplitude and frequency evolution. T Gupta, NJ Cornish. Physical Review D, 2024•APS [arXiv:2404.11719](https://arxiv.org/abs/2404.11719).
6. <a id="ref-6"></a>Model-agnostic gravitational-wave background characterization algorithm. T Knapp, PM Meyers, AI Renzini.Physical Review D, 2025•APS. [arXiv:2507.08095](https://arxiv.org/abs/2507.08095)
7. <a id="ref-7"></a>“Precise analysis of gravitational waves from binary neutron star coalescence using Hilbert–Huang transform based on Akima spline interpolation.” Yoda, Itsuki et al. Progress of Theoretical and Experimental Physics (2023). [DOI:10.1093/ptep/ptad101](https://doi.org/10.1093/ptep%2Fptad101)
<!-- doc:footnotes:end -->
