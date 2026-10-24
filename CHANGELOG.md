# Changelog

Notable changes to `pyakima`. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## Unreleased

### Security

- Require Pillow 13.0.0 or newer for the Matplotlib demos to fix
  CVE-2026-96639.

### Changed

- Drop Python 3.10 support; require Python 3.11 or newer, NumPy 1.23.5 or
  newer, and Numba 0.57 or newer.

## [0.1.0] - 2026-07-10

First public release.

### Added

- Akima spline interpolation with three corner models: `non-rounded`
  (Wodicka/GSL-style), `akima` (close to SciPy `method="akima"`), and `makima`
  (close to SciPy `method="makima"`; the default).
- `AkimaSpline` object-oriented interface for ordinary Python callers, with
  SciPy-like `ext` boundary handling (extrapolate, zero, boundary value, or NaN).
- Numba-friendly helpers -- `make_akima_coeffs`, `cubic_call`,
  `cubic_call_scalar`, `cubic_call_vector`, and `cubic_call_vector_linear` --
  for building and evaluating splines inside fully jitted code.
- Full type annotations (`py.typed`) and numpy-style docstrings.
- Sphinx documentation site, runnable demo subpackage (`pyakima.demos`), and a
  cross-backend speed benchmark.

[0.1.0]: https://github.com/mcdigman/pyakima/releases/tag/v0.1.0
