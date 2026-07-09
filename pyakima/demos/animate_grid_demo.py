"""Render light and dark animated GIFs showing pyakima interpolation on irregular grids.

A single panel draws the reference function ``f(x) = 100 exp(-x**8 / 10) + x**2``
as a thick, faded dashed curve. Each frame resamples it on ``N_POINTS`` control
points whose grid morphs between uniform and concentrated at the sharp shoulder
near ``x = 1.5``, then rebuilds a pyakima makima fit and a SciPy cubic spline.
On the uniform grid the cubic rings and overshoots across the under-sampled
shoulder while makima stays close to the reference; concentrating the points
there rescues both interpolants.

The control-point spacing is the inverse CDF of a uniform-plus-Gaussian density
whose bump height ``alpha`` oscillates as ``sin(pi k / FRAMES) ** 2`` -- alpha=0
gives a uniform grid, alpha=1 the fully concentrated grid. Placing points at
uniform quantiles of a positive density keeps them monotone and pins the
endpoints every frame.

Two variants are written, ``akima_grid_light.gif`` and ``akima_grid_dark.gif``,
so the README can serve each via a theme-aware ``<picture>`` element.

Run from a source checkout::

    pip install -e '.[demos]'                      # scipy, matplotlib, pygsl_lite
    python -m pyakima.demos.animate_grid_demo      # writes assets/akima_grid_*.gif

Copyright 2026 Matthew C. Digman

"""

from typing import Any, cast

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from pyakima import AkimaSpline
from pyakima.demos._theme import ASSETS, FRAMES, rc_params, run_all, save_animation, style_axes

N_POINTS = 11  # control points sampled each frame
X_C = 1.5  # concentration centre (the max-curvature shoulder of f)
WIDTH = 0.6  # concentration width
A_MAX = 8.0  # peak density boost at full concentration
EDGE_AMP = 0.06  # extra quantile swing for the plateau-edge point (index 1)
DOMAIN = (0.0, 10.0)
XF = np.linspace(*DOMAIN, 4001)  # fine grid: reference curve and the density CDF
XS = np.linspace(*DOMAIN, 800)  # dense grid: interpolant evaluation

# The 'cubic' grey preset draws the reference curve, 'akima' orange the SciPy
# cubic, and 'makima' blue the pyakima fit.


def _reference(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(100.0 * np.exp(-(x**8) / 10.0) + x**2, dtype=np.float64)


def _control_x(alpha: float, edge_shift: float = 0.0) -> NDArray[np.float64]:
    # Points at uniform quantiles of a uniform + Gaussian-bump density: alpha=0
    # is uniform, larger alpha concentrates points around X_C. Monotone by
    # construction, with the endpoints pinned at the domain edges. edge_shift
    # nudges the second point's quantile so it swings on and off the plateau;
    # |edge_shift| < 1 / (N_POINTS - 1) keeps it between its neighbours.
    dens = 1.0 + alpha * A_MAX * np.exp(-(((XF - X_C) / WIDTH) ** 2))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(XF))])
    cdf /= cdf[-1]
    quantiles = np.linspace(0.0, 1.0, N_POINTS)
    quantiles[1] += edge_shift
    return np.asarray(np.interp(quantiles, cdf, XF), dtype=np.float64)


def _render(name: str, theme: dict[str, str]) -> None:
    ref, cubic, makima, dot = theme['cubic'], theme['akima'], theme['makima'], theme['dot']
    # rc keys are dynamic strings, not the RcParams key Literal, so cast past the check.
    with plt.rc_context(cast(Any, rc_params(theme))):  # noqa: TC006
        fig, ax = plt.subplots(figsize=(9, 5.6), layout='constrained')

        def frame(k: int) -> None:
            alpha = float(np.sin(np.pi * k / FRAMES) ** 2)  # 0 -> uniform, 1 -> concentrated
            edge = EDGE_AMP * float(np.sin(2 * np.pi * k / FRAMES))  # plateau-edge point sway
            xc = _control_x(alpha, edge)
            yc = _reference(xc)
            ax.clear()
            style_axes(ax)
            ax.plot(XF, _reference(XF), color=ref, ls='--', lw=5, alpha=0.5, label='true f(x)')
            ax.plot(XS, CubicSpline(xc, yc)(XS), color=cubic, ls='--', lw=2.5, label='scipy cubic')
            ax.plot(XS, AkimaSpline(xc, yc, corner_model=2)(XS), color=makima, ls='-', lw=3, label='pyakima makima')
            ax.plot(xc, yc, 'o', color=dot, ms=8)
            ax.set(xlim=DOMAIN, ylim=(-20, 170), xlabel='x', ylabel='f(x)')
            ax.set_title('Irregular grids: makima resists overshoot')
            ax.legend(loc='upper center', frameon=False, ncols=3, fontsize=16)

        save_animation(fig, frame, ASSETS / f'akima_grid_{name}.gif')


if __name__ == '__main__':
    run_all(_render)
