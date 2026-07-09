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

from pathlib import Path
from typing import Any, cast

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from pyakima import AkimaSpline

ASSETS = Path(__file__).resolve().parents[2] / 'assets'

FRAMES = 45
FPS = 15
DPI = 72

N_POINTS = 11  # control points sampled each frame
X_C = 1.5  # concentration centre (the max-curvature shoulder of f)
WIDTH = 0.6  # concentration width
A_MAX = 8.0  # peak density boost at full concentration
EDGE_AMP = 0.06  # extra quantile swing for the plateau-edge point (index 1)
DOMAIN = (0.0, 10.0)
XF = np.linspace(*DOMAIN, 4001)  # fine grid: reference curve and the density CDF
XS = np.linspace(*DOMAIN, 800)  # dense grid: interpolant evaluation

# Same colorblind-safe (Okabe-Ito) presets as animate_demo; here the muted
# 'cubic' grey draws the reference curve, 'akima' orange the SciPy cubic, and
# 'makima' blue the pyakima fit. Keys: bg, fg, dot, makima, akima, nonround, cubic.
THEMES = {
    'light': {
        'bg': '#ffffff',
        'fg': '#1f2328',
        'dot': '#24292f',
        'makima': '#0072b2',
        'akima': '#d55e00',
        'nonround': '#009e73',
        'cubic': '#57606a',
    },
    'dark': {
        'bg': '#0d1117',
        'fg': '#c9d1d9',
        'dot': '#f0f6fc',
        'makima': '#56b4e9',
        'akima': '#e69f00',
        'nonround': '#009e73',
        'cubic': '#8b949e',
    },
}


def _rc(theme: dict[str, str]) -> dict[str, object]:
    bg, fg = theme['bg'], theme['fg']
    return {
        'figure.facecolor': bg,
        'axes.facecolor': bg,
        'savefig.facecolor': bg,
        'text.color': fg,
        'axes.labelcolor': fg,
        'axes.titlecolor': fg,
        'axes.edgecolor': fg,
        'xtick.color': fg,
        'ytick.color': fg,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    }


def _style(ax: Axes) -> None:
    # ticks on all four sides, pointing inward, long enough to read.
    ax.tick_params(which='both', direction='in', top=True, right=True, length=9, width=1.2)


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
    with plt.rc_context(cast(Any, _rc(theme))):  # noqa: TC006
        fig, ax = plt.subplots(figsize=(9, 5.6), layout='constrained')

        def frame(k: int) -> None:
            alpha = float(np.sin(np.pi * k / FRAMES) ** 2)  # 0 -> uniform, 1 -> concentrated
            edge = EDGE_AMP * float(np.sin(2 * np.pi * k / FRAMES))  # plateau-edge point sway
            xc = _control_x(alpha, edge)
            yc = _reference(xc)
            ax.clear()
            _style(ax)
            ax.plot(XF, _reference(XF), color=ref, ls='--', lw=5, alpha=0.5, label='true f(x)')
            ax.plot(XS, CubicSpline(xc, yc)(XS), color=cubic, ls='--', lw=2.5, label='scipy cubic')
            ax.plot(XS, AkimaSpline(xc, yc, corner_model=2)(XS), color=makima, ls='-', lw=3, label='pyakima makima')
            ax.plot(xc, yc, 'o', color=dot, ms=8)
            ax.set(xlim=DOMAIN, ylim=(-20, 170), xlabel='x', ylabel='f(x)')
            ax.set_title('Irregular grids: makima resists overshoot')
            ax.legend(loc='upper center', frameon=False, ncols=3)

        anim = animation.FuncAnimation(fig, frame, frames=FRAMES, interval=50)
        out = ASSETS / f'akima_grid_{name}.gif'
        anim.save(out, writer=animation.PillowWriter(fps=FPS), dpi=DPI)
        plt.close(fig)
    print(f'wrote {out}')


def _run() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        _render(name, theme)


if __name__ == '__main__':
    _run()
