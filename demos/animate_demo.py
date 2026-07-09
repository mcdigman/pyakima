"""Render light and dark animated GIFs showcasing pyakima against a cubic spline.

Top panel: a control point slides up and down, contrasting the pyakima makima
fit (local, low-ringing) with SciPy's natural :class:`CubicSpline`, which swings
well above and below the moving spike. Bottom panel: an asymmetric
piecewise-linear corner whose right arm tilts each frame, zoomed in to expose
the three exported corner models -- ``non-rounded`` keeps the sharp kink while
``akima`` and ``makima`` round it by differing amounts.

Two variants are written, ``akima_demo_light.gif`` and ``akima_demo_dark.gif``,
so the README can serve each via a theme-aware ``<picture>`` element.

Run from a source checkout (``demos/`` ships as an example, not an installed
package)::

    pip install -e '.[demos]'                 # scipy, matplotlib, pygsl_lite
    python -m demos.animate_demo              # writes assets/akima_demo_*.gif

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
from scipy.interpolate import CubicSpline

from pyakima import AkimaSpline

ASSETS = Path(__file__).resolve().parents[1] / 'assets'

FRAMES = 45
FPS = 15
DPI = 72
X = np.arange(9.0)
XS = np.linspace(X[0], X[-1], 601)
SPIKE_BASE = np.array([0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0])
SPIKE = 4  # index of the oscillating spike in the top panel
CORNER_LEFT = np.array([0.0, 3.0, 6.0, 9.0])  # fixed slope-3 arm; apex at x=3

# The three corner curves are keyed by line style so the plot reads without
# relying on colour; each theme supplies a colorblind-safe (Okabe-Ito) palette
# tuned for its background. Keys: bg, fg, dot, makima, akima, nonround, cubic.
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
        'legend.fontsize': 18,
    }


def _style(ax: Axes) -> None:
    # ticks on all four sides, pointing inward, long enough to read.
    ax.tick_params(which='both', direction='in', top=True, right=True, length=9, width=1.2)


def _render(name: str, theme: dict[str, str]) -> None:
    # (label, corner_model, colour, linestyle) -- makima drawn last / thickest.
    models = (
        ('non-rounded', 0, theme['nonround'], ':'),
        ('akima', 1, theme['akima'], '--'),
        ('makima', 2, theme['makima'], '-'),
    )
    with plt.rc_context(cast('Any', _rc(theme))):  # keys are dynamic strings, not the RcParams Literal
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7.7), height_ratios=(3, 2), layout='constrained')

        cubic, makima, dot = theme['cubic'], theme['makima'], theme['dot']

        def frame(k: int) -> None:
            phase = np.sin(2 * np.pi * k / FRAMES)
            for ax in (ax_top, ax_bot):
                ax.clear()
                _style(ax)

            # top: makima (the recommended default) vs a natural cubic spline
            y = SPIKE_BASE.copy()
            y[SPIKE] = 4.0 + 4.0 * phase
            ax_top.plot(XS, CubicSpline(X, y, bc_type='natural')(XS), color=cubic, ls='--', lw=4, label='scipy cubic')
            ax_top.plot(XS, AkimaSpline(X, y, corner_model=2)(XS), color=makima, ls='-', lw=5, label='pyakima makima')
            ax_top.plot(X, y, 'o', color=dot, ms=8, label='control points')
            ax_top.set(ylim=(-3, 9), ylabel='y')
            ax_top.set_title('Less ringing than a cubic spline')
            ax_top.legend(loc='upper right', frameon=False)

            # bottom: three corner models on an asymmetric linear corner (right arm tilts)
            slope = -2.0 + phase  # stays in [-3, -1]: always an asymmetric downward kink
            yc = np.concatenate([CORNER_LEFT, 9.0 + slope * np.arange(1.0, 6.0)])
            for label, model, color, ls in models:
                ax_bot.plot(XS, AkimaSpline(X, yc, corner_model=model)(XS), color=color, ls=ls, lw=4.4, label=label)
            ax_bot.plot(X, yc, 'o', color=dot, ms=8)
            ax_bot.set(xlim=(1.8, 4.4), ylim=(5, 10.4), xlabel='x', ylabel='y')
            ax_bot.set_title('Corner handling at a sharp kink (zoom)')
            ax_bot.legend(loc='upper left', frameon=False)

        anim = animation.FuncAnimation(fig, frame, frames=FRAMES, interval=50)
        out = ASSETS / f'akima_demo_{name}.gif'
        anim.save(out, writer=animation.PillowWriter(fps=FPS), dpi=DPI)
        plt.close(fig)
    print(f'wrote {out}')


def _run() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        _render(name, theme)


if __name__ == '__main__':
    _run()
