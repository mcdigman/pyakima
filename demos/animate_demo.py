"""Render an animated GIF showcasing pyakima against a standard cubic spline.

Top panel: a control point slides up and down, contrasting the pyakima makima
fit (local, low-ringing) with SciPy's natural :class:`CubicSpline`, which swings
well above and below the moving spike. Bottom panel: an asymmetric
piecewise-linear corner whose right arm tilts each frame, zoomed in to expose
the three exported corner models -- ``non-rounded`` keeps the sharp kink while
``akima`` and ``makima`` round it by differing amounts.

Run from a source checkout (``demos/`` ships as an example, not an installed
package)::

    pip install -e '.[demos]'                 # scipy, matplotlib, pygsl_lite
    python -m demos.animate_demo              # writes assets/akima_demo.gif

Copyright 2026 Matthew C. Digman

"""

from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.interpolate import CubicSpline

from pyakima import AkimaSpline

OUT = Path(__file__).resolve().parents[1] / 'assets' / 'akima_demo.gif'

FRAMES = 40
X = np.arange(9.0)
XS = np.linspace(X[0], X[-1], 601)
SPIKE_BASE = np.array([0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0])
SPIKE = 4  # index of the oscillating spike in the top panel
CORNER_LEFT = np.array([0.0, 3.0, 6.0, 9.0])  # fixed slope-3 arm; apex at x=3

# Dark, colorblind-safe (Okabe-Ito) palette; the three curves are also keyed by
# line style so the plot reads without relying on colour.
BG = '#0d1117'
FG = '#c9d1d9'
DOT = '#f0f6fc'
MAKIMA, AKIMA, NONROUND, CUBIC = '#56b4e9', '#e69f00', '#009e73', '#8b949e'
# (label, corner_model, colour, linestyle) -- makima drawn last / thickest.
MODELS = (
    ('non-rounded', 0, NONROUND, ':'),
    ('akima', 1, AKIMA, '--'),
    ('makima', 2, MAKIMA, '-'),
)

DARK = {
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'savefig.facecolor': BG,
    'text.color': FG,
    'axes.labelcolor': FG,
    'axes.titlecolor': FG,
    'axes.edgecolor': FG,
    'xtick.color': FG,
    'ytick.color': FG,
}


def _style(ax: plt.Axes) -> None:
    # ticks on all four sides, pointing inward, long enough to read.
    ax.tick_params(which='both', direction='in', top=True, right=True, length=7, width=1.0, labelsize=11)


def _run() -> None:
    with plt.rc_context(DARK):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 6), height_ratios=(3, 2), layout='constrained')

        def frame(k: int) -> None:
            phase = np.sin(2 * np.pi * k / FRAMES)
            for ax in (ax_top, ax_bot):
                ax.clear()
                _style(ax)

            # top: makima (the recommended default) vs a natural cubic spline
            y = SPIKE_BASE.copy()
            y[SPIKE] = 4.0 + 4.0 * phase
            ax_top.plot(XS, CubicSpline(X, y, bc_type='natural')(XS), color=CUBIC, ls='--', lw=2, label='scipy cubic')
            ax_top.plot(XS, AkimaSpline(X, y, corner_model=2)(XS), color=MAKIMA, ls='-', lw=2.5, label='pyakima makima')
            ax_top.plot(X, y, 'o', color=DOT, ms=6, label='control points')
            ax_top.set(ylim=(-3, 9), ylabel='y')
            ax_top.set_title('Less ringing than a cubic spline')
            ax_top.legend(loc='upper right', fontsize=12, frameon=False)

            # bottom: three corner models on an asymmetric linear corner (right arm tilts)
            slope = -2.0 + phase  # stays in [-3, -1]: always an asymmetric downward kink
            yc = np.concatenate([CORNER_LEFT, 9.0 + slope * np.arange(1.0, 6.0)])
            for name, model, color, ls in MODELS:
                ax_bot.plot(XS, AkimaSpline(X, yc, corner_model=model)(XS), color=color, ls=ls, lw=2.2, label=name)
            ax_bot.plot(X, yc, 'o', color=DOT, ms=6)
            ax_bot.set(xlim=(1.8, 4.4), ylim=(5, 10.4), xlabel='x', ylabel='y')
            ax_bot.set_title('Corner handling at a sharp kink (zoom)')
            ax_bot.legend(loc='upper left', fontsize=12, frameon=False)

        anim = animation.FuncAnimation(fig, frame, frames=FRAMES, interval=50)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        anim.save(OUT, writer=animation.PillowWriter(fps=20), dpi=72)
        plt.close(fig)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    _run()
