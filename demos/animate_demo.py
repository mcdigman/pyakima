"""Render an animated GIF showcasing pyakima against standard splines.

Top panel: a control point slides up and down, contrasting the pyakima Akima
fit (local, overshoot-free) with SciPy's natural :class:`CubicSpline`, which
rings above and below the moving spike. Bottom panel: an asymmetric
piecewise-linear corner whose right arm tilts each frame, zoomed in to expose
the three exported corner models -- ``non-rounded`` keeps the sharp kink while
``akima`` and ``makima`` round it by differing amounts.

Run with the local mamba env::

    mamba run -n DTMCMC-dev python -m demos.animate_demo

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
MODELS = (('non-rounded', 0, 'C1'), ('akima', 1, 'C0'), ('makima', 2, 'C2'))


def _run() -> None:
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 6), height_ratios=(3, 2))

    def frame(k: int) -> None:
        phase = np.sin(2 * np.pi * k / FRAMES)
        for ax in (ax_top, ax_bot):
            ax.clear()

        # top: overshoot vs a standard cubic spline
        y = SPIKE_BASE.copy()
        y[SPIKE] = 4.0 + 4.0 * phase
        ax_top.plot(XS, CubicSpline(X, y)(XS), color='0.6', lw=2, label='scipy CubicSpline')
        ax_top.plot(XS, AkimaSpline(X, y, corner_model=1)(XS), color='C0', lw=2, label='pyakima akima')
        ax_top.plot(X, y, 'ko', ms=6, label='control points')
        ax_top.set(ylim=(-3, 9), title='Akima stays local — no overshoot', ylabel='y')
        ax_top.legend(loc='upper right', fontsize=8)

        # bottom: three corner models on an asymmetric linear corner (right arm tilts)
        slope = -2.0 + phase  # stays in [-3, -1]: always an asymmetric downward kink
        yc = np.concatenate([CORNER_LEFT, 9.0 + slope * np.arange(1.0, 6.0)])
        for name, model, color in MODELS:
            ax_bot.plot(XS, AkimaSpline(X, yc, corner_model=model)(XS), color=color, lw=2, label=name)
        ax_bot.plot(X, yc, 'ko', ms=6)
        ax_bot.set(xlim=(1.8, 4.4), ylim=(5, 10.2), xlabel='x', ylabel='y')
        ax_bot.set_title('Corner models at a sharp kink (zoom)')
        ax_bot.legend(loc='lower left', fontsize=8)

    fig.tight_layout()
    anim = animation.FuncAnimation(fig, frame, frames=FRAMES, interval=50)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=animation.PillowWriter(fps=20), dpi=72)
    plt.close(fig)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    _run()
