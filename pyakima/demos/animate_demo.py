"""Render light and dark animated GIFs showcasing pyakima against a cubic spline.

Top panel: a control point slides up and down, contrasting the pyakima makima
fit (local, low-ringing) with SciPy's natural :class:`CubicSpline`, which swings
well above and below the moving spike. Bottom panel: an asymmetric
piecewise-linear corner whose right arm tilts each frame, zoomed in to expose
the three exported corner models -- ``non-rounded`` keeps the sharp kink while
``akima`` and ``makima`` round it by differing amounts.

Two variants are written, ``akima_demo_light.gif`` and ``akima_demo_dark.gif``,
so the README can serve each via a theme-aware ``<picture>`` element.

Run from a source checkout::

    pip install -e '.[demos]'                 # scipy, matplotlib, pygsl_lite
    python -m pyakima.demos.animate_demo      # writes assets/akima_demo_*.gif

Copyright 2026 Matthew C. Digman

"""

from typing import Any, cast

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from pyakima import AkimaSpline
from pyakima.demos._theme import ASSETS, FRAMES, rc_params, run_all, save_animation, style_axes

X = np.arange(9.0)
XS = np.linspace(X[0], X[-1], 601)
SPIKE_BASE = np.array([0.0, 0.0, 0.0, 2.0, 5.0, 2.0, 0.0, 0.0, 0.0])
SPIKE = 4  # index of the oscillating spike in the top panel
CORNER_LEFT = np.array([0.0, 3.0, 6.0, 9.0])  # fixed slope-3 arm; apex at x=3

# The three corner curves are keyed by line style so the plot reads without
# relying on colour, over the shared Okabe-Ito palette (see _theme.THEMES).


def _render(name: str, theme: dict[str, str]) -> None:
    # (label, corner_model, colour, linestyle) -- makima drawn last / thickest.
    models = (
        ('non-rounded', 0, theme['nonround'], ':'),
        ('akima', 1, theme['akima'], '--'),
        ('makima', 2, theme['makima'], '-'),
    )
    # rc keys are dynamic strings, not the RcParams key Literal, so cast past the check.
    with plt.rc_context(cast(Any, rc_params(theme))):  # noqa: TC006
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7.7), height_ratios=(3, 2), layout='constrained')

        cubic, makima, dot = theme['cubic'], theme['makima'], theme['dot']

        def frame(k: int) -> None:
            phase = np.sin(2 * np.pi * k / FRAMES)
            for ax in (ax_top, ax_bot):
                ax.clear()
                style_axes(ax)

            # top: makima (the recommended default) vs a natural cubic spline
            y = SPIKE_BASE.copy()
            y[SPIKE] = 4.0 + 4.0 * phase
            ax_top.plot(XS, CubicSpline(X, y, bc_type='natural')(XS), color=cubic, ls='--', lw=4, label='scipy cubic')
            ax_top.plot(XS, AkimaSpline(X, y, corner_model=2)(XS), color=makima, ls='-', lw=5, label='pyakima makima')
            ax_top.plot(X, y, 'o', color=dot, ms=8, label='control points')
            ax_top.set(ylim=(-3, 9), ylabel='y')
            ax_top.set_title('Less ringing than a cubic spline')
            ax_top.legend(loc='upper right', frameon=False, fontsize=18)

            # bottom: three corner models on an asymmetric linear corner (right arm tilts)
            slope = -2.0 + phase  # stays in [-3, -1]: always an asymmetric downward kink
            yc = np.concatenate([CORNER_LEFT, 9.0 + slope * np.arange(1.0, 6.0)])
            for label, model, color, ls in models:
                ax_bot.plot(XS, AkimaSpline(X, yc, corner_model=model)(XS), color=color, ls=ls, lw=4.4, label=label)
            ax_bot.plot(X, yc, 'o', color=dot, ms=8)
            ax_bot.set(xlim=(1.8, 4.4), ylim=(5, 10.4), xlabel='x', ylabel='y')
            ax_bot.set_title('Corner handling at a sharp kink (zoom)')
            ax_bot.legend(loc='upper left', frameon=False, fontsize=18)

        save_animation(fig, frame, ASSETS / f'akima_demo_{name}.gif')


if __name__ == '__main__':
    run_all(_render)
