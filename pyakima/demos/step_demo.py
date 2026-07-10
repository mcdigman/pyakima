"""Render light and dark PNGs of pyakima's corner models on a step function.

The step at ``x = 4`` creates sharp corners where the three exported corner
models diverge: ``non-rounded`` (GSL-like) keeps the corner sharp, while the
``akima`` (SciPy-like) and ``makima`` (SciPy-like) models round it by differing
amounts. Every curve here comes from pyakima's own :class:`AkimaSpline`, so this
demo runs on a bare ``pyakima`` install -- unlike speed_demo it needs neither
SciPy nor pygsl_lite. Being static, it is written as a ``.png`` rather than an
animated GIF.

Two variants are written, ``akima_step_light.png`` and ``akima_step_dark.png``,
so the README can serve each via a theme-aware ``<picture>`` element.

Run from a source checkout::

    pip install -e '.[demos]'            # only matplotlib is needed for this demo
    python -m pyakima.demos.step_demo    # writes assets/akima_step_*.png

Copyright 2026 Matthew C. Digman

"""

from typing import Any, cast

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from pyakima import AkimaSpline
from pyakima.demos._theme import ASSETS, DPI, rc_params, run_all, style_axes

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
Y = 2.0 * np.heaviside(X - 4.0, 0.5) - 1.0
XS = np.linspace(1.0, 7.0, 1401)
YS_TRUE = 2.0 * np.heaviside(XS - 4.0, 0.5) - 1.0

# (label, corner_model, palette key, linestyle) -- reuses the shared corner-model
# colours and line styles so the look matches the animated demos, and stays
# readable in greyscale / for colourblind viewers. All three are pyakima fits.
MODELS = (
    ('non-rounded (GSL-like)', 0, 'nonround', ':'),
    ('akima (SciPy-like)', 1, 'akima', '--'),
    ('makima (SciPy-like)', 2, 'makima', '-'),
)


def _render(name: str, theme: dict[str, str]) -> None:
    # rc keys are dynamic strings, not the RcParams key Literal, so cast past the check.
    with plt.rc_context(cast(Any, rc_params(theme))):  # noqa: TC006
        fig, ax = plt.subplots(figsize=(9, 5.6), layout='constrained')
        style_axes(ax)
        ax.plot(XS, YS_TRUE, color=theme['cubic'], ls='--', lw=5, alpha=0.5, label='true step')
        for label, model, key, ls in MODELS:
            ax.plot(XS, AkimaSpline(X, Y, corner_model=model)(XS), color=theme[key], ls=ls, lw=3, label=label)
        ax.plot(X, Y, 'o', color=theme['dot'], ms=8, label='control points')
        ax.set(xlabel='x', ylabel='y')
        ax.set_title('pyakima corner models on a step function')
        ax.legend(loc='upper left', frameon=False, fontsize=16)
        out = ASSETS / f'akima_step_{name}.png'
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    run_all(_render)
