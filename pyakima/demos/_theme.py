"""Shared light/dark presets and helpers for the pyakima.demos animations.

Centralises the colorblind-safe (Okabe-Ito) palette, the Matplotlib rc overrides
and axis styling, the frame-count/fps/dpi, and the render loop shared by the
animation demo scripts so each demo stays free of duplicated boilerplate.

Copyright 2026 Matthew C. Digman

"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ASSETS = Path(__file__).resolve().parents[2] / 'assets'

FRAMES = 45
FPS = 15
DPI = 72

# Colorblind-safe (Okabe-Ito) palette tuned per background.
# Keys: bg, fg, dot, makima, akima, nonround, cubic.
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


def rc_params(theme: dict[str, str]) -> dict[str, object]:
    """Return Matplotlib rc overrides that paint the figure in ``theme``'s colours.

    Returns
    -------
    dict[str, object]
        rc settings to apply with ``matplotlib.pyplot.rc_context``.
    """
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
    }


def style_axes(ax: Axes) -> None:
    """Put inward ticks on all four sides of ``ax``, long enough to read."""
    ax.tick_params(which='both', direction='in', top=True, right=True, length=9, width=1.2)


def save_animation(fig: Figure, frame: Callable[[int], None], out: Path) -> None:
    """Animate ``frame`` over ``FRAMES``, write ``out`` as a looping GIF, and close ``fig``."""
    anim = animation.FuncAnimation(fig, frame, frames=FRAMES, interval=50)
    anim.save(out, writer=animation.PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f'wrote {out}')


def run_all(render: Callable[[str, dict[str, str]], None]) -> None:
    """Ensure the assets directory exists, then render each theme via ``render``."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        render(name, theme)
