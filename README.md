# pyakima

Pure Python Implementation of Akima Splines

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/akima_demo_dark.gif">
  <img alt="Two-panel animation. Top: as one control point slides up and down, the pyakima makima curve hugs the data while a natural cubic spline rings above and below the spike. Bottom: a zoom on a sharp kink where the non-rounded, akima, and makima corner models round the corner by differing amounts." src="assets/akima_demo_light.gif">
</picture>

The top panel slides one control point up and down: the pyakima `makima` fit
stays local and flat on either side of the spike, while a natural cubic spline
rings above and below it. The bottom panel zooms into a sharp kink to show the
three corner models `pyakima` exports — `non-rounded` (GSL), `akima` (SciPy),
and `makima` — which round the corner by differing amounts.

## Regenerating the demo

`pyakima.demos` ships as an example subpackage; run it from a source checkout so
it can write the README assets:

```bash
pip install -e '.[demos]'               # scipy, matplotlib, pygsl_lite
python -m pyakima.demos.animate_demo    # writes assets/akima_demo_{light,dark}.gif
```
