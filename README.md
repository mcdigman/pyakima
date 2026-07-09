# pyakima

Pure Python Implementation of Akima Splines

![Two-panel animation. Top: as one control point slides up and down, the pyakima makima curve hugs the data while a natural cubic spline rings above and below the spike. Bottom: a zoom on a sharp kink where the non-rounded, akima, and makima corner models round the corner by differing amounts.](assets/akima_demo.gif)

The top panel slides one control point up and down: the pyakima `makima` fit
stays local and flat on either side of the spike, while a natural cubic spline
rings above and below it. The bottom panel zooms into a sharp kink to show the
three corner models `pyakima` exports — `non-rounded` (GSL), `akima` (SciPy),
and `makima` — which round the corner by differing amounts.

## Regenerating the demo

`demos/` ships as an example rather than an installed package, so run it from a
source checkout:

```bash
pip install -e '.[demos]'         # scipy, matplotlib, pygsl_lite
python -m demos.animate_demo      # writes assets/akima_demo.gif
```
