# pyakima

Pure Python Implementation of Akima Splines

![pyakima demo: Akima interpolation avoids the overshoot of a cubic spline, with selectable corner handling](docs/_static/akima_demo.gif)

The top panel slides one control point up and down: the Akima fit stays local
and flat on either side of the spike, while a standard cubic spline rings above
and below it. The bottom panel zooms into a sharp kink to show the three corner
models `pyakima` exports — `non-rounded` (GSL), `akima` (SciPy), and `makima` —
which round the corner by differing amounts.

## Regenerating the demo

```bash
pip install pyakima[demos]             # pulls in scipy + matplotlib
python -m pyakima.demos.animate_demo   # writes docs/_static/akima_demo.gif
```
