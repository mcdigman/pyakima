# Examples

Animated comparisons of `pyakima` against SciPy's cubic splines. The images
follow the light/dark toggle in the sidebar.

## Corner models

<img class="only-light" alt="Two-panel animation. Top: as one control point slides up and down, the pyakima makima curve hugs the data while a natural cubic spline rings above and below the spike. Bottom: a zoom on a sharp kink where the non-rounded, akima, and makima corner models round the corner by differing amounts." src="_static/akima_demo_light.gif">
<img class="only-dark" alt="Two-panel animation. Top: as one control point slides up and down, the pyakima makima curve hugs the data while a natural cubic spline rings above and below the spike. Bottom: a zoom on a sharp kink where the non-rounded, akima, and makima corner models round the corner by differing amounts." src="_static/akima_demo_dark.gif">

```{include} ../README.md
:start-after: <!-- doc:corners:start -->
:end-before: <!-- doc:corners:end -->
```

## Irregular grids

<img class="only-light" alt="Single panel animation. As the control points slide between a regular and irregular grid, the pyakima makima curve smoothly hugs the data, while the scipy default cubic spline oscillates so strongly it extends off the plotted y axis." src="_static/akima_grid_light.gif">
<img class="only-dark" alt="Single panel animation. As the control points slide between a regular and irregular grid, the pyakima makima curve smoothly hugs the data, while the scipy default cubic spline oscillates so strongly it extends off the plotted y axis." src="_static/akima_grid_dark.gif">

```{include} ../README.md
:start-after: <!-- doc:grid:start -->
:end-before: <!-- doc:grid:end -->
```

```{include} ../README.md
:start-after: <!-- doc:regen:start -->
:end-before: <!-- doc:regen:end -->
```

```{include} ../README.md
:start-after: <!-- doc:footnotes:start -->
:end-before: <!-- doc:footnotes:end -->
```
