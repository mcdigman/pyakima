# pyakima

Numba-compatible Akima spline interpolation.

`pyakima` provides an {class}`~pyakima.AkimaSpline` object for constructing and
evaluating Akima splines, along with `njit`-compiled helpers that create and
evaluate splines from within other numba-compiled code.

## Installation

```bash
pip install pyakima
```

## Quick start

```python
import numpy as np
from pyakima import AkimaSpline

x = np.linspace(0.0, 10.0, 11)
y = np.sin(x)

spline = AkimaSpline(x, y)
spline(np.array([1.5, 2.5, 3.5]))
```

```{toctree}
:maxdepth: 2
:caption: Contents

api
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
