"""
Demonstrate calling python akima method.

Copyright 2025 Matthew C. Digman

"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import Akima1DInterpolator

from pyakima import AkimaSpline

if __name__ == '__main__':
    # evaluation points
    xs = np.linspace(1.0, 7.0, num=1401)
    ys_expect = 2 * np.heaviside(xs - 4, 0.5) - 1

    # control points
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    y = 2 * np.heaviside(x - 4, 0.5) - 1

    akima_gsl = AkimaSpline(x, y, corner_model=0, denom_small_cut=0.0)
    y_akima_gsl = akima_gsl(xs)

    akima_scipy1 = AkimaSpline(x, y, corner_model=1, denom_small_cut=1.0e-9)
    y_akima_scipy1 = akima_scipy1(xs)

    y_akima_scipy1_actual = Akima1DInterpolator(x, y, extrapolate=False, method='akima')(xs)
    assert_allclose(y_akima_scipy1, y_akima_scipy1_actual, atol=1.0e-14, rtol=1.0e-14)

    akima_scipy2 = AkimaSpline(x, y, corner_model=2, denom_small_cut=0.0)
    y_akima_scipy2 = akima_scipy2(xs)

    y_akima_scipy2_actual = Akima1DInterpolator(x, y, extrapolate=False, method='makima')(xs)
    assert_allclose(y_akima_scipy2, y_akima_scipy2_actual, atol=1.0e-14, rtol=1.0e-14)

    _fig, ax = plt.subplots()
    ax.plot(x, y, 'o', label='data')
    ax.plot(xs, y_akima_scipy1, label='scipy akima')
    ax.plot(xs, y_akima_scipy2, label='scipy makima')
    ax.plot(xs, y_akima_gsl, label='gsl akima')
    ax.plot(xs, ys_expect, 'k--', label='true function', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.legend()
    plt.show()
