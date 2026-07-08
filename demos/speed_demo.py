"""Demonstrate speed of pyakima vs scipy and pygsl."""
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pygsl_lite.spline
from scipy.interpolate import Akima1DInterpolator

from pyakima import AkimaSpline

if __name__ == '__main__':

    xs = np.linspace(0., 10., num=10001)
    ys_expect = np.sin(2 * np.pi * xs)

    x = xs[::25].copy()
    y = ys_expect[::25].copy()

    corner_model = 1
    denom_small_cut = 0.

    akima_my = AkimaSpline(x, y, corner_model=corner_model, denom_small_cut=denom_small_cut)

    akima_gsl = pygsl_lite.spline.akima(x.size)
    akima_gsl.init(x, y)

    akima_scipy = Akima1DInterpolator(x, y, extrapolate=True, method='makima')

    y_akima_scipy = akima_scipy(xs)

    y_akima_gsl = akima_gsl.eval_vector(xs)

    y_akima_my = akima_my(xs)

    n_run = 10000

    t1 = perf_counter()
    for _itrm in range(n_run):
        Akima1DInterpolator(x, y, extrapolate=True)
    t2 = perf_counter()

    print('scipy init time    corner 0 %.3e' % ((t2 - t1) / n_run))

    t1 = perf_counter()
    for _itrm in range(n_run):
        akima_gsl_loc = pygsl_lite.spline.akima(x.size)
        akima_gsl_loc.init(x, y)
    t2 = perf_counter()

    print('gsl   init time    corner 0 %.3e' % ((t2 - t1) / n_run))

    t1 = perf_counter()
    for _itrm in range(n_run):
        AkimaSpline(x, y, corner_model=corner_model)
    t2 = perf_counter()

    print('my    init time    corner 0 %.3e' % ((t2 - t1) / n_run))

    t1 = perf_counter()
    for _itrm in range(n_run):
        akima_scipy(xs)
    t2 = perf_counter()

    print('scipy eval time    corner 0 %.3e' % ((t2 - t1) / n_run))

    t1 = perf_counter()
    for _itrm in range(n_run):
        akima_gsl.eval_vector(xs)
    t2 = perf_counter()

    print('gsl   eval time    corner 0 %.3e' % ((t2 - t1) / n_run))

    t1 = perf_counter()
    for _itrm in range(n_run):
        akima_my(xs)
    t2 = perf_counter()

    print('my    eval time    corner 0 %.3e' % ((t2 - t1) / n_run))

    do_plots = False
    if do_plots:
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o', label='data')
        ax.plot(xs, y_akima_scipy, label='scipy makima')
        ax.plot(xs, y_akima_gsl, label='gsl akima')
        ax.plot(xs, y_akima_my, label='my akima')
        ax.plot(xs, ys_expect, 'k--', label='true function', alpha=0.5)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
