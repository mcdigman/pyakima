"""Python Akima Spline Implementation.

Copyright Matthew Digman 2025

objects defined:

SplineCoeffs: namedtuple storing a spline
AkimaSpline: python object managing creating and evaluating an akima spline

"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, overload

import numba
import numba.core.types
import numba.extending
import numpy as np
from numba import njit

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SplineCoeffs(NamedTuple):
    """Named tuple storing the coefficients that represent a cubic spline.

    Attributes
    ----------
    x : NDArray[np.floating]
        n_control x coordinates of the spline control points.
    y : NDArray[np.floating]
        n_control y coordinates of the spline control points.
    n_control : int
        number of control points.
    a : NDArray[np.floating]
        n_control-1 first (constant) terms of the cubic spline pieces.
    b : NDArray[np.floating]
        n_control-1 second (linear) terms of the cubic spline pieces.
    c : NDArray[np.floating]
        n_control-1 third (quadratic) terms of the cubic spline pieces.
    d : NDArray[np.floating]
        n_control-1 fourth (cubic) terms of the cubic spline pieces.
    """

    x: NDArray[np.floating]
    y: NDArray[np.floating]
    n_control: int
    a: NDArray[np.floating]
    b: NDArray[np.floating]
    c: NDArray[np.floating]
    d: NDArray[np.floating]


@njit()
def akima_create_helper(
    x: NDArray[np.floating], y: NDArray[np.floating], corner_model: int = 0, denom_small_cut: float = 0.0
) -> SplineCoeffs:
    """
    Precompute the coefficients necessary to handle akima splines.

    Parameters
    ----------
    x : NDArray[np.floating]
        monotonically increasing x coordinates of the spline control points (must be at least 5).
    y : NDArray[np.floating]
        y coordinates of the spline control points (size must match x).
        Non-finite y are used in computations as-is (like gsl, unlike scipy which raises),
        which typically produces nans near the non-finite y but keeps the spline usable
        over mostly finite stretches of data.
    corner_model : int
        selection for how corners are handled:
        0 uses the Wodicka non-rounded corner method (near-exact match to gsl when
        denom_small_cut == 0, not differentiable at sharp corners);
        1 uses the default akima method (close to scipy method='akima');
        2 uses the modified/makima method with stabilizing weights that need no special
        corner handling (close to scipy method='makima').
    denom_small_cut : float
        threshold below which the slope denominator is treated as zero and handled
        specially. Usually best left at zero.

    Returns
    -------
    SplineCoeffs
        object representing the computed spline.

    Raises
    ------
    ValueError
        if corner_model is unrecognized.

    Notes
    -----
    Notes on non-rounded corner akima spline implementation:
    (up to floating point differences from different factoring and compilation);
    it is described in algorithm 13.1 in "Akima and Renner Subsplines" from "Numerical Algorithms with C"
    by Engeln-Müllges, Gisela & Uhlig, Frank, ISBN 9783642646829
    See https://link.springer.com/content/pdf/10.1007/978-3-642-61074-5_13.pdf

    Currently, only natural boundary conditions are implemented.

    """
    # enforce required conditions
    n_control: int = x.size
    assert n_control > 4
    assert y.size == x.size

    # calculate the difference
    xdiffs: NDArray[np.floating] = np.diff(x)
    assert np.all(xdiffs > 0.0)

    # set boolean variables to control the loop behavior in each corner model case
    if corner_model == 0:
        # gsl-like non-rounded corner handling
        modified: bool = False
        sharp_corners: bool = True
        # denom_small_cut should be zero to match gsl
    elif corner_model == 1:
        # scipy method='akima'-like corner handling
        modified = False
        sharp_corners = False
        # denom_small_cut should be 10^-9 to match scipy
    elif corner_model == 2:
        # scipy method='makima'-like corner handling
        modified = True
        sharp_corners = False
        # denom_small_cut should be zero to match makima
    else:
        msg = f'Unrecognized option for corner model {corner_model}'
        raise ValueError(msg)

    # the numerically computed local slopes
    m = np.zeros(n_control + 3)

    m[2 : n_control + 1] = np.diff(y) / xdiffs  # local slopes

    # natural boundary conditions
    m[0] = 3 * m[2] - 2 * m[3]
    m[1] = 2 * m[2] - m[3]
    m[n_control + 1] = 2 * m[n_control] - m[n_control - 1]
    m[n_control + 2] = 3 * m[n_control] - 2 * m[n_control - 1]

    t_left = np.zeros(n_control)  # left sided slopes
    t_right = np.zeros(n_control)  # right side slopes (differ from t_left only for non-rounded corner handling)

    # loop through the control points
    for i in range(n_control):
        # w1 and w2 are weights
        if modified:
            # modified akima weights
            w1: float = np.abs(m[i + 3] - m[i + 2]) + np.abs(m[i + 3] + m[i + 2]) / 2.0
            w2: float = np.abs(m[i + 1] - m[i]) + np.abs(m[i + 1] + m[i]) / 2.0
        else:
            # basic akima weights
            w1 = np.abs(m[i + 3] - m[i + 2])
            w2 = np.abs(m[i + 1] - m[i])

        # the denominator of the slope; if denom is zero and m[i+2]!=m[i+1], we have a corner
        denom: float = w1 + w2

        dm2: float = np.abs(m[i + 2] - m[i + 1])  # if denom is zero and m[i+2] == m[i+1], the spline is just flat

        if np.isnan(denom) or ~np.isfinite(dm2):
            # handling for nans
            t_left[i] = np.nan
            t_right[i] = np.nan
            continue

        if dm2 == 0.0:
            # handle flat case
            t_left[i] = m[i + 1]
            t_right[i] = t_left[i]
            continue

        # calculate the denominator cutoff we need with appropriate dimension scaling
        if denom_small_cut == 0.0:
            denom_cut_loc: float = 0.0
        else:
            denom_cut_loc = denom_small_cut * dm2

        if denom <= denom_cut_loc:
            if sharp_corners:
                # gsl-like corner handling
                t_left[i] = m[i + 1]
                t_right[i] = m[i + 2]
            else:
                # scipy method=akima like handling
                # note for modified case, should really only get here if there is effectively no slope
                t_left[i] = (m[i + 1] + m[i + 2]) / 2.0
                t_right[i] = t_left[i]
            continue

        # zero denominator should be trapped by previous checks,
        # but handle anyway in case an edge case slips by to prevent zero division errors
        if w2 == 0.0 or denom == 0.0:
            alpha: float = 0.0
        else:
            # derivative of slope with respect to m, used to interpolate the slope
            alpha = w2 / denom

        # not a special case, so evaluate the slopes in the default manner
        t_left[i] = m[i + 1] + alpha * (m[i + 2] - m[i + 1])
        t_right[i] = t_left[i]

    # create the arrays to store spline coefficients
    a = np.zeros(n_control - 1)
    b = np.zeros(n_control - 1)
    c = np.zeros(n_control - 1)
    d = np.zeros(n_control - 1)

    # store the spline coefficients
    for i in range(n_control - 1):
        a[i] = y[i]
        b[i] = t_right[i]

        # as written, h is the same as xdiff, but might not be in some possible modifications
        h: float = x[i + 1] - x[i]

        c[i] = (3 * m[i + 2] - 2 * t_right[i] - t_left[i + 1]) / h
        d[i] = (t_right[i] + t_left[i + 1] - 2 * m[i + 2]) / h**2

    return SplineCoeffs(x, y, n_control, a, b, c, d)


@overload
def spline_single_knot_eval(xint: float, spline: SplineCoeffs, i: int) -> float: ...
@overload
def spline_single_knot_eval(xint: NDArray[np.floating], spline: SplineCoeffs, i: int) -> NDArray[np.floating]: ...
@njit()
def spline_single_knot_eval(
    xint: float | NDArray[np.floating], spline: SplineCoeffs, i: int
) -> float | NDArray[np.floating]:
    """
    Evaluate the spline from the values at the knot point with index i.

    Do not check whether xint is in the x range covered by the specified knot.

    Parameters
    ----------
    xint : float | NDArray[np.floating]
        scalar or array of x values at which to evaluate the spline.
    spline : SplineCoeffs
        object representing the spline to evaluate.
    i : int
        index of the spline knot to evaluate.

    Returns
    -------
    float | NDArray[np.floating]
        evaluated points of the same shape as xint.
    """
    return (
        spline.a[i]
        + spline.b[i] * (xint - spline.x[i])
        + spline.c[i] * (xint - spline.x[i]) ** 2
        + spline.d[i] * (xint - spline.x[i]) ** 3
    )


@njit()
def cubic_call_scalar(xint: float, spline: SplineCoeffs, ext: int) -> float:
    """
    Call cubic spline with a scalar.

    Linearly searches through control points; more intelligent searches are possible, see e.g. cubic_call_vector

    Parameters
    ----------
    xint : float
        scalar point at which to evaluate the spline.
    spline : SplineCoeffs
        object representing the spline to evaluate.
    ext : int
        flag selecting the method of bounds handling.

    Returns
    -------
    float
        interpolated y value.

    Raises
    ------
    ValueError
        if the extrapolation option is unrecognized.
    """
    n_control = spline.n_control

    # handle out of bounds
    if ext == 0:
        y_bound_low: float = np.nan
        y_bound_high: float = np.nan
    elif ext == 1:
        y_bound_low = 0.0
        y_bound_high = 0.0
    elif ext == 3:
        y_bound_low = float(spline.y[0])
        y_bound_high = float(spline.y[-1])
    elif ext == 4:
        y_bound_low = np.nan
        y_bound_high = np.nan
    else:
        msg = f'Unrecognized option for extrapolation: {ext}'
        raise ValueError(msg)

    # for constant boundary value handling
    if xint < spline.x[0] and ext != 0:
        return y_bound_low
    if xint > spline.x[-1] and ext != 0:
        return y_bound_high
    # find the proper subspline
    for i in range(n_control - 2):
        if xint < spline.x[i + 1]:
            return spline_single_knot_eval(xint, spline, i)

    # always evaluate using last iteration if we get here
    return spline_single_knot_eval(xint, spline, n_control - 2)


@njit()
def cubic_call_vector(xint: NDArray[np.floating], spline: SplineCoeffs, ext: int) -> NDArray[np.floating]:
    """
    Evaluate akima splines with a vector input.

    Note that there are several possible implementations of this method
    that could be best in various circumstances.
    The current implementation can take advantage of
    the assumption that xint is typically likely to be sorted (either forward or reversed)
    but does not require it; if the application was required to be sorted
    (or the specific sort order was known)
    a somewhat more efficient implementation would be possible.
    If instead the input was very likely not to have any particular order,
    it could be better not to even check if it is sorted.
    Especially for small inputs (either in spline.x or xint), or when run on a GPU,
    it might be faster to drop any correlation between values
    of xint so that the evaluation can better proceed in parallel.
    Depending on the application, lazy evaluation of spline coefficients
    could be more efficient, but is not implemented here yet.

    Parameters
    ----------
    xint : NDArray[np.floating]
        array of points at which to evaluate the spline.
    spline : SplineCoeffs
        object representing the spline to evaluate.
    ext : int
        flag selecting the method of bounds handling.

    Returns
    -------
    NDArray[np.floating]
        array of the same size as xint containing interpolated y values.

    Raises
    ------
    ValueError
        if the extrapolation method is unrecognized.
    """
    n_control = spline.n_control

    # boundary value handling
    if ext == 0:
        y_bound_low: float = np.nan
        y_bound_high: float = np.nan
    elif ext == 1:
        y_bound_low = 0.0
        y_bound_high = 0.0
    elif ext == 3:
        y_bound_low = float(spline.y[0])
        y_bound_high = float(spline.y[-1])
    elif ext == 4:
        y_bound_low = np.nan
        y_bound_high = np.nan
    else:
        msg = f'Unrecognized option for extrapolation: {ext}'
        raise ValueError(msg)

    res = np.zeros(xint.size)

    # handle the initial value
    last_idx: int = int(np.searchsorted(spline.x[: n_control - 1], float(xint[0]), side='right') - 1)
    if last_idx > n_control - 2:
        last_idx = n_control - 2
    elif last_idx < 0:
        last_idx = 0
    res[0] = cubic_call_scalar(float(xint[0]), spline, ext)

    # find the proper subspline for later iterations using previous results as a guess for the location
    for j in range(1, xint.size):
        # by using fact that input xint will generally be sorted
        # we can use successive starting guesses to accelerate finding the nearest spline points
        # this speedup will get larger if there is more points; if less it might be better to do a linear search

        x_loc = float(xint[j])

        if x_loc < spline.x[0] and ext != 0:
            res[j] = y_bound_low
            last_idx = 0
            continue

        if x_loc >= spline.x[n_control - 2]:
            if x_loc <= spline.x[n_control - 1] or ext == 0:
                res[j] = spline_single_knot_eval(x_loc, spline, n_control - 2)
                last_idx = n_control - 2
            else:
                res[j] = y_bound_high
                last_idx = n_control - 2
            continue

        if x_loc > xint[j - 1]:
            if x_loc < spline.x[last_idx + 1]:
                i = last_idx
            else:
                i = int(np.searchsorted(spline.x[last_idx : n_control - 1], x_loc, side='right') - 1 + last_idx)
            last_idx = i
        elif x_loc >= spline.x[last_idx]:
            i = last_idx
        else:
            i = int(np.searchsorted(spline.x[:last_idx], x_loc, side='right') - 1)

        last_idx = i

        res[j] = spline_single_knot_eval(x_loc, spline, i)

    return res


@overload
def cubic_call(xint: float, spline: SplineCoeffs, ext: int) -> float: ...
@overload
def cubic_call(xint: NDArray[np.floating], spline: SplineCoeffs, ext: int) -> NDArray[np.floating]: ...
def cubic_call(xint: float | NDArray[np.floating], spline: SplineCoeffs, ext: int) -> float | NDArray[np.floating]:
    """
    Evaluate akima splines with scalar or vector input.

    Note that there are several possible implementations of this method
    that could be best in various circumstances.
    The current implementation can take advantage of
    the assumption that xint is typically likely to be sorted (either forward or reversed)
    but does not require it; if the application was required to be sorted
    (or the specific sort order was known)
    a somewhat more efficient implementation would be possible.
    If instead the input was very likely not to have any particular order,
    it could be better not to even check if it is sorted.
    Especially for small inputs (either in spline.x or xint), or when run on a GPU,
    it might be faster to drop any correlation between values
    of xint so that the evaluation can better proceed in parallel.
    Depending on the application, lazy evaluation of spline coefficients
    could be more efficient, but is not implemented here yet.

    Parameters
    ----------
    xint : float | NDArray[np.floating]
        scalar or array of points at which to evaluate the spline.
    spline : SplineCoeffs
        object representing the spline to evaluate.
    ext : int
        flag selecting the method of bounds handling.

    Returns
    -------
    float | NDArray[np.floating]
        scalar or array of the same size as xint containing interpolated y values.

    Raises
    ------
    TypeError
        if the type of xint is unsupported.
    """
    # implement in the select function
    if isinstance(xint, np.ndarray):
        return cubic_call_vector(xint, spline, ext)
    if isinstance(xint, (float, np.floating, numba.core.types.Float)):
        return cubic_call_scalar(xint, spline, ext)
    msg = 'Unsuported type of input'
    raise TypeError(msg)


@numba.extending.overload(cubic_call)
def _select_cubic_call(xint, spline, ext):  # noqa: ANN001, ANN202 # type: ignore[implicit-any-parameter]
    assert isinstance(ext, numba.core.types.Integer)
    assert isinstance(spline, numba.core.types.NamedTuple)
    if isinstance(xint, numba.core.types.Float):

        def temp(xint, spline, ext):  # noqa: ANN001, ANN202 # type: ignore[reportRedeclaration]
            return cubic_call_scalar(xint, spline, ext)
    elif isinstance(xint, numba.core.types.Array):

        def temp(xint, spline, ext):  # noqa: ANN001, ANN202 # type: ignore[reportRedeclaration]
            return cubic_call_vector(xint, spline, ext)
    else:
        msg = 'Unsuported type of input: ' + str(type(xint))
        raise TypeError(msg)
    return temp


@njit()
def cubic_call_vector_linear(xint: NDArray[np.floating], spline: SplineCoeffs, ext: int) -> NDArray[np.floating]:
    """
    Evaluate akima splines with a scalar input.

    Helper similar to cubic_call_vector (see more discussion there), but make loop iterations uncorrelated.
    This method may be faster if xint not at least partially sorted
    or on some compute architectures (this method is more readily parallelizable)

    Parameters
    ----------
    xint : NDArray[np.floating]
        array of points at which to evaluate the spline.
    spline : SplineCoeffs
        object representing the spline to evaluate.
    ext : int
        flag selecting the method of bounds handling.

    Returns
    -------
    NDArray[np.floating]
        array of the same size as xint containing interpolated y values.
    """
    res = np.zeros(xint.size)
    # iterate over every input point
    for j in range(xint.size):
        # let cubic_call_scalar handle finding the correct subspline and evaluating
        res[j] = cubic_call_scalar(xint[j], spline, ext)
    return res


class AkimaSpline:
    """Python class to manage akima splines.

    Parameters
    ----------
    x : NDArray[np.floating]
        monotonically increasing spline control points (must be at least 5).
    y : NDArray[np.floating]
        values at the spline control points (size must match x).
    ext : int
        flag for the extrapolation method.
    corner_model : int | str
        flag for the corner handling method. Current options are:
        0 or 'non-rounded' for the non-rounded method of Wodicka;
        1 or 'akima' for the method described by Akima (scipy method='akima');
        2 or 'makima' for the modified method with less overshoot (scipy method='makima').
    denom_small_cut : float
        cutoff in the denominator of the spline slopes, below which the spline has a corner.
    linear_vector_calls : int
        affects only the speed of __call__, not the results.
        If 1, linearly search the control points when evaluating at xint;
        if 0, use a search assuming xint may be partly sorted (forward or reverse).

    Raises
    ------
    ValueError
        if the specified model parameters are unrecognized.
    """

    def __init__(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        ext: int = 3,
        corner_model: int | str = 0,
        denom_small_cut: float = np.nan,
        linear_vector_calls: int = 0,
    ) -> None:
        # record the inputs
        assert linear_vector_calls in (0, 1)

        self.ext: int = ext
        self.denom_small_cut: float = denom_small_cut
        self.linear_vector_calls: int = linear_vector_calls

        # parse the input corner model
        if corner_model in ('non-rounded', 0):
            self.corner_model: int = 0
        elif corner_model in ('akima', 1):
            self.corner_model = 1
        elif corner_model in ('makima', 2):
            self.corner_model = 2
        else:
            msg = f'Unrecognized option for corner model: {corner_model}'
            raise ValueError(msg)

        # default values for the denominator cutoff depend on the method
        if np.isnan(self.denom_small_cut):
            if corner_model == 0:
                # match gsl
                self.denom_small_cut = 0.0
            elif self.corner_model == 2:
                # cut is superfluous by design in the modified akima case
                # because the slope is engineered to be zero when the denominator is zero
                self.denom_small_cut = 0.0
            else:
                # match scipy with cut
                self.denom_small_cut = 1.0e-9

        # get the spline object
        self.spline: SplineCoeffs = akima_create_helper(
            x.copy(), y.copy(), corner_model=self.corner_model, denom_small_cut=self.denom_small_cut
        )

    @overload
    def __call__(self, xint: float) -> float: ...
    @overload
    def __call__(self, xint: NDArray[np.floating]) -> NDArray[np.floating]: ...
    def __call__(self, xint: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
        """
        Call the akima spline object.

        Parameters
        ----------
        xint : float | NDArray[np.floating]
            scalar or array of points at which to evaluate the spline.

        Returns
        -------
        float | NDArray[np.floating]
            scalar or array of the same size as xint with the spline evaluated at the requested points.
        """
        if isinstance(xint, np.ndarray) and self.linear_vector_calls == 1:
            return cubic_call_vector_linear(xint, self.spline, self.ext)
        return cubic_call(xint, self.spline, self.ext)
