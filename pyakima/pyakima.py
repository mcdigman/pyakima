"""Python Akima Spline Implementation
C Matthew Digman 2025

objects defined:

SplineCoeffs: namedtuple storing a spline
AkimaSpline: python object managing creating and evaluating an akima spline

"""

from collections import namedtuple

import numpy as np
from numba import njit

SplineCoeffs = namedtuple('SplineCoeffs', ['x', 'y', 'N', 'a', 'b', 'c', 'd'])
SplineCoeffs.__doc__ = """
Named Tuple Object for storing the coefficients to represent a cubic spline
    N:
        scalar integer, number of x coordinates
    x:
        array of N x coordinates of spline control points
    y:
        array of N y coordinates of spline control points
    a:
        array of N-1 first terms in cubic spline expansion at control points
    b:
        array of N-1 second terms in cubic spline expansion at control points
    c:
        array of N-1 third terms in cubic spline expansion at control points
    d:
        array of N-1 fourth terms in cubic spline expansion at control points
"""


class AkimaSpline:
    """Python class to manage akima splines"""

    def __init__(
        self,
        x,
        y,
        ext=3,
        corner_model=0,
        denom_small_cut=np.nan,
        linear_vector_calls=False
    ):
        """Create an akima spline object
        inputs:
            x:
                array of monotonically increasing spline control points (must be at least 5)
            y:
                array of values at spline control points (size must match x)
            ext:
                integer flag for extrapolation method
            corner_model:
                flag for corner handling method. Current options are:
                    0 or 'non-rounded': non-rounded corner handling method of Wodicka, as used in GSL
                    1 or 'akima': corner handling descrbed by Akima, as used in scipy method='akima'
                    2 or 'makima': modified corner handling with less overshoot, as in scipy method='makima'
            denom_small_cut:
                cutoff in denominator of spline slopes, below which spline has a corner. GSL uses 0, scipy uses 1.e-9
            linear_vector_calls:
                affects only speed of __call__, should not change results at all
                if True, linearly search through spline control points when evaluating splines at xint
                If False, try a search assuming xint may be partly sorted (either forward or reverse)
        """

        # record the inputs

        self.ext = ext
        self.denom_small_cut = denom_small_cut
        self.linear_vector_calls = linear_vector_calls

        # parse the input corner model
        if corner_model in ('non-rounded', 0):
            self.corner_model = 0
        elif corner_model in ('akima', 1):
            self.corner_model = 1
        elif corner_model in ('makima', 2):
            self.corner_model = 2
        else:
            raise ValueError('Unrecognized option for corner model')

        # default values for the denominator cutoff depend on the method
        if np.isnan(self.denom_small_cut):
            if corner_model == 0:
                # match gsl
                self.denom_small_cut = 0.
            elif self.corner_model == 2:
                # cut is superfluous by design in the modified akima case
                # because the slope is engineered to be zero when the denominator is zero
                self.denom_small_cut = 0.
            else:
                # match scipy with cut
                self.denom_small_cut = 1.e-9

        # get the spline object
        self.spline = akima_create_helper(
                x.copy(),
                y.copy(),
                corner_model=self.corner_model,
                denom_small_cut=self.denom_small_cut
        )

    def __call__(self, xint):
        """
        call the akima spline object
        inputs:
                xint:
                    either a scalar or array of points at which to evaluate the spline
        outputs:
                res:
                    scalar or array of same size as xint containing the spline evaluated at requested points
        """
        if isinstance(xint, np.ndarray):
            if self.linear_vector_calls:
                return cubic_call_vector_linear(xint, self.spline, self.ext)
            else:
                return cubic_call_vector(xint, self.spline, self.ext)
        else:
            return cubic_call_scalar(xint, self.spline, self.ext)


@njit()
def akima_create_helper(x, y, corner_model=0, denom_small_cut=0.):
    """
    method to precompute the coefficients necessary to handle akima splines.

    inputs:
        x:
            monotonically increasing x coordinates of the spline control points (must be at least 5)

        y:
            the y coordinates of the spline control points (must match size of x)
            Note that this method handles non-finite y the way gsl does,
            which is to use them in computations just like finite values
            This typically produces nans when the spline is evaluated near the non-finite y.
            The gSL handling allows the spline to be computed ~sensibly for data stretches
            that are mostly finite but have a few non-finite values.
            scipy instead detects any non-finite values and throws an error

        denom_small_cut:
            threshold below which the denominator in the slope
            is assumed to be zero and therefore needs different handling.
            GSL only handles it differently if it is 0 exactly,
            while scipy uses a cutoff of 1.e-9 for method='akima'

        corner_model:
            integer selection for how to handle corners.
            corner_model == 0:
                The Wodicka non-rounded corner method,
                which reduces spline misbehavior at sharp corners but makes the spline not differentiable there,
                also creating slight qualitative discontinutity when the denominator of the slope becomes exactly 0.
                If corner_model == 0 and denom_small_cut == 0., this implementation should near-exactly match gsl's
                Further description below.
            corner_model == 1:
                default akima implementation can behave badly at corners,
                and also has a qualitative discontinuity in the splines when the denominator crosses denom_small_cut
                Close match to scipy method='akima'
            corner_model == 2:
                modified akima implementation
                (see https://blogs.mathworks.com/cleve/2019/04/29/makima-piecewise-cubic-interpolation/ and scipy)
                with weights that act as numerical stabilizers, removing qualitative discontinuities such that
                no additional special handling is needed for corners.
                close match to scipy method='makima'
    outputs:
        spline:
            a SplineCoeffs object representing the computed spline

    Notes on non-rounded corner akima spline implementation:
    (up to floating point differences from different factoring and compilation);
    it is described in algorithm 13.1 in "Akima and Renner Subsplines" from "Numerical Algorithms with C"
    by Engeln-Müllges, Gisela & Uhlig, Frank, ISBN 9783642646829
    See https://link.springer.com/content/pdf/10.1007/978-3-642-61074-5_13.pdf

    Currently, only natural boundary conditions are implemented.

    """
    # enforce required conditions
    N = x.size
    assert N > 4
    assert y.size == x.size

    # calculate the difference
    xdiffs = np.diff(x)
    assert np.all(xdiffs > 0.)

    # set boolean variables to control the loop behavior in each corner model case
    if corner_model == 0:
        # gsl-like non-rounded corner handling
        modified = False
        sharp_corners = True
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
        raise ValueError('Unrecognized option for corner model')

    # the numerically computed local slopes
    m = np.zeros(N + 3)

    m[2:N+1] = np.diff(y)/xdiffs  # local slopes

    # natural boundary conditions
    m[0] = 3*m[2] - 2*m[3]
    m[1] = 2*m[2] - m[3]
    m[N+1] = 2*m[N] - m[N-1]
    m[N+2] = 3*m[N] - 2*m[N-1]

    t_left = np.zeros(N)   # left sided slopes
    t_right = np.zeros(N)  # right side slopes (differ from t_left only for non-rounded corner handling)

    # loop through the control points
    for i in range(N):
        # w1 and w2 are weights
        if modified:
            # modified akima weights
            w1 = np.abs(m[i+3] - m[i+2]) + np.abs(m[i+3]+m[i+2])/2.
            w2 = np.abs(m[i+1] - m[i]) + np.abs(m[i+1]+m[i])/2.
        else:
            # basic akima weights
            w1 = np.abs(m[i+3] - m[i+2])
            w2 = np.abs(m[i+1] - m[i])

        # the denominator of the slope; if denom is zero and m[i+2]!=m[i+1], we have a corner
        denom = w1 + w2

        dm2 = np.abs(m[i+2] - m[i+1])  # if denom is zero and m[i+2] == m[i+1], the spline is just flat

        if np.isnan(denom) or ~np.isfinite(dm2):
            # handling for nans
            t_left[i] = np.nan
            t_right[i] = np.nan
            continue

        if dm2 == 0.:
            # handle flat case
            t_left[i] = m[i+1]
            t_right[i] = t_left[i]
            continue

        # calculate the denominator cutoff we need with appropriate dimension scaling
        if denom_small_cut == 0.:
            denom_cut_loc = 0.
        else:
            denom_cut_loc = denom_small_cut * dm2

        if denom <= denom_cut_loc:
            if sharp_corners:
                # gsl-like corner handling
                t_left[i] = m[i+1]
                t_right[i] = m[i+2]
            else:
                # scipy method=akima like handling
                # note for modified case, should really only get here if there is effectively no slope
                t_left[i] = (m[i+1] + m[i+2])/2.
                t_right[i] = t_left[i]
            continue

        # zero denominator should be trapped by previous checks,
        # but handle anyway in case an edge case slips by to prevent zero division errors
        if w2 == 0. or denom == 0.:
            alpha = 0.
        else:
            # derivative of slope with respect to m, used to interpolate the slope
            alpha = w2/denom

        # not a special case, so evaluate the slopes in the default manner
        t_left[i] = m[i+1] + alpha*(m[i+2] - m[i+1])
        t_right[i] = t_left[i]

    # create the arrays to store spline coefficients
    a = np.zeros(N - 1)
    b = np.zeros(N - 1)
    c = np.zeros(N - 1)
    d = np.zeros(N - 1)

    # store the spline coefficients
    for i in range(N - 1):
        a[i] = y[i]
        b[i] = t_right[i]

        # as written, h is the same as xdiff, but might not be in some possible modifications
        h = x[i+1] - x[i]

        c[i] = (3*m[i+2] - 2*t_right[i] - t_left[i+1])/h
        d[i] = (t_right[i] + t_left[i+1] - 2*m[i+2])/h**2

    spline = SplineCoeffs(x, y, N, a, b, c, d)
    return spline


@njit()
def spline_single_knot_eval(xint, spline, i):
    """
    evaluate the spline from the values at the knot point with index i
    without checking whether xint is in the x range covered by the specified knot

    inputs:
        xint:
            array or scalar of x values to evaluate the spline at
        spline:
            a SplineCoeffs object representing the spline with points to evaluate
        i:
            scalar integer index of the spline knot to evaluate the spline at
    outputs:
        res:
            array or scalar of evaluated points of the same shape as xint
    """

    return (
        spline.a[i]
        + spline.b[i] * (xint - spline.x[i])
        + spline.c[i] * (xint - spline.x[i])**2
        + spline.d[i] * (xint - spline.x[i])**3
    )


@njit()
def cubic_call_scalar(xint, spline, ext):
    """
    helper for scalar cubic spline evaluation
    linearly searches through control points; more intelligent searches are possible, see e.g. cubic_call_vector

    inputs:
        xint:
            scalar float point at which to evaluate the spline
        spline:
            a SplineCoeffs object representing the spline with points to evaluate
        ext:
            integer flag to select method of bounds handling
    outputs:
        res:
            scalar float, interpolated y value
    """

    N = spline.N

    # handle out of bounds
    if ext == 1:
        y_bound_low = 0.
        y_bound_high = 0.
    elif ext == 3:
        y_bound_low = spline.y[0]
        y_bound_high = spline.y[-1]
    elif ext == 4:
        y_bound_low = np.nan
        y_bound_high = np.nan
    else:
        raise ValueError('Unrecognized option for extrapolation')

    # for constant boundary value handling
    if xint < spline.x[0]:
        return y_bound_low
    elif xint > spline.x[-1]:
        return y_bound_high
    else:
        # find the proper subspline
        for i in range(N - 2):
            if xint < spline.x[i+1]:
                return spline_single_knot_eval(xint, spline, i)

    # always evaluate using last iteration if we get here
    return spline_single_knot_eval(xint, spline, N - 2)


@njit()
def cubic_call_vector_linear(xint, spline, ext):
    """
    helper similar to cubic_call_vector (see more discussion there), but make loop iterations uncorrelated.
    This method may be faster if xint not at least partially sorted
    or on some compute architectures (this method is more readily parallelizable)

    inputs:
        xint:
            array of points at which to evaluate the spline
        spline:
            a SplineCoeffs object representing the spline with points to evaluate
        ext:
            integer flag to select method of bounds handling
    outputs:
        res:
            array of same size as xint containing interpolated y values
    """
    res = np.zeros(xint.size)
    # iterate over every input point
    for j in range(xint.size):
        # let cubic_call_scalar handle finding the correct subspline and evaluating
        res[j] = cubic_call_scalar(xint[j], spline, ext)
    return res


@njit()
def cubic_call_vector(xint, spline, ext):
    """
    helper vector call method for evaluating the akima splines.
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

    inputs:
        xint:
            array of points at which to evaluate the spline
        spline:
            a SplineCoeffs object representing the spline with points to evaluate
        ext:
            integer flag to select method of bounds handling
    outputs:
        res:
            array of same size as xint containing interpolated y values
    """

    N = spline.N

    # boundary value handling
    if ext == 1:
        y_bound_low = 0.
        y_bound_high = 0.
    elif ext == 3:
        y_bound_low = spline.y[0]
        y_bound_high = spline.y[-1]
    elif ext == 4:
        y_bound_low = np.nan
        y_bound_high = np.nan
    else:
        raise ValueError('Unrecognized option for extrapolation')

    res = np.zeros(xint.size)

    # handle the initial value
    last_idx = np.searchsorted(spline.x[:N-1], xint[0], side='right') - 1
    if last_idx > N - 2:
        last_idx = N - 2
    elif last_idx < 0:
        last_idx = 0
    res[0] = cubic_call_scalar(xint[0], spline, ext)

    # find the proper subspline for later iterations using previous results as a guess for the location
    for j in range(1, xint.size):
        # by using fact that input xint will generally be sorted
        # we can use successive starting guesses to accelerate finding the nearest spline points
        # this speedup will get larger if there is more points; if less it might be better to do a linear search

        # i = np.searchsorted(spline.x[:N-1], xint[j], side='right') - 1

        if xint[j] < spline.x[0]:
            res[j] = y_bound_low
            last_idx = 0
            continue

        if xint[j] >= spline.x[N-2]:
            if xint[j] <= spline.x[N-1]:
                res[j] = spline_single_knot_eval(xint[j], spline, N - 2)
                last_idx = N - 2
            else:
                res[j] = y_bound_high
                last_idx = N - 2
            continue

        if xint[j] > xint[j-1]:
            if xint[j] < spline.x[last_idx+1]:
                i = last_idx
            else:
                i = (
                    np.searchsorted(spline.x[last_idx:N-1], xint[j], side='right')
                    - 1
                    + last_idx
                )
            last_idx = i
        else:
            if xint[j] >= spline.x[last_idx]:
                i = last_idx
            else:
                i = np.searchsorted(spline.x[:last_idx], xint[j], side='right') - 1

        last_idx = i

        res[j] = spline_single_knot_eval(xint[j], spline, i)

    return res
