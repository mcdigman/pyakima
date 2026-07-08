"""Tests for the public pyakima spline helpers."""

# ruff: noqa: D103

from __future__ import annotations

import numpy as np
import pytest

from pyakima.pyakima import (
    AkimaSpline,
    SplineCoeffs,
    akima_create_helper,
    cubic_call,
    cubic_call_scalar,
    cubic_call_vector,
    cubic_call_vector_linear,
    spline_single_knot_eval,
)


def _affine_control_points(dtype: type[np.floating] = np.float64) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(5, dtype=dtype)
    y = 2 * x + dtype(1)
    return x, y


def _nonlinear_control_points(dtype: type[np.floating] = np.float64) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    y = np.array([0.0, 1.0, 0.5, 2.0, -1.0, 3.0], dtype=dtype)
    return x, y


def _assert_same_float_values(
    actual: np.ndarray | np.floating | float,
    expected: np.ndarray | np.floating | float,
    *,
    maxulp: int = 4,
) -> None:
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    assert actual_arr.shape == expected_arr.shape

    actual_nan = np.isnan(actual_arr)
    expected_nan = np.isnan(expected_arr)
    np.testing.assert_array_equal(actual_nan, expected_nan)

    finite = np.isfinite(actual_arr) & np.isfinite(expected_arr)
    if finite.any():
        np.testing.assert_array_max_ulp(actual_arr[finite], expected_arr[finite], maxulp=maxulp)

    nonfinite_not_nan = ~(finite | actual_nan)
    if nonfinite_not_nan.any():
        np.testing.assert_array_equal(actual_arr[nonfinite_not_nan], expected_arr[nonfinite_not_nan])


def _assert_coefficients_equal(actual: SplineCoeffs, expected: SplineCoeffs, *, maxulp: int = 4) -> None:
    assert actual.n_control == expected.n_control
    for name in ('x', 'y', 'a', 'b', 'c', 'd'):
        _assert_same_float_values(getattr(actual, name), getattr(expected, name), maxulp=maxulp)


def _outside_interval_mask(size: int, first: int, last: int) -> np.ndarray:
    mask = np.ones(size, dtype=bool)
    mask[max(first, 0) : min(last + 1, size)] = False
    return mask


def _interval_slice(size: int, first: int, last: int) -> slice:
    return slice(max(first, 0), min(last + 1, size))


def _assert_component_unchanged_outside_interval(
    baseline: SplineCoeffs,
    actual: SplineCoeffs,
    component: str,
    first: int,
    last: int,
) -> None:
    baseline_values = getattr(baseline, component)
    actual_values = getattr(actual, component)
    mask = _outside_interval_mask(baseline_values.size, first, last)
    _assert_same_float_values(actual_values[mask], baseline_values[mask], maxulp=0)


def _power_of_two(dtype: type[np.floating], exponent: int) -> np.floating:
    return dtype(np.ldexp(1.0, exponent))


def _power_of_two_affine_points(
    dtype: type[np.floating],
    *,
    h_exponent: int,
    slope_exponent: int,
) -> tuple[np.ndarray, np.ndarray, np.floating]:
    h = _power_of_two(dtype, h_exponent)
    delta_y = _power_of_two(dtype, h_exponent + slope_exponent)
    index = np.arange(12, dtype=dtype)
    return index * h, index * delta_y, _power_of_two(dtype, slope_exponent)


@pytest.mark.parametrize('corner_model', [0, 1, 2, 'non-rounded', 'akima', 'makima'])
def test_minimum_length_affine_spline_reproduces_line_for_all_corner_models(corner_model: int | str) -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y, ext=0, corner_model=corner_model)
    xint = np.array([-1.0, 0.0, 0.25, 1.5, 3.75, 4.0, 5.0])

    np.testing.assert_array_equal(spline.spline.a, y[:-1])
    np.testing.assert_array_equal(spline.spline.b, np.full(x.size - 1, 2.0))
    np.testing.assert_array_equal(spline.spline.c, np.zeros(x.size - 1))
    np.testing.assert_array_equal(spline.spline.d, np.zeros(x.size - 1))
    np.testing.assert_array_equal(spline(xint), 2 * xint + 1)


def test_create_rejects_length_mismatch_and_nonincreasing_controls() -> None:
    x, y = _affine_control_points()

    with pytest.raises(AssertionError):
        akima_create_helper(x[:4], y[:4])

    with pytest.raises(AssertionError):
        akima_create_helper(x, y[:-1])

    duplicate_x = x.copy()
    duplicate_x[2] = duplicate_x[1]
    with pytest.raises(AssertionError):
        akima_create_helper(duplicate_x, y)

    decreasing_x = x.copy()
    decreasing_x[2] = decreasing_x[1] - 1.0
    with pytest.raises(AssertionError):
        akima_create_helper(decreasing_x, y)

    nan_x = x.copy()
    nan_x[2] = np.nan
    with pytest.raises(AssertionError):
        akima_create_helper(nan_x, y)


@pytest.mark.parametrize(
    ('integer_model', 'string_model', 'denom_small_cut'),
    [(0, 'non-rounded', 0.0), (1, 'akima', 1.0e-9), (2, 'makima', 0.0)],
)
def test_corner_model_string_aliases_match_integer_models_when_cut_is_explicit(
    integer_model: int,
    string_model: str,
    denom_small_cut: float,
) -> None:
    x, y = _nonlinear_control_points()

    int_spline = AkimaSpline(x, y, corner_model=integer_model, denom_small_cut=denom_small_cut)
    string_spline = AkimaSpline(x, y, corner_model=string_model, denom_small_cut=denom_small_cut)

    assert string_spline.corner_model == int_spline.corner_model
    assert string_spline.denom_small_cut == int_spline.denom_small_cut
    _assert_coefficients_equal(string_spline.spline, int_spline.spline)


@pytest.mark.parametrize(
    ('integer_model', 'string_model'),
    [(0, 'non-rounded'), (1, 'akima'), (2, 'makima')],
)
def test_corner_model_string_aliases_use_same_default_denominator_cut_as_integer_models(
    integer_model: int,
    string_model: str,
) -> None:
    x, y = _nonlinear_control_points()

    int_spline = AkimaSpline(x, y, corner_model=integer_model)
    string_spline = AkimaSpline(x, y, corner_model=string_model)

    assert string_spline.corner_model == int_spline.corner_model
    assert string_spline.denom_small_cut == int_spline.denom_small_cut
    _assert_coefficients_equal(string_spline.spline, int_spline.spline)


def test_invalid_corner_model_raises_value_error() -> None:
    x, y = _affine_control_points()

    with pytest.raises(ValueError, match='Unrecognized option for corner model'):
        AkimaSpline(x, y, corner_model=99)

    with pytest.raises(ValueError, match='Unrecognized option for corner model'):
        AkimaSpline(x, y, corner_model='unknown')

    with pytest.raises(ValueError, match='Unrecognized option for corner model'):
        akima_create_helper(x, y, corner_model=99)


def test_explicit_denominator_cut_changes_corner_branch_when_large_enough() -> None:
    x = np.arange(7, dtype=np.float64)
    y = np.array([0.0, 1.0, 4.0, 2.0, 8.0, 9.0, 12.0])

    no_cut = AkimaSpline(x, y, corner_model='akima', denom_small_cut=0.0)
    large_cut = AkimaSpline(x, y, corner_model='akima', denom_small_cut=10.0)

    assert no_cut.denom_small_cut == 0.0
    assert large_cut.denom_small_cut == 10.0
    assert not np.array_equal(no_cut.spline.b, large_cut.spline.b)
    assert not np.array_equal(no_cut.spline.c, large_cut.spline.c)
    assert not np.array_equal(no_cut.spline.d, large_cut.spline.d)


@pytest.mark.parametrize('linear_vector_calls', [0, 1])
def test_linear_vector_calls_keyword_does_not_change_values(linear_vector_calls: int) -> None:
    x, y = _nonlinear_control_points()
    xint = np.array([4.5, 0.0, 2.25, 1.0, -0.5, 5.5, 3.75, 0.25])

    baseline = AkimaSpline(x, y, ext=0, linear_vector_calls=0)(xint)
    actual = AkimaSpline(x, y, ext=0, linear_vector_calls=linear_vector_calls)(xint)

    _assert_same_float_values(actual, baseline, maxulp=4)


def test_invalid_linear_vector_calls_raises_assertion_error() -> None:
    x, y = _affine_control_points()

    with pytest.raises(AssertionError):
        AkimaSpline(x, y, linear_vector_calls=2)


def test_single_knot_eval_accepts_scalar_and_vector_inputs() -> None:
    x, y = _affine_control_points()
    spline = akima_create_helper(x, y)

    scalar = spline_single_knot_eval(np.float64(1.25), spline, 1)
    vector_x = np.array([1.0, 1.25, 1.5, 1.75])
    vector = spline_single_knot_eval(vector_x, spline, 1)

    assert scalar == 3.5
    np.testing.assert_array_equal(vector, 2 * vector_x + 1)


@pytest.mark.parametrize('corner_model', [0, 1, 2])
def test_evaluating_exactly_at_knots_returns_control_values(corner_model: int) -> None:
    x, y = _nonlinear_control_points()
    spline = AkimaSpline(x, y, ext=0, corner_model=corner_model)

    scalar_values = np.array([cubic_call_scalar(float(knot), spline.spline, 0) for knot in x])
    vector_values = cubic_call_vector(x, spline.spline, 0)
    linear_vector_values = cubic_call_vector_linear(x, spline.spline, 0)

    _assert_same_float_values(scalar_values, y, maxulp=4)
    _assert_same_float_values(vector_values, y, maxulp=4)
    _assert_same_float_values(linear_vector_values, y, maxulp=4)
    _assert_same_float_values(spline(x), y, maxulp=4)


@pytest.mark.parametrize(
    ('ext', 'expected'),
    [
        (0, np.array([-1.0, 1.0, 2.0, 9.0, 11.0])),
        (1, np.array([0.0, 1.0, 2.0, 9.0, 0.0])),
        (3, np.array([1.0, 1.0, 2.0, 9.0, 9.0])),
        (4, np.array([np.nan, 1.0, 2.0, 9.0, np.nan])),
    ],
)
def test_extrapolation_modes_for_scalar_vector_and_class_calls(ext: int, expected: np.ndarray) -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y, ext=ext)
    xint = np.array([-1.0, 0.0, 0.5, 4.0, 5.0])

    scalar_values = np.array([cubic_call_scalar(float(point), spline.spline, ext) for point in xint])
    dispatch_values = cubic_call(xint, spline.spline, ext)
    vector_values = cubic_call_vector(xint, spline.spline, ext)
    linear_vector_values = cubic_call_vector_linear(xint, spline.spline, ext)
    class_values = spline(xint)

    _assert_same_float_values(scalar_values, expected)
    _assert_same_float_values(dispatch_values, expected)
    _assert_same_float_values(vector_values, expected)
    _assert_same_float_values(linear_vector_values, expected)
    _assert_same_float_values(class_values, expected)


def test_invalid_extrapolation_mode_raises_for_all_call_paths() -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y)
    xint = np.array([0.5, 1.5])

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        cubic_call_scalar(0.5, spline.spline, 2)

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        cubic_call_vector(xint, spline.spline, 2)

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        cubic_call_vector_linear(xint, spline.spline, 2)

    bad_ext_spline = AkimaSpline(x, y, ext=2)
    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        bad_ext_spline(0.5)

    with pytest.raises(ValueError, match='Unrecognized option for extrapolation'):
        bad_ext_spline(xint)


@pytest.mark.parametrize(
    'xint',
    [
        np.array([-0.5, 0.0, 0.25, 1.75, 3.5, 5.25]),
        np.array([5.25, 3.5, 1.75, 0.25, 0.0, -0.5]),
        np.array([2.2, -0.5, 5.25, 1.1, 1.1, 4.0, 0.0]),
        np.array([2.2]),
        np.array([], dtype=np.float64),
    ],
)
def test_scalar_vector_dispatch_and_linear_vector_paths_agree(xint: np.ndarray) -> None:
    x, y = _nonlinear_control_points()
    spline = AkimaSpline(x, y, ext=0)

    scalar_values = np.array([cubic_call_scalar(float(point), spline.spline, 0) for point in xint])
    vector_values = cubic_call_vector(xint, spline.spline, 0)
    linear_vector_values = cubic_call_vector_linear(xint, spline.spline, 0)
    dispatch_values = cubic_call(xint, spline.spline, 0)
    class_values = spline(xint)
    linear_class_values = AkimaSpline(x, y, ext=0, linear_vector_calls=1)(xint)

    _assert_same_float_values(vector_values, scalar_values, maxulp=4)
    _assert_same_float_values(linear_vector_values, scalar_values, maxulp=4)
    _assert_same_float_values(dispatch_values, scalar_values, maxulp=4)
    _assert_same_float_values(class_values, scalar_values, maxulp=4)
    _assert_same_float_values(linear_class_values, scalar_values, maxulp=4)


def test_cubic_call_rejects_non_float_scalar_and_non_array_inputs() -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y)

    with pytest.raises(TypeError):
        cubic_call(1, spline.spline, 3)

    with pytest.raises(TypeError):
        cubic_call([0.5, 1.5], spline.spline, 3)

    with pytest.raises(TypeError):
        spline(1)


@pytest.mark.parametrize(
    ('ext', 'expected'),
    [
        (0, np.array([np.nan, np.nan, np.nan])),
        (1, np.array([0.0, np.nan, 0.0])),
        (3, np.array([1.0, np.nan, 9.0])),
        (4, np.array([np.nan, np.nan, np.nan])),
    ],
)
def test_nonfinite_evaluation_points(ext: int, expected: np.ndarray) -> None:
    x, y = _affine_control_points()
    spline = AkimaSpline(x, y, ext=ext)
    xint = np.array([-np.inf, np.nan, np.inf])

    scalar_values = np.array([cubic_call_scalar(float(point), spline.spline, ext) for point in xint])
    vector_values = cubic_call_vector(xint, spline.spline, ext)
    linear_vector_values = cubic_call_vector_linear(xint, spline.spline, ext)
    class_values = spline(xint)

    _assert_same_float_values(scalar_values, expected)
    _assert_same_float_values(vector_values, expected)
    _assert_same_float_values(linear_vector_values, expected)
    _assert_same_float_values(class_values, expected)


def test_nonfinite_y_contaminates_local_region_but_leaves_distant_finite_values_usable() -> None:
    x = np.arange(10, dtype=np.float64)
    clean_y = 2 * x + 1
    y_with_nan = clean_y.copy()
    y_with_nan[5] = np.nan
    spline = AkimaSpline(x, y_with_nan, ext=0)

    xint = np.array([0.5, 1.5, 2.5, 7.5, 8.5])
    actual = spline(xint)

    _assert_same_float_values(actual[[0, 1, 4]], (2 * xint + 1)[[0, 1, 4]])
    assert np.isnan(actual[2])
    assert np.isnan(actual[3])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('corner_model', [0, 1, 2])
def test_large_dynamic_range_control_point_has_strict_local_coefficient_kernel(
    dtype: type[np.floating],
    corner_model: int,
) -> None:
    x = np.arange(17, dtype=dtype)
    y = np.array(
        [
            0.0,
            0.25,
            -0.5,
            0.125,
            0.75,
            -0.25,
            0.375,
            -0.625,
            0.5,
            -0.125,
            0.875,
            0.0,
            -0.375,
            0.625,
            -0.75,
            0.25,
            -0.125,
        ],
        dtype=dtype,
    )
    changed = y.copy()
    changed_index = 8
    changed[changed_index] = _power_of_two(dtype, 90 if dtype is np.float32 else 900)

    baseline = akima_create_helper(x, y, corner_model=corner_model)
    large_dynamic_range = akima_create_helper(x, changed, corner_model=corner_model)

    _assert_component_unchanged_outside_interval(baseline, large_dynamic_range, 'a', changed_index, changed_index)
    _assert_component_unchanged_outside_interval(
        baseline,
        large_dynamic_range,
        'b',
        changed_index - 2,
        changed_index + 2,
    )
    for component in ('c', 'd'):
        _assert_component_unchanged_outside_interval(
            baseline,
            large_dynamic_range,
            component,
            changed_index - 3,
            changed_index + 2,
        )

    assert large_dynamic_range.a[changed_index] != baseline.a[changed_index]
    b_kernel = _interval_slice(large_dynamic_range.b.size, changed_index - 2, changed_index + 2)
    cd_kernel = _interval_slice(large_dynamic_range.c.size, changed_index - 3, changed_index + 2)
    assert not np.array_equal(large_dynamic_range.b[b_kernel], baseline.b[b_kernel])
    assert not np.array_equal(large_dynamic_range.c[cd_kernel], baseline.c[cd_kernel])
    assert not np.array_equal(large_dynamic_range.d[cd_kernel], baseline.d[cd_kernel])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_overflowing_control_value_differences_are_confined_to_strict_coefficient_kernel(
    dtype: type[np.floating],
) -> None:
    x = np.arange(16, dtype=dtype)
    y = (np.sin(np.arange(16)) * 0.125).astype(dtype)
    baseline = akima_create_helper(x, y)

    changed = y.copy()
    changed_index = 7
    changed[changed_index] = np.finfo(dtype).max
    changed[changed_index + 1] = -np.finfo(dtype).max
    overflowed = akima_create_helper(x, changed)

    _assert_component_unchanged_outside_interval(baseline, overflowed, 'a', changed_index, changed_index + 1)
    _assert_component_unchanged_outside_interval(baseline, overflowed, 'b', changed_index - 2, changed_index + 3)
    for component in ('c', 'd'):
        _assert_component_unchanged_outside_interval(
            baseline,
            overflowed,
            component,
            changed_index - 3,
            changed_index + 3,
        )

    b_kernel = _interval_slice(overflowed.b.size, changed_index - 2, changed_index + 3)
    cd_kernel = _interval_slice(overflowed.c.size, changed_index - 3, changed_index + 3)
    assert np.any(~np.isfinite(overflowed.b[b_kernel]))
    assert np.any(~np.isfinite(overflowed.c[cd_kernel]))
    assert np.any(~np.isfinite(overflowed.d[cd_kernel]))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_subnormal_x_spacing_division_overflow_returns_nonfinite_local_coefficients(
    dtype: type[np.floating],
) -> None:
    step = np.nextafter(dtype(0), dtype(1), dtype=dtype)
    x = np.array([-3.0, -2.0, -1.0, 0.0, step, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    y = np.array([0.0, 0.5, -0.25, 0.0, 1.0, 0.25, -0.5, 0.75, 0.1, -0.2], dtype=dtype)

    clean_x = x.copy()
    clean_x[4] = dtype(0.5)
    baseline = akima_create_helper(clean_x, y)
    tiny_interval = akima_create_helper(x, y)

    _assert_component_unchanged_outside_interval(baseline, tiny_interval, 'a', 0, -1)
    _assert_component_unchanged_outside_interval(baseline, tiny_interval, 'b', 2, 6)
    for component in ('c', 'd'):
        _assert_component_unchanged_outside_interval(baseline, tiny_interval, component, 1, 6)

    b_kernel = _interval_slice(tiny_interval.b.size, 2, 6)
    cd_kernel = _interval_slice(tiny_interval.c.size, 1, 6)
    assert np.any(~np.isfinite(tiny_interval.b[b_kernel]))
    assert np.any(~np.isfinite(tiny_interval.c[cd_kernel]))
    assert np.any(~np.isfinite(tiny_interval.d[cd_kernel]))


@pytest.mark.parametrize(
    ('dtype', 'h_exponent', 'slope_exponent'),
    [(np.float32, -50, 125), (np.float64, -500, 1021)],
)
def test_near_overflow_affine_spline_keeps_zero_higher_order_coefficients(
    dtype: type[np.floating],
    h_exponent: int,
    slope_exponent: int,
) -> None:
    x, y, slope = _power_of_two_affine_points(dtype, h_exponent=h_exponent, slope_exponent=slope_exponent)

    spline = akima_create_helper(x, y)

    expected_slope = np.full(spline.b.shape, float(slope), dtype=spline.b.dtype)
    _assert_same_float_values(spline.b, expected_slope, maxulp=4)
    np.testing.assert_array_equal(spline.c, np.zeros_like(spline.c))
    np.testing.assert_array_equal(spline.d, np.zeros_like(spline.d))


@pytest.mark.parametrize(
    ('dtype', 'h_exponent', 'slope_exponent'),
    [(np.float32, -50, 127), (np.float64, -500, 1023)],
)
def test_actual_multiplier_overflow_in_affine_spline_produces_ieee_nonfinite_coefficients(
    dtype: type[np.floating],
    h_exponent: int,
    slope_exponent: int,
) -> None:
    x, y, _ = _power_of_two_affine_points(dtype, h_exponent=h_exponent, slope_exponent=slope_exponent)

    spline = akima_create_helper(x, y)

    assert np.all(np.isfinite(spline.a))
    assert np.any(~np.isfinite(spline.b))
    assert np.any(~np.isfinite(spline.c))
    assert np.any(~np.isfinite(spline.d))


def test_single_knot_eval_preserves_ieee_zero_times_infinity() -> None:
    spline = SplineCoeffs(
        x=np.array([0.0, 1.0]),
        y=np.array([1.0, 2.0]),
        n_control=2,
        a=np.array([1.0]),
        b=np.array([np.inf]),
        c=np.array([0.0]),
        d=np.array([0.0]),
    )

    scalar_at_knot = spline_single_knot_eval(0.0, spline, 0)
    vector_values = spline_single_knot_eval(np.array([0.0, 1.0]), spline, 0)

    assert np.isnan(scalar_at_knot)
    assert np.isnan(vector_values[0])
    assert vector_values[1] == np.inf


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_array_output_precision_matches_input_precision(dtype: type[np.floating]) -> None:
    x, y = _affine_control_points(dtype)
    spline = AkimaSpline(x, y)
    xint = np.array([0.5, 1.5], dtype=dtype)

    coeff_dtypes = {spline.spline.a.dtype, spline.spline.b.dtype, spline.spline.c.dtype, spline.spline.d.dtype}
    assert coeff_dtypes == {np.dtype(dtype)}
    assert spline(xint).dtype == np.dtype(dtype)
    assert cubic_call_vector(xint, spline.spline, 3).dtype == np.dtype(dtype)
    assert cubic_call_vector_linear(xint, spline.spline, 3).dtype == np.dtype(dtype)
    assert spline_single_knot_eval(xint, spline.spline, 0).dtype == np.dtype(dtype)


def test_float32_scalar_output_precision_matches_input_precision() -> None:
    x, y = _affine_control_points(np.float32)
    spline = AkimaSpline(x, y)

    assert isinstance(cubic_call(np.float32(0.5), spline.spline, 3), np.float32)
    assert isinstance(spline(np.float32(0.5)), np.float32)
