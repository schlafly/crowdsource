"""
The "cardinal B-spline basis function" for 2nd order.  This function is
C1, and has support only in -0.5 < x < 0.5.

sum_i (c_i * basis1dq(x-g_i)) generates every piecewise C1 quadratic with
knots at g_i, if g_i = i * 1/3.
"""

import numpy


def basis1dq(x):
    ret = numpy.zeros_like(x)
    x = x*3 + 1.5
    m = (x >= 0) & (x < 1)
    ret[m] = x[m]**2/2.
    m = (x >= 1) & (x < 2)
    ret[m] = (-2*x[m]**2. + 6*x[m] - 3)/2.
    m = (x >= 2) & (x < 3)
    ret[m] = (3-x[m])**2./2.
    return ret


def basis2dq(x, y):
    return basis1dq(x)*basis1dq(y)
