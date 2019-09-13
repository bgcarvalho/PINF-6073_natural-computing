"""

Some functions from:
https://gist.github.com/denis-bz/da697d8bc74fae4598bf

Some functions from:
https://en.wikipedia.org/wiki/Test_functions_for_optimization

"""
import numpy as np


def ackley_function(x):
    """David Ackley, 1987.

    Minimum is ZERO at x = [0, 0, ..., 0]

    https://www.sfu.ca/~ssurjano/ackley.html

    """
    n = x.shape[0]
    return np.squeeze(-20 *
        np.exp(-0.2 * np.sqrt((1 / n) * np.dot(x.T, x))) -
        np.exp((1 / n) * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e
    )


def rastrigin_function(x):
    """
    MIN at f(0, ..., 0) = 0

    :param x:
    :return:
    """
    A = 10.0
    n = x.shape[0]
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def sphere_function(x):
    """

    :param x:
    :return:
    """
    return np.sum(x ** 2)


def rosenbrock_function(x):
    """
    As seen in:
    http://en.wikipedia.org/wiki/Rosenbrock_function
    """
    x0 = x[:-1]
    x1 = x[1:]
    return (
        100 * np.sum((x1 - x0 ** 2) ** 2) +
        np.sum((1 - x0) ** 2)
    )


def beale_function(x, y):
    """
    MIN at x=3.0, y=0.5 => 0.0

    :param x:
    :param y:
    :return:
    """
    return (
        (1.5 - x + x * y) ** 2 +
        (2.25 - x + x * y ** 2) ** 2 +
        (2.625 - x + x * y ** 3) ** 2
    )


def goldstein_price_function(x, y):
    """
    MIN f(0.0, -1.0) => 3.0

    :param x:
    :return:
    """
    return (
        1 + ((x + y + 1) ** 2) *
        (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2) *
        (30 + (2 * x - 3 * y) **2 *
         (18 - 32 * x + 12 * x **2 + 48 * y - 36 * x * y + 27 * y ** 2))
    )


def booth_function(x, y):
    """
    MIN f(1, 3) = 0

    :param x:
    :return:
    """
    return ((x + 2 * y - 7) ** 2) * ((2 * x + y - 5) ** 2)


def bukin_function_6(x, y):
    """
    MIN f(-10, 1) = 0
    :param x:
    :return:
    """
    return (
        100 * np.sqrt(np.fabs(y - 0.01 * x ** 2 )) +
        0.01 * np.fabs(x + 10)
    )


def matyas_function(x, y):
    """
    MIN f(0, 0) = 0
    :param x:
    :return:
    """
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def levi_function_13(x, y):
    """
    MIN f(1, 1) = 0
    :param x:
    :param y:
    :return:
    """
    return (
        (np.sin(3 * np.pi * x) ** 2) +
        ((x + 1) ** 2) * (1 + np.sin(3 * np.pi * x) ** 2) +
        ((y - 1) ** 2) * (1 + np.sin(2 * np.pi * y) ** 2)
    )


def himmelblau_function(x, y):
    """
    MIN_1 f(3, 2) = 0
    MIN_2
    MIN_3
    MIN_4

    :param x:
    :return:
    """
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def three_hump_camel_function(x, y):
    """
    MIN f(0, 0) = 0
    :param x:
    :param y:
    :return:
    """
    return (2 * x ** 2 - 1.05 * x ** 4 + (1/6) * x ** 6 + x * y + y ** 2)


def easom_function(x, y):
    """
    MIN f( pi, pi ) = -1
    :param x:
    :param y:
    :return:
    """
    return -np.cos(x) * np.cos(y) * np.exp(-
        (
            (x - np.pi) ** 2 +
            (y - np.pi) ** 2
        )
    )


def cross_in_tray_function(x):
    """

    :param x:
    :return:
    """
    pass


def eggholder_function(x):
    """

    :param x:
    :return:
    """
    pass


def hoelder_table_function(x):
    """

    :param x:
    :return:
    """
    pass


def mccormick_function(x):
    """

    :param x:
    :return:
    """
    pass


def schaffer_function_2(x):
    """

    :param x:
    :return:
    """
    pass


def schaffer_function_4(x):
    """

    :param x:
    :return:
    """
    pass


def styblinski_tang_function(x):
    """
    MIN f(-2.903534, ... -2.903534) > -39.16617 * N
    :param x:
    :return:
    """
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)


def nesterov(x):
    """
    Nesterov's nonsmooth Chebyshev-Rosenbrock function,
    Overton 2011 variant 2
    """
    x = np.asarray_chkfinite(x)
    x0 = x[:-1]
    x1 = x[1:]
    return (
        np.fabs( 1 - x[0] ) / 4
        + sum( abs( x1 - 2 * np.fabs(x0) + 1 ))
    )


def saddle(x):
    x = np.asarray_chkfinite(x) - 1
    return (
        np.mean( np.diff( x ** 2 ))
        + .5 * np.mean( x ** 4 )
    )


def zakharov(x):
    n = x.shape[0]
    j = np.arange(1.0, n + 1.0)
    s2 = np.sum(j * x ) / 2.0
    return np.sum(x ** 2) + s2 ** 2 + s2 ** 4


def trid( x ):
    x = np.asarray_chkfinite(x)
    return np.sum((x - 1) ** 2) - np.sum(x[:-1] * x[1:])


def sum2( x ):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1.0, n + 1)
    return np.sum(j * x ** 2)


def schwefel(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.fabs(x))))


def powell(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    n4 = ((n + 3) // 4) * 4
    if n < n4:
        x = np.append(x, np.zeros(n4 - n))
    x = x.reshape((4, -1))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
    f = np.empty_like( x )
    f[0] = x[0] + 10 * x[1]
    f[1] = np.sqrt(5) * (x[2] - x[3])
    f[2] = (x[1] - 2 * x[2]) ** 2
    f[3] = np.sqrt(10) * (x[0] - x[3]) ** 2
    return np.sum(f ** 2)
