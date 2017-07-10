# -*- encoding: utf-8 -*-
import numpy as np
import numpy.linalg
import scipy
import scipy.linalg

def newton(xk, f, Df, tol = 10**-14, maxit = 10000):
    """
    Newtonapproximation in n-dim (n>=1)

    @param {callable} f       - function that takes at least one argument
    @param {callable} Df      - Jacobianmatrix of f
    @param {ndarray|float} xk - start values
    @param {float} tol        - tolerance (default: 10^-14)
    @param {int} maxit        - max iterations (default: 10'000)

    @return {ndarray|float} - approximated solution for f(xk) = 0 within a tolerance
    @return {int}           - used iterations
    @return {ndarray}       - values of xk in each step
    """

    X = []
    for i in range(1, maxit):
        sk = np.linalg.solve(np.atleast_2d(Df(xk)), np.atleast_1d(f(xk)))
        xk -= sk
        if np.linalg.norm(sk) <= tol*np.linalg.norm(xk):
            break
        X.append(xk)

    return xk[0] if np.size(xk) == 1 else xk, i, np.array(X)

def secant(f, x0, x1, abstol=1e-15, maxit=100):
    r"""Secant iteration for solving f(x) = 0 and obtaining x*.

    @param {callable} f       - function that takes at least one argument
    @param {float} x0         - first start value
    @param {float} x1         - second start value
    @param {float} abstol     - absolute tolerance (default: 10^-15)
    @param {int} maxit        - max iterations (default: 100)

    @return {array}           - values of xk in each step
    """
    x = [x0, x1]
    for i in range(maxit):
        qk = (f(x[-1]) - f(x[-2]))/(x[-1] - x[-2])
        x.append(x[-1] - f(x[-1])/qk)
        if abs(x[-1] - x[-2]) < abstol:
            break
    return array(x)

#########
# Tests #
#########

if __name__ == '__main__':

    def f1(x):
        return x

    def Df1(x):
        return 1

    def f2(x):
        return x

    def Df2(x):
        return np.array([[1., .0], [.0, 1.]])

    def f3(x):
        return np.array([np.exp(x[0]*x[1]) + x[0]**2 + x[1] - 1.2,
                       x[0]**2 + x[1]**2 + x[0] - 0.55])

    def Df3(x):
        return np.array([[x[1]*np.exp(x[0]*x[1]) + 2.0*x[0],  x[0]*np.exp(x[0]*x[1]) + 1.0],
                        [2.0*x[0] + 1.0,  2.0*x[1]]])

    def f4(x):
        return np.array([np.arctan(x)])

    def Df4(x):
        return np.array([1/(x**2 + 1)])

    x1 = 5
    x2 = np.array([5., 3.])
    x3 = np.array([0.6, 0.5])
    x4 = 1.2

    print("1D")
    xstar, i = newton(x1, f1, Df1)[:-1]
    print("Nullstelle:  " + str(xstar))
    print("Iterationen: " + str(i))

    print("2D")
    xstar, i = newton(x2, f2, Df2)[:-1]
    print("Nullstelle:  " + str(xstar))
    print("Iterationen: " + str(i))

    print("2D")
    xstar, i = newton(x3, f3, Df3)[:-1]
    print("Nullstelle:  " + str(xstar))
    print("Iterationen: " + str(i))
