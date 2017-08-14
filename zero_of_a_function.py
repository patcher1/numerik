# -*- encoding: utf-8 -*-
import numpy as np
import numpy.linalg
import scipy
import scipy.linalg

def newton(xk, f, Df, damper=1., tol=10**-14, maxit=10000):
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
    for i in range(1, maxit+1):
        sk = np.linalg.solve(np.atleast_2d(Df(xk)), np.atleast_1d(f(xk)))
        lamk = damper(f, Df, xk, sk) if callable(damper) else damper
        xk -= lamk*sk
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

def fixedpoint(phi, xk, tol=1e-15, maxit=100):
    r"""Fixed-point iteration for solving f(x) = 0 with given phi() for phi(x*) = x*.

    @param {callable} phi     - function for iteration phi(x*)=x*
    @param {float} xk         - start value
    @param {float} tol        - tolerance (default: 10^-15)
    @param {int} maxit        - max iterations (default: 100)

    @return {float}         - approximated solution for phi(x*)=x* within a tolerance
    @return {int}           - used iterations
    @return {ndarray}       - values of xk in each step
    """    
    X = []
    for i in range(1, maxit+1):
        xl = xk
        xk = phi(xk)
        if np.linalg.norm(xk - xl) <= tol*np.linalg.norm(xk):
            break
        X.append(xk)
    return xk, i, np.array(X)

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
        return np.arctan(x)

    def Df4(x):
        return 1/(x**2 + 1)

    """
    F(x) = x*np.exp(x) -1
    """
    phi1 = lambda x: np.exp(-x) # lineare Konv
    phi2 = lambda x: (1 + x)/(1 + np.exp(x)) # quad. Konv
    phi3 = lambda x: x + 1 - x*np.exp(x) # Divergenz

    print(fixedpoint(phi1, 0.1)[0])
    print(fixedpoint(phi2, 0.1)[0])
    print(fixedpoint(phi3, 0.1)[0])

    x1 = 5
    x2 = np.array([5., 3.])
    x3 = np.array([0.6, 0.5])
    x4 = 1.5

    damper = lambda f, Df, xk, sk: 1

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

    print("1D arctan(x)")
    xstar, i = newton(x4, f4, Df4)[:-1]
    print("Nullstelle:  " + str(xstar))
    print("Iterationen: " + str(i))

    def damper(f, Df, xk, sk, maxit=100):
        lam = 1
        for i in range(1, maxit):
            s_bar = np.linalg.solve(np.atleast_2d(Df(xk)), np.atleast_1d(f(xk - lam*sk)))
            if np.linalg.norm(s_bar) <= (1 - lam/2)*np.linalg.norm(sk):
                break
            lam /= 2
        return lam

    print("1D arctan(x) damped")
    xstar, i = newton(150, f4, Df4, damper)[:-1]
    print("Nullstelle:  " + str(xstar))
    print("Iterationen: " + str(i))