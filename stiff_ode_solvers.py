import numpy as np
import numpy.linalg
import scipy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
from ode_solvers import *

def exp_euler(f, Df, y0, t0, T, N):
    """
    Exponentielles Euler Verfahren

    @param {callable} f         - function
    @param {callable} Df        - Jacobimatrix of f
    @param {float} t0           - Anfangszeit
    @param {float} T            - Endzeit
    @param {ndarray|float} y0   - Anfangswert
    @param {int} N              - Anzahl Iterationen

    @return {array} t     - Zeiten
    @return {ndarray} y   - Orte
    """

    t, h = np.linspace(t0, T, N, retstep=True)
    y0 = np.atleast_1d(y0)
    y = np.zeros((N, y0.shape[0]))
    y[0,:] = y0

    for k in range(N-1):
        J = Df(y[k,:])
        x = np.linalg.solve(J, f(y[k,:]))
        y[k+1,:] = y[k,:] + np.dot(scipy.linalg.expm(h*J) - np.eye(np.size(y0)), x)
    return t, y

if __name__ == '__main__':

    rhs = lambda t, y: -4*y*(y - 2)
    rhs = lambda t, y: 5*y*(1 - y)
    y0 = 0.1
    t0 = 0
    T = 5
    Ng = int(T/0.2)
    Nr = int(T/0.52)

    # Butcher scheme for Radau
    Brad = np.array([
        [ 1/3,   5/12, -1/12 ],
        [ 1,     3/4,   1/4  ],
        #------|--------------
        [ 0.0,   3/4,   1/4  ]
    ])

    t1, y1 = runge_kutta(rhs, y0, t0, T, Ng, Brad)
    t2, y2 = runge_kutta(rhs, y0, t0, T, Nr, Brad)

    plt.plot(t1, y1, 'g')
    plt.plot(t2, y2, 'r')
    plt.show()
