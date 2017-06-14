import numpy as np
import numpy.linalg
import scipy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
from ode_solvers import *



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
