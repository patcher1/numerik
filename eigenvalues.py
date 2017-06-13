# -*- encoding: utf-8 -*-
import numpy as np
import numpy.linalg

import scipy
import scipy.linalg

def power_method(A, maxit=100, tol=1e-05):
    """
    @param {ndarray} A      - the matrix (nxn)
    @param {int} maxit      - maximum number of iterations
    @param {float} tol      - tolerance
    @return {array} [ew, i] - highest eigenvalue (ew) and number of needed iterations (i)
    """
    z = np.random.random((A.shape[0], 1))
    z /= np.linalg.norm(z)

    for i in range(maxit):
        w = A.dot(z)
        w /= np.linalg.norm(w)
        if (np.allclose(z, w, tol)): break
        z = w
    return z.T.dot(A).dot(z)[0][0], i

def ipm(A, maxit, tol):
    """
    @param {ndarray} A      - the matrix (nxn)
    @param {int} maxit      - maximum number of iterations
    @param {float} tol      - tolerance
    @return {array} [z, i]  - lowest z and number of needed iterations (i)
    """    
    P, L, U = scipy.linalg.lu(A)
    z = np.random.random((A.shape[0], 1))
    z /= np.linalg.norm(z)

    for i in range(maxit):
        w = np.linalg.solve(U, np.linalg.solve(L, P.T.dot(z)))
        w /= np.linalg.norm(w)
        if (np.allclose(z, w, tol)): break
        z = w
    return z, i

def inverse_power_method(A, maxit=100, tol=1e-05):
    """
    @param {ndarray} A      - the matrix (nxn)
    @param {int} maxit      - maximum number of iterations
    @param {float} tol      - tolerance
    @return {array} [ew, i] - lowest eigenvalue (ew) and number of needed iterations (i)
    """
    z, i = ipm(A, maxit, tol)
    return z.T.dot(A).dot(z)[0][0], i

def shifted_inverse_power_method(A, v, maxit=100, tol=1e-05):
    """
    @param {ndarray} A      - the matrix (nxn)
    @param {float} v        - close number
    @param {int} maxit      - maximum number of iterations
    @param {float} tol      - tolerance
    @return {array} [ew, i] - closest eigenvalue (ew) and number of needed iterations (i)
    """
    z, i = ipm(A - np.eye(A.shape[0]).dot(v), maxit, tol)
    return z.T.dot(A).dot(z)[0][0], i

#########
# Tests #
#########

# Dies hilft mit der Formatierung von Matrizen. Die Anzahl
# Nachkommastellen ist 3 und kann hier ----------------------.
# geändert werden.                                           v
np.set_printoptions(linewidth=200, formatter={'float': '{: 0.3f}'.format})

ew = np.array([100.0, 10.0, 12.0, 0.04, 0.234, 3.92, 72.0, 42.0, 77.0, 32.0])
n = ew.size
Q, _ = np.linalg.qr(np.random.random((n, n)))
A = np.dot(Q.transpose(), np.dot(np.diag(ew), Q))

# TODO Finden sie den grössten Eigenwert vom A mittels Potenzmethode.
print(power_method(A)[0])

# TODO Finden sie den kleinsten Eigenwert vom A mittels inverser Potenzmethode.
print(inverse_power_method(A)[0])

# TODO Finden sie den Eigenwert am nächsten bei 42.
print(shifted_inverse_power_method(A, 40)[0])

