# -*- encoding: utf-8 -*-
import numpy as np
import numpy.linalg
import scipy
import scipy.linalg
import numpy.linalg

##################################
# Hilfsfunktionen aus dem Skript #
##################################

# 6.6.1
def chebexp(y):
    """Efficiently compute coefficients $\alpha_j$ in the Chebychev expansion
    $p = \sum_{j=0}^{n} \alpha_j T_j$ of $p \in \Cp_n$ based on values $y_k$,
    $k=0,\ldots,n$, in Chebychev nodes $t_k$, $k=0,\ldots,n$. These values are
    passed in the row vector $y$.
    """
    # degree of polynomial
    n = y.shape[0] -1

    # create r.h.s vector $z$ by wrapping and componentwise scaling
    t = np.arange(0, 2*n+2)
    z = np.exp(-np.pi*1.0j*n/(n+1.0)*t) * np.hstack([y,y[::-1]])

    # Solve linear system for $\zeta$ with effort $O(n\log n)$
    c = scipy.ifft(z)

    # recover $\gamma_j$
    t = np.arange(-n, n+2)
    b = np.real(np.exp(0.5j*np.pi/(n+1.0)*t) * c)

    # recover coefficients $a$ of Chebyshev expansion
    a = np.hstack([ b[n], 2*b[n+1:2*n+1] ])

    return a

# 6.6.3
def evalchebexp(a,N):
    n = a.shape[0]-1
    c = np.zeros(2*N, dtype=complex)
    c[:n+1] = 1.*a
    c[0] *= 2
    c[n] *= 2
    c[2*N-n+1:] = c[n-1:0:-1]
    z = scipy.ifft(c)*2*N
    y = 1.*z[:N+1]
    x = np.cos(np.arange(N+1)*np.pi/N)
    return x,y

# 5.4.25
def clenshaw(a, x):
    # Grad n des Polynoms
    n = a.shape[0] - 1
    d = np.tile( np.reshape(a,n+1,1), (x.shape[0], 1) )
    d = d.T
    for j in range(n, 1, -1):
        d[j-1,:] = d[j-1,:] + 2.0*x*d[j,:]
        d[j-2,:] = d[j-2,:] - d[j,:]
    y = d[0,:] + x*d[1,:]
    return y


# 7.3.3
def gaussquad(n):
    """Berechnet die Knoten und Gewichte für Gauss-Legendre Quadratur.
    """
    i = np.arange(n-1)
    b = (i+1.) / np.sqrt(4.*(i+1)**2 - 1.)
    J = np.diag(b, -1) + np.diag(b, 1)
    x, ev = np.linalg.eigh(J)
    w = 2 * ev[0,:]**2
    return x, w