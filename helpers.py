# -*- encoding: utf-8 -*-
import numpy as np
import numpy.linalg
import scipy
import scipy.linalg
import numpy.linalg
import numpy.fft
import numpy.linalg
from numpy import zeros, flipud, double, array

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

# 6.4.3
def evaliptrig(y,N):
    n = len(y)
    if (n%2) == 0:
        c = np.fft.ifft(y)
        a = np.zeros(N, dtype=complex)
        a[:n/2] = c[:n/2]
        a[N-n/2:] = c[n/2:]
        v = np.fft.fft(a);
        return v
    else: raise TypeError('odd length')

# 6.4.2
def trigpolycoeff(t, y):
    N = y.shape[0]
    if N%2 == 1:
        n = (N-1.)/2.
        M = np.exp(2*np.pi*1j*np.outer(t[:-1],np.arange(-n,n)))
    else:
        n = N/2.
        M = np.exp(2*np.pi*1j*np.outer(t[:-1],np.arange(-n+1,n)))
    c = np.linalg.solve(M, y[:-1])
    return c

# 8.4.9
def splitting_parameters(method):
    u"""
    :param method: A string specifying the method for time integration.
    :return: Two arrays :math:`a` and :math:`b`.

    ====== ===== =========== =========
    Method Order Authors     Reference
    ====== ===== =========== =========
    LT     1     Lie/Trotter [1]_, [3]_ page 42, equation 5.2
    S2     2     Strang      [2]_, [3]_ page 42, equation 5.3
    SS     2     Strang      [2]_, [3]_ page 42, equation 5.3
    PRKS6  4     Blanes/Moan [4]_ page 318, table 2, 'S6'
    BM42   4     Blanes/Moan [4]_ page 318, table 3, 'SRKNb6'
    Y4     4     Yoshida     [5]_, [3]_ page 40, equation 4.4
    Y61    6     Yoshida     [5]_, [3]_ page 144, equation 3.11
    BM63   6     Blanes/Moan [4]_ page 318, table 3, 'SRKNa14'
    KL6    6     Kahan/Li    [6]_, [3]_ page 144, equation 3.12
    KL8    8     Kahan/Li    [6]_, [3]_ page 145, equation 3.14
    L42    (4,2) McLachlan   [7]_ page 6
    L84    (8,4) McLachlan   [7]_ page 8
    ====== ===== =========== =========

    .. [1] H.F. Trotter, "On the product of semi-groups of operators",
            Proc. Am. Math. Soc.1O (1959) 545-551.

    .. [2] G. Strang, "On the construction and comparison of difference schemes",
            SIAM J. Numer. Anal. 5 (1968) 506-517.

    .. [3] E. Hairer, C. Lubich, and G. Wanner, "Geometric Numerical Integration -
            Structure-Preserving Algorithms for Ordinary Differential Equations",
            Springer-Verlag, New York, 2002.

    .. [4] S. Blanes and P.C. Moan, "Practical Symplectic Partitioned
            Runge-Kutta and Runge-Kutta-Nystrom Methods", J. Computational and
            Applied Mathematics, Volume 142, Issue 2, (2002) 313-330.

    .. [5] H. Yoshida, "Construction of higher order symplectic integrators",
            Phys. Lett. A 150 (1990) 262-268.

    .. [6] W. Kahan and  R.-c. Li, "Composition constants for raising the orders
            of unconventional schemes for ordinary differential equations",
            Math. Comput. 66 (1997) 1089-1099.

    .. [7] R.I. McLachlan, "Composition methods in the presence of small parameters",
            BIT Numerical Mathematics, Volume 35, Issue 2, (1995) 258-268.
    """
    if method == "LT":
        s = 1
        a = zeros(s)
        b = zeros(s)
        a[0] = 1.0
        b[0] = 1.0
    elif method == "S2":
        s = 2
        a = zeros(s)
        b = zeros(s)
        a[1] = 1.0
        b[0] = 0.5
        b[1] = 0.5
    elif method == "SS":
        s = 2
        a = zeros(s)
        b = zeros(s)
        a[0] = 0.5
        a[1] = 0.5
        b[0] = 1.0
    elif method == "L42":
        # Pattern ABA and m = s = 2
        s = 3
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.5773502691896258
        a[0] = 0.5*(1.-a[1])
        a[2] = a[0]
        b[0] = 0.5
        b[1] = 0.5
    elif method == "BM42":
        s = 7
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.245298957184271
        a[2] = 0.604872665711080
        a[3] = 0.5 - a[:3].sum()
        a[4:] = flipud(a[1:4])
        b[0] = 0.0829844064174052
        b[1] = 0.396309801498368
        b[2] = -0.0390563049223486
        b[3] = 1.0 - 2.0*b[:3].sum()
        b[4:] = flipud(b[:3])
    elif method == "Y4":
        s = 4
        a = zeros(s)
        b = zeros(s)
        pp = 3.0
        theta = 1.0/(2.0-2**(1.0/pp))
        vi = -2**(1.0/pp)*theta
        a[0] = 0.0
        a[1] = theta
        a[2] = vi
        a[3] = theta
        b[0] = 0.5*theta
        b[1] = 0.5*(vi+theta)
        b[2:] = flipud(b[:2])
    elif method == "Y61":
        s = 8
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.78451361047755726381949763
        a[2] = 0.23557321335935813368479318
        a[3] = -1.17767998417887100694641568
        a[4] = 1.0 - 2.0*a[1:4].sum()
        a[5:] = flipud(a[1:4])
        b[0] = 0.5*a[1]
        b[1] = 0.5*a[1:3].sum()
        b[2] = 0.5*a[2:4].sum()
        b[3] = 0.5*(1-4*b[1]-a[3])
        b[4:] = flipud(b[0:4])
    elif method == "BM63":
        s = 15
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.09171915262446165
        a[2] = 0.183983170005006
        a[3] = -0.05653436583288827
        a[4] = 0.004914688774712854
        a[5] = 0.143761127168358
        a[6] = 0.328567693746804
        a[7] = 0.5 - a[:7].sum()
        a[8:] = flipud(a[1:8])
        b[0] = 0.0378593198406116
        b[1] = 0.102635633102435
        b[2] = -0.0258678882665587
        b[3] = 0.314241403071447
        b[4] = -0.130144459517415
        b[5] = 0.106417700369543
        b[6] = -0.00879424312851058
        b[7] = 1.0 - 2.0*b[:7].sum()
        b[8:] = flipud(b[:7])
    elif method =="PRKS6":
        s = 7
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.209515106613362
        a[2] = -0.143851773179818
        a[3] = 0.5 - a[:3].sum()
        a[4:] = flipud(a[1:4])
        b[0] = 0.0792036964311957
        b[1] = 0.353172906049774
        b[2] = -0.0420650803577195
        b[3] = 1 - 2*b[:3].sum()
        b[4:] = flipud(b[:3])
    elif method == "KL6":
        s = 10
        a = zeros(s)
        b = zeros(s)
        a[1] = 0.39216144400731413927925056
        a[2] = 0.33259913678935943859974864
        a[3] = -0.70624617255763935980996482
        a[4] = 0.08221359629355080023149045
        a[5] = 1.0 - 2.0*a[1:5].sum()
        a[6:] = flipud(a[1:5])
        b[0] = 0.5*a[1]
        b[1] = 0.5*a[1:3].sum()
        b[2] = 0.5*a[2:4].sum()
        b[3] = 0.5*a[3:5].sum()
        b[4] = 0.5*(1-2*a[1:4].sum()-a[4])
        b[5:] = flipud(b[0:5])
    elif method == "KL8":
        s = 18
        a = zeros(s)
        b = zeros(s)
        a[0] = 0.0
        a[1] = 0.13020248308889008087881763
        a[2] = 0.56116298177510838456196441
        a[3] = -0.38947496264484728640807860
        a[4] = 0.15884190655515560089621075
        a[5] = -0.39590389413323757733623154
        a[6] = 0.18453964097831570709183254
        a[7] = 0.25837438768632204729397911
        a[8] = 0.29501172360931029887096624
        a[9] = -0.60550853383003451169892108
        a[10:] = flipud(a[1:9])
        b[0:-1] = 0.5*(a[:-1]+a[1:])
        b[-1] = 1.*b[0]
    elif method == "L84":
        # Pattern ABA
        s = 6
        a = zeros(s)
        b = zeros(s)
        a[0] = 0.07534696026989288842
        a[1] = 0.51791685468825678230
        a[2] = -0.09326381495814967072
        b[0] = 0.19022593937367661925
        b[1] = 0.84652407044352625706
        b[2] = -1.07350001963440575260
        a[3:] = flipud(a[0:3])
        b[3:-1] = flipud(b[0:2])
    else:
        raise NotImplementedError("Unknown method: " + method)

    return a, b

