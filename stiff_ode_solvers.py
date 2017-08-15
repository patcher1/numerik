import numpy as np
import numpy.linalg
import scipy
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
from ode_solvers import *
from scipy.linalg import expm
from numpy.linalg import solve, norm
from numpy import *

def exp_euler_long(f, Df, y0, t0, T, N):
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

    t, h = linspace(t0, T, N, retstep=True)
    y0 = atleast_1d(y0)
    y = zeros((N, y0.shape[0]))
    y[0,:] = y0

    for k in range(N-1):
        J = Df(y[k,:])
        x = solve(J, f(y[k,:]))
        y[k+1,:] = y[k,:] + dot(expm(h*J) - eye(size(y0)), x)
    return t, y

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
    method = lambda rhs, y, t0, dt: exp_euler_step(f, Df, y, t0, dt)
    return integrate(method, None, y0, t0, T, N)

def exp_euler_step(f, Df, y0, t0, dt):
    x = solve(Df(y0), f(y0))
    return y0 + dot(expm(dt*Df(y0)) - eye(size(y0)), x)


if __name__ == '__main__':

    """
    rhs = lambda t, y: -4*y*(y - 2)
    rhs = lambda t, y: 5*y*(1 - y)
    y0 = 0.1
    t0 = 0
    T = 5
    Ng = int(T/0.2)
    Nr = int(T/0.52)

    # Butcher scheme for Radau
    Brad = array([
        [ 1/3,   5/12, -1/12 ],
        [ 1,     3/4,   1/4  ],
        #------|--------------
        [ 0.0,   3/4,   1/4  ]
    ])

    t1, y1 = runge_kutta(rhs, y0, t0, T, Ng, Brad)
    t2, y2 = runge_kutta(rhs, y0, t0, T, Nr, Brad)

    f = lambda x: x
    dF = lambda x: 1

    t3, y3 = exp_euler(rhs, Df, y0, t0, T, Ng)

    plt.plot(t1, y1, 'g')
    plt.plot(t2, y2, 'r')
    plt.show()
    """


    # exp euler Beispiel (S10A3)
    # TODO Jacobi-Matrix
    Df = lambda y: array([
        [ -2.0*y[0]/y[1],  (y[0]/y[1])**2 + log(y[1]) + 1.0 ],
        [ -1.0,            0.0                              ]
    ])

    # TODO Rechte Seite
    f = lambda y: array([ -y[0]**2/y[1] + y[1]*log(y[1]), -y[0] ])

    # TODO Exakte Loesung
    sol = lambda t: array([array([ -cos(t)*exp(sin(t)), exp(sin(t)) ]) for t in t])

    # Anfangswert
    y0 = array([-1, 1])

    to = 0
    te = 6
    nsteps = 20
    #ts, y = expEV(nsteps, to, te, y0, f, Df)
    ts, y = exp_euler(f, Df, y0, to, te, nsteps)

    t_ex = linspace(to, te, 1000)
    y_ex = sol(t_ex)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(ts, y[:,0], 'r-x', label=r'$y[0]$')
    plt.plot(ts, y[:,1], 'g-x', label=r'$y[1]$')
    plt.plot(t_ex, y_ex[:,0],'r', label=r'$y_{ex}[0$]')
    plt.plot(t_ex, y_ex[:,1],'g', label=r'$y_{ex}[1$]')
    plt.legend(loc='best')
    plt.xlabel('$t$')
    plt.ylabel('$y$')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.semilogy( ts, norm(y-sol(ts), axis=1), label=r'$|| y - y_{ex}||$')
    plt.xlabel('$t$')
    plt.ylabel('Abs. Fehler')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('exp_euler.pdf')
    plt.show()

    # Konvergenzordung

    plt.figure()
    Ns = [24, 48, 96, 192, 384]
    hs = zeros_like(Ns).astype(float)  # Gitterweite.
    errors = []  # Fehler.
    e_abs = zeros_like(Ns).astype(float)  # abs. Fehler
    e_rel = zeros_like(Ns).astype(float)  # rel. Fehler

    # TODO Berechnen Sie die Konvergenzordung.
    for i, N in enumerate(Ns):
        t, y = exp_euler(f, Df, y0, to, te, N)
        hs[i] = t[1] - t[0]
        #e_abs[i] = norm(sol(t) - y).max()
        e_abs[i] = norm(y - sol(t), axis=1).max()
        e_rel[i] = norm(e_abs[i]/y_ex[-1])


    # NOTE, die folgenden Zeilen könnten Ihnen beim plotten helfen.
    plt.loglog(hs, e_abs)
    plt.title('Konvergenzplot')
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.xlabel('$h$')
    plt.ylabel('Abs. Fehler')
    plt.savefig('exp_euler_konvergenz.pdf')
    plt.show()

    # Berechnung der Konvergenzraten
    conv_rate = polyfit(log(hs), log(e_abs), 1)[0]
    print('Exponentielles Eulerverfahren konvergiert mit algebraischer Konvergenzordnung: %.2f' % conv_rate)
