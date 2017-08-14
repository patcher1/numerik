# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from helpers import gaussquad

def trapezoid(f, a, b, N):
    """Zusammengesetzte Trapezregel in 1d.

    Input:
           f : Funktion f(x) welche integiert werden soll.
        a, b : untere/obere Grenze des Integrals.
           N : Anzahl Teilintervalle in der zusammengesetzten Regel.
    """
    # Für geschlossene Quadraturformeln kann man mit einem
    # Trick die Anzahlfunktionsaufrufe reduzieren.
    # Die Beobachtung ist, dass zwei benachbarte Teilintervalle einen
    # gemeinsamen Knoten haben. Anstatt, `f` zweimal für diesen Knoten
    # auszurechnen, summiert man einfach die Gewichte:
    x, h = np.linspace(a, b, N+1, retstep=True)
    return h*(.5*f(a) + sum(f(m) for m in x[1:-1]) + .5*f(b))

def simpson(f, a, b, N):
    """Zusammengesetzte Simpsonregel in 1d.

    Input:
           f : Funktion f(x) welche integiert werden soll.
        a, b : untere/obere Grenze des Integrals.
           N : Anzahl Teilintervalle in der zusammengesetzten Regel.
    """
    x, h = np.linspace(a, b, N+1, retstep=True)
    xm = .5*(x[1:] + x[:-1])
    return h/6.0 * (f(a) + 4.0*sum(f(m) for m in xm) + 2.0*sum(f(z) for z in x[1:-1]) + f(b))

def midpoint(f, a, b, N):
    """Zusammengesetzte Mittelpunktsregel in 1d.

    Input:
           f : Funktion f(x) welche integiert werden soll.
        a, b : untere/obere Grenze des Integrals.
           N : Anzahl Teilintervalle in der zusammengesetzten Regel.
    """
    x, h = np.linspace(a, b, N+1, retstep=True)
    return h*sum(f(m) for m in .5*(x[1:] + x[:-1]))

def two_dim(rule, f, a, b, Nx, c, d, Ny):
    F = lambda y: rule(lambda x: f(x, y), a, b, Nx)
    return rule(F, c, d, Ny) 

def gauss_legendre(f, a, b, n):
    """ Gauss-Legendre Quadratur (nicht zusammengesetzt).

    f:     Funktion f(x)
    a, b:  Obere und untere Grenze des Intervalls.
    n:     Anzahl Quadraturpunkte.
    """

    xs, ws = gaussquad(n) #7.3.3
    x = a + (xs + 1.)*(b-a)*.5
    return np.sum(.5*(b-a)*ws*f(x))

def composite_legendre(f, a, b, N, n=100):
    """ Zusammengesetzte Gauss-Legendre Quadratur.

    f:     Funktion f(x)
    a, b:  Obere und untere Grenze des Intervalls.
    N:     Anzahl Teilintervalle.
    n:     Anzahl Quadraturpunkte pro Teilintervall.
    """
    dx = (b-a)/N
    return sum(gauss_legendre(f, a + i*dx, a + (i+1)*dx, n) for i in range(N))

def mcquad(f, a, b, N, d=1):
    """Berechnet das `d`-dimensionale Integral von `f`.

    Input:
        f : Funktion welche integriert werden soll. Das Argument von `f` ist ein d-dim Array.
     a, b : untere/obere Grenzen des Integrals. Bei mehreren Dimensionen können beide d-dimensinale Arrays sein.
        d : Anzahl Dimensionen.
        N : Anzahl Zufallsvektoren.

    Output:
     mean : Approximation.
    sigma : Standardabweichung. @see 7.6.9
    """

    faccepts = 1 # 0: column vectors ([[a],[b]]) or 1: row vectors ([a,b])
    x = a + (b-a)*np.random.rand(N, d)
    fx = np.array([f(m if faccepts == 1 else m.reshape(d,1)) for m in x])
    vol = np.abs(np.prod(b-a))
    mean = vol*np.sum(fx)/float(N)
    return mean, np.sqrt((vol**2*np.sum(fx**2)/N - mean**2)/(N-1.))


'''
    Adaptive integration. Uses a worse and a better integration method (for example: Simpson and Trapezoid)
    to approximate where smaller steps are needed. Of course in the end, the integral will be calculated with the better method.

    @param {callable} f         - function to integrate
    @param {float} a            - lower bound
    @param {float} b            - upper bound
    @param {int} N              - initial number of intervalls
    @param {callable} psilow    - a quadrature method
    @param {callable} psihigh   - a better quadrature method
    @param {float} rtol         - relative tolerance
    @param {float} atol         - absolute tolerance
    @param {array} ev           - initial evaluation points for approximating the integral

    @return I                   - approximated Integral
    @return ev                  - evaluation points
'''
def adaptQuad(f, a, b, N, psilow, psihigh, rtol=1e-5, atol=1e-5, ev=None):
    ev = np.linspace(a, b, N) if ev == None else ev
    Il = np.zeros(ev.size - 1)
    Ih = np.zeros(ev.size - 1)
    for i in range(ev.size - 1):
        Il[i] = psilow(f, ev[i], ev[i+1], 1)
        Ih[i] = psihigh(f, ev[i], ev[i+1], 1)

    I = np.sum(Ih)                         #We take the better approximation as the Integral
    eloc = np.abs(Ih - Il)
    eglob = np.sum(eloc)

    if eglob > rtol*np.abs(I) and eglob > atol:
        midpoints = .5*(ev[:-1] + ev[1:])
        refcells = np.nonzero(eloc > .9*np.sum(eloc)/np.size(eloc))[0]
        ev = np.sort(np.append(ev, midpoints[refcells]))
        I, ev = adaptQuad(f, a, b, N, psilow, psihigh, rtol, atol, ev)
    return I, ev

#########
# Tests #
#########

def f1(x):
    return 1.0 / (1.0 + 5.0*x**2)

def f2(x):
    return np.sqrt(x)

def f2d(x, y):
    return 1.0 / np.sqrt((x - 2.0) ** 2 + (y - 2.0) ** 2)

def f3(x):
    return 1/(10**(-4)+x**2) #function with extremely large values around 0

if __name__ == "__main__":
    # Testfunktionen
    If1ex = np.arctan(np.sqrt(5.0)) / np.sqrt(5.0)
    If2ex = 2.0 / 3.0
    IF2dex = 1.449394876268660

    a, b = -1, 1
    N = 10
    ev = np.linspace(a, b, N)
    I, ev = adaptQuad(f3, a, b, N, trapezoid, simpson, 1e-3, 1e-3)

    print("Integral: adapt", I)
    print("Integral Simps: ", simpson(f3, a, b, len(ev))) 
    print("EV-Points: ", ev)

    plt.figure()
    x = np.linspace(a, b, 100)
    plt.plot(x, f3(x), 'b')
    plt.plot(ev, f3(ev), 'r^')
    plt.show()

    #print(composite_legendre(f1, 0.0,  1.0, 128))
    #print(If1ex)
    #print(If1ex-composite_legendre(f1, 0.0,  1.0, 128))
