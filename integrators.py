# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from helpers import gaussquad

def trapezoid_rule(f, a, b, N):
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

def simpson_rule(f, a, b, N):
    """Zusammengesetzte Simpsonregel in 1d.

    Input:
           f : Funktion f(x) welche integiert werden soll.
        a, b : untere/obere Grenze des Integrals.
           N : Anzahl Teilintervalle in der zusammengesetzten Regel.
    """
    x, h = np.linspace(a, b, N+1, retstep=True)
    xm = .5*(x[1:] + x[:-1])
    return h/6.0 * (f(a) + 4.0*sum(f(m) for m in xm) + 2.0*sum(f(z) for z in x[1:-1]) + f(b))

def mid_point_rule(f, a, b, N):
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
        f : Funktion welche integriert werden soll. Das Argument von
            `f` ist ein d-dim Array.
     a, b : untere/obere Grenze des Integrals.
        d : Anzahl Dimensionen.
        N : Anzahl Zufallsvektoren.

    Output:
     mean : Approximation.
    sigma : Standardabweichung.
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
    @param {array} i_ev         - initial evaluation points for approximating the integral
    @param {float} rtol         - relative tolerance
    @param {float} atol         - absolute tolerance
    @param {callable} quadw     - a quadrature method
    @param {callable} quadb     - a better quadrature method
'''
def adaptiveQuad(f, a, b, i_ev, rtol, atol, quadw, quadb):
    wLocal = np.zeros(i_ev.size -1)
    bLocal = np.zeros(i_ev.size -1)
    for i in range(i_ev.size -1):
        wLocal[i] = quadw(f,i_ev[i],i_ev[i+1],1)
        bLocal[i] = quadb(f,i_ev[i],i_ev[i+1],1)

    I = np.sum(bLocal)                         #We take the better approximation as the Integral
    local_errors = np.abs(bLocal - wLocal)
    global_error = np.sum(local_errors)

    if global_error > rtol*np.abs(I) and global_error > atol:
        midpoints = 0.5*( i_ev[:-1]+i_ev[1:] )
        refcells = np.nonzero( local_errors > 0.9*np.sum(local_errors)/np.size(local_errors) )[0]
        I, i_ev = adaptiveQuad(f,a,b,np.sort(np.append(i_ev,midpoints[refcells])),rtol,atol,quadw,quadb)
    return I, i_ev
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

    a=-1
    b=1
    num_i_ev=10
    i_ev=np.linspace(a, b, num_i_ev)
    I, ev = adaptiveQuad(f3,a,b,i_ev, 10**-5,10**-5,trapezoid_rule,simpson_rule)

    print("Integral: ",I)
    print("EV-Points: ", ev)

    print(composite_legendre(f1, 0.0,  1.0, 128))
    print(If1ex)
    print(If1ex-composite_legendre(f1, 0.0,  1.0, 128))
