# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

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

def golub_welsch(n):
    """Berechnet die Knoten und Gewichte für Gauss-Legendre Quadratur.
    """
    i = np.arange(n-1)
    b = (i+1.) / np.sqrt(4.*(i+1)**2 - 1.)
    J = np.diag(b, -1) + np.diag(b, 1)
    x, ev = np.linalg.eigh(J)
    w = 2 * ev[0,:]**2
    return x, w

def gauss_legendre(f, a, b, n):
    """ Gauss-Legendre Quadratur (nicht zusammengesetzt).

    f:     Funktion f(x)
    a, b:  Obere und untere Grenze des Intervalls.
    n:     Anzahl Quadraturpunkte.
    """

    xs, ws = golub_welsch(n) #7.3.3
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
    x = a + (b-a)*np.random.rand(N, d)
    fx = np.array([f(m) for m in x])
    mean = np.sum(fx)/N
    return mean, np.sqrt((np.sum(fx**2)/N - mean**2)/(N-1.))

def f1(x):
    return 1.0 / (1.0 + 5.0*x**2)

def f2(x):
    return np.sqrt(x)

def f2d(x, y):
    return 1.0 / np.sqrt((x - 2.0) ** 2 + (y - 2.0) ** 2)


if __name__ == "__main__":
    # Testfunktionen
    If1ex = np.arctan(np.sqrt(5.0)) / np.sqrt(5.0)
    If2ex = 2.0 / 3.0
    IF2dex = 1.449394876268660

    print(composite_legendre(f1, 0.0,  1.0, 128))
    print(If1ex)
    print(If1ex-composite_legendre(f1, 0.0,  1.0, 128))
