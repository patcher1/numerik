# -*- encoding: utf-8 -*-
import numpy as np
import numpy.linalg
import scipy
import scipy.linalg
import numpy.linalg
import matplotlib.pyplot as plt
from helpers import evalchebexp, chebexp, clenshaw
import numpy.random

# Polynominterpolation

def simple(f, a, b, N):
    """
    Interpolation einer Funktion mit aquidistanten Stützstellen

    @param {callable} f          - Funktion zur Approximation
    @param [{int}, {int}] [a, b] - Intervall zur Interpolation
    @param {int} N               - Grad des Polynoms

    @return {callable}  - Interpolations-Polynom
    """ 

    x = np.linspace(a, b, N+1)
    c = my_polyfit(x, f(x), N)
    return lambda v: np.polyval(c, v)

def chebychev(f, a, b, N):
    """
    Interpolation einer Funktion mit Chebychev-Stützstellen direkt evaluiert

    @param {callable} f          - Funktion zur Approximation
    @param [{int}, {int}] [a, b] - Intervall zur Interpolation
    @param {int} N               - Grad des Polynoms

    @return {callable}  - Interpolations-Polynom
    """ 

    x = a + 0.5*(b-a)*(1.0 + np.cos(0.5*np.pi*(2*np.arange(N+1)+1)/(N+1.0))) #Cheby Knoten
    c = my_polyfit(x, f(x), N)
    return lambda v: np.polyval(c, v)

def cheby_with_evalchebexp(f, a, b, N):
    """
    Interpolation einer Funktion mit Chebychev-Stützstellen evaluiert mit evalchebexp

    @param {callable} f          - Funktion zur Approximation
    @param [{int}, {int}] [a, b] - Intervall zur Interpolation
    @param {int} N               - Grad des Polynoms

    @return {array} [x, y]  - x, y Koordinaten
    """     
    x = a + 0.5*(b-a)*(1.0 + np.cos(0.5*np.pi*(2*np.arange(N+1)+1)/(N+1.0))) #Cheby Knoten
    c = chebexp(f(x))
    return evalchebexp(c, N)

def cheby_with_clenshaw(f, a, b, N):
    """
    Interpolation einer Funktion mit Chebychev-Stützstellen evaluiert mit dem Clenshaw-Algorithmus

    @param {callable} f          - Funktion zur Approximation
    @param [{int}, {int}] [a, b] - Intervall zur Interpolation
    @param {int} N               - Grad des Polynoms

    @return {array} [x, y]  - x, y Koordinaten
    """         
    x = a + 0.5*(b-a)*(1.0 + np.cos(0.5*np.pi*(2*np.arange(N+1)+1)/(N+1.0))) #Cheby Knoten
    c = chebexp(f(x))
    return x, clenshaw(c, x)

def my_polyfit(x, y, N):
    return numpy.linalg.solve(numpy.vander(x), y)


def barycentric_weights(x):
    """
    Berechne die baryzentrischen Gewichte zu den Stützstellen x

    @param {array} x    - Stützstellen
    @return {array} w   - baryzentrische Gewichte
    """
    w = np.ones_like(x).astype(float)
    for j, xj in enumerate(x):
        for k, xk in enumerate(x):
            if (k != j): w[j] /= (xj - xk)
    return w

def barycentric(x, fx):
    """
    Baue das Lagrangesche Interpolationspolynom

    @param {array} x    - Stützstellen
    @param {array} fx   - Werte an den Stützstellen
    @return {callable}  - Interpolationspolynom (Achtung beim auswerten kein Broadcasting)
                          Auswerten: px = np.array([p(x) for x in X])
                          Nicht so: px = p(X), falls X ein Array
    """    
    w = barycentric_weights(x)
    return lambda t: np.dot(w/(t - x), fx)/np.sum(w/(t - x))

# Trigonometrische Interpolation
#TODO


#########
# Tests #
#########

if __name__ == '__main__':

    
    #sint = np.sin(t)
    #sp = np.fft.fft(sint)
    #freq = np.fft.fftfreq(t.shape[-1])
    #plt.plot(freq, sp.real, freq, sp.imag)

    """
    x = np.sin(np.arange(64))
    n = np.size(x)
    ft = np.fft.fftshift(np.fft.fft(x))
    m = np.size(np.fft.fft(x))
    #plt.plot(x, np.fft.fft(x))
    plt.vlines(np.arange(n), np.zeros(n), x, colors='g')
    plt.vlines(np.arange(-m/2, m/2), np.zeros(m), ft, colors='r')
    plt.show()
    """
    """
    x = np.arange(64)
    y = np.sin(2*np.pi*x/64) + np.sin(7*2*np.pi*x/64)
    y = y + np.random.randn(len(x)) #distortion
    c = np.fft.fft(y)
    p = np.abs(c)**2
    p /= 64.
    plt.plot(x,y,'-+')
    plt.show()
    plt.bar(x[:32],p[:32])
    plt.show()
    """

    """

    f = lambda x: 1.0/(1.0 + x**2)
    g = lambda x: 0.5*f(5*x) + f(5*x+5)

    l = -1.0
    r = 1.0
    x = np.linspace(l, r, 10**3)
    N = 21

    p1 = simple(g, l, r, N)
    y1 = p1(x)
    p2 = chebychev(g, l, r, N)
    y2 = p2(x)
    x3, y3 = cheby_with_evalchebexp(g, l, r, N)
    x4, y4 = cheby_with_clenshaw(g, l, r, N)


    plt.figure()
    plt.plot(x, g(x), label=r"Original")
    plt.plot(x, y1, label=r"Äquidistant")
    plt.plot(x, y2, label=r"Chebychev, direkt")
    plt.plot(x3, y3, label=r"Chebychev-Abzissen")
    plt.plot(x4, y4, label=r"Clenshaw")
    plt.grid(True)
    plt.xlim(l, r)
    plt.ylim(-2, 2)
    plt.legend(loc='lower center', framealpha=0.8)
    plt.title('Aequidistant interpolation points')
    plt.show()

    """

    # Bsp: Lagrange Basis aus Prüfung H14 A3

    # Polynom aus der Aufgabenstellung
    f = lambda t: 1 + (1 + 2 ** 20) * t ** 4 + t ** 8
    # Stuetzstellen
    t = np.linspace(-4.0, 4.0, 36)
    y = f(t)
    # Auswertungspunkte
    x = np.linspace(t[0], t[-1], 1000)
    # Exakte Funktionswerte
    fvs = f(x)


    # a) Polynominterpolation mit polyfit und polyval
    #coeffs_polyfit = zeros_like(y)
    coeffs_polyfit = np.polyfit(x, fvs, 8)

    ######################################################################
    #                                                                    #
    # TODO: Berechnen Sie hier die Polynom-Koeffizienten mittels polyfit #
    #                                                                    #
    ######################################################################

    print("Coefficients of the polynomial are:")
    print(coeffs_polyfit)
    p = np.polyval(coeffs_polyfit, x)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x, fvs, "r")
    plt.plot(x, p, "b--")
    plt.plot(t, f(t), "k*")
    plt.grid(True)
    plt.legend(["Funktion $f$", 'Interpolationspolynom'], loc="best")
    plt.xlabel("x")

    plt.subplot(2,1,2)
    plt.semilogy(x, abs(fvs - p), 'b--')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("Fehler")
    plt.title('polyval(polyfit)')
    plt.tight_layout()
    plt.savefig('polyfit_polyval.png')


    # b) Baryzentrische Formel
    plt.figure()
    plt.subplot(2,1,1)
    barweight = barycentric_weights(t)

    p = barycentric(t, y)
    px = np.array([p(X) for X in x])

    plt.plot(x, fvs, "r")
    plt.plot(x, px, "b--")
    plt.plot(t, f(t), "k*")
    plt.grid(True)
    plt.legend(["Funktion $f$", 'Interpolationspolynom'], loc="best")
    plt.xlabel("x")

    plt.subplot(2,1,2)
    plt.semilogy(x, abs(fvs - px), 'b--')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("Fehler")
    plt.title('Baryzentrische Koordinaten')
    plt.tight_layout()
    plt.savefig('polyfit_barycentric.png')