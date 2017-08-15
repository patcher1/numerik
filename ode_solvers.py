# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize
import scipy.linalg
import matplotlib.pyplot as plt
from eigenvalues import arnoldi, lanczos, krylov
from helpers import splitting_parameters
from ode45 import ode45

##############
# Solve ODEs #
##############

def integrate(method, f, y0, t0, T, N):
    y = np.empty((N+1,) + np.atleast_1d(y0).shape)
    y[0,...], dt = y0, (T - t0)/N
    for i in range(0, N):
        y[i+1,...] = method(f, y[i,...], t0 + i*dt, dt)
    return np.linspace(t0, T, N+1), y # if the shape of the original y0 must be conserved
    #return np.linspace(t0, T, N+1), y.reshape((y.shape[0], np.size(y0)))

def eE_step(rhs, y0, t0, dt):
    return y0 + dt*rhs(t0, y0)

def eE(rhs, y0, t0, T, N):
    return integrate(eE_step, rhs, y0, t0, T, N)

def iE_step(rhs, y0, t0, dt):
    # Das implizite Eulerverfahren ist
    #     y1 = y0 + dt * rhs(t+dt, y1)
    # Wir müssen diese gleichung nach y1 auflösen.
    F = lambda y1 : y1 - (y0 + dt * rhs(t0 + dt, y1))
    return scipy.optimize.fsolve(F, eE_step(rhs, y0, t0, dt))

def iE(rhs, y0, t0, T, N):
    return integrate(iE_step, rhs, y0, t0, T, N)

def iM_step(rhs, y0, t0, dt):
    # Die implizite Mittelpunktsregel ist
    #    y1 = y0 + dt*rhs(t+0.5*dt, 0.5*(y0 + y1))
    F = lambda y1 : y1 - (y0 + dt*rhs(t0 + .5*dt, .5*(y0 + y1)))
    return scipy.optimize.fsolve(F, eE_step(rhs, y0, t0, dt))

def iM(rhs, y0, t0, T, N):
    return integrate(iM_step, rhs, y0, t0, T, N)

def eM_step(rhs, y0, t0, dt):
    return y0 + dt*rhs(t0 + .5*dt, y0 + .5*dt*rhs(t0, y0))

def eM(rhs, y0, t0, T, N):
    return integrate(eM_step, rhs, y0, t0, T, N)

def vv_step(rhs, xv0, t0, dt):
    xv0 = xv0.reshape((2, -1))
    xv1 = np.empty_like(xv0)
    x0, x1 = xv0[0,:], xv1[0,:]
    v0, v1 = xv0[1,:], xv1[1,:]

    x1[:] = x0 + dt*v0 + .5*dt**2 * rhs(t0, x0)
    v1[:] = v0 + .5*dt*(rhs(t0, x0) + rhs(t0+dt, x1))

    return xv1.reshape(-1)

def vv(rhs, y0, t0, T, N):
    return integrate(vv_step, rhs, y0, t0, T, N)

def magnus(omega, y0, t0, T, N):
    """
    Integrator by Magnus methods

    @param {callable} omega(t, h)   - Omega-Matrix which must accept t and h as parameters
                                      (@see Script 8.8)
           @param {float} t         - current time
           @param {float} h         - length of time step
    @param {array|float} y0         - Startvalues
    @param {float} t0               - Start time
    @param {float} T                - End time
    @param {int} N                  - Number of steps

    @return {ndarray} [t, y]        - t: array of timesteps, y: ndarray of coordinates
    """  
    return integrate(magnus_step, omega, y0, t0, T, N)

def magnus_step(omega, y0, t0, dt):
    exp = np.exp if np.size(y0) == 1 else scipy.linalg.expm
    return np.dot(exp(omega(t0, dt)), y0)

def splitting_step(phi_a, phi_b, y0, t0, dt, a, b):
    y = y0
    for a, b in zip(a, b):
        if (a != 0.0): y = phi_a(y, a*dt)
        if (b != 0.0): y = phi_b(y, b*dt)
    return y

def splitting(phi_a, phi_b, y0, t0, T, N, a, b):
    r"""Generalized splitting method.

    @param {callable} phi_a   - 1st term in rhs
    @param {callable} phi_b   - 2nd term in rhs
    @param {float} y0         - Start value
    @param {float} t0         - Start time
    @param {float} T          - End time
    @param {int} N            - Number of steps
    @param {array} a          - length of phi_a's steps
    @param {array} b          - length of phi_b's steps

    @return {ndarray} [t, y]  - t: array of timesteps, y: ndarray of coordinates
    """
    method = lambda rhs, y, t0, dt: splitting_step(phi_a, phi_b, y, t0, dt, a, b)
    return integrate(method, None, y0, t0, T, N)

def rk(rhs, y0, t0, T, N, B):
    r"""Generalized runge kutta method.

    @param {callable} rhs     - right hand side of ODE
    @param {float} y0         - Start value
    @param {float} t0         - Start time
    @param {float} T          - End time
    @param {int} N            - Number of steps
    @param {ndarray} B        - Butcher Scheme

    @return {ndarray} [t, y]  - t: array of timesteps, y: ndarray of coordinates
    """
    method = lambda rhs, y0, t0, dt: rk_step(rhs, y0, t0, dt, B)
    return integrate(method, rhs, y0, t0, T, N)

# Einzelner Schritt in RK mit Fallunterscheidung    
def rk_step(rhs, y0, t0, dt, B):
    """
    INPUTS
    rhs: Rechte Seite der DGL: dy/dt = f(t,y(t))
    t0: Letzter Zeitpunkt (bzw. letzter Wert der Laufvariable)
    dt: Schrittweite
    y0: Letzte Position
    B: Butcher Schema
    
    OUTPUTS
    y1: Position zum naechsten Zeitpunkt
    """
    # Initialisierung des Vektors k
    A, b, c, s, dim = B[0:-1,1:], B[-1,1:], B[0:-1,0], B.shape[1] - 1, np.size(y0)
    k = np.zeros((s, dim))
    
    # A strikte untere Dreiecksmatrix --> Explizites RK-Verfahren
    if np.array_equal(A, np.tril(A, 1)):
        for i in range(s):
            k[i,:] = rhs(t0 + dt*c[i], y0 + dt*np.dot(A[i,:],k))

    # A nicht-strikte untere Dreiecksmatrix --> Diagonal implizites RK-Verfahren
    elif np.array_equal(A, np.tril(A)):
        for i in range(s):
            F = lambda x: x - rhs(t0 + c[i]*dt, y0 + dt*np.dot(A[i,:], k + x))
            k[i,:] = scipy.optimize.fsolve(F, eE_step(rhs, y0, t0, dt))

    # A keine untere Dreiecksmatrix --> Allgemeines implizites RK-Verfahren
    else:
        # Funktion F mit allen ki als Variablen definieren
        # Weil wir sowohl als Input als auch als Output von fsolve nur einen
        # Vektor und leider keine Matrix brauchen koennen, arbeiten wir mit
        # reshape um die Operationen in den for-Schleifen lesbarer zu machen
        def F(k):
            k = k.reshape(s, dim)
            Fk = np.array([k[i,:] - rhs(t0 + dt*c[i], y0 + dt*np.dot(A[i,:],k)) for i in range(s)])
            return Fk.reshape(s*dim,)
        
        # Startwert fuer fsolve: Verwende eE
        start = eE_step(rhs, y0, t0, dt)
        
        # Loese das Gleichungssystem, um k zu bestimmen
        k = scipy.optimize.fsolve(F, start.reshape(s*dim,)).reshape(s, dim)

    return y0 + dt*np.dot(b, k)

###################################################
# Methods to get y function of linear ODE Systems #
###################################################

"""
Example: linear ODE System
y'(t) = λ*A*y(t)
=> y(t) = e^(λ*t*A)*y0

l = -1j

# By Krylov:
V, H = krylov(A, y0, k)
y = lambda t: V[:,:-1].dot(scipy.linalg.expm(l*t*H))[:,0]

# Analytic
y = lambda t: scipy.linalg.expm(l*t*A).dot(y0) # y(t) = exp(l*t*A)*y0

# Diagonalize
D, V = scipy.linalg.eig(A)
y = lambda t: V.dot(np.diag(np.exp(l*t*D)).dot(scipy.linalg.solve(V, v)))

"""

#########
# Tests #
#########

if __name__ == '__main__':

    # Magnus-Verfahren n. Ordnung
    # DGL: y'(t) = A(t)y(t)
    # Die Mathieu-Gleichung:
    # y'' + (ω^2 + ε*cos(t))*y = 0
    # y(0)=1, y'(0)=0

    # Funktion für den Kommutator AB - BA
    C = lambda A, B: np.dot(A, B) - np.dot(B, A)

    omega = 1.
    eps = 0.25
    y0 = 1
    y0p = 0
    t0 = 0.
    T = 20*np.pi
    N = 10**5
    A = lambda t : np.array([[0, 1],[-(omega**2 + eps*np.cos(t)), 0]])
    z = lambda z, t: np.array([z[1], -(omega**2 + eps*np.cos(t))*z[0]])
    z0 = np.array([y0, y0p])

    # 2. Ordnung (Omega wie im Skript Bsp: 8.8.1)
    O2 = lambda t, h: h*A(t + 0.5*h)
    t2, y2 = magnus(O2, z0, t0, T, N)

    # 4. Ordnung (Omega wie im Skript Bsp: 8.8.2)
    A1 = lambda t, h: A(t + (0.5 - np.sqrt(3)/12)*h)
    A2 = lambda t, h: A(t + (0.5 + np.sqrt(3)/12)*h)
    O4 = lambda t, h: 0.5*h*(A1(t, h) + A2(t, h)) - h**2*(np.sqrt(3)/12)*C(A1(t, h), A2(t, h))
    t4, y4 = magnus(O4, z0, t0, T, N)

    # 6. Ordnung (Omega wie im Skript Bsp: 8.8.4)
    A1 = lambda t, h: A(t + (0.5 - np.sqrt(15)/10)*h)
    A2 = lambda t, h: A(t + 0.5*h)
    A3 = lambda t, h: A(t + (0.5 + np.sqrt(15)/10)*h)
    O6 = lambda t, h: (h/6)*(A1(t, h) + 4*A2(t, h) + A3(t, h)) - (h**2/12)*C(A1(t, h), A3(t, h))
    t6, y6 = magnus(O6, z0, t0, T, N)

    plt.figure()
    plt.plot(t2,y2[:,0],'r-',label='Mag 2. Ordn')
    plt.plot(t4,y4[:,0],'g-',label='Mag 4. Ordn')
    plt.plot(t6,y6[:,0],'b-',label='Mag 6. Ordn')
    plt.xlabel('Zeit t')
    plt.ylabel('Position y(t)')
    plt.grid(True)
    plt.show()


    """
    # Runge Kutta Bsp
    # Beispiel: Gedaempftes Pendel
    f = lambda t, y: np.array([y[1], -82*y[0]-2*y[1]])
    t0 = 0.
    T = 4.
    N = 300
    y0 = np.array([1.,0.])

    #Exakte Loesung
    y = lambda t: np.exp(-t)*np.cos(9*t) # exakte Loesung
    t = np.linspace(t0,T,1000)
    y_exact = y(t)

    # Butcher scheme for explicit Euler
    Bee = np.array([
        [ 0.0,   0.0 ],
        #------|-------
        [ 0.0,   1.0 ]
    ])

    # Butcher scheme for implicit Euler
    Bie = np.array([
        [ 1.0,   1.0 ],
        #------|-------
        [ 0.0,   1.0 ]
    ])

    # Butcher scheme for explicit midpoint
    Bem = np.array([
        [ 0.0,   0.0, 0.0 ],
        [ 0.5,   0.5, 0.0 ],
        #------|------------
        [ 0.0,   0.0, 1.0 ]
    ])

    # Butcher scheme for implicit midpoint
    Bim = np.array([
        [ 0.5,   0.5 ],
        #------|-------
        [ 0.0,   1.0 ]
    ])

    # Butcher scheme for Runge-Kutta 3/8
    B38 = np.array([
        [ 0.0,    0.0,  0.0, 0.0, 0.0 ],
        [ 1/3,    1/3,  0.0, 0.0, 0.0 ],
        [ 2/3,   -1/3,  1.0, 0.0, 0.0 ],
        [ 1.0,    1.0, -1.0, 1.0, 0.0 ],
        #------|------------------------
        [ 0.0,    1/8,  3/8, 3/8, 1/8 ]
    ])

    Btr = np.array([
        [ 0.0,   0.0, 0.0 ],
        [ 1.0,   1.0, 0.0 ],
        #------|------------
        [ 1.0,   0.5, 0.5 ]
    ])


    # Die 5 einfachsten RK-Verfahren ueber RK und dem Butcher-Tableau
    t_eE2, y_eE2 = rk(f, y0, t0, T, N, Bee)
    t_iE2, y_iE2 = rk(f, y0, t0, T, N, Bie)
    t_eM2, y_eM2 = rk(f, y0, t0, T, N, Bem)
    t_iM2, y_iM2 = rk(f, y0, t0, T, N, Bim)
    t_eTR2, y_eTR2 = rk(f, y0, t0, T, N, Btr)

    #t_eE2, y_eE2 = eE(f, y0, t0, T, N)
    #t_iE2, y_iE2 = iE(f, y0, t0, T, N)
    #t_eM2, y_eM2 = eM(f, y0, t0, T, N)
    #t_iM2, y_iM2 = iM(f, y0, t0, T, N)
    #t_eTR2, y_eTR2 = rk(f, y0, t0, T, N, Btr)

    #Plotten
    plt.figure()
    plt.plot(t_eE2,y_eE2[:,0],'r-',label='eE')
    plt.plot(t_iE2,y_iE2[:,0],'b-',label='iE')
    plt.plot(t_eM2,y_eM2[:,0],'m*',label='eM')
    plt.plot(t_iM2,y_iM2[:,0],'c',label='iM')
    plt.plot(t_eTR2,y_eTR2[:,0],'g-',label='eTR')
    plt.plot(t,y_exact,'k--',label='Exakt')
    plt.legend(loc='best')
    plt.title('Die 5 einfachsten RK-Verfahren: Ueber RK')
    plt.xlabel('Zeit t')
    plt.ylabel('Position y(t)')
    plt.ylim(-1,1)
    plt.grid(True)
    plt.show()
    """

    # Splitting example from S11A1
    """
    B = -0.1
    theta = 0.25*np.pi

    # Zur Kontrolle mit ode45.
    def rhs(t, y):
        return np.dot(dRdt(t), np.dot(invR(t), y)) + B*y

    def R(t):
        angle = theta*t
        A = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        return A

    def invR(t):
        return R(-t)

    def dRdt(t):
        angle = theta*t
        A = theta*np.array([[-np.sin(angle), -np.cos(angle)],
                            [np.cos(angle), -np.sin(angle)]])
        return A

    y0 = np.array([1.0, 0.0])
    t0 = 0.0
    t_end = 100.0
    n_steps = 1000

    Phi_rot = lambda y0, t: np.dot(scipy.linalg.expm(np.dot(dRdt(t), invR(t))*t), y0)
    Phi_stretch = lambda y0, t: np.exp(B*t)*y0


    a, b = splitting_parameters('KL8')
    t4, y4 = splitting(Phi_rot, Phi_stretch, y0, t0, t_end, n_steps, a, b)
    plt.plot(y4[:,0], y4[:,1], label='KL8')

    a, b = splitting_parameters('L84')
    t5, y5 = splitting(Phi_rot, Phi_stretch, y0, t0, t_end, n_steps, a, b)
    plt.plot(y5[:,0], y5[:,1], label='L84')

    t_ode45, y_ode45 = ode45(rhs, [t0, t_end], y0)
    plt.plot(y_ode45[:,0], y_ode45[:,1], label='ode45')
    
    plt.legend(loc='best')
    plt.savefig("spiral.pdf")
    plt.show()
    """
