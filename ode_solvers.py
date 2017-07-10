# -*- coding: utf-8 -*-

if __name__ == '__main__':
import scipy.optimize
import scipy.linalg
import matplotlib.pyplot as plt
from eigenvalues import arnoldi, lanczos, krylov

##############
# Solve ODEs #
##############

def integrate(method, rhs, y0, t0, T, N):
    y = np.empty((N+1,) + np.atleast_1d(y0).shape)
    y[0,...], dt = y0, T/N
    for i in range(0, N):
        y[i+1,...] = method(rhs, y[i,...], t0 + i*dt, dt)

    return np.arange(N+1)*dt, y.reshape((y.shape[0], np.size(y0)))

def explicit_euler_step(rhs, y0, t0, dt):
    return y0 + dt*rhs(t0, y0)

def explicit_euler(rhs, y0, t0, T, N):
    return integrate(explicit_euler_step, rhs, y0, t0, T, N)

def implicit_euler_step(rhs, y0, t0, dt):
    # Das implizite Eulerverfahren ist
    #     y1 = y0 + dt * rhs(t+dt, y1)
    # Wir müssen diese gleichung nach y1 auflösen.
    F = lambda y1 : y1 - (y0 + dt * rhs(t0 + dt, y1))
    return scipy.optimize.fsolve(F, explicit_euler_step(rhs, y0, t0, dt))

def implicit_euler(rhs, y0, t0, T, N):
    return integrate(implicit_euler_step, rhs, y0, t0, T, N)

def implicit_mid_point_step(rhs, y0, t0, dt):
    # Die implizite Mittelpunktsregel ist
    #    y1 = y0 + dt*rhs(t+0.5*dt, 0.5*(y0 + y1))
    F = lambda y1 : y1 - (y0 + dt*rhs(t0 + .5*dt, .5*(y0 + y1)))
    return scipy.optimize.fsolve(F, explicit_euler_step(rhs, y0, t0, dt))

def implicit_mid_point(rhs, y0, t0, T, N):
    return integrate(implicit_mid_point_step, rhs, y0, t0, T, N)

def explicit_mid_point_step(rhs, y0, t0, dt):
    return y0 + dt*rhs(t0 + .5*dt, y0 + .5*dt*rhs(t0, y0))

def explicit_mid_point(rhs, y0, t0, T, N):
    return integrate(explicit_mid_point_step, rhs, y0, t0, T, N)

def velocity_verlet_step(rhs, xv0, t0, dt):
    xv0 = xv0.reshape((2, -1))
    xv1 = np.empty_like(xv0)
    x0, x1 = xv0[0,:], xv1[0,:]
    v0, v1 = xv0[1,:], xv1[1,:]

    x1[:] = x0 + dt*v0 + .5*dt**2 * rhs(t0, x0)
    v1[:] = v0 + .5*dt*(rhs(t0, x0) + rhs(t0+dt, x1))

    return xv1.reshape(-1)

def velocity_verlet(rhs, y0, t0, T, N):
    return integrate(velocity_verlet_step, rhs, y0, t0, T, N)

def magnus(omega, y0, t0, T, N):
    """
    Integrator by Magnus methods

    @param {callable} omega(t, h)   - Omega-Matrix which must accept t and h as parameters (@see Script 8.8)
           @param {float} t         - current time
           @param {float} h         - length of time step
    @param {array|float} y0         - Startvalues
    @param {float} t0               - Start time
    @param {float} T                - End time
    @param {int} N                  - Number of steps

    @return {ndarray} [t, y]          - t: array of timesteps, y: ndarray of coordinates
    """  
    return integrate(magnus_step, omega, y0, t0, T, N)

def magnus_step(omega, y0, t0, dt):
    exp = np.exp if np.size(y0) == 1 else scipy.linalg.expm
    return np.dot(exp(omega(t0, dt)), y0)

def runge_kutta(rhs, y0, t0, T, N, B):
    """
    INPUTS:
    rhs: Rechte Seite der DGL: dy/dt = f(t,y(t))
    t0, T: Start- und Endzeitpunkt
    y0: Startvektor
    N: Anzahl Teilintervalle
    B: Butcher Schema
    
    OUTPUTS:
    t: Die Zeiten/Laufvariable zu den approximierten Funktionswerten
    y: Array aus dem Startwert und den N approximierten Funktionswerten
    """

    y = np.zeros((N+1, np.size(y0)))
    t, dt = np.linspace(t0, T, N+1, retstep=True)
    y[0,:] = y0
       
    # Iterative Berechnung der approximierten Funktionswerte y[i+1,:]
    for i in range(N):
        y[i+1,:] = runge_kutta_step(rhs, t[i], y[i,:], dt, B)
    return t, y


# Einzelner Schritt in RK mit Fallunterscheidung    
def runge_kutta_step(rhs, t0, y0, dt, B):
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
    A, b, c, s, dim = B[0:-1,1:], B[-1,1:], B[0:,0], B.shape[1] - 1, np.size(y0)
    k = np.zeros((s, dim))
    
    # A strikte untere Dreiecksmatrix --> Explizites RK-Verfahren
    if np.array_equal(A, np.tril(A, 1)):
        for i in range(s):
            k[i,:] = rhs(t0 + dt*c[i], y0 + dt*np.dot(A[i,:],k))

    # A nicht-strikte untere Dreiecksmatrix --> Diagonal implizites RK-Verfahren
    elif np.array_equal(A, np.tril(A)):
        for i in range(s):
            F = lambda x: x - rhs(t0 + c[i]*dt, y0 + dt*np.dot(A[i,:], k + x))
            k[i,:] = scipy.optimize.fsolve(F, explicit_euler_step(rhs, y0, t0, dt))

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
        start = explicit_euler_step(rhs, y0, t0, dt)
        
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

    """

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

    """
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
    t_eE2, y_eE2 = runge_kutta(f, y0, t0, T, N, Bee)
    t_iE2, y_iE2 = runge_kutta(f, y0, t0, T, N, Bie)
    t_eM2, y_eM2 = runge_kutta(f, y0, t0, T, N, Bem)
    t_iM2, y_iM2 = runge_kutta(f, y0, t0, T, N, Bim)
    t_eTR2, y_eTR2 = runge_kutta(f, y0, t0, T, N, Btr)

    #t_eE2, y_eE2 = explicit_euler(f, y0, t0, T, N)
    #t_iE2, y_iE2 = implicit_euler(f, y0, t0, T, N)
    #t_eM2, y_eM2 = explicit_mid_point(f, y0, t0, T, N)
    #t_iM2, y_iM2 = implicit_mid_point(f, y0, t0, T, N)
    #t_eTR2, y_eTR2 = runge_kutta(f, y0, t0, T, N, Btr)

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
