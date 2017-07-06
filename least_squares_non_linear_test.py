from numpy import *
from scipy.constants import c
import scipy.stats as stats
from numpy.linalg import lstsq, norm
from matplotlib.pyplot import *
from matplotlib.mlab import find

from least_squares_linear import *
from least_squares_non_linear import *


#Assignment 09,2 as a test (GPS)
def gn(x, F, J, tol=1e-6, maxit=100):
    return solve_with_gauss_newton(x,F,J,solve_with_svd,tol,maxit) 


#Flughoehe
rho = 26570*1e3
# Erdradius
re = 6370*1e3

def J(x, X):
    """
    Jacobi Matrix

    Keyword Arguments:
    x --  [ tr, xr, yr, zr]
    X --  [ ts, xs, ys, zs]
    """
    Jr = 2*(x-X)
    Jr[:,0] *= -c**2
    return Jr

def F(x, X):
    """
    Residuenvektor

    Keyword Arguments:
    x --  [ tr, xr, yr, zr]
    X --  [ ts, xs, ys, zs]
    """
    Fr = (x-X)**2
    Fr[:,0] *= -c**2
    Fr = sum(Fr, axis=1) #spacetime intervall (is invariant under Lorentz-Transformation)

    return Fr

def cart_to_longlat(x):
    """
    Keyword Arguments:
    x -- position vector

    Returns:
    longtitude --- in degrees
    long. dir. --- direction O => east, or W => west
    latitude   --- in degrees
    lat. dir.  --- direction N => northern hemisphere, S => southern hemisphere
    """
    r = norm(x)
    theta = arcsin(x[2]/r)
    phi = arctan2(x[1],x[0])

    return rad2deg(phi), 'O' if x[1] > 0 else 'W', rad2deg(abs(theta)), 'N' if theta > 0 else 'S'


# Rotationsmatrizen
def rot_y(alpha):
    return array ([[  cos(alpha), 0, sin(alpha) ],
                   [  0         , 1, 0          ],
                   [ -sin(alpha), 0, cos(alpha) ]])
def rot_z(alpha):
    return array ([[  cos(alpha),  sin(alpha), 0 ],
                   [  -sin(alpha), cos(alpha), 0 ],
                   [  0,            0,         1 ]])


def random_sampling(satellite_positions, dtvar, plot_axes):
    """
    Fehleranalyse
    Plottet die Fehler in der Positionsbestimmung
    fuer 1000 zufaellig ausgewaehlte Untermengen der sichtbaren Satelliten mit
    Normal(0, dtvar)-verteilten Fehlern in der Zeitmessung.

    Keyword Arguments:
    satellite_positions ---
    dtvar               --- Standardabweichung der Messfehler
    plot_axes           --- matplotlib.axes object
    """
    random.seed(0)
    ax = plot_axes

    poe = re*array([1, 1, 1])/sqrt(3)
    # finde die sichtbaren Satelliten
    I = dot(satellite_positions-poe, poe) > 0
    satellites_visible = satellite_positions[I,:]

    distance = norm( poe - satellites_visible, axis=1)
    travel_time = distance/c

    # add noise to measured travel time
    travel_time_measured = travel_time + random.normal(scale=dtvar, size=len(travel_time))
    # measurements
    X = hstack( (travel_time_measured.reshape(-1,1), satellites_visible) )

    x0 = array([0, 0, 0, 0])
    results = {i: [] for i in range(4,13)}

    for i in range(1000):
        idx = find(random.rand(len(X)) < 0.5)
        Xs = X[idx,:]
        if len(Xs) < 4:
            continue
        try:
            vals = gn(x0, lambda x: F(x,Xs), lambda x: J(x,Xs), tol=1e-6, maxit=100)
            x = vals[0]
            has_converged = vals[1]
            if has_converged:
                results[len(Xs)].append( (idx, norm(x[1:]-poe)) )
        except IndexError:
            print("Es ist ein Fehler aufgetreten.")
            print("Die Gauss-Newton Methode `gn` sollte mindestens folgende Werte zurueckgeben: (x, has_converged)")
            print("wobei: x die Loesung und has_converged [True|False] sein muss.")
            print("Passe gegebenenfalls deine Implementation an und versuch es nochmal.")
            return

    for key in list(results.keys()):
        results[key] = sorted( results[key], key= lambda x: x[-1])

    for key, value in results.items():
            ax.semilogy( len(value)*[key], [ v[1] for v in value], 'x')

    grid(True)

    ax.set_xlabel('Anz. Satelliten')
    ax.set_ylabel('Fehler in der Position [m]')

if __name__ == '__main__':

    ## set the satellite positionss
    N = 4 # Anz. Satelliten pro Bahn
    phi = 2*pi/N
    j = arange(0,N).reshape(-1,1)
    p0 = hstack([cos(j*phi), sin(j*phi), 0*j])
    satellite_positions = []
    for i in range(6):
        phi = pi/7
        pos_local = dot( rot_z(phi), p0.T).T
        pos_local = dot( rot_y(deg2rad(55)), pos_local.T).T
        pos_local = dot( rot_z(i*2*pi/6), pos_local.T).T
        satellite_positions.append(pos_local)
    satellite_positions = rho*vstack(satellite_positions)


    # poe = point on earth
    #poe = array([1, 1, 1 ])/sqrt(3)*re
    poe = array([1, 1, 0 ])/sqrt(2)*re
    poe = array([0, 0, 1 ])*re

    # find the visible satellites
    I = dot(satellite_positions-poe, poe) > 0
    satellites_visible = satellite_positions[I,:]

    # set up input data
    distance = norm( poe - satellites_visible, axis=1)
    travel_time = distance/c
    random.seed(0)
    dt = 1e-8 # Abweichung 10 Nanosekunden
    travel_time_measured = travel_time + random.normal(scale=dt, size=len(travel_time))
    # measurements
    X = hstack( (travel_time_measured.reshape(-1,1), satellites_visible) )


    x0 = array([0, 0, 0, 0])
    x, has_converged, it = gn(x0, lambda x: F(x,X), lambda x: J(x,X), tol=1e-8, maxit=100)

    # Ausgabe
    if has_converged:
        print('Gauss-Newton liefert die Loesung, nach %d Iterationen:' % it)
        print('d = %.5e s' % x[0])
        print('x = %.3f km' % (x[1]/1000))
        print('y = %.3f km' % (x[2]/1000))
        print('z = %.3f km' % (x[3]/1000))
        int, OE, lat, SN = cart_to_longlat(x[1:])
        print('Wir befinden uns bei %.2f %s, %.2f %s' % (int, OE, lat, SN ))
    else:
        print('You are on the dark side of the moon.')

    ## Unteraufgabe d)
    figure(figsize=(9,4))
    ax = subplot(141)
    dt = 1e-10
    random_sampling(satellite_positions, dt, ax)
    title(r'$\varepsilon_t = %.2e$' % dt)
    ylim(1e-2, 1e4)
    xlim(3,9)

    ax = subplot(142)
    dt = 1e-8
    random_sampling(satellite_positions, dt, ax)
    title(r'$\varepsilon_t = %.2e$' % dt)
    ylim(1e-2, 1e4)
    xlim(3,9)

    ax = subplot(143)
    dt = 1e-7
    random_sampling(satellite_positions, dt, ax)
    title(r'$\varepsilon_t = %.2e$' % dt)
    ylim(1e-2, 1e4)
    xlim(3,9)

    ax = subplot(144)
    dt = 1e-6
    random_sampling(satellite_positions, dt, ax)
    title(r'$\varepsilon_t = %.2e$' % dt)
    ylim(1e-2, 1e4)
    xlim(3,9)

    tight_layout()
    savefig('error-analysis.pdf')

    show()
