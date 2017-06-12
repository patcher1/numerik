def einschritt_stoermer_verlet(rhs, y0, v0, T, N):
    """ Einschritt Störmer-Verlet.

    Input: rhs     ... Rechte Seite der ODE
           y0      ... 1D Array, Anfangskoordinaten
           v0      ... 1D Array, Anfangsgeschwindigkeiten
           N       ... Anzahl Zeitschritte
           T       ... Endzeit

    Output: t ... Zeit
            y ... 1D Array, Trajektorien
    """

    t, dt = np.linspace(0, T, N+1, retstep=True)
    y, v = np.empty((N+1,) + y0.shape), np.empty((N+1,) + v0.shape)
    y[0,:], v[0,:] = y0, v0

    for i in range(N):
        v12 = v[i,:] + .5*dt*rhs(y[i,:])
        y[i+1,:] = y[i,:] + dt*v12
        v[i+1,:] = v12 + .5*dt*rhs(y[i+1,:])

    return t, y

def zweischritt_stoermer_verlet(rhs, y0, v0, T, N):
    """ Zweischritt Störmer-Verlet.

    Dieses Zweischrittverfahren kommt ohne extra Speicher für die
    Geschwindigkeit aus und halb so vielen Aufrufe der rechten Seite aus.

    Input: rhs     ... Rechte Seite der ODE.
           y0      ... 1D Array, Anfangskoordinaten
           v0      ... 1D Array, Anfangsgeschwindigkeiten
           N       ... Anzahl Zeitschritte
           T       ... Endzeit

    Output: t ... Zeit
            y ... 1D Array, Trajektorien

    """

    t, dt = np.linspace(0, T, N+1, retstep=True)
    y = np.empty((N+1,) + y0.shape)
    y[0,:], y[1,:] = y0, y0 + dt*v0 + .5*dt**2*rhs(y0)

    for i in range(1,N):
        y[i+1,:] = -y[i-1,:] + 2.*y[i,:] + dt**2*rhs(y[i,:])

    return t, y
