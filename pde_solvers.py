# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize

def integrate(method, rhs, y0, T, N):
    y = np.empty((N+1,) + y0.shape)

    t0, dt = 0.0, T/N
    y[0,...] = y0
    for i in range(0, N):
        y[i+1,...] = method(rhs, y[i,...], t0 + i*dt, dt)

    return np.arange(N+1)*dt, y

def explicit_euler_step(rhs, y0, t0, dt):
    return y0 + dt*rhs(t0, y0)

def explicit_euler(rhs, y0, T, N):
    return integrate(explicit_euler_step, rhs, y0, T, N)

def implicit_euler_step(rhs, y0, t0, dt):
    # Das implizite Eulerverfahren ist
    #     y1 = y0 + dt * rhs(t+dt, y1)
    # Wir müssen diese gleichung nach y1 auflösen.
    F = lambda y1 : y1 - (y0 + dt * rhs(t0 + dt, y1))
    return scipy.optimize.fsolve(F, explicit_euler_step(rhs, y0, t0, dt))

def implicit_euler(rhs, y0, T, N):
    return integrate(implicit_euler_step, rhs, y0, T, N)

def implicit_mid_point_step(rhs, y0, t0, dt):
    # Die implizite Mittelpunktsregel ist
    #    y1 = y0 + dt*rhs(t+0.5*dt, 0.5*(y0 + y1))
    F = lambda y1 : y1 - (y0 + dt*rhs(t0 + .5*dt, .5*(y0 + y1)))
    return scipy.optimize.fsolve(F, explicit_euler_step(rhs, y0, t0, dt))

def implicit_mid_point(rhs, y0, T, N):
    return integrate(implicit_mid_point_step, rhs, y0, T, N)

def explicit_mid_point_step(rhs, y0, t0, dt):
    return y0 + dt*rhs(t0 + .5*dt, y0 + .5*dt*rhs(t0, y0))

def explicit_mid_point(rhs, y0, T, N):
    return integrate(explicit_mid_point_step, rhs, y0, T, N)

def velocity_verlet_step(rhs, xv0, t0, dt):
    xv0 = xv0.reshape((2, -1))
    xv1 = np.empty_like(xv0)
    x0, x1 = xv0[0,:], xv1[0,:]
    v0, v1 = xv0[1,:], xv1[1,:]

    x1[:] = x0 + dt*v0 + .5*dt**2 * rhs(t0, x0)
    v1[:] = v0 + .5*dt*(rhs(t0, x0) + rhs(t0+dt, x1))

    return xv1.reshape(-1)

def velocity_verlet(rhs, y0, T, N):
    return integrate(velocity_verlet_step, rhs, y0, T, N)
