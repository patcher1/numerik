# Funktionen

## Verfahren

- `scipy.optimize.newton`: Newton-Verfahren (Sekanten-Verfahren, falls Df nicht gegeben)
- `scipy.optimize.bisect`: Bisektionsverfahren
- `scipy.optimize.fsolve`: n-dim Nullstellensuche
- `numpy.linalg.solve`: A*x=b
- `scipy.linalg.solve_triangular`
- `numpy.linalg.lstsq`: arg min||A*x-b|| lineare Ausgl.
- `scipy.optimize.leastsq`: arg min||f(y)||**2 nicht-lin. Ausgl.
- `numpy.linalg.pinv`: Pseudo-Inverse
- `numpy.linalg.svd`: SVD-Zerlegung
- `numpy.linalg.qr`: QR-Zerlegung, A*x=b <=> R*x=Q.T*b
- `scipy.linalg.qr_delete` lösche lin abhängige Zeilen (Spalten)
- `scipy.linalg.lu`: LU-Zerlegung
- `scipy.linalg.lu_solve`: 
- `numpy.linalg.cholesky`: A*x=b <=> L*y=b und dann L*H*x=y
- `numpy.linalg.cond`: Kondition einer Matrix
- `numpy.linalg.norm`: Norm
- `numpy.linalg.eig`: EW/EV
- `scipy.sparse.linalg.eigs`: Anzahl EW/EV mit Arnoldi/Lancsoz
- `scipy.integrate.quad`: Integrator
- `scipy.integrate.nquad`: n-dim Integrator (ruft rekursiv `quad` auf)
- `scipy.integrate.quadrature`: Gauss Quadratur
- `scipy.linalg.expm`: e hoch Matrize
- `numpy.polynomial.polynomial.polyfit`: gibt Koeffizienten des Polynoms zurück (lowest power first)
- `numpy.polyfit`: gibt Koeffizienten des Polynoms zurück (highest power first) (macht `numpy.linalg.solve(numpy.vander(x), y)`)
- `numpy.polyval`: evaluiert anhand Koeffizienten (von polyfit) ein Polynom an gegebenen Punkten
- `numpy.vander`: generiert eine Vandermonde matrix
- `numpy.fft.fft`: `c=fft(y)` => c=(FN)y (FN: Fouriertrnasformationsmartix)
- `numpy.fft.ifft`: `y=ifft(c)` => y=1/N(FN.H)c
- `numpy.fft.fftshift`: shifte Koeffizienten sodass 0-Frequenz in der Mitte ist (zurück mit `ifftshift`)
- `numpy.diag`: erstellt Diagmatrix aus Array oder diag(x) gibt Diagelemente aus Matrix

## Allgemeine

- `np.linspace(start, stop, num=50)`: np.linspace(11, 151, 15) => [11,21,...,151]
- `np.arange(start, stop, step=1)`: np.arange(11, 152, 10) => [11,21,...,151]
- `np.zeros(shape), np.ones(shape), np.empty(shape)`: gets array of shape with zeros, ones, or uninitialized elements
- `np.dot`: Skalarprodukt
- `np.array`: create numpy array from normal array
- `np.ones_like().astype(float)`: Array with ones
- `np.zeros_like().astype(float)`: Array with zeros
- `np.empty_like().astype(float)`: Array of uninitialized (arbitrary) data

## Plots

- `matplotlib.pyplot.semilog{x,y}`: Make a plot with log scaling on the {x,y} axis.
- `matplotlib.pyplot.plot`: Ex: `plt.plot(x, f(x), "-b", label=r"$f(x)$")`
- `matplotlib.pyplot.loglog`: Make a plot with log scaling on both the x and y axis.
- Example plot
```python
plt.figure()
plt.plot(x, f(x), "-b", label=r"$f(x)$")
plt.grid(True)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.legend()
plt.savefig("figure.eps")
plt.show()
```

## Sonstiges

- `ode45`: Bsp: `ode45(f, [t0, T], y0, **options)` mit `options = {'reltol':1e-8, 'abstol':1e-8, 'initialstep':2e-1, 'stats':'on'}`