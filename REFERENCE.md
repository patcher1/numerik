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

## Allgemeine

- `np.linspace`:
- `np.arange`: np.arange(11, 152, 10) => [11,21,...,151]
- `np.zeros`:
- `np.ones`:
- `np.dot`: Skalarprodukt
- `np.array`:

## Plots

- `matplotlib.pyplot.semilogy`: Make a plot with log scaling on the y axis.