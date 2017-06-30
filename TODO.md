# TODO

# Nullstellen
- [x] Newton-Verfahren
  - [x] 1-dim (1.7.2)
  - [x] 2-dim (1.7.4)
  - [x] n-dim
- [ ] gedämpftes Newton-Verfahren (1.8.6)
- [ ] Bisektionsverfahren (Intervallhalbierungsverfahren) (S7A2g)
- [ ] Sekantenverfahren (1.6.2)

# Ausgleichsrechnung
- [x] lineare
- [ ] nicht-lineare
  - [ ] Newton-Verfahren (S7A1)
    - [ ] n-dim
  - [ ] Gauss-Newton Verfahren (3.2.2) (S9A2c)
- [ ] Least Squares Problem  

# LinAlg

- [ ] Gram-Schmidt-Verfahren (S8A1)
- [ ] Modifiziertes Gram-Schmidt (S8A1)
- [ ] QR-Zerlegung
- [ ] LU-Zerlegung
- [ ] Cholesky-Zerlegung
- [ ] SVD-Zerlegung
- [ ] Konditionszahl
- [ ] Householder Transformation
- [ ] Eigenwerte
  - [x] Potenzmethode (S. 87) (S10A1)
    - [x] grösste (S10A1)
    - [x] kleinste (Inverse Iteration) (S10A1)
      - [ ] Vorkonditionierte inverse Iteration (4.3.17)
  - [ ] Rayleigh-Quotient-Iteration (4.3.12)
  - [x] Krylov-Verfahren (S10A2c) (Bsp)
    - [x] Arnoldi-Iteration (4.4.4, geht aber nicht) (S10A2d)
    - [x] Lanczos-Iteration (4.4.9, geht aber nicht) (S10A2d)

# Interpolation

- [x] Polynomiale
  - [x] Monom Basis
  - [ ] Newton Basis
  - [ ] Lagrange Basis
  - [ ] Baryzentrische Interpolationsformel
- [x] Chebyshev-Interpolation
  - [x] direkt evaluiert
  - [x] mit evalchebexp() evaluiert (6.6.3)
  - [x] mit Clenshaw evaluiert (5.4.25)
- [ ] Trignonometrische
  - [ ] Fourier
  - [ ] DFT

# Integrale

- [ ] Mittelpunktregel (S6A3b)
  - [x] 2-dim
- [ ] Trapezregel
  - [x] 2-dim (S4A2)
- [ ] Simpsonregel
  - [x] 2-dim (S4A2)
- [x] zusammengesetzte
  - [x] Mittelpunktregel (7.2.6)
  - [x] Trapezregel (S4A1)
  - [x] Simpsonregel (7.2.7) (S3A3e, S4A1) 
- [x] Gauss-Legendre Quadratur (S5A1) (Bsp)
  - [x] zusammengesetzte (S5A1)
  - [x] Golub-Welsch (7.3.3)
- [ ] Adaptive Quadratur
- [x] Monte-Carlo (7.7.1)
  - [ ] Control Variates (7.7.2)
  - [ ] Importance Sampling (7.7.5)
  - [x] n-dim (S4A3)
- [ ] R^d Funktionen
  - [ ] Clenshaw-Curtis Quadratur (7.3.5, 7.3.6)

# ODEs

- [x] Explicit Euler (S2A2, S5A2)
- [x] Implicit Euler (S2A2, S5A2)
- [x] Velocity Verlet (S2A2)
- [x] Implicit Mid Point (S2A2, S5A2)
- [x] Explicit Mid Point (S5A2)
- [x] Störmer Verlet (S3A1)
  - [x] einschritt (S3A1)
  - [x] zweischritt (S3A1)
- [ ] Splitting-Verfahren (Bsp)
  - [ ] Lie-Trotter
  - [ ] Strang (S11A1c, 2, 3)
- [x] Runge Kutta Verfahren (S5A2, S6A3c) (Bsp)
  - [x] Gauss-Kollokationsverfahren (Butcher) (S6A3d)
  - [x] Radau + Lobatto QF (Butcher)
- [ ] Magnus Verfahren
- [x] Krylov-Verfahren (S10A2c) (Bsp)

# Steife ODEs

- [ ] exponentielles Rosenbrock-Euler-Verfahren (S10A3) (Bsp)
- [ ] Adaptiver Integrator für steife ODEs und Systeme mit Methoden unterschiedlicher Ordnung (8.6.2)
