import numpy as np
import numpy.linalg as linalg
import scipy.linalg
import zero_of_a_function
import matplotlib.pyplot as plt
#NOT TESTED YET
'''Find the minimum of the residue vector F(x) in the 2-norm. The residue vector is in general a function from R^n -> R
   so we need to find x such that grad(F)(x) = 0 and Hess(F)(x) is positive definite and symmetric.
   Bascially just uses Newton iteration with the Gradient as the function and the Hessian as the Jacobian.

   When to use:
   This method is not often used since the calculation of the Hessian matrix takes a lot of time / is ugly.
   Better use Gauss-Newton if it is not explicitly stated that one should use Newton.
'''
def solve_with_newton(xk,gradient, hessian, tol = 10**-14, maxit = 10000):
    x = newton(xk,gradient, hessian,tol, maxit)
    return x

'''
    Linearizes the minimization problem by approximating F(x) = F(y) + J(y)(x-y) (like a multidimensional tangent-equation)
    After linearization a minimization method from least_squares_linear can be used (QR, SVD)
'''
def solve_with_gauss_newton(xk, F, J, tol = 10**-14, maxit = 10000, equation_minimizer = np.lstsq):
    for i in range(maxit):
        sk = equation_minimizer(J(xk),F(xk))[0]  #Finds sk such that [J(x)*sk - F(x)] is minimal / can't use linalg.solve since J(x) is not necessarily to be invertible
        xk -= sk
        if np.linalg.norm(sk) <= tol*np.linalg.norm(xk):
            break
    return xk
