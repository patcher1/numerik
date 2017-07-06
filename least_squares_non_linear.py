import numpy as np
import numpy.linalg as linalg
import scipy.linalg
import matplotlib.pyplot as plt

from zero_of_a_function import newton

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
    Linearizes the minimization problem by approximating F(xk) = JF(xk)sk in each iteration step.
    After linearization a minimization method from least_squares_linear will be used (QR, SVD)

    @return {ndarray|float} - approximated solution for ||F(xk)|| = 0 within a tolerance
    @return {boolean}       - If the iteration had converged
    @return {int}           - used iterations

'''
def solve_with_gauss_newton(xk, F, J,equation_minimizer, tol = 10**-14, maxit = 10000):
    for i in range(maxit):
        sk = equation_minimizer(J(xk),F(xk))  #Finds sk such that [J(x)*sk - F(x)] is minimal / can't use linalg.solve since J(x) is not necessarily invertible
        xk = xk-sk                            #Change this to xk-=sk for epic fail
        if np.linalg.norm(sk) <= tol*np.linalg.norm(xk):
            return xk, True, i
    return xk, False, maxit

'''print ("xk - sk", xk-sk)
print("sk: ", sk)
print("xk: ",xk)
print("i: ",i)
print("norm: ", np.linalg.norm(sk) )
print("norm xk: ", tol*np.linalg.norm(xk) )
print("tol: ", tol)'''
