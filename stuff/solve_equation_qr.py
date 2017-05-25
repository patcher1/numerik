import numpy as np
import scipy.linalg

def solve_with_qr(A, b):
    Q, R = np.linalg.qr(A)
    btilde = np.dot(Q.T, b)
    x= scipy.linalg.solve(R, btilde) #numpy.linalg or scipy.linalg?
    return x
