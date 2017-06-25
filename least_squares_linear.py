import numpy as np
import numpy.linalg as linalg
import scipy.linalg

import matplotlib.pyplot as plt



#Could be useful if you don't have a lot of base_functions and a lot of data
#Build the coefficient matrix of the linear equation system for which we want to minimize the residue.
def buildMatrix (base_functions, evaluation_points):
    #The matrix A
    A = np.zeros((evaluation_points.size,base_functions.size))

    #Iterate over points where the base functions will be evaluated. Corresponds to a row in A.
    for i in range(0, evaluation_points.size):
        ev_point = evaluation_points[i]
        #Iterate over the base functions, evaluate them and save the result in the matrix. Corresponds to a column in A.
        for j in range(0, base_functions.size):
            A[i,i] = base_functions[j](ev_point)
    return A


#Use this if the mxn matrix is big and sparse
#Uses the fact that the Jacobian matrix / Gradient of ||Ax-b||^2 should be zero if it is a minimum
def solve_with_normal_equation(A, b):
    ATA=np.dot(A.T,A)
    ATb=np.dot(A.T,b)
    x=np.linalg.solve(ATA,ATb)
    return x

#Possible to use this with a non fulll rank matrix, however SVD is better (according to the script).
def solve_with_qr(A, b):
    Q, R = scipy.linalg.qr(A,mode="economic") #economic mode returns a square matrix for R (last rows will all be 0)
    btilde = np.dot(Q.T, b)
    x= scipy.linalg.solve_triangular(R, btilde) #R is an upper triagonal matrix
    return x

#non-readable code copied from solution of series 9
def solve_with_svd(A, b):
    U, sigma, Vh = scipy.linalg.svd(A)
    r = sigma.shape[0]
    U1 = U[:,:r]
    x = np.dot(Vh[:,:r].T, np.dot(np.diag(1/sigma), np.dot(U1.T, b)))
    return x


def exactActivity (t, lambdas , m0):
    activity=0
    for i in range (m0.size):
        activity += m0[i]* lambdas[i]* np.exp (-lambdas[i]*t)
    return activity

#Implement assignemnt 8.2 (radioactive Decomposition)
def test():
    n=4
    number_of_measurements=10**2
    m0=np.floor(np.random.rand(n)*1000)
    lambdas=np.linspace(10**-2,10**-1,n)

    times = np.linspace(0.1 , 7, number_of_measurements)
    activities = np.zeros(times.shape)
    for i in range(number_of_measurements):
        activities[i] = exactActivity(times[i],lambdas,m0)

    noise_amplitude = 1E-7
    noise = 1 + (noise_amplitude *np.random.rand (number_of_measurements) - noise_amplitude )
    noisy_activities = activities*noise #hihihi
    plt.plot (times, activities, "k-", label="Exact Activity")
    plt.plot (times, noisy_activities, "go", alpha=0.5 , label="Noisy")
    plt.grid(True)
    plt.legend()
    plt.show()

    A=np.zeros((number_of_measurements, n))
    for i in range(n):
        A[:,i] = lambdas[i]* np.exp (-lambdas[i]*times)

    #Approximate the m's with least squares methods
    x1=solve_with_normal_equation(A, activities)
    x2=solve_with_qr(A,activities)
    x3=solve_with_svd(A,activities);
    print("Exact")
    print("Exact M's: ",m0)
    print("Normal M's: ",x1)
    print("QR M's: ",x2)
    print("SVD M's: ",x3)

    x1=solve_with_normal_equation(A, noisy_activities)
    x2=solve_with_qr(A,noisy_activities)
    x3=solve_with_svd(A,noisy_activities);
    print("Noisy: ")
    print("Exact M's: ",m0)
    print("Normal M's: ",x1)
    print("QR M's: ",x2)
    print("SVD M's: ",x3)

    return



if __name__ == "__main__":
    test()
