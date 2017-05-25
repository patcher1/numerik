import numpy as np
import numpy.linalg as linalg
import scipy.linalg

import matplotlib.pyplot as plt

def approximate_polynomial(polynom_degree,function, inputs, method ):
    A = np.zeros((inputs.size,polynom_degree))

    for i in range(0, polynom_degree):
        A[:,i] = np.power(inputs, i)

    b = np.array([function(x) for x in inputs])
    x = method(A,b)

    '''professional_solution = linalg.lstsq(A,b)[0]
    print("Polynom Degree: ",polynom_degree,"M: ",inputs.size)
    print("QR: ", x)
    print("LSTSQ: ", professional_solution)
    print ("Differences: ",np.subtract(x,professional_solution))

    print("Inputs: ", inputs)
    print("QR-Values: ",np.polyval(x[::-1], inputs))
    print("LSTSQ-Values: ",np.polyval(professional_solution[::-1], inputs))
    print("b: ",b)'''

    return x

def solve_with_qr(A, b):
    Q, R = np.linalg.qr(A)
    btilde = np.dot(Q.T, b)
    x= scipy.linalg.solve(R, btilde)
    return x



def approximate_runge():
    runge_plot_points = np.linspace(-5, 5, 100)
    numberOfPoints=np.array([20,40])
    start=2
    stop=14
    plt.figure(1,figsize=(12,16))
    f = lambda x: 1/(1+x**2)
    sub_plot=211
    for m in np.nditer(numberOfPoints):
        plt.subplot(sub_plot)
        sub_plot += 1
        for n in range(start, stop+1):
            inputs = np.linspace(-5, 5, m) #m=11 wäre ein Problem wegen Division durch 0

            solution = approximate_polynomial(n,f,inputs,solve_with_qr)
            y = np.polyval(solution[::-1], runge_plot_points)
            plt.plot(runge_plot_points, y, label="Degree: %d "% n)


    plt.plot(runge_plot_points, f(runge_plot_points),label="Runge")
    plt.grid(True)
    plt.legend()
    plt.subplot(211)
    plt.plot(runge_plot_points, f(runge_plot_points), label="Runge")
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('runge_function.pdf')
    return

if __name__ == "__main__":
    approximate_runge()
