# -*- encoding: utf-8 -*-
import numpy as np
import numpy.linalg

import scipy
import scipy.linalg

#################################
# Methods to get one Eigenvalue #
#################################

def power_method(A, maxit=100, tol=1e-05):
    """
    @param {ndarray} A      - the matrix (nxn)
    @param {int} maxit      - maximum number of iterations
    @param {float} tol      - tolerance
    @return {array} [ew, i] - highest eigenvalue (ew) and number of needed iterations (i)
    """
    z = np.random.random((A.shape[0], 1))
    z /= np.linalg.norm(z)

    for i in range(maxit):
        w = A.dot(z)
        w /= np.linalg.norm(w)
        if (np.allclose(z, w, tol)): break
        z = w
    return z.T.dot(A).dot(z)[0][0], i

def ipm(A, maxit, tol):
    """
    @param {ndarray} A      - the matrix (nxn)
    @param {int} maxit      - maximum number of iterations
    @param {float} tol      - tolerance
    @return {array} [z, i]  - lowest z and number of needed iterations (i)
    """    
    P, L, U = scipy.linalg.lu(A)
    z = np.random.random((A.shape[0], 1))
    z /= np.linalg.norm(z)

    for i in range(maxit):
        w = np.linalg.solve(U, np.linalg.solve(L, P.T.dot(z)))
        w /= np.linalg.norm(w)
        if (np.allclose(z, w, tol)): break
        z = w
    return z, i

def inverse_power_method(A, maxit=100, tol=1e-05):
    """
    @param {ndarray} A      - the matrix (nxn)
    @param {int} maxit      - maximum number of iterations
    @param {float} tol      - tolerance
    @return {array} [ew, i] - lowest eigenvalue (ew) and number of needed iterations (i)
    """
    z, i = ipm(A, maxit, tol)
    return z.T.dot(A).dot(z)[0][0], i

def shifted_inverse_power_method(A, n, maxit=100, tol=1e-05):
    """
    @param {ndarray} A      - the matrix (nxn)
    @param {float} n        - close number
    @param {int} maxit      - maximum number of iterations
    @param {float} tol      - tolerance
    @return {array} [ew, i] - closest eigenvalue (ew) and number of needed iterations (i)
    """
    z, i = ipm(A - np.eye(A.shape[0]).dot(n), maxit, tol)
    return z.T.dot(A).dot(z)[0][0], i

##################################
# Methods to get all Eigenvalues #
##################################

def krylov(A, y0, k):
    return lanczos(A, y0, k) if np.array_equal(A, A.conj().T) else arnoldi(A, y0, k)

def arnoldi(A, y0, k):
    """Arnoldi algorithm to compute the Krylov approximation.

    @param {ndarray} A  - the matrix (n,n)
    @param {float} y0   - start vector
    @param {int} k      - number of Krylov steps
    @return {ndarray} V - is the large matrix of shape (n,k+1) containing the orthogonal vectors.
    @return {ndarray} H - is the small matrix of shape (k,k) containing the Krylov approximation of A
    @return {float} Hl  - H[l+1,l], the last value of H, to decide whether EWs of A are EWs of H
    """
    V = np.zeros((A.shape[0], k+1), dtype=np.complexfloating)
    H = np.zeros((k+1, k), dtype=np.complexfloating)

    V[:,0] = y0/np.linalg.norm(y0)
    for l in range(k):
        v_tilde = A.dot(V[:,l])
        for j in range(l):
            H[j,l] = V[:,j].conj().dot(v_tilde)
            v_tilde -= H[j,l]*V[:,j]
        H[l+1,l] = np.linalg.norm(v_tilde)
        if H[l+1,l] == 0: break # Abbruch, weil Kk(A, v) = Kl(A, v)
        V[:,l+1] = v_tilde/H[l+1,l]

    return V, H[:-1,:], H[-1,-1]

def lanczos(A, y0, k):
    """
    Lanczos algorithm to compute the Krylov approximation.

    @param {ndarray} A  - the matrix (n,n)
    @param {float} y0   - start vector
    @param {int} k      - number of Krylov steps
    @return {ndarray} V - is the large matrix of shape (n,k+1) containing the orthogonal vectors.
    @return {ndarray} H - is the small matrix of shape (k,k) containing the Krylov approximation of A
    @return {float} Hl  - H[l+1,l], the last value of H, to decide whether EWs of A are EWs of H
    """   
    V = np.zeros((A.shape[0], k+1), dtype=np.complexfloating)
    a = np.zeros((k,), dtype=np.complexfloating)
    b = np.zeros((k+1,), dtype=np.complexfloating)

    V[:,0] = y0/np.linalg.norm(y0)
    for l in range(k):
        v_tilde = A.dot(V[:,l])
        if l > 0: v_tilde -= b[l]*V[:,l-1]
        a[l] = V[:,l].conj().dot(v_tilde)
        v_tilde -= a[l]*V[:,l]
        b[l+1] = np.linalg.norm(v_tilde)
        if b[l+1] == 0: break # Abbruch, weil Kk(A, v) = Kl(A, v)
        V[:,l+1] = v_tilde/b[l+1]

    b = b[1:-1]
    return V, np.diag(a) + np.diag(b, 1) + np.diag(b, -1), b[-1]

def ews_by_krylov(A, k):
    """
    @param {ndarray} A  - the matrix (nxn)
    @param {int} k      - number of Krylov steps
    @return {array}     - all eigenvalues of A
    """    
    y0 = np.random.random((A.shape[0],))
    y0 /= np.linalg.norm(y0)
    V, H, Hl = krylov(A, y0, min(A.shape[0], k))

    if (Hl == 0 and not np.array_equal(np.diagonal(H, -1), np.zeros(H.shape[0] - 1))):
        # EW of H <=> EW of A
        return scipy.linalg.eigvals(H)
    else:
        return scipy.linalg.eigvals(A)

#########
# Tests #
#########

if __name__ == '__main__':

    # Dies hilft mit der Formatierung von Matrizen. Die Anzahl
    # Nachkommastellen ist 3 und kann hier ----------------------.
    # geändert werden.                                           v
    np.set_printoptions(linewidth=200, formatter={'float': '{: 0.3f}'.format})

    ew = np.array([100.0, 10.0, 12.0, 0.04, 0.234, 3.92, 72.0, 42.0, 77.0, 32.0])
    n = ew.size
    Q, _ = np.linalg.qr(np.random.random((n, n)))
    A = np.dot(Q.transpose(), np.dot(np.diag(ew), Q))

    def construct_matrix(N, kind='minij'):
        def delta(i, j):
            return 1 if i == j else 0

        H = np.zeros((N, N), dtype=np.complexfloating)
        for i in range(N):
            for j in range(N):
                if kind == 'sqrt':
                    H[i, j] = 1 * np.sqrt((i + 1)**2 + (j + 1)**2) + (i + 1)*delta(i, j)
                elif kind == 'dvr':
                    if i != j:
                        t = i - j
                        H[i, j] = 2.0 / t ** 2
                        H[i, j] *= (-1)**t
                elif kind == 'minij':
                    H[i, j] = float(1 + min(i, j))
                else:
                    raise ValueError("Unknown matrix type: "+str(kind))

        if kind == "dvr":
            eps = 0.5
            a = -10.0
            b = 10.0
            h = (b - a) / N
            x = np.linspace(a, b, N)
            H *= (eps / h) ** 2
            H += np.diag(x**2)
            H *= 0.01 / eps

        return H

    N = 2**9

    # TODO: Waehlen Sie die Matrix #
    #kind = 'minij'
    #kind = 'sqrt'
    #kind = 'dvr'

    #A = construct_matrix(N, kind)

    print(A)
    print(ew)
    print("-------------------------")
    
    # TODO Finden sie den grössten Eigenwert vom A mittels Potenzmethode.
    print(power_method(A)[0])

    # TODO Finden sie den kleinsten Eigenwert vom A mittels inverser Potenzmethode.
    print(inverse_power_method(A)[0])

    # TODO Finden sie den Eigenwert am nächsten bei 42.
    print(shifted_inverse_power_method(A, 40)[0])

    # Alle EWs
    #print(ews_by_krylov(A, 50))
    print(np.real(np.around(ews_by_krylov(A, 50), decimals=1)))
