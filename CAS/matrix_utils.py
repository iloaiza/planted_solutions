import numpy as np 
import scipy
from scipy import optimize

def get_imag_symmetric(n):
    '''
    Construct list of i*symmetric matrices kappa_pq based on n*(n-1)/2 
    '''
    kappas = []
    for p in range(n):
        for q in range(p+1, n):
            kappa = np.zeros((n, n), np.complex128)
            kappa[p, q] = 1j
            kappa[q, p] = 1j
            kappas.append(kappa)
    return kappas

def get_anti_symmetric(n):
    '''
    Construct list of anti-symmetric matrices kappa_pq based on n*(n-1)/2 
    '''
    kappas = []
    for p in range(n):
        for q in range(p+1, n):
            kappa = np.zeros((n, n))
            kappa[p, q] = 1
            kappa[q, p] = -1
            kappas.append(kappa)
    return kappas

def construct_anti_symmetric(n, params):
    '''
    Constrcut the nxn anti-symmetric matrix based on the sum of basis with oparams as coefficients
    '''
    real_anti = get_anti_symmetric(n)
    anti_symm = np.zeros((n, n))
    for idx, antisym in enumerate(real_anti):
        anti_symm += params[idx] * antisym
    return anti_symm

def construct_orthogonal(n, params):
    '''
    The parameters are n(n-1)/2 terms that determines the lower diagonals of e^{anti-symmetric}
    '''
    anti_symm = construct_anti_symmetric(n, params)
    return scipy.linalg.expm(anti_symm)

def construct_unitary(n, params):
    '''
    The parameters are n(n-1) terms that determines the lower diagonals of e^{i symm + anti-symmetric}
    '''
    anti_herm = np.zeros((n, n)).astype(np.complex128)
    anti_basis = get_imag_symmetric(n)
    real_anti = get_anti_symmetric(n)
    anti_basis.extend(real_anti)

    for idx, param in enumerate(params):
        anti_herm += param * anti_basis[idx]
    return scipy.linalg.expm(anti_herm)

def get_orthogonal_param(O, tol):
    '''
    Obtain the n(n-1)/2 values v_i that determines the upper diagonals of A where 
    O = e^{A}

    Args:
        O: A real orthogonal matrix 
    
    Returns:
        angles: A n*(n-1)/2 size numpy array of A_{01}, A_{02}, ... coefficients
            where O = e^{A}
    ''' 
    def ortho_cost(params, O):
        O_param = construct_orthogonal(O.shape[0], params)
        diff = O - O_param
        return np.sum(diff * diff)
    
    if np.isclose(np.linalg.det(O), -1):
        O[:, 0] = -O[:, 0]
    n = O.shape[0]
    angs = np.random.rand(int(n*(n-1)/2))
    fun = lambda x: ortho_cost(x, O)
    sol = optimize.minimize(fun, angs, tol=tol)
    return sol.x
