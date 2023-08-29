import numpy as np

from math import log

from openfermion import (
    FermionOperator,
    hermitian_conjugated,
    normal_ordered,
    commutator
)

from openfermion import get_sparse_operator as gso

def get_binary_string(n, N=None):
    z = bin(n)[2:]

    if N is None:
        return z
    
    elif len(z) == N:
        return z

    elif len(z) > N:
        print('string longer than desired length')
        return z
    
    else:
        while len(z) < N:
            z = '0' + z
        return z

def get_SD_expansion(v):
    n_qubits = log(len(v), 2)

    SD = dict()
    for n, coef in enumerate(v):
        if np.abs(coef) > 1e-6:
            bin_string = get_binary_string(n, n_qubits)
            SD[bin_string] = coef
    return SD

def get_eigenvalue(O, v):
    if np.allclose(v, 0):
        return "v is zero"
    
    spectrum = sorted(np.linalg.eig(O)[0])
    w        = O @ v
    for z in spectrum:
        if np.allclose(w, z * v):
            return z
    return "not an eigenvector"

def is_positive_semidefinite(O):
    spectrum = np.linalg.eig(O)[0]
    min_eigenvalue = min(spectrum)
    return min_eigenvalue + pow(10, -6) > 0

def is_ground_state(O, v):
    if np.allclose(v, 0):
        return "v is zero"

    spectrum = np.linalg.eig(O)[0]
    min_eigenvalue = min(spectrum)
    return np.allclose(O @ v, min_eigenvalue * v)

def num(p):
    return FermionOperator('{}^ {}'.format(p, p))

def get_vac(n_qubits):
    v    = np.zeros(2**n_qubits)
    v[0] = 1
    return v

def get_Splus(p):
    return FermionOperator('{}^ {}^'.format(2*p, 2*p+1))
