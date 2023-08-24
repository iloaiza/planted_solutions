import numpy as np

from openfermion import (
    FermionOperator,
    hermitian_conjugated,
    normal_ordered,
    commutator
)

from openfermion import get_sparse_operator as gso

from utils import (
    get_SD_expansion,
    get_vac,
    get_eigenvalue,
    get_Splus
)

from math import log

from numpy.random import uniform

def get_Gamma(coefs):
    Norb = len(coefs)

    Gamma = FermionOperator()
    for i, c in enumerate(coefs):
        Gamma += c * get_Splus(i)

    return gso(Gamma, 2 * Norb).toarray()

def get_geminal_power_function(coefs):
    Gamma = get_Gamma(coefs)

    def geminal_power(Np):
        op = np.identity(Gamma.shape[0])
        for _ in range(Np):
            op = op @ Gamma
        return op
    
    return geminal_power

def get_AGP(Gamma, Np, Norb):
    vac      = get_vac(2 * Norb)
    AGP      = Gamma(Np) @ vac
    return AGP

def get_K(p, q, cp, cq):
    term00 = FermionOperator('{}^ {}'.format(2*p,2*q))
    term01 = FermionOperator('{}^ {}'.format(2*p+1,2*q+1))

    term10 = FermionOperator('{}^ {}'.format(2*q,2*p))
    term11 = FermionOperator('{}^ {}'.format(2*q+1,2*p+1))

    return cp * (term00 + term01) - cq * (term10 + term11)

def get_all_K(coefs):
    Norb = len(coefs)

    all_K    = [ [None for _ in range(Norb)] for _ in range(Norb) ] 
    all_Kdag = [ [None for _ in range(Norb)] for _ in range(Norb) ]

    for p in range(Norb):
        for q in range(Norb):
            K    = get_K(p, q, coefs[p], coefs[q]) 
            Kdag = hermitian_conjugated(K)

            all_K[p][q]    = gso(K, 2 * Norb).toarray()
            all_Kdag[p][q] = gso(Kdag, 2 * Norb).toarray()

    return all_K, all_Kdag
