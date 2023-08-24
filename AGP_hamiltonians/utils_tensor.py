import numpy as np

from openfermion import (
    FermionOperator
)

from openfermion import get_sparse_operator as gso

def onebody_tensor_multiply(obt1, obt2):
    N   = obt1.shape[0]
    tbt = np.zeros([N,N,N,N], dtype=np.complex128)
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    tbt[p,q,r,s] = obt1[p,q] * obt2[r,s]
    return tbt

def chem_ten2op(obt, tbt, N):
    op1 = FermionOperator()
    op2 = FermionOperator()
    for p in range(N):
        for q in range(N):
            term = ((p,1), (q,0))
            coef = obt[p,q]
            op1 += FermionOperator(term, coef)
            for r in range(N):
                for s in range(N):
                    term = ((p,1), (q,0), (r,1), (s,0))
                    coef = tbt[p,q,r,s]
                    op2 += FermionOperator(term, coef)
    return op1, op2, op1 + op2

def chem_tbt2array(tbt):
    N = tbt.shape[0]
    _, Hferm, _ = chem_ten2op(np.zeros([N,N]), tbt, N)
    return gso(Hferm, N).toarray()

def get_K_tensor(p, q, cp, cq, Norb):
    obt = np.zeros([2*Norb, 2*Norb], dtype=np.complex128)

    obt[2*p, 2*q]     += cp
    obt[2*p+1, 2*q+1] += cp

    obt[2*q, 2*p]     += -cq
    obt[2*q+1, 2*p+1] += -cq

    return obt

def get_Kdag_tensor(p, q, cp, cq, Norb):
    obt = np.zeros([2*Norb, 2*Norb], dtype=np.complex128)

    obt[2*q, 2*p]     += cp 
    obt[2*q+1, 2*p+1] += cp

    obt[2*p, 2*q]     += -cq
    obt[2*p+1, 2*q+1] += -cq

    return obt

def get_all_K_tensors(coefs):
    Norb = len(coefs)

    all_K    = [ [None for _ in range(Norb)] for _ in range(Norb) ]
    all_Kdag = [ [None for _ in range(Norb)] for _ in range(Norb) ]

    for p in range(Norb):
        for q in range(Norb):
            Kten    = get_K_tensor(p, q, coefs[p], coefs[q], Norb) 
            Kdagten = get_Kdag_tensor(p, q, coefs[p], coefs[q], Norb)
            
            all_K[p][q]    = Kten
            all_Kdag[p][q] = Kdagten

    return all_K, all_Kdag

def get_KdagK_tensor(p, q, r, s, cp, cq, cr, cs, Norb):
    tbt = np.zeros([2*Norb, 2*Norb, 2*Norb, 2*Norb], dtype=np.complex128)

    tbt[2*q, 2*p, 2*r, 2*s]         += cp * cr
    tbt[2*q, 2*p, 2*r+1, 2*s+1]     += cp * cr
    tbt[2*q, 2*p, 2*s, 2*r]         += cp * -cs
    tbt[2*q, 2*p, 2*s+1, 2*r+1]     += cp * -cs

    tbt[2*q+1, 2*p+1, 2*r, 2*s]     += cp * cr
    tbt[2*q+1, 2*p+1, 2*r+1, 2*s+1] += cp * cr
    tbt[2*q+1, 2*p+1, 2*s, 2*r]     += cp * -cs
    tbt[2*q+1, 2*p+1, 2*s+1, 2*r+1] += cp * -cs

    tbt[2*p, 2*q, 2*r, 2*s]         += -cq * cr
    tbt[2*p, 2*q, 2*r+1, 2*s+1]     += -cq * cr
    tbt[2*p, 2*q, 2*s, 2*r]         += -cq * -cs
    tbt[2*p, 2*q, 2*s+1, 2*r+1]     += -cq * -cs

    tbt[2*p+1, 2*q+1, 2*r, 2*s]     += -cq * cr
    tbt[2*p+1, 2*q+1, 2*r+1, 2*s+1] += -cq * cr
    tbt[2*p+1, 2*q+1, 2*s, 2*r]     += -cq * -cs
    tbt[2*p+1, 2*q+1, 2*s+1, 2*r+1] += -cq * -cs

    return tbt

def get_all_KdagK_tensors(coefs):
    Norb = len(coefs)

    tensors = [ [ [ [None for _ in range(Norb)] for _ in range(Norb) ] for _ in range(Norb) ] for _ in range(Norb) ]

    for i in range(Norb):
        for j in range(i+1, Norb):
            for k in range(Norb):
                for l in range(k+1, Norb):
                    tensors[i][j][k][l] = get_KdagK_tensor(i, j, k, l, coefs[i], coefs[j], coefs[k], coefs[l], Norb)
    
    return tensors
