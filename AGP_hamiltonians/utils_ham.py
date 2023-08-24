import numpy as np

from utils_AGP import (
    get_K,
    get_all_K
)

from utils_tensor import (
    get_K_tensor,
    get_Kdag_tensor,
    get_all_K_tensors,
    get_KdagK_tensor,
    get_all_KdagK_tensors,
    onebody_tensor_multiply
)

from openfermion import (
    hermitian_conjugated,
    FermionOperator
)
from openfermion import get_sparse_operator as gso

from scipy.linalg import expm

EINSUM_PATH = ['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)]

def extract_params(x, Norb, Nterm):
    N_AGP     = Norb
    N_omegas  = Nterm
    N_coefs_A = Norb * (Norb - 1) // 2
    N_coefs   = Nterm * N_coefs_A

    AGPparams = x[ : N_AGP]
    omegas    = x[N_AGP : N_AGP + N_omegas]
    coefs_lin = x[N_AGP + N_omegas : ]

    coefs = list()
    for alpha in range(Nterm):
        coefs.append(coefs_lin[alpha * N_coefs_A : (alpha + 1) * N_coefs_A])

    return AGPparams, omegas, coefs


#
#    Direct Hamiltonian and two-body-tensor calculations
#

def obtain_AGP_hamiltonian_operator(x, Norb, Nterm):
    AGPparams, omegas, coefs = extract_params(x, Norb, Nterm)

    K, Kdag = get_all_K(AGPparams)

    H = np.zeros([2**(2*Norb), 2**(2*Norb)], dtype=np.complex128)
    for alpha in range(Nterm):
        tally_ij = 0
        for i in range(Norb):
            for j in range(i+1, Norb):
                tally_kl = 0
                for k in range(Norb):
                    for l in range(k+1, Norb):
                        op1       = coefs[alpha][tally_ij] * Kdag[i][j]
                        op2       = coefs[alpha][tally_kl] * K[k][l]
                        H        += omegas[alpha] * op1 @ op2
                        tally_kl += 1
                tally_ij += 1
    return H

def obtain_AGP_hamiltonian_tensor(x, Norb, Nterm):
    AGPparams, omegas, coefs = extract_params(x, Norb, Nterm)

    Kten, Kdagten = get_all_K_tensors(AGPparams)

    N   = 2 * Norb
    tbt = np.zeros([N,N,N,N], dtype=np.complex128)

    for alpha in range(Nterm):
        tally_ij = 0
        for i in range(Norb):
            for j in range(i+1, Norb):
                tally_kl = 0
                for k in range(Norb):
                    for l in range(k+1, Norb):
                        obt1      = coefs[alpha][tally_ij] * Kdagten[i][j]
                        obt2      = coefs[alpha][tally_kl] * Kten[k][l]
                        tbt      += omegas[alpha] * onebody_tensor_multiply(obt1, obt2)
                        tally_kl += 1
                tally_ij += 1
    return tbt

#
#    Hamiltonian and two-body-tensor calculations through tildeK
#

def obtain_AGP_hamiltonian_operator_through_tildeK(x, Norb, Nterm):
    AGPparams, omegas, coefs = extract_params(x, Norb, Nterm)

    K, Kdag = get_all_K(AGPparams)

    tildeK = list()
    for alpha in range(Nterm):
        cur_tildeK = np.zeros([2**(2*Norb), 2**(2*Norb)], dtype=np.complex128)
        tally_ij = 0
        for i in range(Norb):
            for j in range(i+1, Norb):
                cur_tildeK += coefs[alpha][tally_ij] * K[i][j]
                tally_ij += 1
        tildeK.append(cur_tildeK)
    tildeKdag = [hermitian_conjugated(op) for op in tildeK]

    H = np.zeros([2**(2*Norb), 2**(2*Norb)], dtype=np.complex128)
    for alpha in range(Nterm):
        H += omegas[alpha] * tildeKdag[alpha] @ tildeK[alpha]
    return H

def obtain_AGP_hamiltonian_tensor_through_tildeK(x, Norb, Nterm):
    AGPparams, omegas, coefs = extract_params(x, Norb, Nterm)

    Kten, Kdagten = get_all_K_tensors(AGPparams)

    N            = 2 * Norb
    tildeKten    = list()
    tildeKdagten = list()
    for alpha in range(Nterm):
        cur_tildeK    = np.zeros([N,N], dtype=np.complex128)
        cur_tildeKdag = np.zeros([N,N], dtype=np.complex128)
        tally_ij      = 0
        for i in range(Norb):
            for j in range(i+1, Norb):
                cur_tildeK    += coefs[alpha][tally_ij] * Kten[i][j]
                cur_tildeKdag += coefs[alpha][tally_ij] * Kdagten[i][j]
                tally_ij      += 1
        tildeKten.append(cur_tildeK)
        tildeKdagten.append(cur_tildeKdag)
    
    tbt = np.zeros([N,N,N,N], dtype=np.complex128)
    for alpha in range(Nterm):
        tbt += omegas[alpha] * onebody_tensor_multiply(tildeKdagten[alpha], tildeKten[alpha])
    return tbt

#
#    orbital rotation implementation
#

def construct_antisymmetric(angles, N):
    X = np.zeros([N,N], dtype=np.complex128)
    tally = 0
    for p in range(N):
        for q in range(p+1, N):
            X[p,q] += angles[tally]
            X[q,p] -= angles[tally]
            tally  += 1
    assert np.allclose(X, -X.T)
    return X

def construct_orthogonal(angles, N):
    return expm(construct_antisymmetric(angles, N))

def construct_orbital_rotation_operator(angles, N):
    X = construct_antisymmetric(angles, N)

    Xferm = FermionOperator()
    for p in range(N):
        for q in range(N):
            term   = ( (p,1), (q,0) )
            coef   = X[p,q]
            Xferm += FermionOperator(term, coef)
    
    Xmat = gso(Xferm, N).toarray()
    return expm(Xmat)

def orbital_rotate(tbt, U):
    return np.einsum('pqrs,pa,qb,rc,sd->abcd', tbt, U, U, U, U)

def unitarily_conjugate(H, U):
    return U.conj().T @ H @ U

#
#    implement Hamiltonian constructors which include orbital rotation
#

def extract_full_params(x, Norb, Nterm):
    N     = 2 * Norb
    N_rot = N * (N - 1) // 2
    
    angles                   = x[ : N_rot] 
    remaining                = x[N_rot : ]
    AGPparams, omegas, coefs = extract_params(remaining, Norb, Nterm)

    return angles, remaining, AGPparams, omegas, coefs

def obtain_AGP_hamiltonian_tensor_rotated(x, Norb, Nterm):
    angles, remaining, AGPparams, omegas, coefs = extract_full_params(x, Norb, Nterm)

    Htbt = obtain_AGP_hamiltonian_tensor(remaining, Norb, Nterm)
    O    = construct_orthogonal(angles, 2 * Norb)
    return orbital_rotate(Htbt, O) 

def obtain_AGP_hamiltonian_operator_rotated(x, Norb, Nterm):
    angles, remaining, AGPparams, omegas, coefs = extract_full_params(x, Norb, Nterm)

    Hmat = obtain_AGP_hamiltonian_operator(remaining, Norb, Nterm)
    Omat = construct_orbital_rotation_operator(angles, 2 * Norb)
    return unitarily_conjugate(Hmat, Omat)

def obtain_AGP_hamiltonian_tensor_rotated_fast(x, Norb, Nterm):
    angles, remaining, AGPparams, omegas, coefs = extract_full_params(x, Norb, Nterm)
    N = 2 * Norb

    tbt = np.zeros([N,N,N,N])
    tally_ij = 0
    for i in range(Norb):
        for j in range(i+1, Norb):
            tally_kl = 0
            for k in range(Norb):
                for l in range(k+1, Norb):

                    coef_ijkl = 0
                    for alpha in range(Nterm):
                        coef_ijkl += omegas[alpha]*coefs[alpha][tally_ij]*coefs[alpha][tally_kl]

                    tbt[2*j,2*i,2*k,2*l]         += coef_ijkl * AGPparams[i] * AGPparams[k]
                    tbt[2*j,2*i,2*k+1,2*l+1]     += coef_ijkl * AGPparams[i] * AGPparams[k]
                    tbt[2*j,2*i,2*l,2*k]         += coef_ijkl * AGPparams[i] * -AGPparams[l]
                    tbt[2*j,2*i,2*l+1,2*k+1]     += coef_ijkl * AGPparams[i] * -AGPparams[l]

                    tbt[2*j+1,2*i+1,2*k,2*l]     += coef_ijkl * AGPparams[i] * AGPparams[k]
                    tbt[2*j+1,2*i+1,2*k+1,2*l+1] += coef_ijkl * AGPparams[i] * AGPparams[k]
                    tbt[2*j+1,2*i+1,2*l,2*k]     += coef_ijkl * AGPparams[i] * -AGPparams[l]
                    tbt[2*j+1,2*i+1,2*l+1,2*k+1] += coef_ijkl * AGPparams[i] * -AGPparams[l]

                    tbt[2*i,2*j,2*k,2*l]         += coef_ijkl * -AGPparams[j] * AGPparams[k]
                    tbt[2*i,2*j,2*k+1,2*l+1]     += coef_ijkl * -AGPparams[j] * AGPparams[k]
                    tbt[2*i,2*j,2*l,2*k]         += coef_ijkl * -AGPparams[j] * -AGPparams[l]
                    tbt[2*i,2*j,2*l+1,2*k+1]     += coef_ijkl * -AGPparams[j] * -AGPparams[l]

                    tbt[2*i+1,2*j+1,2*k,2*l]     += coef_ijkl * -AGPparams[j] * AGPparams[k]
                    tbt[2*i+1,2*j+1,2*k+1,2*l+1] += coef_ijkl * -AGPparams[j] * AGPparams[k]
                    tbt[2*i+1,2*j+1,2*l,2*k]     += coef_ijkl * -AGPparams[j] * -AGPparams[l]
                    tbt[2*i+1,2*j+1,2*l+1,2*k+1] += coef_ijkl * -AGPparams[j] * -AGPparams[l]

                    tally_kl += 1
            tally_ij += 1

    O = construct_orthogonal(angles, N)
    return np.einsum('pqrs,pa,qb,rc,sd->abcd', tbt, O, O, O, O, optimize=EINSUM_PATH)