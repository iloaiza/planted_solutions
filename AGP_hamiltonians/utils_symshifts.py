import numpy as np

from openfermion import (
    FermionOperator
)

from openfermion import get_sparse_operator as gso

from utils_ham import (
    extract_full_params,
    obtain_AGP_hamiltonian_tensor,
    construct_orthogonal,
    orbital_rotate,
    obtain_AGP_hamiltonian_operator,
    construct_orbital_rotation_operator,
    unitarily_conjugate
)

EINSUM_PATH = ['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)]

def total_number_operator(Norb):
    N = 2 * Norb
    op = FermionOperator()
    for p in range(N):
        op += FermionOperator('{}^ {}'.format(p, p))
    return op

def total_number_tensor(Norb, alpha=1):
    N = 2 * Norb
    ten = np.zeros([N,N,N,N])
    for p in range(N):
        ten[p,p,p,p] += alpha
    return ten

def total_number_squared_operator(Norb):
    N = 2 * Norb
    op = FermionOperator()
    for p in range(N):
        for q in range(N):
            op += FermionOperator('{}^ {} {}^ {}'.format(p,p,q,q))
    return op

def total_number_squared_tensor(Norb, beta=1):
    N = 2 * Norb
    ten = np.zeros([N,N,N,N])
    for p in range(N):
        for q in range(N):
            ten[p,p,q,q] += beta
    return ten

def total_seniority_operator(Norb):
    op = FermionOperator()
    for i in range(Norb):
        op += FermionOperator('{}^ {} {}^ {}'.format(2*i,2*i,2*i,2*i))
        op += FermionOperator('{}^ {} {}^ {}'.format(2*i+1,2*i+1,2*i+1,2*i+1))
        op -= FermionOperator('{}^ {} {}^ {}'.format(2*i,2*i,2*i+1,2*i+1))
        op -= FermionOperator('{}^ {} {}^ {}'.format(2*i+1,2*i+1,2*i,2*i))
    return op

def total_seniority_tensor(Norb, gamma=1):
    N = 2 * Norb
    ten = np.zeros([N,N,N,N])
    for i in range(Norb):
        ten[2*i,2*i,2*i,2*i]         += gamma
        ten[2*i+1,2*i+1,2*i+1,2*i+1] += gamma
        ten[2*i,2*i,2*i+1,2*i+1]     -= gamma
        ten[2*i+1,2*i+1,2*i,2*i]     -= gamma
    return ten

def symmetry_shift_operator(alpha, beta, gamma, Norb):
    return (
        alpha * total_number_operator(Norb) +
        beta  * total_number_squared_operator(Norb) +
        gamma * total_seniority_operator(Norb)
    )

def symmetry_shift_tensor(alpha, beta, gamma, Norb):
    return (
        total_number_tensor(Norb, alpha) + 
        total_number_squared_tensor(Norb, beta) + 
        total_seniority_tensor(Norb, gamma)
    )

#
#    implement symmetry shifted AGP hamiltonians
#

def extract_symmetry_shift_params(x):
    return x[:-3], x[-3:]

def obtain_symshift_AGP_hamiltonian_tensor_rotated(x, Norb, Nterm):
    ham_params, sym_params     = extract_symmetry_shift_params(x)
    angles, remaining, _, _, _ = extract_full_params(ham_params, Norb, Nterm)

    Htbt = obtain_AGP_hamiltonian_tensor(remaining, Norb, Nterm)
    Stbt = symmetry_shift_tensor(sym_params[0], sym_params[1], sym_params[2], Norb)
    O    = construct_orthogonal(angles, 2 * Norb)

    return orbital_rotate(Htbt + Stbt, O) 

def obtain_symshift_AGP_hamiltonian_operator_rotated(x, Norb, Nterm):
    ham_params, sym_params     = extract_symmetry_shift_params(x)
    angles, remaining, _, _, _ = extract_full_params(ham_params, Norb, Nterm)

    Hop = obtain_AGP_hamiltonian_operator(remaining, Norb, Nterm)
    Sop = gso(symmetry_shift_operator(sym_params[0], sym_params[1], sym_params[2], Norb), 2 * Norb).toarray()
    Oop = construct_orbital_rotation_operator(angles, 2 * Norb)

    return unitarily_conjugate(Hop + Sop, Oop)

def obtain_symshift_AGP_hamiltonian_tensor_rotated_fast(x, Norb, Nterm):
    ham_params, sym_params                      = extract_symmetry_shift_params(x)
    angles, remaining, AGPparams, omegas, coefs = extract_full_params(ham_params, Norb, Nterm)
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

    #number
    for p in range(N):
        tbt[p,p,p,p] += sym_params[0]

    #number squared
    for p in range(N):
        for q in range(N):
            tbt[p,p,q,q] += sym_params[1]

    #seniority
    for i in range(Norb):
        tbt[2*i,2*i,2*i,2*i]         += sym_params[2]
        tbt[2*i+1,2*i+1,2*i+1,2*i+1] += sym_params[2]
        tbt[2*i,2*i,2*i+1,2*i+1]     += -sym_params[2]
        tbt[2*i+1,2*i+1,2*i,2*i]     += -sym_params[2]

    O = construct_orthogonal(angles, N)
    return np.einsum('pqrs,pa,qb,rc,sd->abcd', tbt, O, O, O, O, optimize=EINSUM_PATH)