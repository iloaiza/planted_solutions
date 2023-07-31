"""Compare original (Before 2021/07/22) implementation of csa against the parallel implementation. 
Results: Not much parallelized. Only changed wr_lr implementation 
"""
path_prefix = "../" 
import sys 
sys.path.append(path_prefix)

import csa_utils as csau
import matrix_utils as matu 
import ferm_utils as feru 
import saveload_utils as sl 
import numpy as np 
import time 
from openfermion import count_qubits

# Original sum_cartans Implementation 
def cartan_orbtransf(tensor, U, complex=True):
    '''
    tensor : ndarray N x N x N x N
        representing V_pqrs of H = V_pqrs a_p+ a_q a_r+ a_s 
    U : ndaaray N x N
        the orbital transformation
    ortho : boolean
        decides whether U is unitary or orthogonal
    '''
    if not complex:
        return np.einsum('al,bl,cm,dm,llmm->abcd', U, U, U, U, tensor)
    else:
        return np.einsum('al,bl,cm,dm,llmm->abcd', U, np.conj(U), U, np.conj(U), tensor)
def rotated_cartan(ctvals, U, n, complex):
    '''
    Generates a rank-4 tensor representing rotated 
    second-order U(N) cartan based on given parameters

    Params
    -------
    ctvals : ndarray n(n+1)/2 or n**2 
        the values on the second-order cartan
    O : ndarray n x n
        the orthogonal matrix
    '''
    # Building cartan matrix
    tbt = csau.build_tbt(n, complex)
    idx = 0

    for a in range(n):
        for b in range(a, n):
            val = ctvals[idx]
            tbt[a, a, b, b] = val
            tbt[b, b, a, a] = val
            idx += 1

    if complex:
        for a in range(n):
            for b in range(a, n):
                val = ctvals[idx]
                tbt[a, a, b, b] += 1j * val
                if a != b:
                    tbt[b, b, a, a] += 1j * val
                idx += 1

    # Rotate cartan matrix
    return cartan_orbtransf(tbt, U, complex)
def sum_cartans(x, n, alpha, complex):
    '''
    Return the sums of rotated cartans.  

    Params
    -------
    x : ndarray (n**2) * alpha
        All the parameters specifying cartans values and the unitaries. 
        Pictorially the parameters are [p1, p2, p3, p_alpha]
        where pi = [cartan_parameters, u_parameters]
    n : number of orbital
    alpha : number of rotated cartans
    '''
    upnum, ctpnum, pnum = csau.get_param_num(n, complex=complex)

    # Constructing tbt
    tbt = csau.build_tbt(n, complex)

    for i in range(alpha):
        curp = x[i * pnum: (i + 1) * pnum]
        if not complex:
            U = matu.construct_orthogonal(n, curp[ctpnum:])
        else:
            U = matu.construct_unitary(n, curp[ctpnum:])
        tbt += rotated_cartan(curp[:ctpnum], U, n, complex=complex)
    return tbt

# Original grad implementation 
def get_w_o(lmbdas, o_params, n):
    '''
    Computing del W_pqrs / del O_ab
    '''
    def get_cartan_matrix(lmbdas, n):
        '''
        Creates a symmetric rank-2 tensor V with each mode with n indices
        where V_ab are diagvals. This represents product of cartan
        subalgebra of U(N) fermion algebra based on E^p_q 

        Params
        -------
        diagvals : ndarray n(n+1)/2
            the values on the second-order cartan.
        n : the number of orbital
        '''
        V = np.zeros((n, n))
        idx = 0
        for a in range(n):
            for b in range(a, n):
                val = lmbdas[idx]
                V[a, b] = val
                V[b, a] = val
                idx += 1
        return V
    wo = np.zeros((n, n, n, n, n, n))
    O = matu.construct_orthogonal(n, o_params)
    lmbda_matrix = get_cartan_matrix(lmbdas, n)

    delta = np.identity(n)
    wo += np.einsum('pa,qb,rm,sm,bm->pqrsab', delta, O, O, O, lmbda_matrix)
    wo += np.einsum('qa,pb,rm,sm,bm->pqrsab', delta, O, O, O, lmbda_matrix)
    wo += np.einsum('ra,pl,ql,sb,lb->pqrsab', delta, O, O, O, lmbda_matrix)
    wo += np.einsum('sa,pl,ql,rb,lb->pqrsab', delta, O, O, O, lmbda_matrix)

    return wo
def get_o_angles(oparams, i, n):
    '''Computing del O_{ab} / del theta_i

    Params
    -------
    oparams : ndarray n(n-1)/2
        The angles that determines a n by n unitary 
    i : int
        The index of the angles to take gradient
    Returns
    -------
    grad : ndarray n x n
        The gradient w.r.t ith angles
    '''
    kappa = matu.get_anti_symmetric(n)[i]
    K = matu.construct_anti_symmetric(n, oparams)
    D, O = np.linalg.eig(K)

    I = O.conj().T @ kappa @ O
    for a in range(n):
        for b in range(n):
            if abs(D[a] - D[b]) > 1e-8:
                I[a, b] *= (np.exp(D[a] - D[b]) - 1) / (D[a] - D[b])
    expD = np.zeros((n, n)).astype(np.complex128)
    for a in range(n):
        expD[a, a] = np.exp(D[a])
    return np.real(O @ I @ expD @ O.conj().T)
def real_theta_gradients(x, diff):
    '''
    Computing del C / del \thetas
    where thetas are angles for the (anti-symm matrices)
    '''
    n = diff.shape[0]
    ctpnum = int(n * (n + 1) / 2)
    opnum = int(n * (n - 1) / 2)

    wo = get_w_o(x[:ctpnum], x[ctpnum:], n)

    ograd = np.zeros(opnum)
    for i in range(opnum):
        otheta = get_o_angles(x[ctpnum:], i, n)
        wtheta = np.einsum('pqrsab,ab->pqrs', wo, otheta)
        ograd[i] = np.sum(2 * diff * wtheta)
    return ograd
def real_diag_gradients(x, diff):
    '''
    Computing del C / del lambdas
    where lambdas are the symmetric diagonal coefficients for cartan elements
    '''
    def compute_wr_lr(n, l, m, O):
        # TODO: Does below works
        wr_lr = np.zeros((n, n, n, n))
        for p in range(n):
            for q in range(n):
                for r in range(n):
                    for s in range(n):
                        wr_lr[p, q, r, s] = O[p, l] * \
                            O[q, l] * O[r, m] * O[s, m]
        # wr_lr = np.einsum('p,q,r,s->pqrs', O[:, l], O[:, l], O[:, m], O[:, m])
        return wr_lr

    n = diff.shape[0]
    ctpnum = int(n * (n + 1) / 2)
    O = matu.construct_orthogonal(n, x[ctpnum:])

    # The gradients of cost w.r.t lambda
    cl = np.zeros(ctpnum)

    # The gradients of cost w.r.t w
    cw = 2 * diff

    idx = 0
    for l in range(n):
        for m in range(l, n):
            wr_lr = compute_wr_lr(n, l, m, O)
            cl[idx] = np.sum(cw * wr_lr)
            if l != m:
                cl[idx] *= 2
            idx += 1
    return cl
def real_ct_grad(x, target, alpha):
    '''
    Params
    -------
    x : ndarray (n**2) * alpha
        All the parameters specifying cartans values and the orthogonal matrices. 
        Pictorially the parameters are [p1, p2, p3, p_alpha]
        where pi = [cartan_params, o_params]
    target : ndarray  n x n x n x n
        The target interaction tensor
    alpha : the number of diagonal terms used
    '''
    n = target.shape[0]
    diff = sum_cartans(x, n, alpha, complex=False) - target

    xgrad = np.zeros(len(x))
    pnum = int(n ** 2)
    ctpnum = int(n * (n + 1) / 2)

    for i in range(alpha):
        curx = x[i * pnum: (i + 1) * pnum]
        diagrad = real_diag_gradients(curx, diff)
        xgrad[i * pnum: i * pnum + ctpnum] = diagrad
        ograd = real_theta_gradients(curx, diff)
        xgrad[i * pnum + ctpnum: (i + 1) * pnum] = ograd
    return xgrad


# Parameters, mol + alpha
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
alpha = 2 if len(sys.argv) < 3 else int(sys.argv[2])
tol = 1e-10 

# Obtain Htbt 
Hf = sl.load_fermionic_hamiltonian(mol)
Htbt = feru.get_chemist_tbt(Hf)
n = count_qubits(Hf) // 2

# Randomize x.
x = np.random.randn(n**2 * alpha) * 2 * np.pi 

# Get original & parallel sum_cartan output 
ori_sumcartan_start = time.time() 
ori_sumcartan = sum_cartans(x, n, alpha, complex=False)
ori_sumcartan_time = time.time() - ori_sumcartan_start

new_sumcartan_start = time.time() 
new_sumcartan = csau.sum_cartans(x, n, alpha, complex=False)
new_sumcartan_time = time.time() - new_sumcartan_start

dif_sumcartan = np.max(np.abs(ori_sumcartan - new_sumcartan))

# Get original & parallel grad output 
ori_grad_start = time.time() 
ori_grad = real_ct_grad(x, Htbt, alpha)
ori_grad_time = time.time() - ori_grad_start

new_grad_start = time.time() 
new_grad = csau.real_ct_grad(x, Htbt, alpha)
new_grad_time = time.time() - new_grad_start

dif_grad = np.max(np.abs(ori_grad - new_grad))

round_digits = 3
print("Max sum_cartans diff: {}".format(dif_sumcartan))
print("Max gradients diff: {}".format(dif_grad))
print("Time elapsed for original sum_cartans: {} s".format(np.round(ori_sumcartan_time, round_digits)))
print("Time elapsed for new sum_cartans: {} s".format(     np.round(new_sumcartan_time, round_digits)))
print("Time elapsed for original grad: {} s".format(np.round(ori_grad_time, round_digits)))
print("Time elapsed for new grad: {} s".format(     np.round(new_grad_time, round_digits)))
assert dif_sumcartan < tol 
assert dif_grad < tol 
