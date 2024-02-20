import numpy as np
from matrix_utils import construct_orthogonal
from ferm_utils import get_ferm_op, get_spin_orbitals, braket, get_fermionic_matrix, get_one_body_terms, get_two_body_tensor
from openfermion import normal_ordered, FermionOperator, InteractionOperator, low_rank_two_body_decomposition
from openfermion.linalg import get_sparse_operator, get_ground_state, get_number_preserving_sparse_operator
from qubit_utils import get_qubit_matrix, get_qubit_hf_expectation
import ferm_utils as fermu

def get_system_details(mol):
    '''
    Return number of electrons, number of csa groups
    '''
    if mol == 'h2':
        return 2, 2
    elif mol == 'lih':
        return 4, 8
    elif mol == 'beh2':
        return 6, 12
    elif mol == 'h2o':
        return 10, 10
    elif mol == 'nh3':
        return 10, 12
    elif mol == 'n2':
        return 14, None 

def get_google_measurement(H:InteractionOperator, tiny=1e-6):
    '''
    Get list of L operators to measure through Google's decomposition method. 
    '''
    # one_bd_sq (L x N x N)
    vals, one_bd_sq, one_bd_offset, _ = low_rank_two_body_decomposition(H.two_body_tensor, truncation_threshold=tiny)

    L = len(vals)
    N = one_bd_sq.shape[1]
    ops = []

    for k in range(L):
        mat = one_bd_sq[k, :, :]
        one_bd = FermionOperator.zero()
        for i in range(N):
            for j in range(N):
                one_bd += FermionOperator(
                    term = (
                        (i, 1), (j, 0)
                    ),
                    coefficient=mat[i, j]
                )
        ops.append(vals[k] * one_bd * one_bd)
    return ops, one_bd_offset

'''
def reproduce_google_meas(H : FermionOperator):
    tbt = get_two_body_tensor(H)
    chem_tbt = composite_index(get_chemist_tbt(tbt))

    assert (np.sum(abs(chem_tbt - chem_tbt.T) > tiny) == 0) 

    # Get eigen decomposition and get descending order
    D, V = np.linalg.eigh(chem_tbt)
    D = np.real(D)
    desidx = abs(D).argsort()[::-1]
    D = D[desidx]
    V = V[:, desidx]

    nsq = D.shape[0]
    approx_tbt = add_index(range(ini), D, V)
    err = np.sum(abs(chem_tbt - approx_tbt))

    while err > tiny:
        ini += 1
        approx_tbt += add_index([ini - 1], D, V)
        err = np.sum(abs(chem_tbt - approx_tbt))

    print("All eigenvalues: \n{}".format(D))
    D = D[:ini]
    print("Reproduced Google's result: \n{}".format(D))

    return ini
'''

def get_hf_variance(ops, hf, n, verbose=False, tiny=1e-7):
    '''
    Obtain the sum of variances of operators in ops
    Use hf (e.g. [0, 0, 1]) as the hartree-fock wavefunction
    '''
    var = 0
    if verbose:
        import time 
    for op in ops:
        if verbose:
            start = time.time()
        opsq = op * op
        cur_var = braket(hf, hf, opsq) - braket(hf, hf, op) ** 2
        assert abs(np.imag(cur_var)) < tiny
        cur_var = np.real(cur_var)
        var += cur_var
        if verbose:
            print('Time taken: {}'.format(time.time() - start))
            print("Current variance: {}".format(cur_var))
    return var 

def get_gs(H:FermionOperator):
    '''
    Return the ground state of H
    '''
    Hmat = get_fermionic_matrix(H)
    val, vecs = np.linalg.eigh(Hmat)
    return vecs[:, 0]

def vector_braket(vecl, vecr, H):
    '''
    Obtain the value of <l|H|r> in vector form
    '''
    return (vecl.T @ H @ vecr).item()

def get_fci_variance(ops, gs, n, verbose=False, tiny=1e-7):
    '''
    Obtain the sum of fci variances of operators in ops
    Use vector gs as the wavefunction
    '''
    var = 0
    for op in ops:
        op = get_fermionic_matrix(op, n)
        opsq = op @ op
        cur_var = vector_braket(gs, gs, opsq) - vector_braket(gs, gs, op) ** 2
        assert abs(np.imag(cur_var)) < tiny
        cur_var = np.real(cur_var)
        var += cur_var
        if verbose:
            print("Current variance: {}".format(cur_var))
    return var

def get_qubit_variance(ops, gs, n, verbose=False, tiny=1e-7):
    '''
    In qubit space, obtain the sum of fci variances of operators in ops
    Use vector gs as the wavefunction
    '''
    var = 0
    for op in ops:
        op = get_qubit_matrix(op, n)
        opsq = op @ op
        cur_var = vector_braket(gs, gs, opsq) - vector_braket(gs, gs, op) ** 2
        assert abs(np.imag(cur_var)) < tiny
        cur_var = np.real(cur_var)
        var += cur_var
        if verbose:
            print("Current variance: {}".format(cur_var))
    return var

def get_qubit_hf_variance(hf, ops):
    '''
    Obtain the variance of operators within ops
    Notice hf needs to be in the correct mapping. Transform before this function if ops in BK.
    '''
    e = 0
    varis = np.zeros(len(ops))
    sq_var = 0
    for i, term in enumerate(ops):
        cur_hf_ev = get_qubit_hf_expectation(hf, term)
        e += cur_hf_ev
        cur_hf_var = max(get_qubit_hf_expectation(hf, term * term) - cur_hf_ev ** 2, 0)
        varis[i] = cur_hf_var
        sq_var += cur_hf_var ** (1/2)
    return e, varis, sq_var

def get_hf_expmap(hf):
    '''
    Obtain mapping from a two-body tensor in chemist order to expectation
    TODO: can be more efficient 
    '''
    norb = len(hf) // 2
    mp = np.zeros((norb, norb, norb, norb))

    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    ### 2x faster 
                    #term = FermionOperator(term=(
                    #            (2*p, 1), (2*q, 0),
                    #            (2*r, 1), (2*s, 0)
                    #        ))
                    #mp[p, q, r, s] += 2*braket(hf, hf, term)
                    #term = FermionOperator(term=(
                    #            (2*p+1, 1), (2*q+1, 0),
                    #            (2*r, 1), (2*s, 0)
                    #        ))
                    # mp[p, q, r, s] += 2*braket(hf, hf, term)
                    for a in range(2):
                        for b in range(2):
                            op = FermionOperator(term=(
                                (2*p+a, 1), (2*q+a, 0),
                                (2*r+b, 1), (2*s+b, 0)
                            ))
                            mp[p, q, r, s] += braket(hf, hf, op)
    return mp

def get_hf_sqmap(hf):
    '''
    Obtain mapping from a two-body tensor in chemist order to expectation of square
    TODO: can be more efficient, in a/b/c/d part 
    '''
    norb = len(hf) // 2
    mp = np.zeros((norb**4, norb**4))

    row = 0
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    col = 0
                    for u in range(norb):
                        for v in range(norb):
                            for j in range(norb):
                                for k in range(norb):
                                    for a in range(2):
                                        for b in range(2):
                                            for c in range(2):
                                                for d in range(2):
                                                    op = FermionOperator(term=(
                                                        (2*p+a, 1), (2*q+a, 0),
                                                        (2*r+b, 1), (2*s+b, 0)
                                                    )) * FermionOperator(term=(
                                                        (2*u+c, 1), (2*v+c, 0),
                                                        (2*j+d, 1), (2*k+d, 0)
                                                    ))
                                                    mp[row, col] += braket(hf, hf, op)
                                    col += 1
                    row += 1
    return mp

def get_tbts_hf_var(tbts, exp_mp, sq_mp):
    '''
    Return the hartree fock variances
    '''
    variances = []
    for tbt in tbts:
        tbtflat = tbt.flatten()
        curvar = tbtflat.T @ sq_mp @ tbtflat - np.sum(tbt * exp_mp) ** 2
        curvar = max(curvar, 0)
        variances.append(curvar)
    variances = np.array(variances)

    return variances

def get_tbts_hf_sqrtvar(tbts, exp_mp, sq_mp, display=True):
    '''
    Return the hartree fock sqrt variance 
    '''
    variances = get_tbts_hf_var(tbts, exp_mp, sq_mp)
    
    if display:
        print("Variances: {}".format(variances))
        print("Sqrt Variances: {}".format(variances ** (1/2)))
    sqrtv = np.sum(variances ** (1/2))

    return sqrtv

def get_one_body_correction(H : FermionOperator, obt):
    '''
    Returning the one body offset from original Hamiltonain 
    and the one body tensor from chemist reordering 
    '''
    org_one_body = get_one_body_terms(H)
    chem_one_body = get_ferm_op(obt, spin_orb=True)
    return org_one_body + chem_one_body

def get_one_body_correction_from_tbt(H: FermionOperator, Htbt=None):
    """ Returning the one body difference from original Hamiltonian and two body tensor. 
    """
    if Htbt is None: 
        Htbt = fermu.get_chemist_tbt(H)
    
    return normal_ordered(H - get_ferm_op(Htbt, spin_orb=False))
    
def tbt_rangom_gen(variance, orb=4, mean=0):
    '''
    Generate at random orb x orb x orb x orb tensor 
    with specified variance and mean 
    '''
    s = np.random.normal(mean, variance, orb**4)
    s = np.reshape(s, (orb, orb, orb, orb))
    s = s + np.einsum('ijkl->lkji', s)
    return s/2

def tbt_variance_estimate(tbt):
    '''
    Estimating the variance by taking the element-wise variance
    '''
    tbtflt = tbt.flatten()
    return np.var(tbtflt)

def single_lambda_variance_estimate(lambda_matrix, nelec):
    '''
    Return 1/4 * (highest_eigval - lowest_eigval)**2  
    
    Args:
        lambda_matrix: norb x norb numpy matrix representing lambda[i, j]*n_i*n_j fermionic operator

    Returns:
        A number representing the variance estimate
    '''    
    norb = lambda_matrix.shape[0]
    ferm_op = get_ferm_op(lambda_matrix, spin_orb=False)
    ferm_mat = get_number_preserving_sparse_operator(ferm_op, num_qubits=2*norb, num_electrons=nelec)
    low, _ = get_ground_state(ferm_mat)
    high, _ = get_ground_state(-ferm_mat)
    high = -high

    return 1/4 * (high - low)**2 

def get_eigvaluebased_cartan_variance_estimate(cartan_matrices, num_electron):
    '''
    Return list of variance estimates based on cartan matrices 
    using equation: 1/4 * (highest_eigval - lowest_eigval)**2  

    Args: 
        cartan_matrices: The list of cartan matrices 
    
    Returns:
        variances_est: The numpy array of variance estimates 
    '''
    variances_est = np.full(len(cartan_matrices), np.nan)
    for idx, cm in enumerate(cartan_matrices):
        variances_est[idx] = single_lambda_variance_estimate(cm, num_electron)
    return variances_est
