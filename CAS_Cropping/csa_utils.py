from matrix_utils import construct_unitary, construct_orthogonal, get_anti_symmetric, construct_anti_symmetric, get_orthogonal_param
from ferm_utils import get_hf, get_ferm_op
from var_utils import get_hf_expmap, get_hf_sqmap
from openfermion import FermionOperator
import numpy as np
from itertools import product
import scipy

def build_tbt(n, complex):
    '''
    Initialize two body tensor matrix
    '''
    if not complex:
        tbt = np.zeros((n, n, n, n))
    else:
        tbt = np.zeros((n, n, n, n), np.complex128)
    return tbt

def get_param_num(n, k, complex = False):
    '''
    Counting the parameters needed, where k is the number of orbitals occupied by CAS Fragments,
    and n-k orbitals are occupied by the CSA Fragments
    '''
    if not complex:
        upnum = int(n * (n - 1) / 2)
    else:
        upnum = n * (n - 1)
    casnum = 0
    for block in k:
        casnum += len(block) ** 4
    pnum = upnum + casnum

    return upnum, casnum, pnum

def get_cartan_parts(x, n, alpha, complex=False):
    '''
    Return the diagonal lambda_ij of each cartan parts in a list 
    '''
    def get_cartan(ctvals, complex, partitions = None):
        '''
        Making cartan matrix given values 
        '''
        tbt = build_tbt(n, complex)
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
        return tbt

    _, ctpnum, pnum = get_param_num(n, complex)
    cartans = []
    for i in range(alpha):
        curp = x[i * pnum: (i + 1) * pnum]
        cartans.append(get_cartan(curp[:ctpnum], complex))
    return cartans


def get_tbt_parts(x, n, alpha, k, complex=False):
    '''
    The two-body tensor of each cartan parts in a list
    '''
    upnum, casnum, pnum = get_param_num(n, k, complex)

    tbts = []
    for i in range(alpha):
        curp = x[i * pnum: (i + 1) * pnum]
        if not complex:
            U = construct_orthogonal(n, curp[casnum:])
        else:
            U = construct_unitary(n, curp[casnum:])
        tbts.append(rotated_cartan(curp[:casnum], U, n, k, complex))
    return tbts

def tbt2ferm(tbts):
    '''
    Transforms a list of tbts to Fermionic Operators 
    '''
    H = FermionOperator.zero()
    for tbt in tbts:
        import pdb
        pdb.set_trace()
        H += get_ferm_op(tbt)
    return H

    
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
        p = np.einsum_path('ak,bl,cm,dn,klmn->abcd', U, U, U, U, tensor)[0]
        return np.einsum('ak,bl,cm,dn,klmn->abcd', U, U, U, U, tensor, optimize = p)
#         al,bl,cm,dm,llmm->abcd
    else:
        p = np.einsum_path('ak,bl,cm,dn,klmn->abcd', U, np.conj(U), U, np.conj(U), tensor)[0]
        return np.einsum('ak,bl,cm,dn,klmn->abcd', U, np.conj(U), U, np.conj(U), tensor, optimize = p)
#         return np.einsum('al,bl,cm,dm,llmm->abcd', U, np.conj(U), U, np.conj(U), tensor)


def rotated_cartan(ctvals, U, n, k, complex, killer = None):
    '''
    Generates a rank-4 tensor representing rotated 
    second-order U(N) cartan based on given parameters

    Params
    -------
    ctvals : ndarray n(n+1)/2 or n**2 
        the values on the second-order cartan
    U : ndarray n x n
        the orthogonal matrix
    n : Total Orbital Number
    k : Number of orbitals occupied by CAS fragments
    '''
    # Building cartan matrix
    tbt = build_tbt(n, complex)
    idx = 0
    for block in k:
        for a in block:
            for b in block:
                for c in block:
                    for d in block:
                        tbt [a,b,c,d] = ctvals[idx]
                        idx += 1

#     if killer:
#         for t in killer.terms:
            
                    

    # Rotate cartan matrix
    return cartan_orbtransf(tbt, U, complex)

def sum_cartans(x, n, k, alpha, complex):
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
    upnum, casnum, pnum = get_param_num(n, k, complex=complex)

    # Constructing tbt
    tbt = build_tbt(n, complex)

    for i in range(alpha):
        curp = x[i * pnum: (i + 1) * pnum]
        if not complex:
            U = construct_orthogonal(n, curp[casnum:])
        else:
            U = construct_unitary(n, curp[casnum:])
        tbt += rotated_cartan(curp[:casnum], U, n, k, complex)
    return tbt


def tbt_cost(x_tbt, target_tbt, complex):
    '''
    Return the squared cost to minimize
    '''
    diff = target_tbt - x_tbt

    if complex:
        diffr, diffi = np.real(diff), np.imag(diff)
        cost = np.sum(diffr * diffr) + np.sum(diffi * diffi)
    else:
        cost = np.sum(diff * diff)
    return cost


def ct_decomp_cost(x, k, target_tbt, alpha, complex):
    '''
    Return the cost to minimize in cartan decomposition
    '''
    n = target_tbt.shape[0]
    x_tbt = sum_cartans(x, n, k, alpha, complex)

    return tbt_cost(x_tbt, target_tbt, complex)

# def ct_decomp_cost_killer(x, k, target_tbt, alpha, complex):
# #     TODO:
#     '''
#     Return the cost to minimize in cartan decomposition, ignoring killer terms
#     '''
#     n = target_tbt.shape[0]
#     x_tbt = sum_cartans(x, n, k, alpha, complex)
#     for orbs in k:
#         for l in orbs:
#             for p, q in product([i for i in range(n) if i not in orbs], repeat = 2):
#                 target_tbt[l,l,p,q] = 0
#                 target_tbt[p,q,l,l] = 0
#     return tbt_cost(x_tbt, target_tbt, complex)

def get_cas_matrix(lmbdas, n, k):
    """
    Get the CAS matrix defined by the lambdas, with the CAS blocks as a list k.
    """
    cas = np.zeros((n,n,n,n))
    idx = 0
    for block in k:
        for i in block:
            for j in block:
                for k in block:
                    for l in block:
                        cas[i,j,k,l] = lmbdas[idx]
                        idx += 1
    return cas

def get_w_o(lmbdas, o_params, n, k):
    '''
    Computing del W_pqrs / del O_ab
    '''
    wo = np.zeros((n, n, n, n, n, n))
    O = construct_orthogonal(n, o_params)
    lmbda_matrix = get_cas_matrix(lmbdas, n, k)

    delta = np.identity(n)
    wo += np.einsum('ajkl,jq,kr,ls,pb->pqrsab', lmbda_matrix, O, O, O, delta)
    wo += np.einsum('iakl,ip,kr,ls,qb->pqrsab', lmbda_matrix, O, O, O, delta)
    wo += np.einsum('ijal,ip,jq,ls,rb->pqrsab', lmbda_matrix, O, O, O, delta)
    wo += np.einsum('ijka,ip,jq,kr,sb->pqrsab', lmbda_matrix, O, O, O, delta)

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
    kappa = get_anti_symmetric(n)[i]
    K = construct_anti_symmetric(n, oparams)
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


def real_theta_gradients(x, k, diff):
    '''
    Computing del C / del \thetas
    where thetas are angles for the (anti-symm matrices)
    '''
    n = diff.shape[0]
    casnum = 0
    for block in k:
        casnum += len(block) ** 4
    opnum = int(n * (n - 1) / 2)
    
#     print(x.shape)
#     print(casnum)
    wo = get_w_o(x[:casnum], x[casnum:], n, k)

    ograd = np.zeros(opnum)
    for i in range(opnum):
        otheta = get_o_angles(x[casnum:], i, n)
        wtheta = np.einsum('pqrsab,ab->pqrs', wo, otheta)
        ograd[i] = np.sum(2 * diff * wtheta)
    return ograd


def real_diag_gradients(x, k, diff):
    '''
    Computing del C / del lambdas
    where lambdas are the coefficients for different CAS blocks
    '''
    n = diff.shape[0]
    casnum = 0
    for block in k:
        casnum += len(block) ** 4
    O = construct_orthogonal(n, x[casnum:])

    # The gradients of cost w.r.t lambda
    cl = np.zeros(casnum)

    # The gradients of cost w.r.t w
    cw = 2 * diff

    idx = 0
#     for l in range(n):
#         for m in range(l, n):
#             wr_lr = np.einsum('p,q,r,s->pqrs', O[:, l], O[:, l], O[:, m], O[:, m])
#             cl[idx] = np.sum(cw * wr_lr)
#             if l != m:
#                 cl[idx] *= 2
#             idx += 1
    for block in k:
        for i in block:
            for j in block:
                for k in block:
                    for l in block:
                        wr_lr = np.einsum('p,q,r,s->pqrs', O[:, i], O[:, j], O[:, k], O[:, l])
                        cl[idx] = np.sum(cw * wr_lr)
                        idx += 1
    return cl


def real_cas_grad(x, k, target, alpha):
    '''
    Params
    -------
    x : ndarray (n**2) * alpha
        All the parameters specifying cartans values and the orthogonal matrices. 
        Pictorially the parameters are [p1, p2, p3, p_alpha]
        where pi = [CAS_params, o_params]
    k : the list of SIs, each SI contains the orbitals contained in each CAS block
    Union of SI = range(n)
    target : ndarray  n x n x n x n
        The target interaction tensor
    alpha : the number of diagonal terms used
    '''
    n = target.shape[0]
    diff = sum_cartans(x, n, k, alpha, complex=False) - target

    xgrad = np.zeros(len(x))
    upnum, casnum, pnum = get_param_num(n, k = k, complex=False)

    for i in range(alpha):
        curx = x[i * pnum: (i + 1) * pnum]
        diagrad = real_diag_gradients(curx, k, diff)
        xgrad[i * pnum: i * pnum + casnum] = diagrad
        ograd = real_theta_gradients(curx,k, diff)
        xgrad[i * pnum + casnum: (i + 1) * pnum] = ograd
    return xgrad


def csa(target, k, alpha=1, tol=1e-6, grad=False, killer = False, complex=False, x0=None, maxiter=10000, disp=False):
    '''
    Decompose two-body terms using cartan subalgebra and orthogonal transformation
    '''
    # Computing proper tolerance
    n = target.shape[0]
    enum = n ** 4
    fun_tol = (tol / enum) ** 2
    if killer:
        for orbs in k:
            for l in orbs:
                for p, q in product([i for i in range(n) if i not in orbs], repeat = 2):
                    target[l,l,p,q] = 0
                    target[p,q,l,l] = 0
    def fun(x): return ct_decomp_cost(x, k, target, alpha, complex)
    upnum, casnum, pnum = get_param_num(n, k = k, complex=complex)

    # Randomly initiate angles.
    if x0 is None:
        x0 = np.zeros(pnum * alpha)
        for i in range(alpha):
            x0[pnum * i + casnum: pnum *
                (i + 1)] = np.random.rand(upnum) * 2 * np.pi
    
    options = {
        "disp": True,
        "maxiter": maxiter
    }

    # TODO: Include complex gradient
    if grad and not complex:
        def gradfun(x): return real_cas_grad(x, k, target, alpha)
        return scipy.optimize.minimize(fun, x0, options=options, tol=fun_tol, method='BFGS', jac = gradfun)
    else:
        return scipy.optimize.minimize(fun, x0, options=options, tol=fun_tol)


def get_precomputed_variance(tbt, tbtev_ln, tbtev_sq):
    """Use the precomputed expectation values, compute the variance of the given two-body tensor (tbt). 

    Args:
        tbt (ndarray N^4): The two-body tensor representing the operator of interest. 
        tbtev_ln (ndarray N^4): The expectation values of corresponding entries. tbtev_ln[i, j, k, l] = < tbt[i, j, k, l] > 
        tbtev_sq (ndarray N^8): The expectation values of squared tbt. 
            tbtev_sq[i,j,k,l,a,b,c,d] = < tbt[i,j,k,l] * tbt[a,b,c,d] > 

    Returns:
        variance (float): The variance of operator represented by the two body tensor. 
    """
    # Get <tbt>
    ev_ln = np.einsum("ijkl,ijkl->", tbt, tbtev_ln)

    # Get <tbt^2>
    ev_sq = np.einsum("ijkl,abcd,ijklabcd->", tbt, tbt, tbtev_sq)

    return ev_sq - ev_ln**2


def var_theta_gradients(x, n):
    """ Compute del C / del lambdas 
    where lambdas are the symmetric diagonal coefficients for cartan elements

    Args:
        x (ndarray): The n**2 parameters of current two-body tensor. 
        n (int): Number of orbitals. 

    Returns:
        wt (ndarray): The gradient del W_{pqrs} / del theta of shape (p, q, r, s, n_theta)
    """
    # Setup
    opnum, ctpnum, _ = get_param_num(n, complex=False)
    cl = np.zeros(ctpnum)

    # Build del W_{pqrs} / del O_{ab}
    wo = get_w_o(x[:ctpnum], x[ctpnum:], n)

    wt = np.zeros((n, n, n, n, opnum))
    for i in range(opnum):
        # Build del O_{ab} / del theta
        otheta = get_o_angles(x[ctpnum:], i, n)
        wt[:, :, :, :, i] = np.einsum('pqrsab,ab->pqrs', wo, otheta)
    return wt


def var_diag_gradients(x, n):
    """ Compute del C / del lambdas 
    where lambdas are the symmetric diagonal coefficients for cartan elements

    Args:
        x (ndarray): The n**2 parameters of current two-body tensor. 
        n (int): Number of orbitals. 

    Returns:
        wl (ndarray): The gradient del W_{pqrs} / del lambda of shape (p, q, r, s, n_diag) 
    """
    # Setup
    _, ctpnum, _ = get_param_num(n, complex=False)
    wl = np.zeros((n, n, n, n, ctpnum))

    # Compute del W / del lambda
    idx = 0
    O = construct_orthogonal(n, x[ctpnum:])
    for l in range(n):
        for m in range(l, n):
            scale = 1 if l == m else 2
            wl[:, :, :, :, idx] = scale * np.einsum(
                'p,q,r,s->pqrs', O[:, l], O[:, l], O[:, m], O[:, m])
            idx += 1
    return wl


def var_grad(x, target, alpha, var_weight, tbtev_ln, tbtev_sq):
    """ Compute the gradient for VCSA 
    Args:
        x (ndarray (n**2) * alpha): The parameters specifying cartans values and the unitaries. 
        target (ndarray N^4):  The 4-rank two body tensor to approximate. 
        alpha (int): The number of CSA fragments to use. 
        var_weight (float): The weight multiplied to the variance terms. 
        tbtev_ln (ndarray N^4): The expectation values of corresponding entries. 
            tbtev_ln[i, j, k, l] = < tbt[i, j, k, l] > 
        tbtev_sq (ndarray N^8): The expectation values of squared tbt. 
            tbtev_sq[i,j,k,l,a,b,c,d] = < tbt[i,j,k,l] * tbt[a,b,c,d] > 
    Returns:
        v_grad (ndarray): The gradient for each parameter in x. 
    """
    n = target.shape[0]

    # Compute sqrt(v_i) and \sum_i sqrt(v_i)
    tbts = get_tbt_parts(x, n, alpha, complex=False)
    sqrt_vs = np.zeros(alpha)
    for i in range(alpha):
        curv = get_precomputed_variance(tbts[i], tbtev_ln, tbtev_sq)
        curv = max(0, np.real(curv))
        sqrt_vs[i] = curv ** (1 / 2)

    # Building W_{pqrs} - V_{pqrs}
    tbt_sum = np.zeros((n, n, n, n))
    for i in range(alpha):
        tbt_sum += tbts[i]
    diff = tbt_sum - target

    # Compute v_grad
    v_grad = np.zeros(len(x))
    upnum, ctpnum, pnum = get_param_num(n, complex=False)
    for i in range(alpha):
        # For numerical stability
        if sqrt_vs[i] < 1e-6:
            cur_weight = 0
        else:
            cur_weight = var_weight * np.sum(sqrt_vs) / sqrt_vs[i]

        # Compute del C / del W_{pqrs}
        cw = 2 * diff + cur_weight * (
            np.einsum('ijkl,abcdijkl->abcd', tbts[i], tbtev_sq) +
            np.einsum('pqrs,pqrsabcd->abcd', tbts[i], tbtev_sq) -
            2 * np.einsum('pqrs,pqrs->', tbtev_ln, tbts[i]) * tbtev_ln
        )

        # Get (p, q, r, s, n_diag/theta) array for del W_{pqrs} / del_lambda/theta
        curx = x[i * pnum: (i + 1) * pnum]
        diag_grad = var_diag_gradients(curx, n)
        diag_grad = np.einsum("pqrs,pqrsi->i", cw, diag_grad)
        v_grad[i * pnum: i * pnum + ctpnum] = np.real(diag_grad)
        ograd = var_theta_gradients(curx, n)
        ograd = np.einsum("pqrs,pqrsi->i", cw, ograd)
        v_grad[i * pnum + ctpnum: (i + 1) * pnum] = np.real(ograd)

    return v_grad


def var_decomp_cost(x, target, alpha, var_weight, tbtev_ln, tbtev_sq):
    ''' Compute the cost: sum_of_squared_tbt_difference + var_weight * (sum of sqrt variance)^2 

    Args:
        x (ndarray (n**2) * alpha): The parameters specifying cartans values and the unitaries. 
        target (ndarray N^4):  The 4-rank two body tensor to approximate. 
        alpha (int): The number of CSA fragments to use. 
        var_weight (float): The weight multiplied to the variance terms. 
        tbtev_ln (ndarray N^4): The expectation values of corresponding entries. 
            tbtev_ln[i, j, k, l] = < tbt[i, j, k, l] > 
        tbtev_sq (ndarray N^8): The expectation values of squared tbt. 
            tbtev_sq[i,j,k,l,a,b,c,d] = < tbt[i,j,k,l] * tbt[a,b,c,d] > 

    Returns:
        cost (float): The cost sum_of_squared_tbt_difference + var_weight * (sum of sqrt variance)^2 
    '''
    norb = target.shape[0]
    total_tbt = build_tbt(norb, complex=False)
    upnum, ctpnum, pnum = get_param_num(norb, complex=False)

    # Building variance
    tbts = get_tbt_parts(x, norb, alpha)
    sqrtv = 0
    for tbt in tbts:
        curvar = get_precomputed_variance(tbt, tbtev_ln, tbtev_sq)
        curvar = max(0, curvar)
        sqrtv += curvar ** (1 / 2)
        total_tbt += tbt
    sq_sumsqrtv = np.real(sqrtv) ** 2

    diff_cost = tbt_cost(total_tbt, target, complex=False)

    return diff_cost + var_weight * sq_sumsqrtv


def varcsa(target, tbtev_ln, tbtev_sq, alpha, var_weight, tol=1e-6, grad=False, options=None, x0=None):
    '''Decompose two-body terms using cartan subalgebra and orthogonal transformatio 
    with HF variance added in the penalty term. 
    The cost function will be
        sum_of_squared_tbt_difference + var_weight * (sum of sqrt variance)^2 

    Args:
        target (ndarray N^4): The 4-rank two body tensor that the optimization will try to approximate. 
        tbtev_ln (ndarray N^4): The expectation values of corresponding entries. 
            tbtev_ln[i, j, k, l] = < tbt[i, j, k, l] > 
        tbtev_sq (ndarray N^8): The expectation values of squared tbt. 
            tbtev_sq[i,j,k,l,a,b,c,d] = < tbt[i,j,k,l] * tbt[a,b,c,d] > 
        alpha (int): The number of CSA fragments to use. 
        var_weight (float): The weight multiplied to the variance terms. 
        tol: The optimization tolerance. 
        grad (bool): Whether to use gradient
        options (Dict): options for scipy
        x0 (ndarray): The initial parameters. 

    Returns: 
        sol: The scipy's optimizer's solution. sol.x, sol.fun are the converged converged solution and cost function value. 
    '''
    norb = target.shape[0]
    enum = norb ** 4
    fun_tol = (tol / enum) ** 2

    def fun(x): return var_decomp_cost(
        x, target, alpha, var_weight, tbtev_ln, tbtev_sq)

    if x0 is None: 
        # Initiate starting point. Randomize angle.
        upnum, ctpnum, pnum = get_param_num(norb, complex=False)
        x0 = np.random.rand(pnum * alpha)
        for i in range(alpha):
            x0[pnum * i + ctpnum: pnum *
                (i + 1)] = np.random.rand(upnum) * 2 * np.pi

    if grad:
        def gradfun(x): 
            return var_grad(x, target, alpha, var_weight, tbtev_ln, tbtev_sq)
        return scipy.optimize.minimize(fun, x0, tol=fun_tol, method='BFGS', jac=gradfun, options=options)
    else:
        return scipy.optimize.minimize(fun, x0, tol=fun_tol, method='BFGS', options=options)


def get_one_bd_sq(tbt, tol, verbose=False):
    '''
    Return the list of Lpq in matrix form. 

    Args:
        tbt: The 4-rank two body tensor g_{pqrs} over orbital
        tol: Eigenvalue cutoff
        verbose (bool): Whether to print out SVD's eigenvalues. 

    Returns:
        Ls: list of Ls where g_{pqrs} = sum_k L^k_pq L^k_rs
    '''
    # Prepare composite indices
    norb = tbt.shape[0]
    comp_tbt = np.reshape(tbt, (norb**2, norb**2))

    # Compute SVD
    D, V = np.linalg.eigh(comp_tbt)

    # Trim low eigenvalues and order
    D, V = D[abs(D) > tol], V[:, abs(D) > tol]
    descend_idx = abs(D).argsort()[::-1]
    D, V = D[descend_idx], V[:, descend_idx]

    if verbose:
        print("1st SVD eigenvalues: {}".format(D))

    # Collect all one-body sq matrices
    Ls = []
    for i in range(len(D)):
        cur_l = (np.complex(D[i])**(1 / 2)) * np.reshape(V[:, i], (norb, norb))
        Ls.append(cur_l)
    return Ls


def get_svdcsa_sol(tbt, tol=1e-6):
    '''
    Return the csa solution vector based on google's SVD method 

    Args:
        tbt: The 4-rank two body tensor g_{pqrs} over orbital
        tol: Tolerance for eliminating terms from google's SVD method 

    Returns:
        csa_x: The solution vector [lambda_i, theta_i, ...] representing 
            the csa equivalent of google's solution
        L: # of measurable terms, i enumerates to L.
    '''
    def get_solvec_from_one_bd_sq(one_bd_sq):
        '''
        Return the lambda and theta coefficients from one_bd_sq 

        Args:
            one_bd_sq: The matrix L_{pq} over orbital

        Returns:
            sol: [lambda, theta] representing matrix L_{pq}
        '''
        norb = one_bd_sq.shape[0]
        W, O = np.linalg.eigh(one_bd_sq)

        opnum, ctpnum, _ = get_param_num(norb, complex=False)

        # Building lambda
        lmbds = np.full(ctpnum, np.nan)
        idx = 0
        for a in range(norb):
            for b in range(a, norb):
                lmbds[idx] = W[a] * W[b]
                idx += 1

        # Building opnum
        thetas = get_orthogonal_param(O, 1e-11)

        return np.concatenate([lmbds, thetas], axis=0)

    norb = tbt.shape[0]
    list_one_bd_sq = get_one_bd_sq(tbt, tol=tol)

    # Building csa_x
    L = len(list_one_bd_sq)
    _, _, pnum = get_param_num(norb, complex=False)
    csa_x = np.full(pnum * L, np.nan)
    for i in range(L):
        csa_x[i * pnum:(i + 1) *
              pnum] = get_solvec_from_one_bd_sq(list_one_bd_sq[i])
    return csa_x, L


def get_svd_fragments(tbt, tol=1e-8, verbose=False):
    '''Return a list of Fermionic Operators, each computed by SVD proceduare and is measurable using orbital rotation. 
    The two body tensor (tbt) is assumed to be in orbital, and fermionic operator in spin-orbitals will be returned. 
    So count_qubits(fragments[i]) == 2 * tbt.shape[0], 

    Args:
        tbt: The 4-rank two body tensor g_{pqrs} over orbital
        tol: Eigenvalue cutoff
        verbose (bool): Whether to print out SVD's eigenvalues. 

    Returns:
        fragments: list of FermionicOperator hat{L_k} where tbt_{pqrs} = sum_k L^k_pq L^k_rs, 
            hat{L_k} = L^k_pq L^k_rs a_p^+ a_q a_r^+ a_s 
    '''
    Ls = get_one_bd_sq(tbt, tol=tol, verbose=verbose)
    Lops = []
    for L in Ls:
        one_bd_sq_op = get_ferm_op(L)
        Lops.append(one_bd_sq_op * one_bd_sq_op)
    return Lops


def read_grad_tbts(mol, norb, alpha, prefix=''):
    '''
    Reading the tbts results that gradients converged to
    '''
    x = np.load(prefix + 'grad_res/' + mol + '/' +
                mol + '_' + str(alpha) + '_True.npy')
    tbts = get_tbt_parts(x, norb, alpha)
    return tbts


def build_restr_cartan_tensor(c, cartan):
    """Build the 4-rank tensor representing the cartan specified. 

    Args:
        c (float): The coefficients in front of the cartan
        cartan (ndarray NxN): The cartan matrix C where C[i, j] represents C[i, j] ni nj cartan operator. 

    Returns:
        cartan_tensor (ndarray NxNxNxN): The 4-rank tensor with cartan_tensor[i,i,j,j] = c * cartan[i, j]
    """
    n = cartan.shape[0]
    i_indices, j_indices = np.nonzero(cartan)

    cartan_tensor = build_tbt(n, complex=False)
    for k in range(len(i_indices)):
        i, j = i_indices[k], j_indices[k]
        cartan_tensor[i, i, j, j] = c * cartan[i, j]
    return cartan_tensor


def restr_sum_cartans(x, cartans):
    """Return of 4-rank tensor represented as the sum of rotated cartans specified. 

    Args: 
        x (ndarray (1+n*(n-1)/2)*alpha): All the parameters specifying each Cartan fragment.  
            The parameters are [p1, p2, ..., p_alpha]
            where pi = [ci, o_params]
        cartans (List[ndarray NxN]): A list of rank-2, symmetric cartan matrices representing forms of cartans. 

    Returns:
        tbt (ndarray NxNxNxN): The 4-rank tensor represented by x and cartans' forms. 
    """
    n = cartans[0].shape[0]
    upnum, _, _ = get_param_num(n, complex=False)
    pnum = upnum + 1

    # Initializing zero tbt
    tbt = build_tbt(n, complex=False)

    for i in range(len(cartans)):
        curparam = x[i * pnum: (i + 1) * pnum]
        O = construct_orthogonal(n, curparam[1:])
        ct_tsr = build_restr_cartan_tensor(curparam[0], cartans[i])
        rot_ct_tsr = cartan_orbtransf(ct_tsr, O, complex=False)

        tbt += rot_ct_tsr
    return tbt


def restr_csa_cost(x, cartans, target):
    """Obtain the L2 norm of difference between target tensor and the tensor represented by x and cartans. 

    Args:
        x (ndarray (1+n*(n-1)/2)*alpha): All the parameters specifying each Cartan fragment. 
            The parameters are [p1, p2, ..., p_alpha]
            where pi = [ci, o_params]
        cartans (List[ndarray NxN]): A list of rank-2, symmetric cartan matrices representing forms of cartans. 
        target (ndarray NxNxNxN): The target 4-rank tensor. 

    Returns:
        cost (float): The L2 norm of difference between target tensor and represented tensor. 
    """
    x_tensor = restr_sum_cartans(x, cartans)
    return tbt_cost(x_tensor, target, complex=False)


def real_diag_gradients_ij(o_params, i, j, diff):
    '''
    Computing ∂C / ∂\lambdas
    where lambdas are the symmetric diagonal coefficients for cartan elements

    Args:
        o_params (ndarray, len=n*(n-1)/2): All the angles used for the current cartan. 
        i, j (int): The index for ninj to get the coefficient gradient w.r.t. 
        diff (ndarray, size=(n**4)): The difference between Hamiltonian (4 rank tensor V_{pqrs}) and what the current x's parameters represent (W_{pqrs}) . 

    Returns:
        cl: ∂C / ∂\lambdas_{ij}
    '''
    def compute_wr_lr(n, l, m, O):
        wr_lr = np.einsum('p,q,r,s->pqrs', O[:, l], O[:, l], O[:, m], O[:, m])
        return wr_lr

    n = diff.shape[0]
    O = construct_orthogonal(n, o_params)

    # The gradients of cost w.r.t w
    cw = 2 * diff

    # Compute only for i, j
    wr_lr = compute_wr_lr(n, i, j, O)
    cl = np.sum(cw * wr_lr)
    return cl


def restr_coeff_gradients(single_x, cartan, diff):
    """Get the gradients for the single coefficient of the current restricted cartan elements. 

    Args:
        single_x (ndarray (1+n*(n-1)/2)): The solution array for a single restricted cartan. 
            It looks like [c, o_params] where c is the coefficient and o_params are the unitary angles.
        cartan (ndarray NxN): The NxN matrix indicating the form of the current cartan. 
        diff (ndarray NxNxNxN): The numerical different between target and represented two body tensors. diff = represented - target. 

    Returns:
        grads (float): The gradients for the coefficient. 
    """
    grads = 0

    # Loop over cartan.
    i_indices, j_indices = np.nonzero(cartan)
    for k in range(len(i_indices)):
        i, j = i_indices[k], j_indices[k]
        grads += cartan[i, j] * \
            real_diag_gradients_ij(single_x[1:], i, j, diff)
    return grads


def restr_theta_gradients(single_x, cartan, diff):
    """Get the gradients for the angle of the current restricted cartan elements. 

    Args:
        single_x (ndarray (1+n*(n-1)/2)): The solution array for a single restricted cartan. 
            It looks like [c, o_params] where c is the coefficient and o_params are the unitary angles.
        cartan (ndarray NxN): The NxN matrix indicating the form of the current cartan. 
        diff (ndarray NxNxNxN): The numerical different between target and represented two body tensors. diff = represented - target. 

    Returns:
        grads (ndarray n*(n-1)/2): The gradients for the angles. 
    """
    def get_w_o(coeff_cartans, o_params, n):
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
        O = construct_orthogonal(n, o_params)

        delta = np.identity(n)
        wo += np.einsum('pa,qb,rm,sm,bm->pqrsab',
                        delta, O, O, O, coeff_cartans)
        wo += np.einsum('qa,pb,rm,sm,bm->pqrsab',
                        delta, O, O, O, coeff_cartans)
        wo += np.einsum('ra,pl,ql,sb,lb->pqrsab',
                        delta, O, O, O, coeff_cartans)
        wo += np.einsum('sa,pl,ql,rb,lb->pqrsab',
                        delta, O, O, O, coeff_cartans)

        return wo

    def get_o_angles(oparams, i, n):
        '''
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
        kappa = get_anti_symmetric(n)[i]
        K = construct_anti_symmetric(n, oparams)
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

    n = diff.shape[0]
    opnum = int(n * (n - 1) / 2)

    wo = get_w_o(single_x[0] * cartan, single_x[1:], n)

    ograd = np.zeros(opnum)
    for i in range(opnum):
        otheta = get_o_angles(single_x[1:], i, n)
        wtheta = np.einsum('pqrsab,ab->pqrs', wo, otheta)
        ograd[i] = np.sum(2 * diff * wtheta)
    return ograd


def restr_cartan_gradient(x, cartans, target):
    """Obtain the gradient for restricted CSA's solution array. 

    Args:
        x (ndarray (1+n*(n-1)/2)*alpha): All the parameters specifying each Cartan fragment. 
            The parameters are [p1, p2, ..., p_alpha]
            where pi = [ci, o_params]
        cartans (List[ndarray NxN]): A list of rank-2, symmetric cartan matrices representing forms of cartans. 
        target (ndarray NxNxNxN): The target 4-rank tensor. 

    Returns:
        grads (ndarray (float)): The gradients. 
    """
    n = target.shape[0]
    diff = restr_sum_cartans(x, cartans) - target

    grads = np.zeros(len(x))
    upnum, _, _ = get_param_num(n, complex=False)
    pnum = upnum + 1

    for i in range(len(cartans)):
        curx = x[i * pnum: (i + 1) * pnum]
        coeff_grad = restr_coeff_gradients(curx, cartans[i], diff)
        grads[i * pnum:i * pnum + 1] = coeff_grad
        ograd = restr_theta_gradients(curx, cartans[i], diff)
        grads[i * pnum + 1: (i + 1) * pnum] = ograd
    return grads


def restr_csa(target, cartans, tol=1e-6, grad=False):
    """Perform a CSA optimiation where the form of the cartans is fixed.

    Args:
        target (ndarray NxNxNxN): The 4-rank two body tensor that the optimization will try to approximate. 
        cartans (List[ndarray NxN]): A list of cartan matrices Ci with coefficients specifying the form of cartan. 
            The represented cartans are sum_{ij} C[i, j] ni nj, where C[i, j] are symmetric. 
        tol: The optimization tolerance. 

    Returns:
        sol: The scipy's optimizer's solution. sol.x, sol.fun are the converged converged solution and cost function value. 
    """
    # Computing proper tolerance
    n = target.shape[0]
    entry_num = n**4
    fun_tol = (tol / entry_num) ** 2

    def fun(x): return restr_csa_cost(x, cartans, target)

    # Randomly initiate angles
    upnum, _, _ = get_param_num(n, complex=False)
    pnum = upnum + 1
    alpha = len(cartans)
    x0 = np.zeros(pnum * alpha)
    for i in range(alpha):
        x0[pnum * i + 1:pnum * (i + 1)] = np.random.rand(upnum) * 2 * np.pi

    if grad:
        def gradfun(x): return restr_cartan_gradient(x, cartans, target)
        return scipy.optimize.minimize(fun, x0, tol=fun_tol, method='BFGS', jac=gradfun)
    else:
        return scipy.optimize.minimize(fun, x0, tol=fun_tol)


def get_restr_reflections(reflection_idx, n_orbitals, n_cartans=1):
    """Return a list of the reflections as NxN matrices. 

    Args:
        reflection_idx (int): Specifies the type of reflection wanted. 
            0: n_i n_i 
            1: n_i n_j + n_j n_i 
            2: -2n_i + n_i n_j + n_j n_i 
            3: -2n_i - 2n_j + 2(n_i n_j + n_j n_i)
            4: -2n_i - 2n_j + (n_i n_j + n_j n_i)
            5: n_i + 0.5(n_jn_k + j<->k) − 0.5(n_in_k + i<->k)
            6: n_i + 0.5(n_kn_j + <->) − 0.5(n_kn_i + <->) − 0.5(n_jn_i + <->) 
            7: ni + nj − 0.5(n_kn_i + <->) − 0.5(n_in_j + <->) 
            8: ni + nk + nj − 0.5(n_in_j + <->) − 0.5(n_kn_i + <->) − 0.5(n_kn_j + <->)
        n_orbitals (int): The number of orbitals (N). 
        n_cartans (int): How many time this matrix is repeated in the list. 

    Returns:
        cartans: The list of reflections as NxN matrices. 
    """
    # Make reflection
    reflection = np.zeros((n_orbitals, n_orbitals))
    if reflection_idx == 0:
        reflection[0, 0] = 1
    elif reflection_idx == 1:
        reflection[0, 1], reflection[1, 0] = 1, 1
    elif reflection_idx == 2:
        reflection[0, 0] = -2
        reflection[0, 1], reflection[1, 0] = 1, 1
    elif reflection_idx == 3:
        reflection[0, 0] = -2
        reflection[1, 1] = -2
        reflection[0, 1], reflection[1, 0] = 2, 2
    elif reflection_idx == 4:
        reflection[0, 0] = -2
        reflection[1, 1] = -2
        reflection[0, 1], reflection[1, 0] = 1, 1
    elif reflection_idx == 5:
        reflection[0, 0] = 1
        reflection[1, 2], reflection[2, 1] = 0.5, 0.5
        reflection[0, 2], reflection[2, 0] = -0.5, -0.5
    elif reflection_idx == 6:
        reflection[0, 0] = 1
        reflection[1, 2], reflection[2, 1] = 0.5, 0.5
        reflection[0, 2], reflection[2, 0] = -0.5, -0.5
        reflection[0, 1], reflection[1, 0] = -0.5, -0.5
    elif reflection_idx == 7:
        reflection[0, 0] = 1
        reflection[1, 1] = 1
        reflection[0, 2], reflection[2, 0] = -0.5, -0.5
        reflection[0, 1], reflection[1, 0] = -0.5, -0.5
    elif reflection_idx == 8:
        reflection[0, 0] = 1
        reflection[1, 1] = 1
        reflection[2, 2] = 1
        reflection[0, 1], reflection[1, 0] = -0.5, -0.5
        reflection[0, 2], reflection[2, 0] = -0.5, -0.5
        reflection[1, 2], reflection[2, 1] = -0.5, -0.5

    return [reflection for i in range(n_cartans)]

def compute_cas_fragment(Htbt, k, Hf, spin_orb, gs, mol):
    """
    Compute the CAS fragment for the given two body tensor Htbt and the split k, store the information of 
    CAS split, tbt difference norm, relative norm, ground state variance into nested lists in 
    ./run/planted_solution/mol.pkl and store the computed fragments as a dict of the form 
    {CAS_split: computed_parameters} in ./run/planted_solution/mol Hamiltonians.pkl, where computed_parameters
    can be input into sum_cartans to retrieve the planted_solution tbt to compute the final Hamiltonian 
    operator.
    """
    import openfermion as of
    import pickle
    with open("./planted_solution/" + mol + " Hamiltonians.pkl", "rb") as f:
        Hamiltonians = pickle.load(f)
    if str(k) in Hamiltonians:
        return
    sol = csa(Htbt,k = k, alpha=1, tol=1e-5, grad=True)
    cur_tbt = sum_cartans(sol.x, spin_orb, k, alpha=1, complex=False)
    two_norm = np.sum((Htbt - cur_tbt) * (Htbt - cur_tbt))

    relative_norm = two_norm / np.sum(Htbt * Htbt)
    planted_H = get_ferm_op(cur_tbt, True) + of.FermionOperator("", Hf.terms[()])
    sparse_H = of.linalg.get_sparse_operator(planted_H)
    var = np.real(of.linalg.variance(sparse_H, gs))
    D_0 = of.linalg.eigenspectrum(Hf)
    D_0.sort()

    D_1 = np.real(of.linalg.eigenspectrum(planted_H))
    D_1.sort()
    eigen_spectrum_norm = np.linalg.norm(D_1 - D_0)
    lis = [[k, two_norm, relative_norm, var, eigen_spectrum_norm]]
    with open("./planted_solution/" + mol + ".pkl", "rb") as f:
        result = pickle.load(f)
    result += lis
    Hamiltonians[str(k)] = sol.x
    with open("./planted_solution/" + mol + ".pkl", "wb") as f:
        pickle.dump(result, f)
    with open("./planted_solution/" + mol + " Hamiltonians.pkl", "wb") as f:
        pickle.dump(Hamiltonians, f) 
