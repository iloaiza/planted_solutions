#from fermutils in TBFrags
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner, reverse_jordan_wigner, normal_ordered
import numpy as np

def get_chemist_tbt(H : FermionOperator, n = None, spin_orb=False):
    '''
    Obtain the 4-rank tensor that represents two body interaction in H. 
    In chemist ordering a^ a a^ a. 
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    # getting N^4 phy_tbt and then (N/2)^4 chem_tbt 
    phy_tbt = get_two_body_tensor(H, n)
    chem_tbt = np.transpose(phy_tbt, [0, 3, 1, 2])

    if spin_orb:
        return chem_tbt

    # Spin-orbital to orbital 
    n_orb = phy_tbt.shape[0]
    n_orb = n_orb // 2
    alpha_indices = list(range(0, n_orb * 2, 2))
    beta_indices = list(range(1, n_orb * 2, 2))
    chem_tbt = chem_tbt[
        np.ix_(alpha_indices, alpha_indices,
                    beta_indices, beta_indices)]

    return chem_tbt

def get_two_body_tensor(H : FermionOperator, n = None):
    '''
    Obtain the 4-rank tensor that represents two body interaction in H. 
    In physics ordering a^ a^ a a 
    '''
    # number of spin orbitals 
    if n is None:
        n = get_spin_orbitals(H)

    tbt = np.zeros((n, n, n, n), dtype = 'complex_')
    for term, val in H.terms.items():
        if len(term) == 4:
            tbt[
                term[0][0], term[1][0],
                term[2][0], term[3][0]
            ] = val
    return tbt 
def get_spin_orbitals(H : FermionOperator):
    '''
    Obtain the number of spin orbitals of H
    '''
    n = -1 
    for term, val in H.terms.items():
        if len(term) == 4:
            n = max([
                n, term[0][0], term[1][0],
                term[2][0], term[3][0]
            ])
        elif len(term) == 2:
            n = max([
                n, term[0][0], term[1][0]])
    n += 1 
    return n

def get_ferm_op_one(obt, spin_orb):
    '''
    Return the corresponding fermionic operators based on one body tensor
    '''
    n = obt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            if not spin_orb:
                for a in range(2):
                    op += FermionOperator(
                        term = (
                            (2*i+a, 1), (2*j+a, 0)
                        ), coefficient=obt[i, j]
                    )
            else:
                op += FermionOperator(
                    term = (
                        (i, 1), (j, 0)
                    ), coefficient=obt[i, j]
                )
    return op 

def get_ferm_op_two(tbt, spin_orb):
    '''
    Return the corresponding fermionic operators based on tbt (two body tensor)
    This tensor can index over spin-orbtals or orbitals
    '''
    n = tbt.shape[0]
    op = FermionOperator.zero()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n): 
                    if not spin_orb:
                        for a in range(2):
                            for b in range(2):
                                op += FermionOperator(
                                    term = (
                                        (2*i+a, 1), (2*j+a, 0),
                                        (2*k+b, 1), (2*l+b, 0)
                                    ), coefficient=tbt[i, j, k, l]
                                )
                    else:
                        op += FermionOperator(
                            term=(
                                (i, 1), (j, 0),
                                (k, 1), (l, 0)
                            ), coefficient=tbt[i, j, k, l]
                        )
    return op

def get_ferm_op(tsr, spin_orb=False):
    '''
    Return the corresponding fermionic operators based on the tensor
    This tensor can index over spin-orbtals or orbitals
    '''
    if len(tsr.shape) == 4:
        return get_ferm_op_two(tsr, spin_orb)
    elif len(tsr.shape) == 2:
        return get_ferm_op_one(tsr, spin_orb)

def get_obt(H : FermionOperator, n = None, spin_orb=False, tiny=1e-12):
    '''
    Obtain the 2-rank tensor that represents one body interaction in H. 
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    # getting N^2 phy_tbt and then (N/2)^2 chem_tbt 
    if n is None:
        n = get_spin_orbitals(H)
    
    obt = np.zeros((n,n), dtype = 'complex_')
    for term, val in H.terms.items():
        if len(term) == 2:
            #print("Term is {}".format(term))
            if term[0][1] == 1 and term[1][1] == 0:
                obt[term[0][0], term[1][0]] = val
            elif term[1][1] == 1 and term[0][1] == 0:
                obt[term[1][0], term[0][0]] = -val
            else:
                print("Warning, one-body operator has double creation/annihilation operators!")
                quit()

    if spin_orb:
        return obt

    # Spin-orbital to orbital 
    n_orb = obt.shape[0]
    n_orb = n_orb // 2

    obt_red_uu = np.zeros((n_orb, n_orb), dtype = 'complex_')
    obt_red_dd = np.zeros((n_orb, n_orb), dtype = 'complex_')
    obt_red_ud = np.zeros((n_orb, n_orb), dtype = 'complex_')
    obt_red_du = np.zeros((n_orb, n_orb), dtype = 'complex_')
    for i in range(n_orb):
        for j in range(n_orb):
            obt_red_uu[i,j] = obt[2*i, 2*j]
            obt_red_dd[i,j] = obt[2*i+1, 2*j+1]
            obt_red_ud = obt[2*i, 2*j+1]
            obt_red_du = obt[2*i+1, 2*j]

    if np.sum(np.abs(obt_red_du)) + np.sum(np.abs(obt_red_ud)) != 0:
        print("Warning, operator to one-body transformation ran with spin_orb=false, but spin-orbit couplings are not 0!")
    if np.sum(np.abs(obt_red_uu - obt_red_dd)) > tiny:
        print("Warning, operator to one-body transformation ran with spin_orb=false, but isn't symmetric to spin-flips")
        print("obt_uu - obt_dd = {}".format(obt_red_uu - obt_red_dd))

    obt = (obt_red_uu + obt_red_dd) / 2

    return obt

def get_tensors(hamiltonian):
    hamiltonian = hamiltonian
    c = hamiltonian.constant
    hamiltonian_c = hamiltonian - c
    tbt = get_chemist_tbt(hamiltonian_c, spin_orb=False)
    op = get_ferm_op(tbt, False)
    h1b = hamiltonian_c - op
    h1b = reverse_jordan_wigner(jordan_wigner(h1b))
    obt = get_obt(h1b, spin_orb=False)
    return (obt, tbt, c)

def get_obt(H : FermionOperator, n = None, spin_orb=False, tiny=1e-12):
    '''
    Obtain the 2-rank tensor that represents one body interaction in H. 
    In addition, simplify tensor assuming symmetry between alpha/beta coefficients
    '''
    # getting N^2 phy_tbt and then (N/2)^2 chem_tbt 
    if n is None:
        n = get_spin_orbitals(H)
    
    obt = np.zeros((n,n))
    for term, val in H.terms.items():
        if len(term) == 2:
            if term[0][1] == 1 and term[1][1] == 0:
                obt[term[0][0], term[1][0]] = val.real
            elif term[1][1] == 1 and term[0][1] == 0:
                obt[term[1][0], term[0][0]] = -val.real
            else:
                print("Warning, one-body operator has double creation/annihilation operators!")
                quit()

    if spin_orb:
        return obt

    # Spin-orbital to orbital 
    n_orb = obt.shape[0]
    n_orb = n_orb // 2

    obt_red_uu = np.zeros((n_orb, n_orb))
    obt_red_dd = np.zeros((n_orb, n_orb))
    obt_red_ud = np.zeros((n_orb, n_orb))
    obt_red_du = np.zeros((n_orb, n_orb))
    for i in range(n_orb):
        for j in range(n_orb):
            obt_red_uu[i,j] = obt[2*i, 2*j]
            obt_red_dd[i,j] = obt[2*i+1, 2*j+1]
            obt_red_ud = obt[2*i, 2*j+1]
            obt_red_du = obt[2*i+1, 2*j]

    if np.sum(np.abs(obt_red_du)) + np.sum(np.abs(obt_red_ud)) != 0:
        print("Warning, operator to one-body transformation ran with spin_orb=false, but spin-orbit couplings are not 0!")
    if np.sum(np.abs(obt_red_uu - obt_red_dd)) > tiny:
        print("Warning, operator to one-body transformation ran with spin_orb=false, but isn't symmetric to spin-flips")
        print("obt_uu - obt_dd = {}".format(obt_red_uu - obt_red_dd))

    obt = (obt_red_uu + obt_red_dd) / 2

    return obt

def onebody_to_twobody(obt):
    """
    converts obt to tbt using idempotency of number operators: obt[p,p] = tbt[p,p,p,p]
    assumes obt is real-symmetric
    note that the association of tbt's to a given obt is not unique
    """
    
    N    = obt.shape[0]
    D, U = np.linalg.eig(obt)

    U    = U.T # note that my implementation of orbital rotations is U.T @ X @ U, so this line is needed
    tbt  = np.zeros([N,N,N,N])
    for p in range(N):
        tbt[p,p,p,p] = D[p]
    chem_tbt = np.einsum('pqrs,pa,qb,rc,sd->abcd', tbt, U, U, U, U)
    
    return chem_tbt
 