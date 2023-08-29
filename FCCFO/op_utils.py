from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
from openfermion.utils.commutators import commutator
import itertools


iEj = lambda i, j: str(i)+'^ '+str(j)+' '
ijE = lambda i, j: str(i)+'^ '+str(j)+'^ '
Eij = lambda i, j: str(i)+' '+str(j)+' '
ijEkl = lambda i, j, k, l: ijE(i, j) + Eij(k, l)

n = lambda i: FermionOperator(iEj(i, i))
nn = lambda i, j: n(i)*n(j)
number_op = lambda k: sum([n(i) for i in range(k)]) 

iEjxkEl = lambda i, j, k, l: FermionOperator(iEj(i, j) + iEj(k, l))
sym_iEj = lambda i, j: FermionOperator(iEj(i, j)) + FermionOperator(iEj(j, i))
anti_iEj = lambda i, j: FermionOperator(iEj(i, j)) - FermionOperator(iEj(j, i))

gA_r = lambda i, j, k, l: iEjxkEl(j, i, l, k) + iEjxkEl(l, k, j, i) + iEjxkEl(i, j, k, l) + iEjxkEl(k, l, i, j)
gA_i = lambda i, j, k, l: iEjxkEl(j, i, l, k) + iEjxkEl(l, k, j, i) - iEjxkEl(i, j, k, l) - iEjxkEl(k, l, i, j)
gB_r = lambda i, j, k, l: iEjxkEl(j, i, k, l) + iEjxkEl(k, l, j, i) + iEjxkEl(i, j, l, k) + iEjxkEl(l, k, i, j)
gB_i = lambda i, j, k, l: iEjxkEl(j, i, k, l) + iEjxkEl(k, l, j, i) - iEjxkEl(i, j, l, k) - iEjxkEl(l, k, i, j)
gAB = lambda i, j, k, l: gA_r(i, j, k, l) + gB_r(i, j, k, l)
gA_B = lambda i, j, k, l: gA_r(i, j, k, l) - gB_r(i, j, k, l)
gAB_s = lambda i, j, k, l: t_s(gA_r, i, j, k, l) + t_s(gB_r, i, j, k, l)
gA_B_s = lambda i, j, k, l: t_s(gA_r, i, j, k, l) - t_s(gB_r, i, j, k, l)
gAB_3 = lambda p, q, r: n(p)*(sym_iEj(q, r)) + (sym_iEj(r, q))*n(p)
gA_B_3 = lambda p, p2, q: n(p)*(anti_iEj(p2, q)) + (anti_iEj(p2, q))*n(p)
gAB_3_s = lambda p, q, r: gAB_3(2*p, 2*q, 2*r) + gAB_3(2*p + 1, 2*q, 2*r) + gAB_3(2*p, 2*q + 1, 2*r + 1) + gAB_3(2*p + 1, 2*q + 1, 2*r + 1)
gAB_3_s_list = lambda p, q, r: [gAB_3(2*p, 2*q, 2*r), gAB_3(2*p + 1, 2*q, 2*r), gAB_3(2*p, 2*q + 1, 2*r + 1), gAB_3(2*p + 1, 2*q + 1, 2*r + 1)]

#spin
o_s = lambda f, i, j: f(2*i, 2*j) + f(2*i + 1, 2*j + 1)
t_s = lambda f, i, j, k, l: f(2*i, 2*j, 2*k, 2*l) + f(2*i + 1, 2*j + 1, 2*k, 2*l) + f(2*i, 2*j, 2*k + 1, 2*l + 1) + f(2*i + 1, 2*j + 1, 2*k + 1, 2*l + 1)
t_s_list = lambda f, i, j, k, l: list([f(2*i, 2*j, 2*k, 2*l), f(2*i + 1, 2*j + 1, 2*k, 2*l), f(2*i, 2*j, 2*k + 1, 2*l + 1) , f(2*i + 1, 2*j + 1, 2*k + 1, 2*l + 1)])

get_f = lambda a, f: f(a[0], a[1], a[2], a[3])
gA_s = lambda i, j, k, l: t_s(gA_r, i, j, k, l)
gB_s = lambda i, j, k, l: t_s(gB_r, i, j, k, l)
gA_s_list = lambda i, j, k, l: t_s_list(gA_r, i, j, k, l)
gB_s_list = lambda i, j, k, l: t_s_list(gB_r, i, j, k, l)
gAB_s_list = lambda i, j, k, l: t_s_list(gAB, i, j, k, l)
gA_B_s_list = lambda i, j, k, l: t_s_list(gA_B, i, j, k, l)

symsym = lambda i, j, k, l: sym_iEj(i, j)*sym_iEj(k, l)
antianti = lambda i, j, k, l: anti_iEj(i, j)*anti_iEj(k, l)
dd_s = lambda f, i, j: f(2*i, 2*j, 2*i, 2*j) + f(2*i + 1, 2*j + 1, 2*i, 2*j) + f(2*i, 2*j, 2*i + 1, 2*j + 1) + f(2*i + 1, 2*j + 1, 2*i + 1, 2*j + 1)

#checking if the terms of the hamiltonian two body tensor commute:
def check_pauli_commutation(op_list, verbose = True):

    combs = list(itertools.combinations(op_list, 2))
    for a, b in combs:
        if commutator(a, b).induced_norm() != 0.0:
            if verbose:
                print('{} and {} don\'t commute!'.format(a, b))
            return False
    return True

def check_self_commuting(op, verbose = True):
    """
    checks if all the Pauli words on jordan-wigner transforming the fermionic operator commute (fully-commuting set)
    """
    op_list = get_pauli_list(op)
    if verbose:
        print('Number of terms: {}'.format(len(op_list)))
    a = check_pauli_commutation(op_list, verbose)
    if verbose:
        print('Commuting: {}'.format(a))
    return a

def get_pauli_list(op):
    """
    returns list of Pauli operators
    """
    if type(op) == FermionOperator:
        op_jw = jordan_wigner(op)
    else:
        op_jw = op
    
    op_list = [a for a in op_jw.get_operators()]
    op_list_unit = [QubitOperator(list(a.terms.keys())[0]) for a in op_list]
    return op_list_unit

def get_n_pauli(op):
    """
    returns number of Paulis
    """
    return len(get_pauli_list(op))

def check_pair_commuting(op1, op2, verbose = False, all = True):
    if type(op1) == FermionOperator:
        op1, op2 = jordan_wigner(op1), jordan_wigner(op2)
    if commutator(op1, op2).induced_norm() != 0.0:
        if verbose:
            print('Operators dont commute!')
        return False
    
    if all == False:
        return True
    #else they commute
    #check if each Pauli commutes
    if verbose:
        print('Operators commute. Checking if completely commuting...')
    op_list = get_pauli_list(op1) + get_pauli_list(op2)
    a = check_pauli_commutation(op_list, False)
    if verbose:
        print('Commuting: {}'.format(a))
    return a

def check_all_Z(op):
    '''
    Checks if all Z

    Input:
    op : QubitOperator
    '''
    op_list = list(op.terms.keys())
    for i in range(len(op_list)):
        for j in range(len(op_list[i])):
            if op_list[i][j][1] != 'Z':
                return False
    return True