#fermionic term grouping for Hamiltonians

from openfermion import count_qubits
from openfermion.ops import FermionOperator
from openfermion.transforms import normal_ordered
from openfermion.hamiltonians import sz_operator, s_squared_operator
from openfermion.utils import hermitian_conjugated
from openfermion.linalg import get_ground_state, variance, get_sparse_operator
import itertools
import numpy as np
from op_utils import *
from ferm_utils import get_tensors


def group_commutes_with_super_group(g, sg, a = 2, b = 3, verbose = False):
    op1 = g[a]*g[b]
    op2 = sum([sg[i][a]*sg[i][b] for i in range(len(sg))])
    return check_pair_commuting(op1, op2, verbose = verbose, all = True)

def group_sorted_insertion_grouping(groups, oi = 2, ci = 3, verbose = False):
    '''
    Obtain a list of commuting Pauli operator groups using the sorted insertion algorithm.
    '''
    sorted_groups = sorted(groups, key=lambda x: np.abs(x[ci]), reverse=True)
    super_groups = [] #Initialize groups
    for group in sorted_groups:
        found_super_group = False
        for idx, super_group in enumerate(super_groups):
            commute = group_commutes_with_super_group(group, super_group, oi, ci, verbose = verbose)
            if commute: # Add term if it commutes with the current group.
                super_groups[idx].append(group)
                found_super_group = True
                break
        if not found_super_group: super_groups.append([group, ]) #Initiate new group that does not commute with any existing.
    return super_groups

def classify_ind(ind):
    if len(ind) == 2:
        if ind[0] != ind[1]:
            return 6
    if len(ind) == 4:
        s = set(ind)
        n_dist = len(s)
        if n_dist == 4:
            return 1 #pqrs, prqs, psqr
        if n_dist == 3:
            #if ind[0] == ind[1] or ind[2] == ind[3]:
            #    return 2 #ppqr, qrpp
            return 2 #pqpr, prpq, qppr, qprp, prqp
        if n_dist == 2:
            if ind.count(ind[0]) == 2:
                #if ind[0] == ind[1]:
                #    return 4 #ppqq
                #return 5 #pqpq
                return 3 #ppqq, pqpq
            return 4 #pppq
        return 5 #pppp

def fermgroup(hamiltonian, method = 'symFG', tol = 1e-8, verbose = False):
    if method == 'symFG':
        return symFG(hamiltonian, tol, verbose)
    if method == 'nFG':
        return nFG(hamiltonian, tol, verbose)

def nFG(Hamiltonian, tol = 1e-8, verbose = False):
    '''
    Return fermion groups by normal ordering and grouping commuting sets
    
    '''
    op = normal_ordered(Hamiltonian)
    n_qubits = count_qubits(op)

    e, gs = get_ground_state(get_sparse_operator(op, n_qubits)) #for variance estimates
    const = op.constant
    op = normal_ordered(op - const)

    frags = []
    while len(list(op.terms)) > 0:
        ind = list(op.terms)[0]

        op1 = FermionOperator(ind)
        op2 = normal_ordered(hermitian_conjugated(op1))
        if op1 != op2:
            op1 = op1 + op2 #symmetrizing
        
        if np.abs(op.terms[ind]) > tol:
            frags.append([op1, op.terms[ind]])

            f = check_self_commuting(op1, verbose = False)
            if f == False:
                print('Fragments not self commuting, error!\nFragment: {}'.format(op1))
        
        op1 = op.terms[ind]*(op1)
        op = normal_ordered(op - op1)
    
    #frags now contains [symmetric_ferm_op, coeff]
    sg = group_sorted_insertion_grouping(frags, 0, 1, verbose = False) #grouping number preserving fragments

    if verbose:
        print('Obtained new frags')
    
    frags = []
    for a in sg:
        frag = sum([a[i][0]*a[i][1] for i in range(len(a))])
        f = check_self_commuting(frag, verbose = verbose)
        if f == False:
            print('error! Fragment not fully commuting.')
        frags.append(frag)
    
    frags_sparse = [get_sparse_operator(frag, n_qubits) for frag in frags] #finding variance
    var = sum([np.sqrt(variance(op, gs)) for op in frags_sparse])

    if verbose:
        print('Normal ordered Fermionic grouping variance: {}'.format(var))
    return frags, var, sg

def symFG(Hamiltonian, tol = 1e-8, verbose = False):
    obt, tbt, const = get_tensors(Hamiltonian)
    #generate set of all ind
    n_orb = tbt.shape[0]
    n_qubits = count_qubits(Hamiltonian)

    groups = []
    subgroups = {}
    for i in range(1, 7):
        subgroups[i] = []
    
    #make set of all possible combinations of pqrs
    sets_all = list(itertools.combinations_with_replacement(list(range(n_orb)), 4)) + list(itertools.combinations(list(range(n_orb)), 2))

    #get groups
    for ind in sets_all:
        c = classify_ind(ind)
        if c == 1:
            #pqrs
            A = tbt[ind[0]][ind[1]][ind[2]][ind[3]]
            B = tbt[ind[0]][ind[1]][ind[3]][ind[2]]
            ABs = get_f(ind, gAB_s)
            A_Bs = get_f(ind, gA_B_s)
            subgroups[c].append([ind, c, ABs, (A+B)/2, True, True])
            subgroups[c].append([ind, c, A_Bs, (A-B)/2, True, True])

            #prqs
            ind = [ind[0], ind[2], ind[1], ind[3]]
            A = tbt[ind[0]][ind[1]][ind[2]][ind[3]]
            B = tbt[ind[0]][ind[1]][ind[3]][ind[2]]
            ABs = get_f(ind, gAB_s)
            A_Bs = get_f(ind, gA_B_s)
            subgroups[c].append([ind, c, ABs, (A+B)/2, True, True])
            subgroups[c].append([ind, c, A_Bs, (A-B)/2, True, True])

            #psqr
            ind = [ind[0], ind[3], ind[2], ind[1]]
            A = tbt[ind[0]][ind[1]][ind[2]][ind[3]]
            B = tbt[ind[0]][ind[1]][ind[3]][ind[2]]
            ABs = get_f(ind, gAB_s)
            A_Bs = get_f(ind, gA_B_s)
            subgroups[c].append([ind, c, ABs, (A+B)/2, True, True])
            subgroups[c].append([ind, c, A_Bs, (A-B)/2, True, True])
        if c == 2:
            #ppqr
            #need to identify pqr
            s = np.sort(ind)
            if s[0] == s[1]:
                p = s[0]
                q = s[2]
                r = s[3]
            elif s[1] == s[2]:
                p = s[1]
                q = s[0]
                r = s[3]
            else:
                p = s[2]
                q = s[0]
                r = s[1]
            A = tbt[p][p][q][r]
            #B = tbt[ind[0]][ind[1]][ind[3]][ind[2]]
            As = gAB_3_s(p, q, r)
            subgroups[c].append([ind, c, As, A, True, True])
            #pqpr
            A = tbt[p][q][p][r]
            B = tbt[p][q][r][p]
            pqr = [p, q, p, r]
            l1 = get_f(pqr, gAB_s_list)
            l2 = get_f(pqr, gA_B_s_list)
            subgroups[c].append([pqr, c, l1[0]+l1[3], (A+B)/2, True, False])
            subgroups[c].append([pqr, c, l1[1], (A+B)/2, False, False])
            subgroups[c].append([pqr, c, l1[2], (A+B)/2, False, False])

            subgroups[c].append([pqr, c, l2[0]+l2[3], (A-B)/2, True, False])
            subgroups[c].append([pqr, c, l2[1], (A-B)/2, False, False])
            subgroups[c].append([pqr, c, l2[2], (A-B)/2, False, False])
        if c == 3:
            #check
            #ppqq
            if ind[0] == ind[1]:
                p = ind[0]
                q = ind[2]
            else:
                p = ind[0]
                q = ind[1]
            A = tbt[p][p][q][q] + tbt[q][q][p][p]
            op = nn(2*p, 2*q) + nn(2*p + 1, 2*q) + nn(2*p, 2*q + 1) + nn(2*p + 1, 2*q + 1)
            subgroups[c].append([(p, p, q, q), c, op, A, True, True])

            #pqpq
            A = tbt[p][q][p][q]
            B = tbt[p][q][q][p]
            ss = dd_s(symsym, p, q)
            aa = dd_s(antianti, p, q)
            subgroups[c].append([ind, c, ss, (A+B)/2, True, True])
            subgroups[c].append([ind, c, aa, (A-B)/2, True, True])
        if c == 4:
            #check
            #pppq
            s = set(ind)
            if ind.count(ind[0]) == 3:
                p = ind[0]
                s.remove(ind[0])
                q = s.pop()
            else:
                q = ind[0]
                s.remove(ind[0])
                p = s.pop()
            A = tbt[ind[0]][ind[1]][ind[2]][ind[3]]
            l = gAB_3_s_list(p, p, q)
            subgroups[c].append([ind, c, l[0] + l[3], A, True, False])
            subgroups[c].append([ind, c, l[1], A, False, False])
            subgroups[c].append([ind, c, l[2], A, False, False])
        if c == 5:
            #check
            #pppp
            A = tbt[ind[0]][ind[1]][ind[2]][ind[3]]
            B = obt[ind[0]][ind[0]]
            op = nn(2*ind[0], 2*ind[0]) + nn(2*ind[0] + 1, 2*ind[0]) + nn(2*ind[0], 2*ind[0] + 1) + nn(2*ind[0] + 1, 2*ind[0] + 1)
            op2 = nn(2*ind[0], 2*ind[0]) + nn(2*ind[0] + 1, 2*ind[0] + 1)
            subgroups[c].append([ind, c, op, A, True, True])
            subgroups[c].append([ind, c, op2, B, True, True])
        if c == 6:
            #pq (1 electron term)
            A = obt[ind[0]][ind[1]]
            op = o_s(sym_iEj, ind[0], ind[1])
            subgroups[c].append([ind, c, op, A, True, True])
    
    ng = sum([len(subgroups[i]) for i in range(1, 6)])
    if verbose:
        print('Finished forming preliminary groups. Total number of groups: {}'.format(ng))
    #print('\n Removing groups with negligible coefficients...')
    subgroups_mod = {}
    for i in range(1, 7):
        l = []
        for j in range(len(subgroups[i])):
            if np.abs(subgroups[i][j][3]) > tol:
                l.append(subgroups[i][j])
        subgroups_mod[i] = l
    
    ng_mod = sum([len(subgroups_mod[i]) for i in range(1, 7)])
    if verbose:
        print('Removed negligible groups. Total number of groups: {}'.format(ng_mod))

    #sort insertion for each symmetry set
    #evaluate initial variance without grouping 
    No_sym = []
    Both_sym = []
    H_frags_init = []
    for i in range(1, 7):
        for j in range(len(subgroups_mod[i])):
            if subgroups_mod[i][j][4] == True and subgroups_mod[i][j][5] == True:
                Both_sym.append(subgroups_mod[i][j][:4])
            else:
                No_sym.append(subgroups_mod[i][j][:4])
            H_frags_init.append(subgroups_mod[i][j][2]*subgroups_mod[i][j][3])
    #sort with coefficient
    #subgroups
    A = group_sorted_insertion_grouping(Both_sym)
    B = group_sorted_insertion_grouping(No_sym)

    if verbose:
        print('Super grouping completed. Verifying results...')
    #verify results - 
    n_op = number_op(n_orb*2)
    sz = sz_operator(n_orb)
    s2 = s_squared_operator(n_orb)
    H_frags = []
    for a in A:
        op = sum([a[i][2]*a[i][3] for i in range(len(a))])
        n_comm = check_pair_commuting(op, n_op, verbose = False, all = False)
        sz_comm = check_pair_commuting(op, sz, verbose = False, all = False)
        s2_comm = check_pair_commuting(op, s2, verbose = False, all = False)
        if n_comm == False or sz_comm == False or s2_comm == False:
            print(a, n_comm, sz_comm, s2_comm)
            if verbose:
                print('Error, misgrouped in all symmetry preserving!')
            return
        else:
            H_frags.append(op)
    for a in B:
        op = sum([a[i][2]*a[i][3] for i in range(len(a))])
        n_comm = check_pair_commuting(op, n_op, verbose = False, all = False)
        if n_comm == False:
            if verbose:
                print('Error, misgrouped in number preserving only!')
            return
        else:
            H_frags.append(op)
    if verbose:
        print('Results verified successfully!')
        print('{} groups with only Sz, S2 and number symmetry.'.format(len(A)))
        print('{} groups with only number symmetry.'.format(len(B)))
    H_full = sum(H_frags) + const

    #getting number of Paulis
    n_paulis = [get_n_pauli(op) for op in H_frags]

    #ground states:
    H_full_sparse = get_sparse_operator(H_full, 2*n_orb)
    H_frags_sparse = [get_sparse_operator(op, 2*n_orb) for op in H_frags]
    H_frags_init_sparse = [get_sparse_operator(op, 2*n_orb) for op in H_frags_init]

    if verbose:
        print('Getting ground state...')
    e, gs = get_ground_state(H_full_sparse)

    if verbose:
        print('Ground state evaluated, ground state = {}'.format(e))
    
    frag_variances = [np.abs(variance(op, gs)) for op in H_frags_sparse]
    max_var = max(frag_variances)

    if verbose:
        print('Maximum fragment variance: {}'.format(max_var))
    
    #print(H_frags[frag_variances.index(max_var)])
    max_s = max(n_paulis)
    var = sum([np.sqrt(variance(op, gs)) for op in H_frags_sparse])
    var_init = sum([np.sqrt(variance(op, gs)) for op in H_frags_init_sparse])
    
    if verbose:
        print('Variance: {}'.format(var))
        print('Initial variance without supergroups: {}'.format(var_init))
    return H_frags, var, A, B
