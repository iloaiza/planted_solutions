"""Construct CAS Hamiltonians with cropping
"""
import saveload_utils as sl
import ferm_utils as feru
import csa_utils as csau
import var_utils as varu
import openfermion as of
import numpy as np
from sdstate import *
from itertools import product
import random
import h5py
import sys
import os
from matrix_utils import construct_orthogonal
import pickle

### Parameters
tol = 1e-5
balance_strength = 2
save = False
method_name = 'CAS-Cropping'
# Number of spatial orbitals in a block
block_size = 6
# Number of electrons per block
ne_per_block = 6
# +- difference in number of electrons per block
ne_range = 2
# Running Full CI to check compute the ground state, takes exponentially amount of time to execute
FCI = False
# Checking symmetries of the planted Hamiltonian, very costly
check_symmetry = False
# File path and name
path = "hamiltonians_catalysts/"
file_name = "2_co2_6-311++G___12_9d464efb-b312-45f8-b0ba-8c42663059dc.hdf5"

def construct_blocks(b: int, spin_orbs: int):
    """Construct CAS blocks of size b * 2 for spin_orbs number of orbitals"""
    b = b * 2
    k = []
    tmp = [0]
    for i in range(1, spin_orbs):
        if i % b == 0:
            k.append(tmp)
            tmp = [i]
        else:
            tmp.append(i)
    if len(tmp) != 0:
        k.append(tmp)
    return k
        
def get_truncated_cas_tbt(Htbt, k, casnum):
#     Trunctate the original Hamiltonian two body tensor into the cas block structures
    cas_tbt = np.zeros(Htbt.shape)
    cas_x = np.zeros(casnum)
    idx = 0
    for block in k:
        for a in block:
            for b in block:
                for c in block:
                    for d in block:
                        cas_tbt [a,b,c,d] = Htbt [a,b,c,d]
                        cas_x[idx] = Htbt[a,b,c,d]
                        idx += 1
    return cas_tbt, cas_x

def in_orbs(term, orbs):
    """Return if the term is a local excitation operator within orbs"""
    if len(term) == 2:
        return term[0][0] in orbs and term[1][0] in orbs
    elif len(term) == 4:
        return term[0][0] in orbs and term[1][0] in orbs and term[2][0] in orbs and term[3][0] in orbs
    return False

def transform_orbs(term, orbs):
    """Transform the operator term to align the orbs starting from 0"""
#     pass
    if len(term) == 2:
        return ((orbs.index(term[0][0]), 1), (orbs.index(term[1][0]), 0))
    if len(term) == 4:
        return ((orbs.index(term[0][0]), 1), (orbs.index(term[1][0]), 0), 
               (orbs.index(term[2][0]), 1), (orbs.index(term[3][0]), 0))   
    return None

def solve_enums(cas_tbt, k, ne_per_block = 0, ne_range = 0, balance_t = 10):
    """Solve for number of electrons in each CAS block with FCI within the block""" 
    e_nums = []
    states = []
    E_cas = 0
    for orbs in k:
        s = orbs[0]
        t = orbs[-1] + 1
        norbs = len(orbs)
        ne = min(ne_per_block + random.randint(-ne_range, ne_range), norbs - 1)
        print(f"Ne within current block: {ne}")
#         Construct (Ne^-ne)^2 terms in matrix, to enforce structure of states
        if ne_per_block != 0:
            balance_tbt = np.zeros([norbs, norbs,  norbs, norbs,])
            for p, q in product(range(norbs), repeat = 2):
                balance_tbt[p,p,q,q] += 1
            for p in range(len(orbs)):
                balance_tbt[p,p,p,p] -= 2 * ne
#             Construct 2e tensor to enforce the Ne in the ground state.
            strength = balance_t * (1 + random.random())
#             tmp_tbt = np.add(tmp_tbt, balance_tbt)
        flag = True
        while flag:
            balance_tbt *= strength
            cas_tbt[s:t, s:t, s:t, s:t] = np.add(cas_tbt[s:t, s:t, s:t, s:t], balance_tbt)
            tmp = feru.get_ferm_op(cas_tbt[s:t, s:t, s:t, s:t], True)
            sparse_H_tmp = of.get_sparse_operator(tmp)
            tmp_E_min, t_sol = of.get_ground_state(sparse_H_tmp)
            st = sdstate(n_qubit = len(orbs))
            for i in range(len(t_sol)):
                if np.linalg.norm(t_sol[i]) > np.finfo(np.float32).eps:
                    st += sdstate(s = i, coeff = t_sol[i])
#             print(f"state norm: {st.norm()}")
            st.normalize()
            E_st = st.exp(tmp)
            flag = False
            for sd in st.dic:
                ne_computed = bin(sd)[2:].count('1')
                if ne_computed != ne:
#                     print("Not enough balance, adding more terms")
                    flag = True
                    break
#             flag = False
        print(f"E_min: {tmp_E_min} for orbs: {orbs}")
        print(f"current state Energy: {E_st}")
        E_cas += E_st
        states.append(st)
        e_nums.append(ne)                
    return e_nums, states, E_cas

# Killer Construction
def construct_killer(k, e_num, n = 0, const = 1e-2, t = 1e2, n_killer = 5):
    """ Construct a killer operator for CAS Hamiltonian, based on cas block structure of k and the size of killer is 
    given in k, the number of electrons in each CAS block of the ground state
    is specified by e_nums. t is the strength of quadratic balancing terms for the killer with respect to k,
    n_killer specifies the number of operators O to choose.
    """
    if not n:
        n = max([max(orbs) for orbs in k])
    killer = of.FermionOperator.zero()
    for i in range(len(k)):
        orbs = k[i]
        outside_orbs = [j for j in range(n) if j not in orbs]
    #     Construct Ne
        Ne = sum([of.FermionOperator("{}^ {}".format(i, i)) for i in orbs])
    #     Construct O, for O as combination of Epq which preserves Sz and S2
        if len(outside_orbs) >= 4:
            tmp = 0
            while tmp < n_killer:
                p, q = random.sample(outside_orbs, 2)
                if abs(p - q) > 1:
#                     Constructing symmetry conserved killers
                    O = of.FermionOperator.zero()
                    if p % 2 != 0:
                        p -= 1
                    if q % 2 != 0:
                        q -= 1
                    ferm_op = of.FermionOperator("{}^ {}".format(p, q)) + of.FermionOperator("{}^ {}".format(q, p))
                    O += ferm_op
                    O += of.hermitian_conjugated(ferm_op)
                    ferm_op = of.FermionOperator("{}^ {}".format(p + 1, q + 1)) + of.FermionOperator("{}^ {}".format(q + 1, p + 1))
                    O += ferm_op
                    O += of.hermitian_conjugated(ferm_op)
                    killer += (1 + np.random.rand()) * const * O * (Ne - e_nums[i])
                    tmp += 1
        killer += t * (1 + np.random.rand()) * const * ((Ne - e_nums[i]) ** 2)
    return killer

def construct_orbs(key: str):
#     Contruct k from the given key
    count = 0
    lis = key.split("-")
    k = []
    for i in lis:
        tmp = int(i)
        k.append(list(range(count, count + tmp)))
        count += tmp
    return k

if __name__ == "__main__":   
    for file_name in os.listdir(path):
        ps_path = "planted_solutions/"
        f_name = file_name.split(".")[0] + ".pkl"
#         if os.path.exists(ps_path + f_name):
#             continue
        with h5py.File(path + file_name, mode="r") as h5f:
            attributes = dict(h5f.attrs.items())
            one_body = np.array(h5f["one_body_tensor"])
            two_body = np.array(h5f["two_body_tensor"])
    #     Construct a single 2e tensor to represent the Hamiltonian with idempotent transformation
        spin_orbs = one_body.shape[0]
        print(f"Number of spin orbitals: {spin_orbs}")
        spatial_orbs = spin_orbs // 2
        onebody_tbt = feru.onebody_to_twobody(one_body)
        Htbt = np.add(two_body, onebody_tbt)

        k = construct_blocks(block_size, spin_orbs)
        print(f"orbital splliting: {k}")
        upnum, casnum, pnum = csau.get_param_num(spin_orbs, k, complex = False)

        cas_tbt, cas_x = get_truncated_cas_tbt(Htbt, k, casnum)
    #     cas_tbt_tmp = copy.deepcopy(cas_tbt)
        e_nums, states, E_cas = solve_enums(cas_tbt, k, ne_per_block = ne_per_block,
                                            ne_range = ne_range, balance_t = balance_strength)
    #     assert np.allclose(cas_tbt_tmp, cas_tbt), "changed"
        print(f"e_nums:{e_nums}")
        print(f"E_cas: {E_cas}")
#         sd_sol = sdstate()

#         for st in states:
#             sd_sol = sd_sol.concatenate(st)
    # The following code segment checks the state energy for the full Hamiltonian, takes exponential space 
    # and time with respect to the number of blocks
    #     E_sol = sd_sol.exp(cas_tbt)
    #     print(f"Double check ground state energy: {E_sol}")

        # Checking ground state with FCI
        # Warning: This takes exponential time to run
        #     Checking H_cas symmetries
        if check_symmetry or FCI:
            H_cas = feru.get_ferm_op(cas_tbt, True)
        if check_symmetry:
            Sz = of.hamiltonians.sz_operator(spatial_orbs)
            S2 = of.hamiltonians.s_squared_operator(spatial_orbs)
            assert of.FermionOperator.zero() == of.normal_ordered(of.commutator(Sz, H_cas)), "Sz symmetry broken"
            assert of.FermionOperator.zero() == of.normal_ordered(of.commutator(S2, H_cas)), "S2 symmetry broken"

        if FCI:
            E_min, sol = of.get_ground_state(of.get_sparse_operator(H_cas))
            print(f"FCI Energy: {E_min}")
            tmp_st = sdstate(n_qubit = spin_orbs)
            for s in range(len(sol)):
                if sol[s] > np.finfo(np.float32).eps:
                    tmp_st += sdstate(s, sol[s])
            #         print(bin(s))
            print(tmp_st.norm())
            tmp_st.normalize()
            print(tmp_st.exp(H_cas))

        cas_killer = construct_killer(k, e_nums, n = spin_orbs)
        if check_symmetry:
            assert of.FermionOperator.zero() == of.normal_ordered(of.commutator(Sz, cas_killer)), "Killer broke Sz symmetry"
            assert of.FermionOperator.zero() == of.normal_ordered(of.commutator(S2, cas_killer)), "S2 symmetry broken"

        # Checking: if FCI of killer gives same result. Warning; takes exponential time 
        if FCI:
            sparse_with_killer = of.get_sparse_operator(cas_killer + H_cas)
            killer_Emin, killer_sol = of.get_ground_state(sparse_with_killer)
            print(f"FCI Energy solution with killer: {killer_Emin}")
            sd_Emin = sd_sol.exp(cas_tbt) + sd_sol.exp(cas_killer)
            print(f"difference with CAS energy: {sd_Emin - killer_Emin}")

        # Checking: if killer does not change ground state
#         killer_error = sd_sol.exp(cas_killer)
#         print(f"Solution Energy shift by killer: {killer_error}")
    #     killer_E_sol = sd_sol.exp(H_cas + cas_killer)
    #     print(f"Solution Energy with killer: {killer_E_sol}")

        planted_sol = {}
        planted_sol["E_min"] = E_cas
        planted_sol["e_nums"] = e_nums
        planted_sol["sol"] = states
        planted_sol["killer"] = cas_killer
        planted_sol["cas_x"] = cas_x
        planted_sol["k"] = k
        planted_sol["casnum"] = casnum
        planted_sol["pnum"] = pnum
        planted_sol["upnum"] = upnum
        planted_sol["spin_orbs"] = spin_orbs
        # print(planted_sol)
        ps_path = "planted_solutions/"
        f_name = file_name.split(".")[0] + ".pkl"
        print(ps_path +f_name)

        l = list(map(len, k))
        l = list(map(str, l))
        key = "-".join(l)
        print(key)
        if os.path.exists(ps_path + f_name):
            with open(ps_path + f_name, 'rb') as handle:
                dic = pickle.load(handle)
        else:
            dic = {}

        with open(ps_path + f_name, 'wb') as handle:
            dic[key] = planted_sol
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)