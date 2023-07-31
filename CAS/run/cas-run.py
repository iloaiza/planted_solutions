"""Perform CAS-CAS Decomposition
Usage: python cas_run.py (mol) 
"""
import sys
sys.path.append("../")
import pickle
import io

import saveload_utils as sl
import ferm_utils as feru
import csa_utils as csau
import var_utils as varu
import openfermion as of
import numpy as np
import copy
import time 

def partitions(n: int):
    """
    Return the all possible partition of the number n.
    """
    # base case of recursion: zero is the sum of the empty list
    if n == 0:
        yield []
        return
    
    # modify partitions of n-1 to form partitions of n
    for p in partitions(n-1):
        yield [1] + p
        if p and (len(p) < 2 or p[1] > p[0]):
            yield [p[0] + 1] + p[1:]

def partition_to_orbitals(partition: list[int]) -> list:
    """
    Return a orbital partitions from number partition of the total number of spin orbitals
    >>> partition_to_orbitals([2,2])
    [[0, 1], [2, 3]]
    >>> partition_to_orbitals([3,2,2])
    [[0, 1, 2], [3, 4], [5, 6]]
    """
    lis = [list(range(0+sum(partition[:i]),partition[i]+sum(partition[:i]))) for i in range(len(partition))]
    return lis

def valid_orbital_partitions(n: int) -> list[list[int]]:
    """
    Return the valid CAS orbital partitions with the number of spin orbitals n. 
    A partition is valid if it has more than one block and each block has at least 2 orbitals
    """     
    valid_partition = [i for i in list(partitions(n)) if min(i) > 1 and max(i) < n]
    valid_orbitals =[partition_to_orbitals(i) for i in valid_partition]
    return valid_orbitals
      
# def compute_cas_fragment(Htbt, k):
#     sol = csau.csa(Htbt,k = k, alpha=1, tol=tol, grad=True)
#     cur_tbt = csau.sum_cartans(sol.x, spin_orb, k, alpha=1, complex=False)
#     two_norm = np.sum((Htbt - cur_tbt) * (Htbt - cur_tbt))

#     relative_norm = two_norm / np.sum(Htbt * Htbt)
#     planted_H = feru.get_ferm_op(cur_tbt, True) + of.FermionOperator("", Hf.terms[()])
#     sparse_H = of.linalg.get_sparse_operator(planted_H)
#     var = np.real(of.linalg.variance(sparse_H, gs))
#     D_0 = of.linalg.eigenspectrum(Hf)
#     D_0.sort()

#     D_1 = np.real(of.linalg.eigenspectrum(planted_H))
#     D_1.sort()
#     eigen_spectrum_norm = np.linalg.norm(D_1 - D_0)
#     lis = [[k, two_norm, relative_norm, var, eigen_spectrum_norm]]
#     with open("./planted_solution/" + mol + ".pkl", "rb") as f:
#         result = pickle.load(f)
#     with open("./planted_solution/" + mol + " Hamiltonians.pkl", "rb") as f:
#         Hamiltonians = pickle.load(f)
#     result += lis
#     Hamiltonians[str(k)] = sol.x
#     with open("./planted_solution/" + mol + ".pkl", "wb") as f:
#         pickle.dump(result, f)
#     with open("./planted_solution/" + mol + " Hamiltonians.pkl", "wb") as f:
#         pickle.dump(Hamiltonians, f)    
        
        
# for k in valid_orbital_partitions(spin_orb):
#     compute_fragment(Htbt, k)
if __name__ == "__main__":
    ### Parameters
    mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
    tol = 1e-5
    save = False
    method_name = 'CAS-CAS'

    # Get two-body tensor
    Hf = sl.load_fermionic_hamiltonian(mol)
    _, gs = of.linalg.get_ground_state(of.linalg.get_sparse_operator(Hf))
    spin_orb = of.count_qubits(Hf)  
    spatial_orb = spin_orb // 2
    Htbt = feru.get_chemist_tbt(Hf, spin_orb, spin_orb = True)
    one_body = varu.get_one_body_correction_from_tbt(Hf, feru.get_chemist_tbt(Hf))
    # feru.get_one_body_terms(Hf)
    # varu.get_one_body_correction_from_tbt(Hf, Htbt)


    onebody_matrix = feru.get_obt(one_body, n = spin_orb, spin_orb = True)
    onebody_tbt = feru.onebody_to_twobody(onebody_matrix)
    # print(onebody_tbt.shape)
    # print(Htbt.shape)

    Htbt = np.add(Htbt, onebody_tbt)
    recombined = feru.get_ferm_op(Htbt, True)
    print("Initial Norm: {}".format(np.sum(Htbt * Htbt)))
    title = ["Partition","Norm of tbt", "relative tbt-norm", "Variance", "2-norm of eigenvalue spectrum"]
    result = [title]
    Hamiltonians = {}

    with open("./planted_solution/" + mol + ".pkl", "wb") as f:
        pickle.dump(result, f)
    # Hamiltonians[str(k)] = sol.x
    with open("./planted_solution/" + mol + " Hamiltonians.pkl", "wb") as f:
        pickle.dump(Hamiltonians, f)  
    from multiprocessing import Pool
    with Pool() as pool:
      pool.starmap(csau.compute_cas_fragment, [(Htbt, k, Hf, spin_orb, gs, mol) for k in valid_orbital_partitions(spin_orb)])
    print("done")
