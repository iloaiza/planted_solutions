import csa_utils as csau
import ferm_utils as feru
import openfermion as of
import pickle
import json

import numpy as np

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

def load_Hamiltonian(mol, k):
    """Loads a Hamiltonian file stored under the folder planted_solutions, for a molecule mol 
    with an orbital splitting of k. The returned Hamiltonian is in the form openfermion.FermionOperator
    """
    k.sort()
    try:
        with open("./run/planted_solution/" + mol + " Hamiltonians.pkl", "rb") as f:
            dic = pickle.load(f)  
    except:
        print("Hamiltonian of molecule {} not found".format(mol))
        return
    orbitals = partition_to_orbitals(k)
    if str(orbitals) not in dic:
        print("Partition not found")
        print("available partitions:")
        for key in dic:
            print([len(i) for i in json.loads(key)])
        return
    cur_tbt = csau.sum_cartans(dic[str(orbitals)], sum(k), orbitals, 1, complex=False)
    planted_H = feru.get_ferm_op(cur_tbt, True)
    return planted_H

if __name__ == "__main__":
    print(load_Hamiltonian("h2", [1,3]))
    print(load_Hamiltonian("h2", [2,2]))
    print(load_Hamiltonian("h4", [2,2,2,2]))
