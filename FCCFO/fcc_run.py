#Creating the clique with not all number operators with the clique swapping algorithm based on the input molecular Hamiltonian
import openfermion as of
import numpy as np
import saveload_utils as sl
from fermigroups import fermgroup 
from clique_swap import *
from op_utils import *
import pickle as pkl
import reference_state_utils as ref
import sys

if __name__ == "__main__":
    mol = "h2" if len(sys.argv) < 2 else sys.argv[1]
    try:
        H = sl.load_fermionic_hamiltonian(mol, prefix='./')
    except:
        print("Molecule {} does not exist in ./ham_lib".format(mol))
        exit()
    H = of.transforms.chemist_ordered(H)

    n_paulis = len(of.jordan_wigner(H).terms) - 1
    print("Number of Pauli products to measure: {}".format(n_paulis))
    n_ferms = len(H.terms) - 1
    print("Number of Fermionic products to measure: {}".format(n_ferms))

    FG = fermgroup(H, 'nFG', verbose=False)

    opt_clique, opt_2norm = swapping_cliques(FG[0])
    with open("./planted_solutions/" + mol + ".pkl", "wb") as f:
        pkl.dump(sum(opt_clique), f)