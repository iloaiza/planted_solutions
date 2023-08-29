import openfermion as of
import numpy as np
import saveload_utils as sl
from fermigroups import fermgroup 
from op_utils import *
import pickle as pkl
import reference_state_utils as ref
import sys

if __name__ == "__main__":
    with open("./planted_solutions/out.txt", "w") as f:
        f.write('')
    for mol in ["h2", "h4", "lih", "h2o", "beh2", "nh3", "n2"]:
#         mol = "h2" if len(sys.argv) < 2 else sys.argv[1]
        try:
            H = sl.load_fermionic_hamiltonian(mol, prefix='./')
        except:
            print("Molecule " + mol + " not found")
        H = of.transforms.chemist_ordered(H)
        sparse_H = of.get_sparse_operator(H)
        _, gs = of.linalg.get_ground_state(of.linalg.get_sparse_operator(H))
        var = np.real(of.linalg.variance(sparse_H, gs))
        eigen0 = of.linalg.eigenspectrum(H)
        # eigen0.sort()
        try:
            with open("./planted_solutions/" + mol + ".pkl", "rb") as f:
                opt_clique = pkl.load(f)  
        except:
            print("Molecule " + mol + " not found in ./planted_solutions")

        planted_H = sum(opt_clique)
        sparse_clique = of.linalg.get_sparse_operator(planted_H)
        var_clique = np.real(of.linalg.variance(sparse_clique, gs))
        eigen_clique = of.linalg.eigenspectrum(planted_H)
        eigen_clique.sort()
        eigen0.sort()
        norm_clique = np.linalg.norm(eigen_clique - eigen0)
        norm_clique = round(norm_clique, 3)
        var_clique = round(var_clique, 3)
        s = mol + "\nFCCFO & -- & -- & {} & {} \\\\".format(str(var_clique),str(norm_clique))
        print(s)
        with open("./planted_solutions/out.txt", "a") as f:
            f.write(s + '\n')