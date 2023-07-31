""" This records the time it needs to run one iteration of VCSA 
Usage: python time_vcsa.py (mol) (alpha) (maxiter)
"""
path_prefix = "../../"
import sys
sys.path.append(path_prefix)

import saveload_utils as sl
import ferm_utils as feru
import csa_utils as csau
import var_utils as varu
import openfermion as of
import numpy as np
import time

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
alpha = 1 if len(sys.argv) < 3 else int(sys.argv[2])
maxiter = 10 if len(sys.argv) < 4 else int(sys.argv[3])
tol = 1e-7
var_weight = 0.05

# Get two-body tensor
Hf = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
norb = of.count_qubits(Hf) // 2
Htbt = feru.get_chemist_tbt(Hf)
one_body = varu.get_one_body_correction_from_tbt(Hf, Htbt)

# Get tbtev_ln/sq
tbtev_ln = sl.load_tbt_variance("ev", mol, geo=1.0, wfs_type='hf',
                                path_prefix=path_prefix)
tbtev_sq = sl.load_tbt_variance("sq", mol, geo=1.0, wfs_type='hf',
                                verbose=False, path_prefix=path_prefix)

# Run variance-CSA
vcsa_start = time.time()
sol = csau.varcsa(Htbt, tbtev_ln, tbtev_sq, alpha, var_weight=var_weight,
                  tol=tol, grad=True, options={"disp": True, 'maxiter': maxiter})
vcsa_time = time.time() - vcsa_start
print("VCSA done. Total Time elapsed: {}".format(round(vcsa_time)))


# Print
print("Molecule: {}".format(mol))
print("Method: VCSA")
print("Number of CSA fragments: {}".format(alpha))
print("Number of iterations: {}".format(maxiter))
print("Weight of variance term in cost: {}".format(var_weight))
print("Tolerance: {}".format(tol))
print()

print("Time it takes to do one iteration: {}".format(round(vcsa_time / maxiter)))
