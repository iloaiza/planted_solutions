"""
Performing Greedy CSA for k steps. The rest are measured through SVD. 

The one body terms are in GCSA grouping. 
Usage: python gcsa_svd_run.py (mol) (num_greedy_steps).  

TODO: Compute one-body terms from difference between tbt and original. Put one-body term in CSA grouping. 

TODO: Save logfiles in StringIO. Example in gcsa_fc_run.py
"""
import sys
sys.path.append('../')

import saveload_utils as sl
import ferm_utils as fermu
import csa_utils as csau
import var_utils as varu
import time
import numpy as np
import io
from openfermion import FermionOperator, normal_ordered, hermitian_conjugated

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
gd_k = 1 if len(sys.argv) < 3 else int(sys.argv[2])
save = True
svd_tol = 1e-6

# Setting up log file
log_string = None
if save:
    log_string = io.StringIO()
    sys.stdout = log_string

# Beginning file description
print("Mol: {}".format(mol))
print("Greedy step: {}".format(gd_k))

# Preps
Hf = sl.load_fermionic_hamiltonian(mol)
H = sl.load_interaction_hamiltonian(mol)
Htbt = fermu.get_chemist_tbt(Hf)
one_body = varu.get_one_body_correction_from_tbt(Hf, Htbt)
n = fermu.get_spin_orbitals(Hf)
start = time.time()

# Runs
tbts = []
gd_start = time.time()
x_len = (n // 2) ** 2  # Length of a single cartan elements coefficients array
gd_xs = np.zeros((gd_k * x_len))

# For each greedy step, minimize remaining norm using one cartan fragment
for i in range(gd_k):
    sol = csau.csa(Htbt, alpha=1, grad=True)
    gd_xs[i * x_len:(i + 1) * x_len] = sol.x
    cur_tbt = csau.sum_cartans(sol.x, n // 2, alpha=1, complex=False)
    tbts.append(cur_tbt)
    Htbt -= cur_tbt
    print("Step: {}. Current norm: {}\n".format(i + 1, np.sum(Htbt * Htbt)))
gd_time = time.time() - gd_start
print("Greedy Step done. Time elapsed: {}s".format(round(gd_time)))

# Compute Fermionic grouping from GCSA
gcsa_grouping = []
for tbt in tbts:
    gcsa_grouping.append(fermu.get_ferm_op(tbt))

# SVD
svd_fragments = []
if np.sum(Htbt * Htbt) > 1e-8:
    print("Calculating SVD fragments. ")
    svd_fragments = csau.get_svd_fragments(Htbt, tol=svd_tol, verbose=True)
gcsa_grouping.extend(svd_fragments)

# adding one body term
gcsa_grouping.insert(0, one_body)

# Computing sum of operators. Make sure they sums to H.
H_grouped = FermionOperator.zero()
for gcsa_part in gcsa_grouping:
    H_grouped += gcsa_part 
    # Assert all are hermitian 
    assert gcsa_part - hermitian_conjugated(gcsa_part) == FermionOperator.zero() 

diff = normal_ordered(Hf - H_grouped)
print("Should be 0. H - measured: {}".format(diff))

print("Total time elapsed: {}s".format(round(time.time() - start)))

# Saving results
if save:
    print("Saving results. ")
    sl.save_csa_sols(
        sol_array=gd_xs, grouping=gcsa_grouping,
        n_qub=n, mol=mol, geo=1.0, method='GCSA_SVD', log_str_io=log_string
    )
