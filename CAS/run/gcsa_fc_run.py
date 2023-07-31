"""
Performing Greedy CSA for k steps. The rest are measured through FC. 

The one body terms are in GCSA grouping. 
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
from openfermion import FermionOperator, normal_ordered
import openfermion as of 
import qubit_utils as qubu

# Parameters
mol = 'lih'
gd_k = 2
save = True
method_name = "GCSA_FC"

# Setting up log file
log_str_io = None
if save: 
    log_str_io = io.StringIO()
    sys.stdout = log_str_io

# Beginning file description
print("Mol: {}".format(mol))
print("Greedy step: {}".format(gd_k))
print("Method: {}".format(method_name))
print("Qubit Transformation: BK")
print() 

# Preps
Hf = sl.load_fermionic_hamiltonian(mol)
H = sl.load_interaction_hamiltonian(mol)
Htbt = fermu.get_chemist_tbt(Hf)
one_body = varu.get_one_body_correction_from_tbt(Hf, Htbt)
n = fermu.get_spin_orbitals(Hf)

# Runs. Initiating place to save gradient solution vector. 
tbts = []
start = time.time()
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
grad_time = time.time() - start
print("Gradient Optimization done. Time elapsed: {}s".format(round(grad_time)))

# Compute Fermionic grouping from GCSA
gcsa_grouping = []
for tbt in tbts:
    gcsa_grouping.append(fermu.get_ferm_op(tbt))
gcsa_grouping.insert(0, one_body) # Insert one-body term in GCSA 

# FC 
H_remain = fermu.get_ferm_op(Htbt, spin_orb=False)
H_remain_bk = of.bravyi_kitaev(H_remain, n)
fc_fragments = qubu.get_fc_group(H_remain_bk)

# Computing sum of operators. Make sure they sums to H.
H_grouped = FermionOperator.zero() 
for gcsa_part in gcsa_grouping: H_grouped += gcsa_part

# Transform to bk then: for fc_part in fc_fragments: H_grouped += fc_part
H_grouped = of.bravyi_kitaev(H_grouped, n)
for fc_part in fc_fragments: H_grouped += fc_part 

diff = of.bravyi_kitaev(Hf, n) - H_grouped
print("Should be 0. H - measured: {}".format(diff))

print("Total time elapsed: {}s".format(round(time.time() - start)))
# Saving results
if save:
    print("Saving results. ")
    sl.save_csa_sols(
        sol_array=gd_xs, grouping=gcsa_grouping, grouping_remains=fc_fragments, 
        n_qub=n, mol=mol, geo=1.0, method=method_name, log_str_io=log_str_io
    )
