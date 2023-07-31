"""Perform CSA greedily for the first K steps. 
Then perform regular CSA on the terms left. 

Usage: python gcsa_run.py (mol) (num_greedy_steps) (final_alpha) (num_cas_parts). 
"""
import sys
sys.path.append("../")
import io

import saveload_utils as sl
import ferm_utils as feru
import csa_utils as csau
import var_utils as varu
import openfermion as of
import numpy as np
import time 

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
num_greedy_steps = 2 if len(sys.argv) < 3 else int(sys.argv[2])
final_alpha = 1 if len(sys.argv) < 4 else int(sys.argv[3])
k = 0 if len(sys.argv) < 5 else int(sys.argv[4])
tol = 1e-8
save = False
method_name = 'GCSA'
compute_hf_variance = True

# Setting up log file
log_string = None
if save:
    log_string = io.StringIO()
    sys.stdout = log_string

# Get two-body tensor
Hf = sl.load_fermionic_hamiltonian(mol)
norb = of.count_qubits(Hf) // 2
Htbt = feru.get_chemist_tbt(Hf)
one_body = varu.get_one_body_correction_from_tbt(Hf, Htbt)
print("Initial Norm: {}".format(np.sum(Htbt * Htbt)))

# Run greedy CSA
all_tbts = []
sol_array = np.array([])
for t in range(num_greedy_steps):
    gd_start = time.time()
    sol = csau.csa(Htbt,k = k, alpha=1, tol=tol, grad=True)
    sol_array = np.concatenate([sol_array, sol.x], axis=0)
    cur_tbt = csau.sum_cartans(sol.x, norb, k, alpha=1, complex=False)
    all_tbts.append(cur_tbt)
    Htbt -= cur_tbt
    gd_time = round(time.time() - gd_start)
    print("Greedy Step: {}. Current norm: {}. Time elapsed: {} sec".format(
        t + 1, np.sum(Htbt * Htbt), gd_time))
greedy_norm = np.sum(Htbt * Htbt)

# Run CSA
fcsa_start = time.time() 
sol = csau.csa(Htbt, k = k, alpha=final_alpha, tol=tol, grad=True)
extra_dict = {}
extra_dict['final_sol_array'] = sol.x
fcsa_time = round(time.time() - fcsa_start)
print("Final CSA done. Time elapsed: {} sec".format(fcsa_time))

# Change this. Collect individual tbt
final_csa_tbts = csau.get_tbt_parts(sol.x, norb, alpha=final_alpha, k = k)
all_tbts.extend(final_csa_tbts)
for tbt in final_csa_tbts:
    Htbt -= tbt
final_norm = np.sum(Htbt * Htbt)

# Collect fragments.
frags = []
for tbt in all_tbts:
    frags.append(feru.get_ferm_op(tbt, spin_orb=False))
frags.insert(0, one_body)

# Print
print("Molecule: {}".format(mol))
print("Method: {}".format(method_name))
print("Number of greedy step: {}".format(num_greedy_steps))
print("Number of fragments for final step: {}".format(final_alpha))
print("Tolerance: {}".format(tol))
print()

print("Norm after greedy step: {}".format(greedy_norm))
print("Norm after final CSA step: {}".format(final_norm))
print("The variances below are only for two-body terms. ")

# Compute HF variance
if compute_hf_variance:
    # Get tbtev_ln/sq
    tbtev_ln = sl.load_tbt_variance("ev", mol, geo=1.0, wfs_type='hf')
    tbtev_sq = sl.load_tbt_variance("sq", mol, geo=1.0, wfs_type='hf')

    hf_variances = []
    for tbt in all_tbts:
        curvar = csau.get_precomputed_variance(tbt, tbtev_ln, tbtev_sq)
        curvar = max(0, np.real_if_close(curvar))
        hf_variances.append(curvar)
    hf_variances = np.array(hf_variances)
    print("HF variances: {}".format(hf_variances))
    print("HF optimal metric: {}".format(np.sum(hf_variances**(1 / 2))**2))
    print()

# Save
if save:
    sl.save_csa_sols(sol_array=sol_array, grouping=frags, n_qub=norb * 2,
                     mol=mol, geo=1.0, method=method_name, log_str_io=log_string, ext_dict=extra_dict)
