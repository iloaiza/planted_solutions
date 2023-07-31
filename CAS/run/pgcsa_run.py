"""Perform CSA greedily for the first K steps until L1 norm is reached. 
Usage: python pgcsa_run.py (mol) (num_greedy_steps) (l1_norm_tol) 
    (compute_hf T/F) (load T/F) (load_alpha). 
"""
path_prefix = "../"
import sys
sys.path.append(path_prefix)
import io 

import saveload_utils as sl
import numpy as np
import ferm_utils as feru
import csa_utils as csau
import var_utils as varu
import openfermion as of
import time

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
num_greedy_steps = 2 if len(sys.argv) < 3 else int(sys.argv[2])
l1_norm_tol = 2.5e-6 if len(sys.argv) < 4 else float(sys.argv[3])
compute_hf_variance = False if len(sys.argv) < 5 else sys.argv[4] == 'T'
load = False if len(sys.argv) < 6 else sys.argv[5] == 'T'
load_alpha = 1 if len(sys.argv) < 7 else int(sys.argv[6])
tol = 1e-7
save = True
method_name = 'PGCSA'

# Setting up log file
log_string = None
if save:
    log_string = io.StringIO()
    sys.stdout = log_string
print("Calculations begins. Date: {}".format(sl.get_current_time()))

# Get two-body tensor
Hf = sl.load_fermionic_hamiltonian(mol)
norb = of.count_qubits(Hf) // 2
Htbt = feru.get_chemist_tbt(Hf)
one_body = varu.get_one_body_correction_from_tbt(Hf, Htbt)

# Get loaded x0. Update Htbt. 
if load:
    print("Loading previous CSA solutions")
    csa_dict = sl.load_csa_sols(
        mol, geo=1.0, method=method_name, alpha=load_alpha, prefix=path_prefix, verbose=True)
    sol_array = csa_dict['sol_array']
    all_tbts = csau.get_tbt_parts(sol_array, norb, load_alpha)
    Htbt -= np.sum(all_tbts, axis=0)
else:
    sol_array = np.array([])
    all_tbts = []
l1_norm = np.sum(abs(Htbt))
print("Initial L1 Norm: {}".format(l1_norm))

# Run greedy CSA
for k in range(num_greedy_steps):
    # Terminate at certain norm 
    if l1_norm < l1_norm_tol:
        print("Early termination. L1-norm tolerance reached. ")
        break 
    gd_start = time.time()
    sol = csau.csa(Htbt, alpha=1, tol=tol, grad=True)
    sol_array = np.concatenate([sol_array, sol.x], axis=0)
    cur_tbt = csau.sum_cartans(sol.x, norb, alpha=1, complex=False)
    all_tbts.append(cur_tbt)
    Htbt -= cur_tbt
    gd_time = round(time.time() - gd_start)
    l2_norm = np.sum(Htbt * Htbt); l1_norm = np.sum(abs(Htbt))
    print("Greedy Step: {}. Current norm: {}. L1-Norm: {}. Time elapsed: {} sec".format(
        k + 1, l2_norm, l1_norm, gd_time))
    sys.stdout.flush()

# Collect fragments.
frags = []
for tbt in all_tbts:
    frags.append(feru.get_ferm_op(tbt, spin_orb=False))
frags.insert(0, one_body)
alpha = len(frags) - 1

# Print
print()
print("Molecule: {}".format(mol))
print("Method: {}".format(method_name))
print("Number of max greedy step: {}".format(num_greedy_steps))
print("Number of CSA fragments: {}".format(alpha))
print("Optimization tolerance: {}".format(tol))
print("L1 norm termination tolerance: {}".format(l1_norm_tol))
if load:
    print("Loading from previous solution with alpha={}".format(load_alpha))
print()

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
    print("The variances below are only for two-body terms. ")
    print("HF variances: {}".format(hf_variances))
    print("HF optimal metric: {}".format(np.sum(hf_variances**(1 / 2))**2))
    print()

# Save
if save:
    sl.save_csa_sols(sol_array=sol_array, grouping=frags, n_qub=norb * 2,
                     mol=mol, geo=1.0, method=method_name, log_str_io=log_string)
