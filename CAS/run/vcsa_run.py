"""Perform Variance CSA 
Usage: python vcsa_run.py (mol) (var_weight) (alpha) (load) (maxiter)
"""
path_prefix = "../"
import sys
sys.path.append(path_prefix)
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
var_weight = 0.05 if len(sys.argv) < 3 else float(sys.argv[2])
alpha = 1 if len(sys.argv) < 4 else int(sys.argv[3])
load = False if len(sys.argv) < 5 else sys.argv[4] == 'True'
maxiter = 1000 if len(sys.argv) < 6 else int(sys.argv[5])
l1_norm_tol = 2.5e-6
max_num_gsteps = 300
tol = 1e-7
save = True
method_name = "VCSA"

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
print("Initial L1 Norm: {}".format(np.sum(np.abs(Htbt))))

# Get tbtev_ln/sq
tbtev_ln = sl.load_tbt_variance("ev", mol, geo=1.0, wfs_type='hf', verbose=False)
tbtev_sq = sl.load_tbt_variance("sq", mol, geo=1.0, wfs_type='hf', verbose=False)

# Run variance-CSA
vcsa_start = time.time()
if load:
    # Get loaded x0
    print("Loading previous CSA solutions")
    csa_dict = sl.load_csa_sols(
        mol, geo=1.0, method=method_name, alpha=alpha, prefix=path_prefix, verbose=True)
    x0 = csa_dict['sol_array']
    sol = csau.varcsa(Htbt, tbtev_ln, tbtev_sq, alpha, var_weight=var_weight,
                      tol=tol, grad=True, options={"disp": True, 'maxiter': maxiter}, x0=x0)
else:
    sol = csau.varcsa(Htbt, tbtev_ln, tbtev_sq, alpha, var_weight=var_weight,
                      tol=tol, grad=True, options={"disp": True, 'maxiter': maxiter})
vcsa_time = time.time() - vcsa_start
print("VCSA done. Time elapsed: {}".format(round(vcsa_time)))

# Collect individual tbt
sol_array = np.copy(sol.x)
all_tbts = csau.get_tbt_parts(sol_array, norb, alpha=alpha)
for tbt in all_tbts:
    Htbt -= tbt
l1_norm = np.sum(np.abs(Htbt))
v_norm = l1_norm
print("After VCSA. L1 Norm: {}".format(v_norm))

# Run GCSA 
for k in range(max_num_gsteps):
    # Terminate at certain norm 
    if l1_norm < l1_norm_tol:
        print("Early termination. L1-norm tolerance reached.\n")
        break 
    gd_start = time.time()
    sol = csau.csa(Htbt, alpha=1, tol=tol, grad=True)
    sol_array = np.concatenate([sol_array, sol.x], axis=0)
    cur_tbt = csau.sum_cartans(sol.x, norb, alpha=1, complex=False)
    all_tbts.append(cur_tbt)
    Htbt -= cur_tbt
    gd_time = round(time.time() - gd_start)
    l1_norm = np.sum(abs(Htbt))
    print("Greedy Step {}: L1-Norm: {}. Time elapsed: {} sec".format(
        k + 1, l1_norm, gd_time))
    sys.stdout.flush()
g_norm = l1_norm

# Collect fragments.
frags = []
for tbt in all_tbts:
    frags.append(feru.get_ferm_op(tbt, spin_orb=False))
frags.insert(0, one_body)

# Compute HF variance
hf_variances = []
for tbt in all_tbts:
    curvar = csau.get_precomputed_variance(tbt, tbtev_ln, tbtev_sq)
    curvar = max(0, np.real_if_close(curvar))
    hf_variances.append(curvar)
hf_variances = np.array(hf_variances)

# Print
print("Molecule: {}".format(mol))
print("Method: {}".format(method_name))
print("Number of fragments for VCSA step: {}".format(alpha))
print("Number of max greedy step: {}".format(max_num_gsteps))
print("Number of final CSA fragments (VCSA + GCSA): {}".format(len(frags)-1))
print("Weight of variance term in cost: {}".format(var_weight))
print("Optimization tolerance: {}".format(tol))
print("L1 norm termination tolerance: {}".format(l1_norm_tol))
print("Loading from previous solution: {}".format(load))
print()

print("L1 Norm after VCSA: {}".format(v_norm))
print("L1 Norm after GCSA: {}".format(g_norm))

print("The variances below are only for two-body terms. ")

print("HF variances: {}".format(hf_variances))
print("HF optimal metric: {}".format(np.sum(hf_variances**(1 / 2))**2))
print()

# Save
if save:
    sl.save_csa_sols(sol_array=sol_array, grouping=frags, n_qub=norb * 2,
                     mol=mol, geo=1.0, method=method_name, log_str_io=log_string)
