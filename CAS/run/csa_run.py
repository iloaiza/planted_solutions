"""Perform vanilla CSA 
Usage: python csa_run.py (mol) (alpha) (max_iter) (load T/F)
"""
path_prefix = "../"
import sys
sys.path.append(path_prefix)

import pickle
import time 
import numpy as np
import saveload_utils as sl 
import io 
from csa_utils import csa, sum_cartans, get_tbt_parts
from ferm_utils import get_chemist_tbt, get_ferm_op
from openfermion import count_qubits
from var_utils import get_one_body_correction_from_tbt

### Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
alpha = 2 if len(sys.argv) < 3 else int(sys.argv[2])
maxiter = 1 if len(sys.argv) < 4 else int(sys.argv[3])
load = False if len(sys.argv) < 5 else sys.argv[4] == 'T'
tol = 1e-5
save = False
method_name = 'CSA-CAS'

# Setting up log file
log_string = None
if save:
    log_string = io.StringIO()
    sys.stdout = log_string
print("Calculations begins. Date: {}".format(sl.get_current_time()))

### Preps
Hf = sl.load_fermionic_hamiltonian(mol)
norb = count_qubits(Hf) // 2
Htbt = get_chemist_tbt(Hf)
one_body = get_one_body_correction_from_tbt(Hf, Htbt)

# Get loaded x0. Update Htbt. 
if load:
    print("Loading previous CSA solutions")
    csa_dict = sl.load_csa_sols(
        mol, geo=1.0, method=method_name, alpha=alpha, prefix=path_prefix, verbose=True)
    sol_array = csa_dict['sol_array']
    tbt_left = Htbt - sum_cartans(sol_array, Htbt.shape[0], alpha, complex=False)
    print("Initial L1-Norm: {}".format(np.sum(np.abs(tbt_left))))
else:
    sol_array = None 
    print("Initial L1-Norm: {}".format(np.sum(np.abs(Htbt))))

### Runs 
start = time.time()
sol = csa(Htbt, alpha=alpha, grad=True, tol=tol, x0=sol_array, maxiter=maxiter)
all_tbts = get_tbt_parts(sol.x, norb, alpha)

# Collect fragments.
frags = []
for tbt in all_tbts:
    frags.append(get_ferm_op(tbt, spin_orb=False))
frags.insert(0, one_body)
alpha = len(frags) - 1

### Beginning file description
print()
print("Molecule: {}".format(mol))
print("Number of orbitals: {}".format(norb))
print("Method: {}".format(method_name))
print("Number of CSA fragments: {}".format(alpha))
print("Number of max iterations: {}".format(maxiter))
print("Loading? {}".format(load))

### Print 
tbt_left = Htbt - sum_cartans(sol.x, Htbt.shape[0], alpha, complex=False)
print("Remaining L2-Norm: {}".format(np.sum(tbt_left * tbt_left) ** (1/2)))
print("Remaining L1-Norm: {}".format(np.sum(np.abs(tbt_left))))

time_used_s = time.time() - start
print("Time elapsed: {} mins".format(np.round(time_used_s / 60, 2)))

# Save
if save:
    sl.save_csa_sols(sol_array=sol.x, grouping=frags, n_qub=norb * 2,
                     mol=mol, geo=1.0, method=method_name, log_str_io=log_string)
