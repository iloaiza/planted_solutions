"""Perform restricted greedy CSA  
"""
import sys
sys.path.append("../")
import unittest
import io

import saveload_utils as sl
import ferm_utils as feru
import csa_utils as csau
import var_utils as varu
import numpy as np
import openfermion as of

# Parameters
mol = 'h2'
tol = 1e-7
save = False
max_alpha = 100
grad = True
reflection_indices = [0, 1, 3]
spin_orb = False

# Setting up log file
log_strlog_string_io = None
if save:
    log_string = io.StringIO()
    sys.stdout = log_string

# Prep
method = "RESTR_GREEDY_CSA"
Hf = sl.load_fermionic_hamiltonian(mol)
n_qubits = of.count_qubits(Hf)
target_tbt = feru.get_chemist_tbt(Hf, spin_orb=spin_orb)
one_body = varu.get_one_body_correction_from_tbt(Hf, target_tbt)

if spin_orb: n_orbitals = n_qubits 
else: n_orbitals = n_qubits // 2

possible_cartans = []  # List of list of cartans to use.
for idx in reflection_indices:
    possible_cartans.append(csau.get_restr_reflections(
        idx, n_orbitals=n_orbitals))

# Beginning file description
print("Mol: {}".format(mol))
print("Method: {}".format(method))
print("Reflections indices used: {}".format(reflection_indices))
print("Using gradient?: {}".format(grad))
print("Using spin-orbital: {}".format(spin_orb))
print()

# Begin greedy
sol_array = np.array([])
cartan_used = []  # List of list of cartans.
tbts_used = []  # List of 4-rank tensor
alpha = 0
cur_norm = np.sum(target_tbt * target_tbt)**(1 / 2)

# Perform greedy until tol.
while cur_norm > tol and alpha < max_alpha:
    # Loop over possible cartans and go with the lowest.
    lowest_x, lowest_cartans, lowest_tbt_used, lowest_norm = None, None, None, np.inf
    for cartans in possible_cartans:
        # Perform one greedy step, update tbt and saves results.
        sol = csau.restr_csa(target_tbt, cartans, tol=tol, grad=grad)
        cur_tbt = csau.restr_sum_cartans(sol.x, cartans)
        tmp_tbt = target_tbt - cur_tbt
        tmp_norm = np.sum(tmp_tbt * tmp_tbt)**(1 / 2)
        if tmp_norm < lowest_norm:
            lowest_x = sol.x
            lowest_cartans = cartans
            lowest_tbt_used = cur_tbt
            lowest_target_tbt = tmp_tbt
            lowest_norm = tmp_norm

    # Update with the lowest
    sol_array = np.concatenate([sol_array, lowest_x], axis=0)
    cartan_used.append(lowest_cartans)
    tbts_used.append(lowest_tbt_used)
    target_tbt = lowest_target_tbt
    cur_norm = lowest_norm
    alpha += 1
    print("Alpha: {}. Current Norm: {}".format(alpha, cur_norm))
print("# of reflections needed: {}".format(alpha))

# Build grouping
restr_csa_grouping = []
for tbt in tbts_used:
    restr_csa_grouping.append(feru.get_ferm_op(tbt))
restr_csa_grouping.insert(0, one_body)  # Insert one-body term in GCSA

# Save log and results.
if save:
    sl.save_restr_csa_sols(sol_array, cartans=cartan_used, grouping=restr_csa_grouping, n_qubits=n_qubits,
                           mol=mol, geo=1.0, method=method.lower(), path_prefix='../', log_string=log_string)
