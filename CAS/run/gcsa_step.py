"""Perform CSA greedily for k steps for all possible number of cas fragments of the molecule. 
Then perform regular CSA on the terms left. 

Usage: python gcsa_all.py (mol) (num_greedy_steps) (final_alpha) (Trial_num). 
"""
import sys
sys.path.append("../")
import pickle
import io

import saveload_utils as sl
import ferm_utils as feru
import csa_utils as csau
import var_utils as varu
import openfermion as of
import numpy as np
import copy
import time 

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
num_greedy_steps = 5 if len(sys.argv) < 3 else int(sys.argv[2])
final_alpha = 1 if len(sys.argv) < 4 else int(sys.argv[3])
num_trial = 3 if len(sys.argv) < 5 else int(sys.argv[4])
# k = 0 if len(sys.argv) < 5 else int(sys.argv[4])
tol = 1e-8

# Get two-body tensor
Hf = sl.load_fermionic_hamiltonian(mol)
norb = of.count_qubits(Hf) // 2
Htbt = feru.get_chemist_tbt(Hf)
one_body = varu.get_one_body_correction_from_tbt(Hf, Htbt)
print("Initial Norm: {}".format(np.sum(Htbt * Htbt)))

# Run greedy CSA
result = []
result.append(["CasNum","Greedy Step","Two Norm"])
all_tbts = []
sol_array = np.array([])
n = Htbt.shape[0]
# Htbt = feru.get_chemist_tbt(Hf)
for t in range(num_greedy_steps):

#   Consider each possible split for k < n
    best_norm = 1e5 
    for k in range(n + 1):      
        gd_start = time.time()
        for i in range(num_trial):
            trial_best_norm = 1e5
#             trial_best_tbt 
            sol = csau.csa(Htbt,k = k, alpha=1, tol=tol, grad=True)
            cur_tbt = csau.sum_cartans(sol.x, norb, k, alpha=1, complex=False)
            
            
            two_norm = np.sum((Htbt - cur_tbt) * (Htbt - cur_tbt))
#           Update Trial_Best_Norm:
            if two_norm < trial_best_norm:
                trial_best_tbt = cur_tbt
                trial_best_norm = two_norm
#       Save the best fragment with split at k for k != n, if the fragment is not full CAS orbitals 
        if trial_best_norm < best_norm and k != n:
            best_tbt = trial_best_tbt
            best_norm = trial_best_norm
#       Update the information of trials       
        gd_time = round(time.time() - gd_start)
        print("CasNum: {}. Trial Number: {}. Greedy Step: {}. Best 2-norm: {}. Time elapsed: {} sec".format(k, num_trial, t + 1, trial_best_norm, gd_time))
        result.append([k, t+1, two_norm])
#   Updating the Htbt after this decomposition step

    Htbt -= best_tbt
    print("Current Norm:" + str(np.sum(Htbt * Htbt)))
    print("saving fragments")
    with open("./computed/" + mol + "#" + str(num_greedy_steps), "wb") as f:
        pickle.dump(result, f)
print(result)

with open("./computed/" + mol + "#" + str(num_greedy_steps), "wb") as f:
    pickle.dump(result, f)

# # Save
# if save:
#     sl.save_csa_sols(sol_array=sol_array, grouping=frags, n_qub=norb * 2,
#                      mol=mol, geo=1.0, method=method_name, log_str_io=log_string, ext_dict=extra_dict)
