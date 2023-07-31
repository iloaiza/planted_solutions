"""Perform CSA greedily for k steps for all possible number of cas fragments of the molecule. 
Then perform regular CSA on the terms left. 

Usage: python gcsa_all.py (mol) (num_greedy_steps) (final_alpha) (Trial_num). 
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
Htbt_0 = feru.get_chemist_tbt(Hf)
one_body = varu.get_one_body_correction_from_tbt(Hf, Htbt_0)
print("Initial Norm: {}".format(np.sum(Htbt_0 * Htbt_0)))

# Run greedy CSA
result = []
result.append(["CasNum","Greedy Step","Two Norm"])
all_tbts = []
sol_array = np.array([])
n = Htbt_0.shape[0]
Htbt = feru.get_chemist_tbt(Hf)
for k in range(n+1):
    Htbt -= Htbt
    Htbt += Htbt_0
    for t in range(num_greedy_steps):
        best_norm = 1e5 
        gd_start = time.time()
        for i in range(num_trial):
        
            sol = csau.csa(Htbt,k = k, alpha=1, tol=tol, grad=True)
            cur_tbt = csau.sum_cartans(sol.x, norb, k, alpha=1, complex=False)
            
            
            two_norm = np.sum((Htbt - cur_tbt) * (Htbt - cur_tbt))
            if two_norm < best_norm:
                best_tbt = cur_tbt
                best_norm = two_norm
        Htbt -= best_tbt
        gd_time = round(time.time() - gd_start)
        print("CasNum: {}. Trial Number: {}. Greedy Step: {}. Current norm: {}. Time elapsed: {} sec".format(k, num_trial, t + 1, np.sum(Htbt * Htbt), gd_time))
        result.append([k, t+1, two_norm])
print(result)

import pickle
with open("./computed/" + mol + "#" + str(num_greedy_steps), "wb") as f:
    pickle.dump(result, f)
# # Run CSA
# fcsa_start = time.time() 
# sol = csau.csa(Htbt, k = k, alpha=final_alpha, tol=tol, grad=True)
# extra_dict = {}
# extra_dict['final_sol_array'] = sol.x
# fcsa_time = round(time.time() - fcsa_start)
# print("Final CSA done. Time elapsed: {} sec".format(fcsa_time))

# # Change this. Collect individual tbt
# final_csa_tbts = csau.get_tbt_parts(sol.x, norb, alpha=final_alpha, k = k)
# all_tbts.extend(final_csa_tbts)
# for tbt in final_csa_tbts:
#     Htbt -= tbt
# final_norm = np.sum(Htbt * Htbt)

# # Collect fragments.
# frags = []
# for tbt in all_tbts:
#     frags.append(feru.get_ferm_op(tbt, spin_orb=False))
# frags.insert(0, one_body)

# # Print
# print("Molecule: {}".format(mol))
# print("Method: {}".format(method_name))
# print("Number of greedy step: {}".format(num_greedy_steps))
# print("Number of fragments for final step: {}".format(final_alpha))
# print("Tolerance: {}".format(tol))
# print()

# print("Norm after greedy step: {}".format(greedy_norm))
# print("Norm after final CSA step: {}".format(final_norm))
# print("The variances below are only for two-body terms. ")

# # Compute HF variance
# if compute_hf_variance:
#     # Get tbtev_ln/sq
#     tbtev_ln = sl.load_tbt_variance("ev", mol, geo=1.0, wfs_type='hf')
#     tbtev_sq = sl.load_tbt_variance("sq", mol, geo=1.0, wfs_type='hf')

#     hf_variances = []
#     for tbt in all_tbts:
#         curvar = csau.get_precomputed_variance(tbt, tbtev_ln, tbtev_sq)
#         curvar = max(0, np.real_if_close(curvar))
#         hf_variances.append(curvar)
#     hf_variances = np.array(hf_variances)
#     print("HF variances: {}".format(hf_variances))
#     print("HF optimal metric: {}".format(np.sum(hf_variances**(1 / 2))**2))
#     print()

# # Save
# if save:
#     sl.save_csa_sols(sol_array=sol_array, grouping=frags, n_qub=norb * 2,
#                      mol=mol, geo=1.0, method=method_name, log_str_io=log_string, ext_dict=extra_dict)
