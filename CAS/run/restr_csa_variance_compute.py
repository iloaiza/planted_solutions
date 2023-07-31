"""Compute restricted CSA's variances. Can compute for both greedy and full optimization. 
"""
import sys
sys.path.append("../")
import pickle
import io

import numpy as np
import time
import saveload_utils as sl
import openfermion as of

# Parameters
mol = 'h2'
wfs = 'fci'
geo = 1.0
alpha = 4
method = 'RESTR_GREEDY_CSA'
save = True

# Prepare logging
if save:
    log_string = io.StringIO()
    sys.stdout = log_string

# Display run's details
print("Molecule: {}".format(mol))
print("Wavefucntion: {}".format(wfs))
print("Geometries: {}".format(geo))
print("Method: {}".format(method))
print("Number of CSA fragments: {}".format(alpha))
print()

# Prep
restr_csa_sol = sl.load_restr_csa_sols(mol, geo, method.lower(), alpha)
restr_csa_fragments = restr_csa_sol['grouping']

h_ferm = sl.load_fermionic_hamiltonian(mol)
h_reconstructed = of.FermionOperator.zero()
n_qubits = of.count_qubits(h_ferm)

if wfs == 'hf':
    print("Not implemented yet. ")
    quit()
else:
    _, psi = of.get_ground_state(of.get_sparse_operator(h_ferm, n_qubits))

# Runs
group_vars = np.full(len(restr_csa_fragments), np.nan)
for idx, frag in enumerate(restr_csa_fragments):
    curvar = of.variance(of.get_sparse_operator(frag, n_qubits), psi)
    group_vars[idx] = np.real_if_close(curvar, tol=1e4)
    h_reconstructed += frag

print("H - \sum H_i: {}".format(of.normal_ordered(h_ferm - h_reconstructed)))

if save:
    sl.save_variance_result(mol, wfs=wfs, method=method.lower(
    ), geometry=geo, groups=restr_csa_fragments, variances=group_vars, path_prefix='../')

    sl.prepare_path(mol, wfs, method=method.lower(), prefix='../')
    log_fpath = sl.get_logging_path_and_name(
        mol, geo, wfs, method.lower(), '../')
    with open(log_fpath, 'w') as f:
        print(log_string.getvalue(), file=f) 
