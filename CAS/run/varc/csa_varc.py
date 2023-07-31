"""Obtain the difference CSA's fragments FCI variances. Use FCI wavefunction.
This also checks that all fragments sums up to the Hamiltonian as expected. 

Usage: python csa_varc.py (mol) (method) (csa_nfrag) 
"""
path_prefix = "../../"
import sys
sys.path.append(path_prefix)
import numpy as np
import io
import time

import saveload_utils as sl
import openfermion as of
import csa_utils as csau

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
method = 'CSA' if len(sys.argv) < 3 else sys.argv[2]
csa_nfrag = 2 if len(sys.argv) < 4 else int(sys.argv[3])

wfs = 'fci'
geo = 1.0
save = False
check_operator_sum = True

# Prepare logging
if save:
    log_string = io.StringIO()
    sys.stdout = log_string

# Display run's details
print("Molecule: {}".format(mol))
print("Wavefunction: {}".format(wfs))
print("Geometries: {}".format(geo))
print("Method: {}".format(method))
print("Number of CSA fragments: {}".format(csa_nfrag))
print()

# Prep
csa_sol = sl.load_csa_sols(
    mol, geo, method, csa_nfrag, prefix=path_prefix)
csa_fragments = csa_sol['grouping']

csa_tbts = csau.get_tbt_parts(csa_sol['sol_array'],
                              csa_sol['n_qubits'] // 2, csa_sol['csa_alpha'])
n_qubits = csa_sol['n_qubits']

if check_operator_sum:
    h_ferm = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
    h_reconstructed = of.FermionOperator.zero()

# Get wfs
psi = sl.load_ground_state(mol, 'FM', path_prefix=path_prefix, verbose=True)

# Loading FCI tbtev_ln/sq
tbtev_ln = sl.load_tbt_variance(
    "ev", mol, geo=1.0, wfs_type='fci', path_prefix=path_prefix)
tbtev_sq = sl.load_tbt_variance(
    "sq", mol, geo=1.0, wfs_type='fci', path_prefix=path_prefix, verbose=False)

# Compute FCI variance
group_vars = np.full(len(csa_fragments), np.nan)
for idx in range(len(csa_fragments)):
    frag = csa_fragments[idx]
    start = time.time()
    if idx == 0:
        curvar = of.variance(of.get_sparse_operator(frag, n_qubits), psi)
        if abs(np.imag(curvar)) > 1e-7:
            print("Current variance has non-zero complex part: {}".format(curvar))
        group_vars[idx] = np.real(curvar)
    else:
        group_vars[idx] = csau.get_precomputed_variance(
            csa_tbts[idx - 1], tbtev_ln, tbtev_sq)
    if check_operator_sum:
        h_reconstructed += frag

if check_operator_sum:
    print("Should be 0. H - \sum H_i: {}".format(of.normal_ordered(h_ferm - h_reconstructed)))
print("Optimal metric: {}".format(np.sum(group_vars**(1 / 2))**2))

if save:
    sl.save_variance_result(mol, wfs=wfs, method=method.lower(), path_prefix=path_prefix,
                            geometry=geo, groups=csa_fragments, variances=group_vars)
    log_fpath = sl.get_logging_path_and_name(
        mol, geo, wfs, method.lower(), prefix=path_prefix)
    with open(log_fpath, 'w') as f:
        print(log_string.getvalue(), file=f)
