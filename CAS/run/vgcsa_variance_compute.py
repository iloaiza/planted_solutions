"""Obtain the VGCSA fragments FCI variances. Use either FCI or HF wavefunction.
This also checks that all fragments sums up to the Hamiltonian as expected. 

Usage: python vgcsa_variance_compute.py (mol) (wfs) (csa_nfrag) 
"""
import sys
sys.path.append("../")
import numpy as np
import io

from qubit_utils import get_qwc_group
import saveload_utils as sl
import openfermion as of
import csa_utils as csau 

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
csa_nfrag = 2 if len(sys.argv) < 3 else int(sys.argv[2])
wfs = 'fci' 
geo = 1.0
method = 'VGCSA'
save = True
check_precomputed = True 

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
vgcsa_sol = sl.load_csa_sols(mol, geo, method, csa_nfrag)
vgcsa_fragments = vgcsa_sol['grouping']
vgcsa_tbts = csau.get_tbt_parts(vgcsa_sol['sol_array'],
                                vgcsa_sol['n_qubits'] // 2, vgcsa_sol['csa_alpha'])
h_ferm = sl.load_fermionic_hamiltonian(mol)
h_reconstructed = of.FermionOperator.zero()
n_qubits = of.count_qubits(h_ferm)


# Get wfs 
_, psi = of.get_ground_state(of.get_sparse_operator(h_ferm, n_qubits))

# Loading FCI tbtev_ln/sq
tbtev_ln = sl.load_tbt_variance("ev", mol, geo=1.0, wfs_type='fci')
tbtev_sq = sl.load_tbt_variance(
    "sq", mol, geo=1.0, wfs_type='fci', verbose=False)

# Compute FCI variance
group_vars = np.full(len(vgcsa_fragments), np.nan)
for idx, frag in enumerate(vgcsa_fragments):
    if idx == 0:
        curvar = of.variance(of.get_sparse_operator(frag, n_qubits), psi)
        group_vars[idx] = np.real_if_close(curvar, tol=1e4)
    else:
        group_vars[idx] = csau.get_precomputed_variance(vgcsa_tbts[idx-1], tbtev_ln, tbtev_sq)
        if check_precomputed and idx < 3:
            print("Checking {}th fragment variance.".format(idx))
            curvar = of.variance(of.get_sparse_operator(frag, n_qubits), psi)
            print("Computed var: {}".format(curvar))
            print("Precomputed var: {}".format(group_vars[idx]))
            assert np.isclose(curvar, group_vars[idx])
            print("Check passed.")
    h_reconstructed += frag
print("Should be 0. H - \sum H_i: {}".format(of.normal_ordered(h_ferm - h_reconstructed)))
print("Optimal metric: {}".format(np.sum(group_vars**(1 / 2))**2))

if save:
    sl.save_variance_result(mol, wfs=wfs, method=method.lower(), 
                            geometry=geo, groups=vgcsa_fragments, variances=group_vars)
    log_fpath = sl.get_logging_path_and_name(
        mol, geo, wfs, method.lower(), '../')
    with open(log_fpath, 'w') as f:
        print(log_string.getvalue(), file=f) 
