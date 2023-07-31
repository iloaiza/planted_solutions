"""Obtain the GCSA fragments FCI variances. 
This also checks that all fragments sums up to the Hamiltonian as expected. 

Usage: python gcsa_variance_compute.py (mol) (csa_nfrag) 
TODO: Extend to HF. 
"""
import sys
sys.path.append("../")
import numpy as np
import io

from qubit_utils import get_qwc_group
import saveload_utils as sl
import openfermion as of

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
csa_nfrag = 6 if len(sys.argv) < 3 else int(sys.argv[2])
geo = 1.0
wfs = 'fci'
method = 'GCSA'
save = True

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
gcsa_sol = sl.load_csa_sols(mol, geo, method, csa_nfrag)
gcsa_fragments = gcsa_sol['grouping']
h_ferm = sl.load_fermionic_hamiltonian(mol)
h_reconstructed = of.FermionOperator.zero()
n_qubits = of.count_qubits(h_ferm)

if wfs == 'hf':
    print("Not implemented yet. ")
    quit()
else:
    _, psi = of.get_ground_state(of.get_sparse_operator(h_ferm, n_qubits))

# Compute FCI variance
# Runs
group_vars = np.full(len(gcsa_fragments), np.nan)
for idx, frag in enumerate(gcsa_fragments):
    curvar = of.variance(of.get_sparse_operator(frag, n_qubits), psi)
    group_vars[idx] = np.real_if_close(curvar, tol=1e4)
    h_reconstructed += frag
print("Should be 0. H - \sum H_i: {}".format(of.normal_ordered(h_ferm - h_reconstructed)))
print("Optimal metric: {}".format(np.sum(group_vars**(1 / 2))**2))

if save:
    sl.save_variance_result(mol, wfs=wfs, method=method.lower(), 
                            geometry=geo, groups=gcsa_fragments, variances=group_vars)
    log_fpath = sl.get_logging_path_and_name(
        mol, geo, wfs, method.lower(), '../')
    with open(log_fpath, 'w') as f:
        print(log_string.getvalue(), file=f) 
