"""
Obtain the SVD fragments FCI variances. Use either FCI or HF wavefunction 
TODO: Extend to HF. 
"""
import sys
sys.path.append("../")
import pickle 
import numpy as np 

from ham_utils import get_system
from var_utils import get_google_measurement, get_one_body_correction
from ferm_utils import get_ferm_op
from saveload_utils import get_logging_path_and_name, save_variance_result, prepare_path, load_fermionic_hamiltonian, load_interaction_hamiltonian
from openfermion import get_ground_state, get_sparse_operator, count_qubits, FermionOperator, variance, normal_ordered
import time 

### Parameters 
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
wfs = 'fci'
geo = 1.0 
save=True 

### Prepare logging 
method='svd'
if save: 
    prepare_path(mol, wfs, method=method, prefix='../')
    log_fpath = get_logging_path_and_name(mol, geo, wfs, method, '../')
    sys.stdout = open(log_fpath, 'w')

### Display run's details 
print("Molecule: {}".format(mol))
print("Wavefucntion: {}".format(wfs))
print("Geometries: {}".format(geo))
print("Method: {}".format(method))
print() 

start = time.time() 
### Prep 
h_int = load_interaction_hamiltonian(mol, prefix='../')
h_ferm = load_fermionic_hamiltonian(mol, prefix='../')

if wfs == 'hf':
    print("Not implemented yet. ")
    quit()
else: 
    _, psi = get_ground_state(get_sparse_operator(h_ferm)) 
n_qubits = count_qubits(h_ferm)

### Start computing variances 
groups, offset = get_google_measurement(h_int)
one_body_corr = get_one_body_correction(h_ferm, offset)
group_vars = np.full(len(groups)+1, np.nan)
h_meas = FermionOperator.zero() 
for idx, frag in enumerate(groups):
    cur_var = variance(get_sparse_operator(frag, n_qubits), psi)
    group_vars[idx] = np.real_if_close(cur_var, tol=1e6)
    h_meas += frag

# One body terms 
cur_var = variance(get_sparse_operator(one_body_corr, n_qubits), psi)
group_vars[-1] = np.real_if_close(cur_var, tol=1e8)
h_meas += one_body_corr
groups.insert(0, one_body_corr)
total_time = time.time() - start 

print("Time elapsed: {} s".format(round(total_time)))
print("{} parts - H_fermionic: {}".format(method.upper(), normal_ordered(h_meas - h_ferm)))
print("Groups Vars: {}".format(group_vars))
print("Sum over sqrt: {}".format(np.sum(group_vars**(1/2))))
print() 

if save: 
    save_variance_result(mol, wfs=wfs, method=method, geometry=geo, groups=groups, variances=group_vars, path_prefix='../')
    sys.stdout.close()
