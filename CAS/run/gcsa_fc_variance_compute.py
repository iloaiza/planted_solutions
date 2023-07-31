"""
Compute the GCSA_FC fragments variances. Use either FCI or HF wavefunction 
TODO: Extend to HF. 
TODO: Test expectation value of measured parts sum up correctly. 
TODO: 
"""

import sys
sys.path.append("../")
import pickle 
import numpy as np 
import time 

from qubit_utils import get_qwc_group
import saveload_utils as sl
from saveload_utils import get_logging_path_and_name, save_variance_result, prepare_path, load_csa_sols
from openfermion import bravyi_kitaev, get_ground_state, get_sparse_operator, count_qubits, QubitOperator, variance
import openfermion as of 

### Parameters 
mol = 'beh2'
wfs = 'fci'
geo = 1.0 
alpha = 8

### Prepare logging 
method='GCSA_FC'
prepare_path(mol, wfs, method=method.lower(), prefix='../')
log_fpath = get_logging_path_and_name(mol, geo, wfs, method.lower(), '../')
sys.stdout = open(log_fpath, 'w')

### Display run's details 
print("Molecule: {}".format(mol))
print("Wavefucntion: {}".format(wfs))
print("Geometries: {}".format(geo))
print("Method: {}".format(method))
print("Qubit Transformation: BK")
print() 

### Prep 
csa_sol = load_csa_sols(mol, geo, method, alpha)
csa_fragments = csa_sol['grouping']
fc_fragments = csa_sol['grouping_remains']
print("CSA fragments: {}".format(len(csa_fragments)))
print("FC fragments: {}".format(len(fc_fragments)))

h_ferm = sl.load_fermionic_hamiltonian(mol)
h_bk = bravyi_kitaev(h_ferm)

if wfs == 'hf':
    print("Not implemented yet. ")
    quit()
else: 
    _, ferm_psi = get_ground_state(get_sparse_operator(h_ferm)) 
    _, bk_psi = get_ground_state(get_sparse_operator(h_bk))
n_qubits = count_qubits(h_ferm)

### Start computing variances 
# CSA part 
group_vars = np.full(len(csa_fragments)+len(fc_fragments), np.nan)
h_meas = of.FermionOperator.zero() # Making sure sum H_i = H
ev = 0 # Making sure sum <H_i> = <H> 
for idx, csa_frag in enumerate(csa_fragments):
    sparse_frag = get_sparse_operator(csa_frag, n_qubits)
    cur_var = variance(sparse_frag, ferm_psi)
    group_vars[idx] = np.real_if_close(cur_var, tol=1e4)
    
    ev += np.real_if_close(of.expectation(sparse_frag, ferm_psi))
    h_meas += csa_frag

# FC part
h_meas = bravyi_kitaev(h_meas, n_qubits) 
for idx, fc_frag in enumerate(fc_fragments):
    sparse_frag = get_sparse_operator(fc_frag, n_qubits)
    cur_var = variance(sparse_frag, bk_psi)
    group_vars[idx+len(csa_fragments)] = np.real_if_close(cur_var, tol=1e4)
    
    ev += np.real_if_close(of.expectation(sparse_frag, bk_psi))
    h_meas += fc_frag

groups = csa_fragments
groups.extend(fc_fragments)

save_variance_result(mol, wfs=wfs, method=method.lower(), geometry=geo, groups=groups, variances=group_vars, path_prefix='../')
print(group_vars)
print("{} parts - H_BK: {}".format(method.upper(), h_meas - h_bk))
print("sum_i <H_i>: {}".format(ev))
print("Sum over sqrt: {}".format(np.sum(group_vars**(1/2))))
print() 

sys.stdout.close()