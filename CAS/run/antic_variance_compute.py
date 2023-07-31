"""
Compute the FC fragments variances. Use either FCI or HF wavefunction 
TODO: Extend to HF. 
"""
import sys
sys.path.append("../")
import pickle 
import numpy as np 
import time 

from qubit_utils import recursive_largest_first
from ham_utils import get_system
from saveload_utils import get_logging_path_and_name, save_variance_result, prepare_path
from openfermion import bravyi_kitaev, get_ground_state, get_sparse_operator, count_qubits, QubitOperator, variance, commutator

def get_antic_group(H:QubitOperator):
    '''
    Return a list of commuting fragments of H
    '''
    # Building list of operators
    pws = []
    vals = []
    for pw, val in H.terms.items():
        pws.append(QubitOperator(term=pw, coefficient=1))
        vals.append(val)

    # Building commuting matrix and find commuting set
    pnum = len(pws)
    anticomm_mat = np.zeros((pnum, pnum))
    for i in range(pnum):
        for j in range(i+1, pnum):
            if commutator(pws[i], pws[j]) != QubitOperator.zero():
                anticomm_mat[i, j] = 1
    anticomm_mat = np.identity(pnum) + anticomm_mat + anticomm_mat.T 
    colors = recursive_largest_first(1 - anticomm_mat)

    comm_list = [QubitOperator.zero() for i in range(len(colors))]
    for key, indices in colors.items():
        for idx in indices:
            comm_list[key - 1] += pws[idx] * vals[idx]
    return comm_list

### Parameters 
mol = 'h2'
wfs = 'fci'
geo = 1.0 

### Prepare logging 
method='antic'
prepare_path(mol, wfs, method=method, prefix='../')
log_fpath = get_logging_path_and_name(mol, geo, wfs, 'fc', '../')
sys.stdout = open(log_fpath, 'w')

### Display run's details 
print("Molecule: {}".format(mol))
print("Wavefucntion: {}".format(wfs))
print("Geometries: {}".format(geo))
print("Method: {}".format(method))
print() 

### Prep 
h_ferm = get_system(mol, ferm=True, geometry=geo)
h_bk = bravyi_kitaev(h_ferm)

if wfs == 'hf':
    print("Not implemented yet. ")
    quit()
else: 
    _, psi = get_ground_state(get_sparse_operator(h_bk)) 
n_qubits = count_qubits(h_bk)

### Start computing variances 
groups = get_antic_group(h_bk)
group_vars = np.full(len(groups), np.nan)
h_meas = QubitOperator.zero() 
for idx, frag in enumerate(groups):
    cur_var = variance(get_sparse_operator(frag, n_qubits), psi)
    group_vars[idx] = np.real_if_close(cur_var, tol=1e6)
    h_meas += frag

save_variance_result(mol, wfs=wfs, method=method, geometry=geo, groups=groups, variances=group_vars, path_prefix='../')
print("{} parts - H_bk: {}".format(method.upper(), h_meas - h_bk))
print("Sum over sqrt: {}".format(np.sum(group_vars**(1/2))))
print() 

sys.stdout.close()