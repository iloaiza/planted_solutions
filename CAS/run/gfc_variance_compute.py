"""
Compute the Greedy FC fragments FCI variances. Contants are removed before grouping. 
"""
import sys
sys.path.append("../")
import pickle 
import numpy as np 
import time 

from qubit_utils import get_greedy_grouping
from ham_utils import get_system
import saveload_utils as sl 
from openfermion import bravyi_kitaev, get_ground_state, get_sparse_operator, count_qubits, QubitOperator, variance

### Parameters 
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
wfs = 'fci'
geo = 1.0 

### Prepare logging 
method='gfc'
sl.prepare_path(mol, wfs, method=method, prefix='../')
log_fpath = sl.get_logging_path_and_name(mol, geo, wfs, method, '../')
sys.stdout = open(log_fpath, 'w')

### Display run's details 
print("Molecule: {}".format(mol))
print("Wavefucntion: {}".format(wfs))
print("Geometries: {}".format(geo))
print("Method: {}".format(method))
print() 

### Prep 
h_ferm = sl.load_fermionic_hamiltonian(mol, prefix='../')
h_bk = bravyi_kitaev(h_ferm)
h_bk_noconstant = h_bk - h_bk.constant; h_bk_noconstant.compress() 

psi = sl.load_ground_state(mol, 'BK')
n_qubits = count_qubits(h_bk)

### Start computing variances 
groups = get_greedy_grouping(h_bk_noconstant, commutativity='fc')
group_vars = np.full(len(groups), np.nan)
h_meas = QubitOperator.zero() 
for idx, frag in enumerate(groups):
    cur_var = variance(get_sparse_operator(frag, n_qubits), psi)
    group_vars[idx] = np.real_if_close(cur_var, tol=1e6)
    h_meas += frag

sl.save_variance_result(mol, wfs=wfs, method=method, geometry=geo, groups=groups, variances=group_vars, path_prefix='../')
print("{} parts - H_bk: {}".format(method.upper(), h_meas - h_bk))
print("Sum over sqrt: {}".format(np.sum(group_vars**(1/2))))
print() 

sys.stdout.close()
