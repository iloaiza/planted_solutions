"""
Obtain the CSA fragments FCI variances. Use either FCI or HF wavefunction 
TODO: Extend to HF. 
"""
import sys
sys.path.append("../")
import pickle 
import numpy as np 

from ham_utils import get_system
from csa_utils import read_grad_tbts
from ferm_utils import get_chemist_tbt, get_ferm_op
from var_utils import get_system_details
from saveload_utils import get_logging_path_and_name, save_variance_result, prepare_path, load_fermionic_hamiltonian
from openfermion import count_qubits, get_ground_state, get_sparse_operator, variance, normal_ordered, FermionOperator

### Parameters 
mol = 'h2'
wfs = 'fci'
geo = 1.0 

### Prepare logging 
method='csa'
prepare_path(mol, wfs, method=method, prefix='../')
log_fpath = get_logging_path_and_name(mol, geo, wfs, method, '../')
sys.stdout = open(log_fpath, 'w')

### Display run's details 
print("Molecule: {}".format(mol))
print("Wavefucntion: {}".format(wfs))
print("Geometries: {}".format(geo))
print("Method: {}".format(method))
print() 

### Prep
_, alpha = get_system_details(mol)
h_ferm = load_fermionic_hamiltonian(mol, prefix='../')

if wfs == 'hf':
    print("Not implemented yet. ")
    quit()
else: 
    _, psi = get_ground_state(get_sparse_operator(h_ferm)) 

### Get grouping 
norb = count_qubits(h_ferm) // 2
csa_tbts = read_grad_tbts(mol, norb, alpha, prefix='../')
groups = [] 
for tbt in csa_tbts:
    groups.append(get_ferm_op(tbt))

h_remain = h_ferm
for op in groups:
    h_remain = h_remain - op
groups.append(h_remain)

print("H - CSA_H. Should be one_body terms:\n{}\n".format(normal_ordered(h_remain)))

### Show sqrt variances 
group_vars = np.full(len(groups), np.nan)
h_meas = FermionOperator.zero() 
for idx, frag in enumerate(groups):
    cur_var = variance(get_sparse_operator(frag, n_qubits=norb*2), psi)
    group_vars[idx] = np.real_if_close(cur_var, tol=1e6)
    h_meas += frag

save_variance_result(mol, wfs=wfs, method=method, geometry=geo, groups=groups, variances=group_vars, path_prefix='../')
print("{} parts - H_fermionic: {}".format(method.upper(), normal_ordered(h_meas - h_ferm)))
print("Sum over sqrt: {}".format(np.sum(group_vars**(1/2))))
print() 

sys.stdout.close()