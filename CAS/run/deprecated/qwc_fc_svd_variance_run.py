"""
Obtain the QWC/FC/SVD fragments FCI variances. This was initially done for Zapata's data. 
"""
import sys
sys.path.append("../")
import pickle 
import numpy as np 
import time 

from qubit_utils import get_qwc_group, get_fc_group
from ham_utils import get_system
from var_utils import get_google_measurement, get_one_body_correction
from ferm_utils import get_ferm_op
from openfermion import bravyi_kitaev, variance, get_ground_state, get_sparse_operator, QubitOperator, count_qubits, FermionOperator, normal_ordered

### Parameters 
mol = 'nh3'
geo = 1.0 

print("Molecule: {}".format(mol))
print("Geometries: {}".format(geo))
print() 

### Prep 
h_int = get_system(mol, ferm=False, geometry=geo)
h_ferm = get_system(mol, ferm=True, geometry=geo)

gs_e, gs_ferm = get_ground_state(get_sparse_operator(h_ferm))
h_bk = bravyi_kitaev(h_ferm)
_, gs_bk = get_ground_state(get_sparse_operator(h_bk)) 
n_qubits = count_qubits(h_bk)

### FC 
fc_groups = get_fc_group(h_bk)
fc_vars = np.full(len(fc_groups), np.nan)
print(fc_groups)
h_fc = QubitOperator.zero() 

# FC Obtain variance & Check 
for idx, fc_part in enumerate(fc_groups):
    cur_var = variance(get_sparse_operator(fc_part, n_qubits), gs_bk)
    fc_vars[idx] = np.real_if_close(cur_var, tol=1e8)
    h_fc += fc_part
print("FC parts - H_bk: {}".format(h_fc - h_bk))
print("Sum over sqrt: {}".format(np.sum(fc_vars**(1/2))))
print() 

### QWC
qwc_groups = get_qwc_group(h_bk)
qwc_vars = np.full(len(qwc_groups), np.nan)
h_qwc = QubitOperator.zero() 
for idx, qwc_part in enumerate(qwc_groups):
    cur_var = variance(get_sparse_operator(qwc_part, n_qubits), gs_bk)
    qwc_vars[idx] = np.real_if_close(cur_var, tol=1e8)
    h_qwc += qwc_part

print("QWC parts - H_bk: {}".format(h_qwc - h_bk))
print("Sum over sqrt: {}".format(np.sum(qwc_vars**(1/2))))
print() 

### SVD
svd_groups, svd_offset = get_google_measurement(h_int)
one_body_corr = get_one_body_correction(h_ferm, svd_offset)
svd_vars = np.full(len(svd_groups)+1, np.nan)
h_svd = FermionOperator.zero()

# Two body terms 
for idx, svd_part in enumerate(svd_groups):
    cur_var = variance(get_sparse_operator(svd_part, n_qubits), gs_ferm)
    svd_vars[idx] = np.real_if_close(cur_var, tol=1e8)
    h_svd += svd_part

# One body terms 
cur_var = variance(get_sparse_operator(one_body_corr, n_qubits), gs_ferm)
svd_vars[-1] = np.real_if_close(cur_var, tol=1e8)
h_svd += one_body_corr

print("SVD parts - H_fermionic: {}".format(normal_ordered(h_svd - h_ferm)))
print("Sum over sqrt: {}".format(np.sum(svd_vars**(1/2))))
print() 
