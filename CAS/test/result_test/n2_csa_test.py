"""Testing the saved CSA solutions. 
"""
path_prefix = "../../"
import sys
sys.path.append(path_prefix)

import saveload_utils as sl
from openfermion import normal_ordered
import csa_utils as csau
import ferm_utils as feru
import numpy as np 

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
method = 'csa' if len(sys.argv) < 3 else sys.argv[2]
csa_nfrag = 2 if len(sys.argv) < 4 else int(sys.argv[3])
geo = 1.0 

# Loading solutions 
frags = sl.load_variance_result(mol, geometry=1.0, wfs='fci', method=method, data_type='grps', path_prefix=path_prefix)
variances = sl.load_variance_result(mol, geometry=1.0, wfs='fci', method=method, data_type='vars', path_prefix=path_prefix)
csa_sol = sl.load_csa_sols(
    mol, geo, method.upper(), csa_nfrag, prefix=path_prefix)

# Display run's details
print("Molecule: {}".format(mol))
print("Method: {}".format(method))
print("Number of CSA fragments: {}".format(len(variances)))

# Check one-body terms 
print("Should be one body:\n{}".format(frags[0]))

# Check first term variance same accross methods 
print("First term variance: {}".format(variances[0]))

# Check sum of operator l1-norm 
h_ferm = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
Htbt = feru.get_chemist_tbt(h_ferm)

for frag in frags:
    h_ferm -= frag 
h_ferm = normal_ordered(h_ferm)
l1_norm = 0
for term, val in h_ferm.terms.items():
    l1_norm += abs(val)
print("Operator remains' L1-norm: {}".format(l1_norm))

# Check tbt l1-norm 
csa_fragments = csa_sol['grouping']
csa_tbts = csau.get_tbt_parts(csa_sol['sol_array'],
                              csa_sol['n_qubits'] // 2, csa_sol['csa_alpha'])
n_qubits = csa_sol['n_qubits']

for tbt in csa_tbts:
    Htbt -= tbt 
print("TBT remains' L1-norm: {}".format(np.sum(np.abs(Htbt))))

remain_tbt2op = feru.get_ferm_op(Htbt)
l1_norm = 0
for term, val in remain_tbt2op.terms.items():
    l1_norm += abs(val)
print("Tbt to operator remains' L1-norm: {}".format(l1_norm))

# Check few first VAR
print("First 20 variances:\n{}".format(variances[:20]))
