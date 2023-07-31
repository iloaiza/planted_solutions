"""Loading fragments from saved results and check <H> = \sum <H_i> 
""" 
path_prefix = "../../"
import sys
sys.path.append(path_prefix)

import saveload_utils as sl
from openfermion import get_sparse_operator, expectation, count_qubits, bravyi_kitaev
from numpy import isclose

# Parameters 
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
method = 'csa' if len(sys.argv) < 3 else sys.argv[2]

# Load Hamiltonain 
Hf = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
n_qubits = count_qubits(Hf)

# Load FM ground state 
gs = sl.load_ground_state(mol, 'FM', path_prefix=path_prefix, verbose=True)

# Load fragments
frags = sl.load_variance_result(mol, geometry=1.0, wfs='fci', method=method, data_type='grps', path_prefix=path_prefix)

# Compute <H>
fm_expected = expectation(get_sparse_operator(Hf, n_qubits), gs)

# Sum over <H_i> 
fm_received = 0
for frag in frags:
    fm_received += expectation(get_sparse_operator(frag, n_qubits), gs)

# Compare 
print("<H>: {}".format(fm_expected))
print("\sum_i <H_i>: {}".format(fm_received))
print("Are they close: {}".format(isclose(fm_expected, fm_received)))
