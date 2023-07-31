"""Checking whether splitting the one-body operator into diagonal and non-diagonal part help with the variance. 
"""
path_prefix = "../"
import sys 
sys.path.append(path_prefix)
import saveload_utils as sl
import openfermion as of 
import numpy as np 
import ferm_utils as feru
import var_utils as varu

# Parameters 
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]

# Get FCI 
Hf = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
n_qubits = of.count_qubits(Hf)
gs_e, gs = of.get_ground_state(of.get_sparse_operator(Hf, n_qubits=n_qubits))

# Get one-body operator. 
Htbt = feru.get_chemist_tbt(Hf)
one_body = varu.get_one_body_correction_from_tbt(Hf, Htbt)
one_body = one_body - one_body.constant; one_body.compress() # Remove constant 

# Get one-body's variance. 
all_ob_var = of.variance(of.get_sparse_operator(one_body, n_qubits=n_qubits), gs)

# Get diagonal one-body operator.
diagonal_ob = of.FermionOperator().zero()
for term, val in one_body.terms.items():
    assert len(term) == 2 
    if term[0][0] == term[1][0]:
        diagonal_ob += of.FermionOperator(term=term, coefficient=val)

# Get non-diagonal. 
non_diagonal_ob = one_body - diagonal_ob

# Display optimal metric 
split_ob_vars = np.zeros(2)
split_ob_vars[0] = of.variance(of.get_sparse_operator(diagonal_ob, n_qubits=n_qubits), gs)
split_ob_vars[1] = of.variance(of.get_sparse_operator(non_diagonal_ob, n_qubits=n_qubits), gs)

print("Optimal metric. (sum sqrt)^2")
print("One body terms: {}".format(one_body))
print("Diagonal One body terms: {}".format(diagonal_ob))
print("Nondiagonal One body terms: {}".format(non_diagonal_ob))
print("Measuring all one-body: {}".format(all_ob_var))
print("Splitting diagonal: {}\n{}".format(split_ob_vars, np.sum(split_ob_vars**(1/2))**2))
