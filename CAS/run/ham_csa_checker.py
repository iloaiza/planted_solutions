"""
Loading hamiltonians from ham_lib and loading CSA converged results from grad_res. 
Checking whether they are the same up to one body terms. 

CHECK TODO: 
    1. H2
    2. LiH
    3. H2O
    4. BeH2 
    5. NH3 
"""
import sys
sys.path.append("../")

import openfermion as of 
import saveload_utils as sl
import csa_utils as csau 
import var_utils as varu 
import ferm_utils as feru 

### Parameters 
mol = 'nh3'

# Loading from ham_lib 
h_lib = sl.load_fermionic_hamiltonian(mol)

# Load CSA result from grad_res 
norb = of.count_qubits(h_lib) // 2
_, alpha = varu.get_system_details(mol)
csa_tbts = csau.read_grad_tbts(mol, norb, alpha, prefix='../')

h_csa = of.FermionOperator.zero() 
for tbt in csa_tbts:
    h_csa += feru.get_ferm_op(tbt)

diff = of.normal_ordered(h_lib - h_csa)
print("H_library - H_CSA. Should be one body term:\n{}\n".format(diff))
