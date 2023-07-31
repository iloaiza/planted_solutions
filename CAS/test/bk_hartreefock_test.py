"""Testing whether the bravyi-kitaev hartree fock wavefunction is generated correctly 
"""
path_prefix = "../"
import sys 
sys.path.append(path_prefix)

import saveload_utils as sl 
import openfermion as of 
import qubit_utils as qubu 
import ferm_utils as feru 
from var_utils import get_system_details 
from numpy import isclose

# Parameters 
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
n_electrons, _ = get_system_details(mol)

# Get Hamiltonians 
Hf = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
Hq = of.bravyi_kitaev(Hf)
n_qubits = of.count_qubits(Hf)

# Get Fermionic HF <H> 
ferm_hf = feru.get_openfermion_hf(n_qubits, n_electrons)
ferm_ev = of.expectation(of.get_sparse_operator(Hf, n_qubits), ferm_hf)

# Get Bravyi-Kitaev HF <H> 
bk_hf = qubu.get_openfermion_bk_hf(n_qubits, n_electrons)
bk_ev = of.expectation(of.get_sparse_operator(Hq, n_qubits), bk_hf)

# Printout 
print("Fermionic <H>    : {}".format(ferm_ev))
print("Bravyi-Kitaev <H>: {}".format(bk_ev))
assert isclose(ferm_ev, bk_ev)
