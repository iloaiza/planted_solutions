"""
Generate Hamiltonians and store 

Usage: python ham_gen (mol)
"""
path_prefix = "../../"
import sys
sys.path.append(path_prefix)
from ham_utils import get_system
from openfermion import count_qubits
import pickle 

mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
basis = 'sto3g' if len(sys.argv) < 3 else sys.argv[2]
H = get_system(mol, basis=basis)
Hf = get_system(mol, ferm=True, basis=basis)
print("Number of qubits: {}".format(count_qubits(Hf)))

if basis == 'sto3g': 
    mol_fname = mol
else:
    mol_fname = "{}_{}".format(mol, basis)

fname = path_prefix + 'ham_lib/' + mol_fname + '_fer.bin'
with open(fname, 'wb') as f:
    pickle.dump(Hf, f)

fname = path_prefix + 'ham_lib/' + mol_fname + '_int.bin'
with open(fname, 'wb') as f:
    pickle.dump(H, f)
