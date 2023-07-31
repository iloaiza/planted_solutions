"""Testing the saved expectation values of pauli-words  
"""
path_prefix = "../../"
import sys 
sys.path.append(path_prefix)

import openfermion as of 
import saveload_utils as sl 
import numpy as np 
import var_utils as varu
import qubit_utils as qubu

# Parameters 
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
ntries = 10 if len(sys.argv) < 3 else int(sys.argv[2])
wfs_type = 'fci' if len(sys.argv) < 4 else sys.argv[3]
tf = 'bk'

# Loading PW ev 
pw_ev = sl.load_pauliword_ev(mol, tf, wfs_type, prefix=path_prefix)

# Loading wfs 
if wfs_type == 'fci':
    wfs = sl.load_ground_state(mol, tf.upper(), path_prefix=path_prefix)
    n_qubits = int(np.log2(wfs.shape))
else:
    Hf = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
    n_qubits = of.count_qubits(Hf)
    nelec, _ = varu.get_system_details(mol)
    wfs = qubu.get_openfermion_bk_hf(n_qubits, nelec)

# Get keys 
pw_words = np.array(list(pw_ev), dtype=tuple)

# For ntries. Test 
for i in range(ntries):
    pw_word = np.random.choice(pw_words)
    pw = of.QubitOperator(pw_word)

    received = pw_ev[pw_word]
    expected = of.expectation(of.get_sparse_operator(pw, n_qubits), wfs)

    if not np.isclose(received, expected): 
        print("Incorrect ev found for pauli-word: {}".format(pw))
        print("Received: {}".format(received))
        print("Expected: {}".format(expected))
        quit() 

print("Test passed")