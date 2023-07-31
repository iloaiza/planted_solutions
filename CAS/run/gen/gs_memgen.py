"""
Generate Ground State of Hamiltonians and store 

Usage: python gs_memgen (mol) (ham_form)
"""
path_prefix = "../../"
import sys
sys.path.append(path_prefix) 

import openfermion as of 
import saveload_utils as sl 
import numpy as np 
import scipy as sp 
import time 
import io 

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
ham_form = 'FM' if len(sys.argv) < 3 else sys.argv[2].upper()

# Setting up log file
log_string = io.StringIO()
sys.stdout = log_string
print("Calculations begins. Date: {}".format(sl.get_current_time()))

# Get Hamiltonain 
H = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
if ham_form == 'BK':
    H = of.bravyi_kitaev(H)

# Get sparse 
# H_sparse = of.get_sparse_operator(H)
n_qubits = of.count_qubits(H)
H_sparse = sp.sparse.csr_matrix((2**n_qubits, 2**n_qubits), dtype=np.complex)
print("Total number of terms: {}".format(len(H.terms)))
idx = 1 
for term, val in H.terms.items():
    curterm = of.get_sparse_operator(of.QubitOperator(term=term, coefficient=val), n_qubits=n_qubits)
    H_sparse += curterm
    idx += 1 
    if idx % 50 == 0:
        print("{}th term done".format(idx))

# Get WFS 
gs_start = time.time()
gs_e, gs = of.get_ground_state(H_sparse) 
gs_time = time.time() - gs_start
print("Ground state obtained. Time elapsed: {} seconds".format(round(gs_time)))
print("Ground state energy: {}".format(gs_e))

# Saving 
sl.save_ground_state(gs, mol, ham_form, path_prefix=path_prefix, log_str_io=log_string)
