### Deprecated 

import sys
sys.path.append('../')
import pickle
import numpy as np
import time 
from ferm_utils import get_spin_orbitals, get_fermionic_matrix
from openfermion import jordan_wigner, QubitOperator
from qubit_utils import get_qubit_matrix
from var_utils import get_system_details
import scipy as sp

### Parameters
mol = 'h2'
nelec, _ = get_system_details(mol) # only needed if run fermionic 
tiny = 1e-6
ferm = False # whether do fci from fermionic or JW
test = False

### Preps
with open('../ham_lib/'+mol+'_fer.bin', 'rb') as f:
    Hf = pickle.load(f)
n = get_spin_orbitals(Hf)

logfname = mol+'_fci.log'
solfname = mol+'_fci.bin'
sys.stdout = open(logfname, 'w')

### Beginning file description
print("System: {}".format(mol))
print("Number of electrons: {}".format(nelec))
print("Number of spin-orbitals: {}".format(n))
print("Method. Is Fermionic? (JW otherwise) : {}".format(ferm))

### Testing
if test:
    jw = jordan_wigner(Hf)
    npauli = len(jw.terms.items())
    tjw = QubitOperator.zero()
    count, total = 0, 2
    for pw, val in jw.terms.items():
        count += 1 
        tjw += QubitOperator(term=pw, coefficient=val)
        if count == total:
            break
    
    start = time.time()
    tmat = get_qubit_matrix(tjw, n)
    tTime = time.time() - start
    print("Time elapsed to get representation: {}".format(time.time() - start))
    print("Projected time for fci: {}".format(tTime * npauli / total))
    quit()
    v, w = sp.linalg.eigh(tmat)
    tTime = time.time() - start
    print("Time elapsed: {}".format(tTime))
    print("Pauli-words computed: {}".format(total))
    print("Pauli-words needed: {}".format(npauli))
    print("Projected time for fci: {}".format(tTime * npauli / total))
    quit()

### Runs 
start = time.time()
if ferm:
    ### Fermionic gs 
    fermat = get_fermionic_matrix(Hf, n, nelec)
    v, w = sp.linalg.eigh(fermat)
else:
    ### JW gs
    jwmat = get_qubit_matrix(jordan_wigner(Hf), n)
    v, w = sp.linalg.eigh(jwmat)

gs = w[:, 0]
print("\nFCI done. Time elapsed: {}".format(time.time() - start))
print("Ground state energy: {}".format(v[0]))
print("Tiny threshold for HF config to drop: {}".format(tiny))
print("Number of non-zero hfs: {}".format(np.sum(abs(gs) > tiny)))

with open(solfname, 'wb') as f:
    pickle.dump(gs, f)
    