"""Testing random elements of the precomputed tbt. 
"""
prefix_path = "../"
import sys
sys.path.append(prefix_path)

import saveload_utils as sl
import var_utils as varu
import ferm_utils as feru
import openfermion as of
import numpy as np 

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
wfs_name = 'hf' if len(sys.argv) < 3 else sys.argv[2]
ntries = 1000 if len(sys.argv) < 4 else int(sys.argv[3])
ev_or_sq = 'sq' 

# Get norb 
Hf = sl.load_fermionic_hamiltonian(mol, prefix=prefix_path)
n_qubits = of.count_qubits(Hf)
norb = n_qubits // 2

# Load correct wavefunction
if wfs_name == 'fci':
    wfs = sl.load_ground_state(
        mol, 'FM', path_prefix=prefix_path, verbose=True)
else:
    nelec, _ = varu.get_system_details(mol)
    del Hf
    wfs = feru.get_openfermion_hf(n_qubits, nelec)

# Loading the tbt
tbt = sl.load_tbt_variance(ev_or_sq, mol, geo=1.0,
                           wfs_type=wfs_name, path_prefix=prefix_path)

# Get number of elements
nelem = tbt.size 

# Get random indices
indices = np.random.choice(nelem, size=ntries, replace=False)

# Loop over indices. Get corresponding operator.
for idx in indices:
    d = idx % norb
    c = idx % norb ** 2 // norb
    b = idx % norb ** 3 // norb**2
    a = idx % norb ** 4 // norb**3
    l = idx % norb ** 5 // norb**4
    k = idx % norb ** 6 // norb**5
    j = idx % norb ** 7 // norb**6
    i = idx // norb**7
    cur_tbt_l = np.zeros((norb, norb, norb, norb))
    cur_tbt_r = np.zeros((norb, norb, norb, norb))
    cur_tbt_l[i, j, k, l], cur_tbt_r[a, b, c, d] = 1, 1
    cur_op = feru.get_ferm_op(cur_tbt_l) * feru.get_ferm_op(cur_tbt_r)

    # Check against value
    received = tbt[i, j, k, l, a, b, c, d]
    expected = of.expectation(of.get_sparse_operator(cur_op, n_qubits), wfs)

    assert received == expected
