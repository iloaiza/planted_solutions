"""This file times the amount of time to compute variance using openfermion's functions on two-body operators. 
It also checks the expectation value <H> to see if wfs or Hamiltonian is constructed correctly. 

Should be used as: python time_variance_comp.py (mol) (wfs). 
"""
import sys
path_prefix = "../../"
sys.path.append(path_prefix)

import time
import openfermion as of
import numpy as np
import ferm_utils as feru
import var_utils as varu
import saveload_utils as sl

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
wfs_name = 'fci' if len(sys.argv) < 3 else sys.argv[2]
n_single_term = 10  # number of single term to average over computing time
parallel_speedup = 30

# Load the specified hamiltonian
Hf = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
n_qubits = of.count_qubits(Hf)

# Get wfs. Time it.
wfs_start = time.time()
if wfs_name == 'fci':
    _, wfs = of.get_ground_state(of.get_sparse_operator(Hf))
else:
    nelec, _ = varu.get_system_details(mol)
    wfs = feru.get_openfermion_hf(n_qubits, nelec)
wfs_time = time.time() - wfs_start

# Generate random tbt.
n_orbitals = n_qubits // 2
random_tbt = np.random.rand(n_orbitals, n_orbitals, n_orbitals, n_orbitals)

# Get operator version of tbt. Time it.
get_op_start = time.time()
random_op = feru.get_ferm_op(random_tbt)
get_op_time = time.time() - get_op_start

# Get variance of the tbt. Time it.
get_var_start = time.time()
variance = of.variance(of.get_sparse_operator(random_op, 2 * n_orbitals), wfs)
get_var_time = time.time() - get_var_start

get_single_var_start = time.time()
for i in range(n_single_term):
    # Generate single term tbt
    single_term_tbt = np.zeros(
        (n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    single_term_tbt[np.random.choice(n_orbitals), np.random.choice(n_orbitals),
                    np.random.choice(n_orbitals), np.random.choice(n_orbitals)] = 1
    single_term_op = feru.get_ferm_op(single_term_tbt)

    # Get variance of the single term tbt.
    variance = of.variance(of.get_sparse_operator(
        single_term_op, 2 * n_orbitals), wfs)
get_single_var_time = time.time() - get_single_var_start
get_single_var_time = get_single_var_time / n_single_term

# Display everything
print("Molecule: {}".format(mol))
print("Wavefunction: {}".format(wfs_name))
print("<H>: {}".format(of.expectation(of.get_sparse_operator(Hf), wfs)))
print("Time taken to obtain wfs: {} min".format(round(wfs_time / 60, 2)))
print("Time taken to obtain random operator from tbt: {} min".format(
    round(get_op_time / 60, 2)))
print("Time taken to obtain variance with random operator and wfs: {} min".format(
    round(get_var_time / 60, 2)))
print("Time taken to obtain variance with single two body operator: {} min".format(
    round(get_single_var_time / 60, 3)))
sq_time = get_single_var_time * n_orbitals**8 # Show total compute time
print("Expected computing time for sq_ev: {} hrs".format(
    round(sq_time/3600)))
print("Expected computing time with {}x speed up from parallel computing: {} hrs".format(
    parallel_speedup, round(sq_time/parallel_speedup/3600, 2)))
