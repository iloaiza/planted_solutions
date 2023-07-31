"""Compute all of the expectation between of pauli-words and products of commuting pauli-words in specified molecule.
Save the results in a dictionary Dict[tuple, float] that accepts a tuple of the two pauli-words and
returns their covariance.
"""
path_prefix = "../../"
import sys
sys.path.append(path_prefix)

import saveload_utils as sl
import openfermion as of
import numpy as np
import qubit_utils as qubu
import var_utils as varu
import ferm_utils as feru
import time
import io
import copy
from pathos import multiprocessing as mp
from itertools import combinations_with_replacement

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
tf = 'bk' if len(sys.argv) < 3 else sys.argv[2]
wfs_type = 'fci' if len(sys.argv) < 4 else sys.argv[3]
geo = 1.0
save = True

# Prepare log file
if save:
    log_string = io.StringIO()
    sys.stdout = log_string
else:
    log_string = None

# Get Hamiltonain as a list of pauli-words
Hf = sl.load_fermionic_hamiltonian(mol, prefix=path_prefix)
Hq = of.bravyi_kitaev(Hf) if tf == 'bk' else of.jordan_wigner(Hf)
n_qubits = of.count_qubits(Hq)
pws = qubu.get_pauliword_list(Hq)

# Get wfs
if wfs_type == 'fci':
    wfs = sl.load_ground_state(mol, tf.upper(), path_prefix=path_prefix)
else:
    nelec, _ = varu.get_system_details(mol)
    wfs = qubu.get_openfermion_bk_hf(n_qubits, nelec)


def get_expectation_value(pw):
    pw = qubu.get_pauli_word(pw)
    return np.real_if_close(of.expectation(of.get_sparse_operator(pw, n_qubits), wfs)).item()

def get_prod_ev(ipw, jpw):
    ipw, jpw = qubu.get_pauli_word(ipw), qubu.get_pauli_word(jpw)
    if of.commutator(ipw, jpw) != of.QubitOperator.zero():
        return None
    else:
        return get_expectation_value(ipw * jpw)


# Build expectation values individual pauli words
ev_start = time.time()
ev_dict = {}
with mp.Pool(mp.cpu_count()) as pool:
    evs = pool.map(get_expectation_value, pws)
    pool.close()
    pool.join()
for idx, pw in enumerate(pws):
    ev_dict[qubu.get_pauli_word_tuple(pw)] = evs[idx]

# Build expectation values for commuting products.
pw_pairs = combinations_with_replacement(pws, 2)

with mp.Pool(mp.cpu_count()) as pool:
    evs = pool.starmap(get_prod_ev, copy.copy(pw_pairs))
    pool.close()
    pool.join()
for idx, pw_pair in enumerate(pw_pairs):
    if evs[idx] != None:  # If the pauli-words are commuting.
        prod_pw = qubu.get_pauli_word(pw_pair[0]) * qubu.get_pauli_word(pw_pair[1])
        ev_dict[qubu.get_pauli_word_tuple(prod_pw)] = evs[idx]
ev_time = time.time() - ev_start

# Print information
print("Molecule: {}".format(mol))
print("Molecular geometry: {}".format(geo))
print("Type of transformation: {}".format(tf))
print("Type of wfs: {}".format(wfs_type))
print("Time elapsed to compute the expectation values: {} mins".format(
    round(ev_time / 60)))
print("Number of PWs stored: {}".format(len(ev_dict)))

# Save the dictionary under scratch/
sl.save_pauliword_ev(ev_dict, mol, tf, wfs_type,
                     geo=geo, prefix=path_prefix, log_string=log_string)
