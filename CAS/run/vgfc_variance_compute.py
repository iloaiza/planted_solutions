"""Obtain the VGFC fragments from considering commutativity, weights, 
and covariance between new Pauli product and the group it can be added to. 

Usage: python vgfc_variance_compute (mol) (grouping_weight) (sq_or_ln) (w_or_v)
"""
path_prefix = "../"
import sys
sys.path.append(path_prefix)
import io

import qubit_utils as qubu
import saveload_utils as sl
import numpy as np
from var_utils import get_system_details
from openfermion import bravyi_kitaev, variance, get_sparse_operator, count_qubits, expectation

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
grouping_weight = 0.05 if len(sys.argv) < 3 else float(sys.argv[2])
sq_or_ln = 'sq' if len(sys.argv) < 4 else sys.argv[3]
w_or_v = 'w' if len(sys.argv) < 5 else sys.argv[4]
grp_wfs = 'hf' if len(sys.argv) < 6 else sys.argv[5]

use_evdict = True
save = True
check = False

# Fixed parameters
wfs_type = 'fci'
geo = 1.0
method_name = 'VGFC'

# Prepare logging
if save:
    log_string = io.StringIO()
    sys.stdout = log_string

# Display run's details
print("Molecule: {}".format(mol))
print("Wavefunction for variance: {}".format(wfs_type))
print("Geometries: {}".format(geo))
print("Method: {}".format(method_name))
print("Grouping weight: {}".format(grouping_weight))
print("Wavefunction for grouping: {}".format(grp_wfs))
print("Squared or linear weight for grouping choice: {}".format(sq_or_ln))
print("Use pw's weight or variance for comparison. w_or_v: {}".format(w_or_v))
print()

# Get Hamiltonian
h_ferm = sl.load_fermionic_hamiltonian(mol)
h_bk = bravyi_kitaev(h_ferm)
n_qubits = count_qubits(h_bk)
n_electrons, _ = get_system_details(mol)

# Get GS. 
ground_state = sl.load_ground_state(mol, 'BK', path_prefix=path_prefix)
hf_state = qubu.get_openfermion_bk_hf(n_qubits, n_electrons)
if use_evdict: ev_dict = sl.load_pauliword_ev(mol, 'bk', wfs_type=grp_wfs)
else: ev_dict = None 

# Determine grouping
pws = qubu.get_pauliword_list(h_bk, ignore_identity=True)
pws.sort(key=qubu.get_pauli_word_coefficients_size, reverse=True)

frags = []
for pw in pws:
    group_found = False
    for frag in frags:
        commute = True
        for pw_frag in qubu.get_pauliword_list(frag):
            if not qubu.is_commuting(pw, pw_frag):
                commute = False
                break
        if commute:
            # Compare covariance with weights
            pw_weight = qubu.get_pauli_word_coefficients_size(pw)
            pw_var = qubu.get_covariance(pw, pw, ev_dict=ev_dict)

            # Get covariance 
            cov = qubu.get_covariance(frag, pw, ev_dict=ev_dict, wfs=hf_state)

            # Checks 
            if check:
                expected = qubu.get_covariance(frag, pw, ev_dict=None, wfs=hf_state)
                print("Expected: {}".format(expected))
                print("Received: {}".format(cov))
                if not np.isclose(expected, cov): 
                    raise(ValueError("Error found with frag: {}\nAnd pw: {}".format(frag, pw)))

            # Compare against pw's weight or variance 
            if w_or_v == 'w': pw_comp = pw_weight
            else: pw_comp = pw_var

            # Use squared or linear depending on choice
            if sq_or_ln == 'sq': pw_comp = pw_comp**2
            else: pw_comp = np.abs(pw_comp)

            if grouping_weight * cov < pw_comp:
                frag += pw
                group_found = True
                break
    if not group_found:
        frags.append(pw)

# Checks
for frag in frags:
    h_bk -= frag
print("H remains. Should be constant. : {}".format(h_bk))

# Compute variance
frag_vars = np.full(len(frags), np.nan)
for idx, frag in enumerate(frags):
    cur_var = variance(get_sparse_operator(frag, n_qubits), ground_state)
    frag_vars[idx] = np.real_if_close(cur_var, tol=1e6)

print("FCI variances:\n{}".format(frag_vars))
print("Optimal metric: {}".format(np.sum(frag_vars**(1 / 2))**2))

# Save results
if save:
    sl.save_variance_result(mol, wfs_type, method=method_name.lower(), geometry=geo, 
                            groups=frags, variances=frag_vars, path_prefix=path_prefix)
    log_fpath = sl.get_logging_path_and_name(
        mol, geo, wfs_type, method_name.lower(), path_prefix)
    with open(log_fpath, 'w') as f:
        print(log_string.getvalue(), file=f) 
