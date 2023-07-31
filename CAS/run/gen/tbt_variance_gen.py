"""Precompute the expectation value of two-body tensor and their squares for faster variance computation during VCSA calculations. 
"""
prefix_path = "../../"
import sys
sys.path.append(prefix_path)
import io
import time

import saveload_utils as sl
import openfermion as of
import numpy as np
import var_utils as varu
import ferm_utils as feru
from pathos import multiprocessing as mp

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
wfs_name = 'hf' if len(sys.argv) < 3 else sys.argv[2]
ev_or_sq = 'ev' if len(sys.argv) < 4 else sys.argv[3]
total_partition = 1 if len(sys.argv) < 5 else int(sys.argv[4])
cur_partition = 0 if len(sys.argv) < 6 else int(sys.argv[5])
cur_partition = max(min(total_partition - 1, cur_partition), 0)
save = True

# Setup postfix.
if total_partition > 1:
    post_fix = "_{}-{}".format(cur_partition, total_partition)
else:
    post_fix = ""

# Setup log file
log_string = None
if save:
    # Prepare log file
    log_string = io.StringIO()
    sys.stdout = log_string

# Get Hamiltonian
Hf = sl.load_fermionic_hamiltonian(mol, prefix=prefix_path)
n_qubits = of.count_qubits(Hf)
n_orbitals = n_qubits // 2

# Get WFS
if wfs_name == 'fci':
    wfs = sl.load_ground_state(mol, 'FM', path_prefix=prefix_path, verbose=True)
else:
    nelec, _ = varu.get_system_details(mol)
    wfs = feru.get_openfermion_hf(n_qubits, nelec)

# Parallelized functions


def ev_unit(orb_index, norb, wfs):
    """Compute the expectation value of tbt[a, b, c, d] where orb_index = aN^3 + bN^2 + cN + d
    """
    d = orb_index % norb
    c = orb_index % norb ** 2 // norb
    b = orb_index % norb ** 3 // norb**2
    a = orb_index // norb**3

    cur_tbt = np.zeros((norb, norb, norb, norb))
    cur_tbt[a, b, c, d] = 1
    cur_op = feru.get_ferm_op(cur_tbt)
    return of.expectation(of.get_sparse_operator(cur_op, 2 * norb), wfs)


def ev_parallelized(orb_index): return ev_unit(orb_index, n_orbitals, wfs)


def sq_unit(orb_index, norb, wfs):
    """Compute the expectation value of tbt[i,j,k,l]*tbt[a,b,c,d] where 
    orb_index = iN^7 + jN^6 + kN^5 + lN^4 + aN^3 + bN^2 + cN + d
    """
    d = orb_index % norb
    c = orb_index % norb ** 2 // norb
    b = orb_index % norb ** 3 // norb**2
    a = orb_index % norb ** 4 // norb**3
    l = orb_index % norb ** 5 // norb**4
    k = orb_index % norb ** 6 // norb**5
    j = orb_index % norb ** 7 // norb**6
    i = orb_index // norb**7

    cur_tbt_l = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    cur_tbt_r = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    cur_tbt_l[i, j, k, l], cur_tbt_r[a, b, c, d] = 1, 1
    cur_op = feru.get_ferm_op(cur_tbt_l) * feru.get_ferm_op(cur_tbt_r)
    return of.expectation(of.get_sparse_operator(cur_op, 2 * norb), wfs)


def sq_parallelized(orb_index): return sq_unit(orb_index, n_orbitals, wfs)


tensor_compute_start = time.time()
# Compute Tensor as requested.
if ev_or_sq == 'ev':
    with mp.Pool(mp.cpu_count()) as pool:
        idx_per_calcuation = n_orbitals**4 // total_partition
        start_idx, end_idx = idx_per_calcuation * \
            cur_partition, idx_per_calcuation * (cur_partition + 1)
        if cur_partition == total_partition - 1:
            end_idx = n_orbitals**4
        ev = pool.map(ev_parallelized, range(start_idx, end_idx))
        pool.close()
        pool.join()
    tensor = np.full((n_orbitals, n_orbitals, n_orbitals, n_orbitals),
                     fill_value=np.nan, dtype=np.complex)
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            for k in range(n_orbitals):
                for l in range(n_orbitals):
                    idx = i * n_orbitals**3 + j * n_orbitals**2 + k * n_orbitals + l
                    if idx < end_idx and idx >= start_idx:
                        tensor[i, j, k, l] = ev[idx - start_idx]
else:
    with mp.Pool(mp.cpu_count()) as pool:
        idx_per_calcuation = n_orbitals**8 // total_partition
        start_idx, end_idx = idx_per_calcuation * \
            cur_partition, idx_per_calcuation * (cur_partition + 1)
        if cur_partition == total_partition - 1:
            end_idx = n_orbitals**8
        sq = pool.map(sq_parallelized, range(start_idx, end_idx))
        pool.close()
        pool.join()
    tensor = np.full((n_orbitals, n_orbitals, n_orbitals, n_orbitals,
                       n_orbitals, n_orbitals, n_orbitals, n_orbitals),
                      fill_value=np.nan, dtype=np.complex)
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            for k in range(n_orbitals):
                for l in range(n_orbitals):
                    for a in range(n_orbitals):
                        for b in range(n_orbitals):
                            for c in range(n_orbitals):
                                for d in range(n_orbitals):
                                    # Prepare current tbt
                                    idx = i * n_orbitals**7 + j * n_orbitals**6 + \
                                        k * n_orbitals**5 + l * n_orbitals**4
                                    idx += a * n_orbitals**3 + b * n_orbitals**2 + c * n_orbitals + d
                                    if idx < end_idx and idx >= start_idx:
                                        tensor[i, j, k, l, a,
                                               b, c, d] = sq[idx - start_idx]
tensor_compute_time = time.time() - tensor_compute_start

# Print log file.
print("Molecule: {}".format(mol))
print("Type of wfs: {}".format(wfs_name))
if total_partition > 1:
    print("Total partitioned calculations: {}".format(total_partition))
    print("Current calculation's index: {}".format(cur_partition))
print("Computing expectation of linear term (ev) or squared terms (sq): {}".format(ev_or_sq))
print("Time elapsed to compute the tensor: {} min".format(
    round(tensor_compute_time / 60, 2)))

# Save tensor. Save log.
if save:
    sl.save_tbt_variance(tensor, ev_or_sq, mol, geo=1.0, path_prefix=prefix_path,
                         wfs_type=wfs_name, log_string=log_string, file_post_fix=post_fix)
