"""Precompute the expectation value of two-body tensor and their squares for faster variance computation during VCSA calculations. 

 Usage: ./tbt_variance_parts_gen.sh (mol) (total_partition) (cur_partition) 
                            (part_total_partition) (part_cur_partition)
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
mol = 'n2' if len(sys.argv) < 2 else sys.argv[1]
total_partition = 2 if len(sys.argv) < 3 else int(sys.argv[2])
cur_partition = 0 if len(sys.argv) < 4 else int(sys.argv[3])
parts_total_partition = 2 if len(sys.argv) < 5 else int(sys.argv[4])
parts_cur_partition = 0 if len(sys.argv) < 6 else int(sys.argv[5])

cur_partition = max(min(total_partition - 1, cur_partition), 0)
parts_cur_partition = max(min(parts_total_partition - 1, parts_cur_partition), 0)
save = True

# Fixed parameter
ev_or_sq = 'sq'
wfs_name = 'hf'

# Setup postfix.
assert total_partition > 1; assert parts_total_partition > 1; 
post_fix = "_{}-{}_{}-{}".format(cur_partition, total_partition, parts_cur_partition, parts_total_partition)

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

# Get wfs 
nelec, _ = varu.get_system_details(mol)
wfs = feru.get_openfermion_hf(n_qubits, nelec)

# Parallelized functions
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
with mp.Pool(mp.cpu_count()) as pool:
    idx_per_calcuation = n_orbitals**8 // total_partition
    start_idx, end_idx = idx_per_calcuation * \
        cur_partition, idx_per_calcuation * (cur_partition + 1)
    if cur_partition == total_partition - 1: end_idx = n_orbitals**8

    # Further partitioning calculation
    idx_per_part_calculation = idx_per_calcuation // parts_total_partition
    part_start_idx = start_idx + parts_cur_partition * idx_per_part_calculation 
    part_end_idx = start_idx + (parts_cur_partition + 1)*idx_per_part_calculation
    if parts_cur_partition == parts_total_partition - 1: part_end_idx = end_idx

    sq = pool.map(sq_parallelized, range(part_start_idx, part_end_idx))
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
                                if idx < part_end_idx and idx >= part_start_idx:
                                    tensor[i, j, k, l, a,
                                            b, c, d] = sq[idx - part_start_idx]
tensor_compute_time = time.time() - tensor_compute_start

# Print log file.
print("Molecule: {}".format(mol))
print("Type of wfs: {}".format(wfs_name))
if total_partition > 1:
    print("Total partitioned calculations: {}".format(total_partition))
    print("Current calculation's index: {}".format(cur_partition))
    print("Part partition: {}".format(parts_total_partition))
    print("Part current index: {}".format(parts_cur_partition))
print("Computing expectation of linear term (ev) or squared terms (sq): {}".format(ev_or_sq))
print("Time elapsed to compute the tensor: {} min".format(
    round(tensor_compute_time / 60, 2)))

# Save tensor. Save log.
if save:
    sl.save_tbt_variance(tensor, ev_or_sq, mol, geo=1.0, path_prefix=prefix_path,
                         wfs_type=wfs_name, log_string=log_string, file_post_fix=post_fix)
