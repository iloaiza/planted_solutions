"""Put back the squared matrix. Compute variance. 
"""
path_prefix = "../../"
import sys
sys.path.append(path_prefix)

import saveload_utils as sl
from scipy.sparse import load_npz, csc_matrix, hstack, vstack
from openfermion import expectation
from numpy import conj
import os 
import gc

# Parameters
frag_idx = 1 if len(sys.argv) < 2 else int(sys.argv[1])
total_partition = 8 if len(sys.argv) < 3 else int(sys.argv[2])
save = True 

# Setup
frag_path = path_prefix + "scratch/n2_fragparts/frag={}/".format(frag_idx)
frag_sq_path = path_prefix + \
    "scratch/n2_fragparts/frag_sq={}/".format(frag_idx)


def collect_matrix_expectation(path, wfs):
    part_dim = wfs.shape[0] // total_partition
    ev = 0
    for i in range(total_partition):
        for j in range(total_partition):
            cur_fname = "i={}_j={}_total={}.npz".format(i, j, total_partition)
            with open(path + cur_fname, "rb") as f:
                cur_matrix = load_npz(f)

            ev += conj(wfs[i * part_dim:(i + 1) * part_dim].T) @ \
                cur_matrix @ wfs[j * part_dim:(j + 1) * part_dim]
            del cur_matrix
            gc.collect() 
    return ev

# Get wfs
psi = sl.load_ground_state('n2', 'FM', path_prefix=path_prefix, verbose=True)

# Get original sparse matrix
ev = collect_matrix_expectation(frag_path, psi)
print("Frag ev collected ")

# Get squared sparse matrix
sq = collect_matrix_expectation(frag_sq_path, psi)
print("Fragsq ev collected ")

if save:
    folder_path = path_prefix+"n2_variances/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    f = open(folder_path+"frag_idx={}.txt".format(frag_idx), 'w')
    sys.stdout = f
    
# Compute variance
print("Molecule: N2")
print("Method: PGCSA")
print("Fragment's index: {}".format(frag_idx))
print("Fragment's total partition ratio: {}".format(total_partition))
print("Variance: {}".format(sq - ev**2))
