"""Combine the precomputed expectation value of two-body tensor and their squares. 
"""
prefix_path = "../../"
import sys
sys.path.append(prefix_path)

import numpy as np
import saveload_utils as sl
import gc 

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
wfs_name = 'hf' if len(sys.argv) < 3 else sys.argv[2]
ev_or_sq = 'ev' if len(sys.argv) < 4 else sys.argv[3]
total_partition = 2 if len(sys.argv) < 5 else int(sys.argv[4])
save = True
check = False

# Load first 
postfix = "_{}-{}".format(0, total_partition)
combined_tensor = sl.load_tbt_variance(ev_or_sq, mol, geo=1.0, wfs_type=wfs_name,
                                path_prefix=prefix_path, file_post_fix=postfix, verbose=False)
np.nan_to_num(combined_tensor, copy=False, nan=0.0)

# Loading the tensors
for i in range(1, total_partition):
    postfix = "_{}-{}".format(i, total_partition)
    tensor = sl.load_tbt_variance(ev_or_sq, mol, geo=1.0, wfs_type=wfs_name,
                                  path_prefix=prefix_path, file_post_fix=postfix, verbose=False)
    np.nan_to_num(tensor, copy=False, nan=0.0)
    combined_tensor += tensor 
    del tensor; gc.collect()


# Loading the tensors
#tensors = []
#for i in range(total_partition):
#    postfix = "_{}-{}".format(i, total_partition)
#    tensor = sl.load_tbt_variance(ev_or_sq, mol, geo=1.0, wfs_type=wfs_name,
#                                  path_prefix=prefix_path, file_post_fix=postfix, verbose=False)
#    tensors.append(tensor)
## Get n_orbitals
#n_orbitals = tensors[0].shape[0]
## Build the tensor
#if ev_or_sq == 'ev':
#    combined_tensor = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals),
#                               dtype=np.complex_)
#    for i in range(n_orbitals):
#        for j in range(n_orbitals):
#            for k in range(n_orbitals):
#                for l in range(n_orbitals):
#                    vals = np.zeros(len(tensors), dtype=np.complex_)
#                    for t in range(len(tensors)):
#                        vals[t] = tensors[t][i, j, k, l]
#                    # Assert only one is not nan.
#                    assert(np.sum(np.isnan(vals) == False) == 1)
#                    combined_tensor[i, j, k, l] = vals[np.isnan(vals) == False]
#else:
#    combined_tensor = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals,
#                                n_orbitals, n_orbitals, n_orbitals, n_orbitals),
#                               dtype=np.complex_)
#    for i in range(n_orbitals):
#        for j in range(n_orbitals):
#            for k in range(n_orbitals):
#                for l in range(n_orbitals):
#                    for a in range(n_orbitals):
#                        for b in range(n_orbitals):
#                            for c in range(n_orbitals):
#                                for d in range(n_orbitals):
#                                    vals = np.zeros(len(tensors), dtype=np.complex_)
#                                    for t in range(len(tensors)):
#                                        vals[t] = tensors[t][i, j, k, l,
#                                                             a, b, c, d]
#                                    # Assert only one is not nan.
#                                    assert(np.sum(np.isnan(vals) == False) == 1)
#                                    combined_tensor[i, j, k, l, a, b, c, d] = \
#                                        vals[np.isnan(vals) == False]
#
# Save if required
if save:
    postfix = "_all-{}".format(total_partition)
    sl.save_tbt_variance(combined_tensor, tensor_type=ev_or_sq, mol=mol, geo=1.0, wfs_type=wfs_name, path_prefix=prefix_path, file_post_fix=postfix, verbose=True)

# Check against existing one if required
if check:
    check_tsr = sl.load_tbt_variance(tensor_type=ev_or_sq, mol=mol, geo=1.0, wfs_type=wfs_name, path_prefix=prefix_path)
    assert(np.isclose(check_tsr, combined_tensor).all())
