"""Combine the precomputed expectation value of two-body tensor and their squares. 

Only used for parts. 
"""
prefix_path = "../../"
import sys
sys.path.append(prefix_path)

import numpy as np
import saveload_utils as sl
import gc 

# Parameters
mol = 'n2' if len(sys.argv) < 2 else sys.argv[1]
total_partition = 2 if len(sys.argv) < 3 else int(sys.argv[2])
cur_partition = 0 if len(sys.argv) < 4 else int(sys.argv[3])
parts_total_partition = 2 if len(sys.argv) < 5 else int(sys.argv[4])
save = True
check = True

# Fixed parameter
ev_or_sq = 'sq'
wfs_name = 'hf'

# Load first 
postfix = "_{}-{}_{}-{}".format(cur_partition, total_partition, 0, parts_total_partition)
combined_tensor = sl.load_tbt_variance(ev_or_sq, mol, geo=1.0, wfs_type=wfs_name,
                                path_prefix=prefix_path, file_post_fix=postfix, verbose=False)

# Loading the tensors
for i in range(1, parts_total_partition):
    postfix = "_{}-{}_{}-{}".format(cur_partition, total_partition, i, parts_total_partition)
    tensor = sl.load_tbt_variance(ev_or_sq, mol, geo=1.0, wfs_type=wfs_name,
                                  path_prefix=prefix_path, file_post_fix=postfix, verbose=False)
    non_nan_indices = np.where(~np.isnan(tensor))
    for a, b, c, d, e, f, g, h in zip(*non_nan_indices):
        assert np.isnan(combined_tensor[a, b, c, d, e, f, g, h])
        combined_tensor[a, b, c, d, e, f, g, h] = tensor[a, b, c, d, e, f, g, h]
    del tensor; gc.collect()


# Check against existing one if required
if check:
    check_tsr = sl.load_tbt_variance(tensor_type=ev_or_sq, mol=mol, geo=1.0, wfs_type=wfs_name, path_prefix=prefix_path)
    n_orbitals = check_tsr.shape[0]

    # Find start & end idx 
    idx_per_calcuation = n_orbitals**8 // total_partition
    start_idx, end_idx = idx_per_calcuation * \
        cur_partition, idx_per_calcuation * (cur_partition + 1)
    if cur_partition == total_partition - 1: end_idx = n_orbitals**8
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            for k in range(n_orbitals):
                for l in range(n_orbitals):
                    for a in range(n_orbitals):
                        for b in range(n_orbitals):
                            for c in range(n_orbitals):
                                for d in range(n_orbitals):
                                    idx = i * n_orbitals**7 + j * n_orbitals**6 + \
                                        k * n_orbitals**5 + l * n_orbitals**4
                                    idx += a * n_orbitals**3 + b * n_orbitals**2 + c * n_orbitals + d
                                    if idx < end_idx and idx >= start_idx:
                                        saved_value = check_tsr[i, j, k, l, a, b, c, d]
                                        comp_value = combined_tensor[i, j, k, l, a, b, c, d]
                                        assert(np.isclose(saved_value, comp_value))

# Save if required
if save:
    postfix = "_{}-{}-all".format(cur_partition, total_partition)
    sl.save_tbt_variance(combined_tensor, tensor_type=ev_or_sq, mol=mol, geo=1.0, wfs_type=wfs_name, path_prefix=prefix_path, file_post_fix=postfix, verbose=True)
