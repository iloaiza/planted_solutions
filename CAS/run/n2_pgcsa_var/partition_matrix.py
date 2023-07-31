"""Partition the sparse matrix of n2's fragments. Save it under 
    scratch/n2_fragparts/frag=${frag_idx}/i=${i_frag}$_j=${j_frag}_total=${total_partition}.npz 
"""
path_prefix = "../../"
import sys 
sys.path.append(path_prefix)

from openfermion import get_sparse_operator, FermionOperator
from scipy.sparse import save_npz, load_npz
import saveload_utils as sl
import os 
import gc 

# Paramters 
frag_idx = 1 if len(sys.argv) < 2 else int(sys.argv[1]) # The index of CSA frag to compute 
total_partition = 8 if len(sys.argv) < 3 else int(sys.argv[2]) # The fraction ratio of frag matrix
# The fraction ratio of terms in frag calculated at once 
frag_partition = 1 if len(sys.argv) < 4 else int(sys.argv[3]) 

# Get fragment 
pgcsa_sol = sl.load_csa_sols('n2', geo=1.0, method='PGCSA', alpha=60, prefix=path_prefix)
pgcsa_fragments = pgcsa_sol['grouping']
frag = pgcsa_fragments[frag_idx]
del pgcsa_sol; del pgcsa_fragments
gc.collect() 

# Define fname 
folder_path = path_prefix + "scratch/n2_fragparts/frag={}/".format(frag_idx)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if frag_partition < 2:
    # Get sparse 
    frag_sparse = get_sparse_operator(frag)
    part_dim = frag_sparse.shape[0] / total_partition
    assert float.is_integer(part_dim)
    part_dim = int(part_dim)

    for i in range(total_partition):
        for j in range(total_partition):
            fname = "i={}_j={}_total={}.npz".format(i, j, total_partition)
            sparse_save = frag_sparse[i*part_dim:(i+1)*part_dim, j*part_dim:(j+1)*part_dim]
            with open(folder_path+fname, "wb") as f:
                save_npz(f, sparse_save)
            print("Done. i: {}. j: {}. Frag partition idx: {}".format(i, j, frag_idx))
else:
    # Easier data structure to handle 
    frag_dict = frag.terms
    frag_tup = list(frag_dict)
    del frag; gc.collect()

    # Determine number of terms to compute. 
    frag_nterm = len(frag_tup)
    frag_nterm_periter = frag_nterm // frag_partition
    for frag_idx in range(frag_partition):
        # Collect operations in current frag 
        cur_start = frag_idx * frag_nterm_periter
        cur_end = cur_start + frag_nterm_periter
        if frag_idx == frag_partition-1: cur_end = frag_nterm

        # Build current frag 
        cur_frag = FermionOperator.zero()
        for op_idx in range(cur_start, cur_end):
            op_tuple = frag_tup[op_idx]
            cur_frag += FermionOperator(term=op_tuple, coefficient=frag_dict[op_tuple])

        # Get Fermionic Sparse Operator 
        cur_frag_sparse = get_sparse_operator(cur_frag)
        part_dim = cur_frag_sparse.shape[0] / total_partition
        assert float.is_integer(part_dim)
        part_dim = int(part_dim)
            
        # Load and sum if frag_idx > 0. Then save. 
        for i in range(total_partition):
            for j in range(total_partition):
                fname = "i={}_j={}_total={}.npz".format(i, j, total_partition)
                sparse_save = cur_frag_sparse[i*part_dim:(i+1)*part_dim, j*part_dim:(j+1)*part_dim]
                if frag_idx > 0:
                    with open(folder_path+fname, "rb") as f: 
                        sparse_save += load_npz(f)
                with open(folder_path+fname, "wb") as f:
                    save_npz(f, sparse_save)
                print("Done. i: {}. j: {}. Frag partition idx: {}".format(i, j, frag_idx))
