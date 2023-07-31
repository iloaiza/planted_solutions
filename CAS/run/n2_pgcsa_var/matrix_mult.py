"""Use the partitioned parts to do matrix multiplication.
Perform MxM = Msq. Compute only the Msq[matrix_sq_i, matrix_sq_j] parts and save under
    scratch/n2_fragparts/frag_sq=${frag_idx}/i=${i_frag}$_j=${j_frag}_total=${total_partition}.npz 
"""
path_prefix = "../../"
import sys 
sys.path.append(path_prefix)
from scipy.sparse import save_npz, load_npz, csc_matrix
import os 

# Parameters 
frag_idx = 1 if len(sys.argv) < 2 else int(sys.argv[1])
total_partition = 8 if len(sys.argv) < 3 else int(sys.argv[2])
matrix_sq_i = 0 if len(sys.argv) < 4 else int(sys.argv[3])
matrix_sq_j = 0 if len(sys.argv) < 5 else int(sys.argv[4])

# Setup 
frag_path = path_prefix + "scratch/n2_fragparts/frag={}/".format(frag_idx)
frag_sq_path = path_prefix + "scratch/n2_fragparts/frag_sq={}/".format(frag_idx)
if not os.path.exists(frag_sq_path):
    os.makedirs(frag_sq_path)

# Do matrix multiplication by parts. 
cur_matrix = None 
for k in range(total_partition):
    ifname = "i={}_j={}_total={}.npz".format(matrix_sq_i, k, total_partition)
    jfname = "i={}_j={}_total={}.npz".format(k, matrix_sq_j, total_partition)

    # Get partitioned matrices 
    with open(frag_path+ifname, "rb") as f:
        ik_matrix = load_npz(f)
    with open(frag_path+jfname, "rb") as f:
        kj_matrix = load_npz(f)
    
    if cur_matrix is None:
        cur_matrix = csc_matrix(ik_matrix.shape)
    
    cur_matrix += ik_matrix @ kj_matrix
    print("Step: {}".format(k+1))

# Save 
fname = "i={}_j={}_total={}.npz".format(matrix_sq_i, matrix_sq_j, total_partition)
with open(frag_sq_path+fname, "wb") as f: 
    save_npz(f, cur_matrix)
