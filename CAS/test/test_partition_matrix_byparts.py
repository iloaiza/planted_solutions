"""Testing the memory efficient implementation of partiioning matrix by parts 
"""
import sys 
from scipy.sparse import load_npz
import numpy as np 

# Parameters 
frag_idx = 1
total_partition = 8 

# Folder name 
expected_folder = "../scratch/n2_fragparts/frag={}_backup/".format(frag_idx)
received_folder = "../scratch/n2_fragparts/frag={}/".format(frag_idx)

# Check each matrix 
for i in range(total_partition):
    for j in range(total_partition):
        file_name = "i={}_j={}_total={}.npz".format(i, j, total_partition)
        with open(expected_folder+file_name, "rb") as f:
            expected_mat = load_npz(f)
        with open(received_folder+file_name, "rb") as f:
            received_mat = load_npz(f)
        max_diff = np.max(np.abs(expected_mat - received_mat))
        print("i: {}. j:{}. Max diff: {}. Passed: {}".format(i, j, max_diff, max_diff < 1e-8))
