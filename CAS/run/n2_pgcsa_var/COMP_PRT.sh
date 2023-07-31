#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --job-name COMPUTE_N2
# Usage: ./COMPUTE_N2.sh (frag_idx) (total_partition) (frag_matrix_partition)

cd $CSA_DIR/run/n2_pgcsa_var/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

# Paritions n2 matrix 
python partition_matrix.py ${1} ${2} ${3}
seq_iter=$((${2} - 1))

# Do matrix multiplication by parts 
for i in `seq 0 ${seq_iter}`
do 
for j in `seq 0 ${seq_iter}`
do
python matrix_mult.py ${1} ${2} ${i} ${j}
done 
done  
