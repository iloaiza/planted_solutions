#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --job-name COMPUTE_N2_VAR
# Usage: ./COMPUTE_N2_VAR.sh (frag_idx) (total_partition)

cd $CSA_DIR/run/n2_pgcsa_var/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python compute_variance.py ${1} ${2}
