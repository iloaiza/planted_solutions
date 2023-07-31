#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name PGCSA_OP
# Usage: ./PGCSA_OP.sh (mol) (num_greedy_steps) (l1_norm_tol) 
#    (compute_hf T/F) (load T/F) (load_alpha).

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python pgcsa_run.py ${1} ${2} ${3} ${4} ${5} ${6} 
