#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=22:00:00
#SBATCH --job-name VGCSA_OP
# Usage: ./VGCSA_OP.sh (mol) (num_vgsteps) (var_weight) 
#        (l1_norm_tol) (max_num_gsteps) (load T/F) (load_alpha). 

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python vgcsa_run.py ${1} ${2} ${3} ${4} ${5} ${6} ${7}
