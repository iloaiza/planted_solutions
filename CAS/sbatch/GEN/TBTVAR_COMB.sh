#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --job-name TBTVAR_COMB
# Usage: ./TBTVAR_GEN.sh (mol) (wfs_type) (ev_or_sq) (total_partition) 
cd $CSA_DIR/run/gen/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python tbt_variance_combine.py ${1} ${2} ${3} ${4} 
