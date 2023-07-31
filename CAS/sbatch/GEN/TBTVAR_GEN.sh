#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name TBTVAR_GEN
# Usage: ./TBTVAR_GEN.sh (mol) (wfs_type) (ev_or_sq) (total_partition) (cur_partition)

cd $CSA_DIR/run/gen/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python tbt_variance_gen.py ${1} ${2} ${3} ${4} ${5}
