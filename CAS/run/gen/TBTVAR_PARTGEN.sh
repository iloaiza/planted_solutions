#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=16:00:00
#SBATCH --job-name TBTVAR_PARTGEN
# Usage: ./TBTVAR_PARTGEN.sh (mol) (total_partition) (cur_partition) 
#                            (part_total_partition) (part_cur_partition)

cd $CSA_DIR/run/gen/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python tbt_variance_parts_gen.py ${1} ${2} ${3} ${4} ${5}
