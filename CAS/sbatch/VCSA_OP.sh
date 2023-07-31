#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name VCSA_OP
# Usage: ./VCSA_OP.sh (mol) (var_weight) (alpha) (load) (maxiter). 

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python vcsa_run.py ${1} ${2} ${3} ${4} ${5}
