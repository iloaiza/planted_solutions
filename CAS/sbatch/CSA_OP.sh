#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --job-name CSA_optimization
# Usage: python csa_run.py (mol) (alpha) (max_iter) (load T/F)

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python csa_run.py ${1} ${2} ${3} ${4}
