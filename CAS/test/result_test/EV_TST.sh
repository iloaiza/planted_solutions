#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=06:00:00
#SBATCH --job-name EV_TST
# Usage: python ev_test.py (mol) (method)

cd $CSA_DIR/test/result_test/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python ev_test.py ${1} ${2} 
