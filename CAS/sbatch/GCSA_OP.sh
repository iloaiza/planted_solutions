#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --job-name GCSA_OP
# Usage: ./GCSA_OP.sh (mol) (num_greedy_step) (num_fullcsa_fragments)

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python gcsa_run.py ${1} ${2} ${3}
