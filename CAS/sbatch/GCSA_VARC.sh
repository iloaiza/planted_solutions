#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --job-name GCSA_VARC
# Usage: ./GCSA_VARC.sh (mol) (num_csa_fragments)

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python gcsa_variance_compute.py ${1} ${2} 
