#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=06:00:00
#SBATCH --job-name GFC_VARC
# Usage: ./GFC_VARC.sh (mol) 

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python gfc_variance_compute.py ${1} 
