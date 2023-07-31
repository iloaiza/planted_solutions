#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --job-name GQWC_VARC
# Usage: ./GQWC_VARC.sh (mol) 

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python gqwc_variance_compute.py ${1} 
