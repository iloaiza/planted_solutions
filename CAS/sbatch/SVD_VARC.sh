#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --job-name SVD_VARC
# Usage: ./SVD_VARC.sh (mol)

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python svd_variance_compute.py ${1}
