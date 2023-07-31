#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=06:00:00
#SBATCH --job-name FC_VARC
# Usage: ./FC_VARC.sh (mol) (color_alg)

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python fc_variance_compute.py ${1} ${2}
