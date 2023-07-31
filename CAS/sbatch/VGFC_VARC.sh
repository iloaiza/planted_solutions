#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=06:00:00
#SBATCH --job-name VGFC_VARC
# Usage: ./VGFC_VARC.sh (mol) (grouping_weight) (sq_or_ln) (w_or_v) (hf_or_fci)

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python vgfc_variance_compute.py ${1} ${2} ${3} ${4} ${5}
