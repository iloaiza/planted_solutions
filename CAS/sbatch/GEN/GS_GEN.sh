#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --job-name GS_GEN
# Usage: /GS_GEN.sh (mol) (ham_form)

cd $CSA_DIR/run/gen/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python gs_gen.py ${1} ${2}
