#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --job-name GS_MEMGEN
# Usage: /GS_MEMGEN.sh (mol) (ham_form)

cd $CSA_DIR/run/gen/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python gs_memgen.py ${1} ${2}
