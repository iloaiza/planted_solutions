#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --job-name PWEV_GEN
# Usage: /PWEV_GEN.sh (mol) (bk_or_jw) (wfs_type)

cd $CSA_DIR/run/gen/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python pw_ev_gen.py ${1} ${2} ${3}
