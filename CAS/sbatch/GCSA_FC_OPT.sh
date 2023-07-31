#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --job-name GCSA_FC_optimization


cd $CSA_DIR/run/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python gcsa_fc_run.py 
