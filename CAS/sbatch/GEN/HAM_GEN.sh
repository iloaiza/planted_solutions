#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=06:00:00
#SBATCH --job-name HAM_GEN

cd $CSA_DIR/run/gen/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python ham_gen.py