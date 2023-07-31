#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --job-name HAM_CHECK

cd $CSA_DIR/ham_lib/
source ~/.virtualenvs/qchem/bin/activate # Setup environment 

python ham_read.py $ham > $ham.txt 