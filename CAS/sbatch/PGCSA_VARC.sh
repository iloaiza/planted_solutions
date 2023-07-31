#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --job-name PGCSA_VARC
# Usage: ./PGCSA_VARC.sh (mol) (num_csa_fragments) (use_precomputed T/F) (start_idx) (n_frag_compute)

cd $CSA_DIR/run/
source ~/.virtualenvs/qchem3.7/bin/activate # Setup environment 

python pgcsa_variance_compute.py ${1} ${2} ${3} ${4} ${5}
