"""
generate random positive semi-definite Hamiltonian for which |AGP> are ground states

CLA: python main_randomHam.py (Norb) (Nterm) (filetag)
"""

from utils_randomHam import (
    generate_RandomHam_and_SolutionKey,
    save_RandomHam_and_SolutionKey
)

import sys

if __name__ == '__main__':
    Norb    = int(sys.argv[1]) 
    Nterm   = int(sys.argv[2])
    filetag = sys.argv[3]

    Htbt, Key = generate_RandomHam_and_SolutionKey(Norb, Nterm)

    save_RandomHam_and_SolutionKey(Htbt, Key, filetag)