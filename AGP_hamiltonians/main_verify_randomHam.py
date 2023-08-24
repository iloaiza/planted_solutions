"""
load randomly generated positive semi-definite AGP solvable Hamiltonian and verify that
information from solution key correctly generates the ground states

CLA: python main_verify_randomHam.py (filetag)
"""

from utils_randomHam import (
    load_RandomHam_and_SolutionKey
)

from utils_verifier import (
    verify_SolutionKey
)

import sys

if __name__ == '__main__':
    filetag = sys.argv[1]

    Htbt, Key = load_RandomHam_and_SolutionKey(filetag)

    verify_SolutionKey(Htbt, Key)