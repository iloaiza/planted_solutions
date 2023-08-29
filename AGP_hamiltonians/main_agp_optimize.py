"""
do optimization of AGP solvable Hamiltonian given electronic Hamiltonian

CLA: python main_agp_optimize.py (moltag)
"""

import numpy as np

from utils_optimize import (
    load_electronic_hamiltonian_combined_tensor,
    evaluate_cost_function,
    obtain_AGP_hamiltonian_fragment,
    save_optimized_parameters
)

from utils_ham import (
    obtain_AGP_hamiltonian_tensor_rotated
)

import sys

from openfermion import get_sparse_operator as gso

from utils_tensor import chem_ten2op

if __name__ == '__main__':
    moltag    = sys.argv[1]  

    target    = load_electronic_hamiltonian_combined_tensor(moltag)

    op_result = obtain_AGP_hamiltonian_fragment(target)

    save_optimized_parameters(op_result.x, moltag)

    #
    #    process results
    #

    N     = target.shape[0] 
    Norb  = N // 2
    Nterm = max([12, Norb*(Norb-1)//2])

    init_norm = np.sum(target*target)
    fini_norm = evaluate_cost_function(op_result.x, target, Norb, Nterm)

    print(f"""
        initial norm : {init_norm}
        final norm   : {fini_norm} 
    """)

    