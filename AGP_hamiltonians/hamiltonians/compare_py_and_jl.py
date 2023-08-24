import numpy as np

import pickle

import sys

from openfermion import (
    FermionOperator,
    normal_ordered
)

if __name__ == '__main__':
    moltag = sys.argv[1]

    jl_filename_H   = f'julia/{moltag}/Hferm' 
    jl_filename_ten = f'julia/{moltag}/tensors'

    with open(jl_filename_H, 'rb') as f:
        Hferm_jl = pickle.load(f)

    with open(jl_filename_ten, 'rb') as f:
        obt_jl, tbt_jl = pickle.load(f)

    py_filename_H   = f'{moltag}/Hferm' 
    py_filename_ten = f'{moltag}/chem_tensors'

    with open(py_filename_H, 'rb') as f:
        Hferm_py = pickle.load(f)

    with open(py_filename_ten, 'rb') as f:
        obt_py, tbt_py = pickle.load(f)

    print(normal_ordered(Hferm_jl - Hferm_py) == FermionOperator().zero())
    print(np.allclose(obt_jl, obt_py))
    print(np.allclose(tbt_jl, tbt_py))