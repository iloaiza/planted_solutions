import numpy as np

from openfermion import get_sparse_operator as gso

from utils_tensor import (
    chem_ten2op
)

from utils_AGP import (
    get_geminal_power_function,
    get_AGP
)

from utils_ham import (
    construct_orbital_rotation_operator
)

from utils import (
    get_eigenvalue,
    is_positive_semidefinite,
    is_ground_state
)

def verify_SolutionKey(Htbt, Key):
    # generate Hamiltonian operator from Htbt
    N           = Htbt.shape[0]
    _, Hferm, _ = chem_ten2op(np.zeros([N,N]), Htbt, N)
    Hop         = gso(Hferm, N).toarray()

    # generate |AGP> creation operator and orbital rotation operator from Key
    angles    = Key[0]
    AGPparams = Key[1]

    Omat      = construct_orbital_rotation_operator(angles, N) 
    Gamma     = get_geminal_power_function(AGPparams)

    # verify that H is positive semi-definite
    print(f"\nHamiltonian is positive semi-definite : {is_positive_semidefinite(Hop)}\n\n")

    # verify that for all 0 <= Np <= Norb, |AGP(Np)> is ground state of H with eigenvalue 0
    Norb = len(AGPparams)
    for Np in range(Norb + 1 + 2):
        AGP = get_AGP(Gamma, Np, Norb)
        print(f"{Np} AGP eigenvalue      : {get_eigenvalue(Hop, Omat.T @ AGP)}")
        print(f"{Np} AGP is ground state : {is_ground_state(Hop, Omat.T @ AGP)}\n\n")

    # finish
    print("verification complete\n\n")
    return None