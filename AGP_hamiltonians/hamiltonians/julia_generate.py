from juliacall import Main as jl
import numpy as np
import pickle

# jl.seval('import Pkg')
# jl.seval('Pkg.add("QuantumMAMBO")')
jl.seval("using QuantumMAMBO")

mambo = jl.QuantumMAMBO
from openfermion import FermionOperator

#EXAMPLE OF HAMILTONIAN GENERATION FOR LCU SYSTEMS
# H,ne = mambo.SAVELOAD_HAM("h2","", False) #accepts h2, lih, beh2, h2o, nh3, and n2
# Hof = mambo.to_OF(H) #Openfermion Hamiltonian object

N_QUBITS = {
    'h2'   : 4,
    'lih'  : 12,
    'beh2' : 14,
    'h2o'  : 14,
    'nh3'  : 16,
    'n2'   : 20
}

if __name__ == '__main__':
    import sys
    moltag  = sys.argv[1] 
    N       = N_QUBITS[moltag]

    Hjl, ne = mambo.SAVELOAD_HAM(moltag, '', False)
    Hferm   = mambo.to_OF(Hjl)

    const   = 0 
    obt     = np.zeros([N,N])
    tbt     = np.zeros([N,N,N,N]) 

    for term, coef in Hferm.terms.items():
        if len(term) == 0:
            const = coef
        elif len(term) == 2:
            obt[
                term[0][0], term[1][0]
            ] = coef
        elif len(term) == 4:
            tbt[
                term[0][0], term[1][0], term[2][0], term[3][0]
            ] = coef
        else:
            print('oh noo :(')

    op_filename  = f'julia/{moltag}/Hferm'
    tbt_filename = f'julia/{moltag}/tensors' 

    with open(op_filename, 'wb') as f:
        pickle.dump(Hferm, f)

    with open(tbt_filename, 'wb') as f:
        pickle.dump([obt, tbt], f)

    print('complete!')