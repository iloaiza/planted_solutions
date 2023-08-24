import numpy as np

from openfermion import (
    MolecularData,
    FermionOperator,
    get_fermion_operator,
    normal_ordered,
    get_chemist_two_body_coefficients
)

from openfermionpyscf import run_pyscf

from math import (
    radians,
    sin,
    cos
)

rH2O     = 1.0 
thetaH2O = radians(107.6/2)
xH2O     = rH2O * sin(thetaH2O)
yH2O     = rH2O * cos(thetaH2O)

thetaNH3 = radians(107) 
cosNH3   = cos(thetaNH3) 
sinNH3   = sin(thetaNH3)
oneNH3   = (cosNH3 - cosNH3**2) / sinNH3
twoNH3   = np.sqrt(1 - cosNH3**2 - oneNH3**2)

GEOMETRIES = {
    'h2'   : [ ['H', [0,0,0]], ['H' , [0,0,1]] ],
    'h4'   : [ ['H', [0,0,0]], ['H' , [0,0,1]], ['H', [0,0,2]], ['H', [0,0,3]] ],
    'lih'  : [ ['H', [0,0,0]], ['Li', [0,0,1]] ],
    'beh2' : [ ['H', [0,0,0]], ['Be', [0,0,1]], ['H', [0,0,2]] ],
    'h2o'  : [ ['O', [0,0,0]], ['H' , [-xH2O,yH2O,0]], ['H', [xH2O,yH2O,0]] ],
    'nh3'  : [ ['H', [0,0,1]], ['H', [0,sinNH3,cosNH3]], ['H', [twoNH3,oneNH3,cosNH3]], ['N', [0,0,0]] ]
}

def obtain_molecular_hamiltonian(moltag):
    geo = GEOMETRIES[moltag]
    mol = MolecularData(geo, 'sto3g', 1, 0)
    mol = run_pyscf(mol, run_scf=True)
    Hmol = mol.get_molecular_hamiltonian()
    return Hmol, get_fermion_operator(Hmol)    

def obtain_phys_representation(Hmol):
    N, const, obt, tbt = Hmol.n_qubits, Hmol.constant, Hmol.one_body_tensor, Hmol.two_body_tensor

    H1 = FermionOperator()
    H2 = FermionOperator()
    for p in range(N):
        for q in range(N):
            term = ((p,1), (q,0))
            coef = obt[p,q]
            H1  += FermionOperator(term, coef)
            for r in range(N):
                for s in range(N):
                    term = ((p,1), (q,1), (r,0), (s,0))
                    coef = tbt[p,q,r,s]
                    H2  += FermionOperator(term, coef)

    return const, obt, tbt, H1, H2

def convert_phys_to_chem(obt_phys, tbt_phys, H1phys, H2phys):
    N = obt_phys.shape[0]

    tbt_chem = np.transpose(tbt_phys, [0,3,1,2]) 
    H2chem   = FermionOperator()
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    term    = ((p,1), (q,0), (r,1), (s,0))
                    coef    = tbt_chem[p,q,r,s]
                    H2chem += FermionOperator(term, coef)
    
    H1epsilon   = normal_ordered(H2phys - H2chem)
    obt_epsilon = np.zeros([N,N])
    for term, coef in H1epsilon.terms.items():
        if len(term) != 2:
            print('oh nooooo :(')
        else:
            obt_epsilon[
                term[0][0], term[1][0]
            ] = coef
    
    obt_chem = obt_phys + obt_epsilon
    H1chem = FermionOperator()
    for p in range(N):
        for q in range(N):
            term    = ((p,1), (q,0))
            coef    = obt_chem[p,q]
            H1chem += FermionOperator(term, coef)

    return obt_chem, tbt_chem, H1chem, H2chem

def obtain_chemist_tensors_via_openfermion(obt_phys, tbt_phys):
    """
    should return the same obt_chem and tbt_chem as previous function 
    (this exists as a debugger of the above function)
    """
    N                      = tbt_phys.shape[0]
    obt_epsilon, tbt_chemC = get_chemist_two_body_coefficients(tbt_phys, spin_basis=True)

    tbt_chem = np.zeros([N,N,N,N])
    for i in range(N//2):
        for j in range(N//2):
            for k in range(N//2):
                for l in range(N//2):
                    for sigma in [0,1]:
                        for tau in [0,1]:
                            tbt_chem[
                                2*i+sigma, 2*j+sigma, 2*k+tau, 2*l+tau
                            ] = tbt_chemC[i,j,k,l]
    
    return obt_phys + obt_epsilon, tbt_chem

def check_onebody_and_twobody_equivalence(onebody, twobody):
    H1 = FermionOperator()
    H2 = FermionOperator()

    N = onebody.shape[0]
    for p in range(N):
        for q in range(N):
            term = ((p,1), (q,0))
            coef = onebody[p,q]
            H1  += FermionOperator(term, coef)
            for r in range(N):
                for s in range(N):
                    term = ((p,1), (q,0), (r,1), (s,0))
                    coef = twobody[p,q,r,s]
                    H2  += FermionOperator(term, coef)
    assert normal_ordered(H1 - H2) == FermionOperator().zero()
    return None

def obtain_fully_twobody_form(obt_chem, tbt_chem):
    assert np.allclose(obt_chem, obt_chem.T)

    N    = obt_chem.shape[0] 
    D, U = np.linalg.eigh(obt_chem)
    U    = U.T
    
    twobody_form = np.zeros([N,N,N,N])
    for p in range(N):
        twobody_form[p,p,p,p] = D[p]
    twobody_form = np.einsum('pqrs,pa,qb,rc,sd->abcd', twobody_form, U, U, U, U)

    check_onebody_and_twobody_equivalence(obt_chem, twobody_form)

    full_tbt = tbt_chem + twobody_form
    H_full   = FermionOperator()
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    term = ( (p,1), (q,0), (r,1), (s,0) )
                    coef = full_tbt[p,q,r,s]
                    H_full += FermionOperator(term, coef)
    
    return full_tbt, H_full

if __name__ == '__main__':
    import sys
    import pickle
    moltag = sys.argv[1]

    Hmol, Hferm                               = obtain_molecular_hamiltonian(moltag)
    const, obt_phys, tbt_phys, H1phys, H2phys = obtain_phys_representation(Hmol)
    obt_chem, tbt_chem, H1chem, H2chem        = convert_phys_to_chem(obt_phys, tbt_phys, H1phys, H2phys)
    obt_chem_OF, tbt_chem_OF                  = obtain_chemist_tensors_via_openfermion(obt_phys, tbt_phys)
    tbt_full, H_full                          = obtain_fully_twobody_form(obt_chem, tbt_chem) 

    assert normal_ordered(Hferm - (const+H1phys+H2phys)) == FermionOperator().zero()
    assert normal_ordered(Hferm - (const+H1chem+H2chem)) == FermionOperator().zero()
    assert normal_ordered((H1phys+H2phys) - (H1chem+H2chem)) == FermionOperator().zero()
    # assert normal_ordered(Hferm - (const+H_full)) == FermionOperator().zero()
    # assert normal_ordered((H1phys+H2phys) - H_full) == FermionOperator().zero()
    # assert normal_ordered((H1chem+H2chem) - H_full) == FermionOperator().zero()
    
    assert np.allclose(obt_chem, obt_chem_OF)
    assert np.allclose(tbt_chem, tbt_chem_OF)

    op_filename   = f'{moltag}/Hferm'
    phys_filename = f'{moltag}/phys_tensors'
    chem_filename = f'{moltag}/chem_tensors'
    full_filename = f'{moltag}/full_tensor'

    with open(op_filename, 'wb') as f:
        pickle.dump(Hferm, f)

    with open(phys_filename, 'wb') as f:
        pickle.dump([obt_phys, tbt_phys], f)

    with open(chem_filename, 'wb') as f:
        pickle.dump([obt_chem, tbt_chem], f)

    with open(full_filename, 'wb') as f:
        pickle.dump(tbt_full, f)

    print('complete!')
