import tequila as tq
from tequila.hamiltonian import QubitHamiltonian
from tequila.grouping.fermionic_functions import n_elec
from openfermion import reverse_jordan_wigner, normal_ordered
import numpy as np
from fermigroups import fermgroup 

def prepare_test_hamiltonian(geometry = None):
    '''
    Return a test hamiltonian.
    '''
    trafo = "JordanWigner"

    if geometry == None:
        geometry="H 0.0 0.0 0.0 \n H 0.0 0.0 1. \n H 0.0 0.0 2. \n H 0.0 0.0 3."

    mol = tq.chemistry.Molecule(
                            geometry=geometry,
                            basis_set="sto3g",
                            transformation=trafo,
                            backend='pyscf'
                                 )
    H = mol.make_hamiltonian()
    Hq = H.to_openfermion()
    Hferm = reverse_jordan_wigner(Hq)
    return mol, H, Hferm, len(Hq.terms) - 1

R = 2.35
theta = 107.6
geometry = [('H', (-R*np.sin(np.deg2rad(theta/2)), 0., -R*np.cos(np.deg2rad(theta/2)))),
            ('O', (0., 0., 0.)),
            ('H', (R*np.sin(np.deg2rad(theta/2)), 0., -R*np.cos(np.deg2rad(theta/2))))]

def get_string_geo(geom):
    st = ''
    for atm in geometry:
        st += atm[0]
        for coord in atm[1]:
            st += ' {}'.format(coord)
        st += ' \n '
    return st[:-4]

geometry = get_string_geo(geometry)

mol, H, Hferm, n_paulis = prepare_test_hamiltonian(geometry)
print("Number of Pauli products to measure: {}".format(n_paulis))

FG = fermgroup(Hferm, 'nFG', verbose=True)
print(normal_ordered(sum(FG[0]) - Hferm))