import numpy as np

from numpy.random import uniform

import pickle
import os

from utils_ham import (
    obtain_AGP_hamiltonian_tensor_rotated
)

def generate_RandomHam_and_SolutionKey(Norb, Nterm):
    N = 2 * Norb

    angles    = uniform(-np.pi/2, np.pi/2, N*(N-1)//2) 
    AGPparams = uniform(-1, 1, Norb)
    omegas    = uniform(0, 1, Nterm)
    coefs     = uniform(-1, 1, [Nterm, Norb*(Norb-1)//2])
    x         = np.concatenate([angles, AGPparams, omegas, coefs.flatten()])

    Htbt = obtain_AGP_hamiltonian_tensor_rotated(x, Norb, Nterm)

    return Htbt, (angles, AGPparams)

def save_RandomHam_and_SolutionKey(Htbt, Key, filetag):
    os.makedirs(f'randomHam/{filetag}')

    Hfilename = f'randomHam/{filetag}/tensor'
    Kfilename = f'randomHam/{filetag}/key'

    with open(Hfilename, 'wb') as f:
        pickle.dump(Htbt, f)

    with open(Kfilename, 'wb') as f:
        pickle.dump(Key, f)

    return None

def load_RandomHam_and_SolutionKey(filetag):
    Hfilename = f'randomHam/{filetag}/tensor'
    Kfilename = f'randomHam/{filetag}/key'

    with open(Hfilename, 'rb') as f:
        Htbt = pickle.load(f)

    with open(Kfilename, 'rb') as f:
        Key = pickle.load(f)

    return Htbt, Key

