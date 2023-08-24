import numpy as np

from scipy.optimize import minimize

import pickle

from utils_ham import (
    obtain_AGP_hamiltonian_tensor_rotated_fast
)

from numpy.random import uniform

def num_params(Norb, Nterm):
    N          = 2 * Norb

    angles_num = N * (N - 1) // 2
    AGP_num    = Norb
    omegas_num = Nterm
    coefsA_num = Norb * (Norb - 1) // 2
    coefs_num  = Nterm * coefsA_num
    total_num  = angles_num + AGP_num + omegas_num + coefs_num

    return angles_num, AGP_num, omegas_num, coefsA_num, coefs_num, total_num

def load_electronic_hamiltonian_combined_tensor(moltag):
    filename = f'hamiltonians/{moltag}/full_tensor'
    with open(filename, 'rb') as f:
        tbt = pickle.load(f)
    return tbt

def evaluate_cost_function(x, target_tbt, Norb, Nterm):
    fragment_tbt = obtain_AGP_hamiltonian_tensor_rotated_fast(x, Norb, Nterm)
    diff         = target_tbt - fragment_tbt

    val = np.sum(diff * diff)
    return np.real(val)   

def obtain_AGP_hamiltonian_fragment(target_tbt):
    #obtain various counts
    N     = target_tbt.shape[0]
    Norb  = N // 2
    Nterm = max([12, Norb*(Norb-1)//2])

    angles_num, AGP_num, omegas_num, coefsA_num, coefs_num, total_num = num_params(Norb, Nterm)

    # obtain initial parameters
    #     - random initial angles
    #     - 0 AGPparams
    #     - 2 omega params
    #     - 0 coefs params
    
    x0 = list(uniform(-1, 1, angles_num)) + list(uniform(-1, 1, AGP_num)) + list(uniform(0,1,omegas_num)) + list(uniform(-1,1,coefs_num))
    # x0                                                           = np.zeros(total_num)
    # x0[ : angles_num]                                            = uniform(-np.pi/2, np.pi/2, angles_num)
    # x0[angles_num + AGP_num : angles_num + AGP_num + omegas_num] = 2 * np.ones(omegas_num)

    # tolerance and other optimization setup
    # tol     = 1e-6 
    # enum    = N ** 4
    # fun_tol = (tol / enum) ** 2
    fun_tol = 1e-9

    options = {
        'maxiter' : 10000,
        'disp'    : False
    }

    #get cost function
    def cost(x):
        return evaluate_cost_function(x, target_tbt, Norb, Nterm)
    
    #define constraints: omegas > 0
    bounds = (
        [(-np.inf, np.inf) for _ in range(angles_num)] + 
        [(-np.inf, np.inf) for _ in range(AGP_num)] + 
        [(0, np.inf) for _ in range(omegas_num)] + 
        [(-np.inf, np.inf) for _ in range(coefs_num)]
    )

    #optimize
    return minimize(cost, x0, method='L-BFGS-B', bounds=bounds, tol=fun_tol, options=options)

def save_optimized_parameters(x, moltag):
    filename = f'results/agp_optimized/{moltag}/params'
    with open(filename, 'wb') as f:
        pickle.dump(x, f)
    return None    