""" Testing VCSA gradients. 
"""
path_prefix = "../"
import sys 
sys.path.append(path_prefix)
import saveload_utils as sl 
import ferm_utils as feru 
import csa_utils as csau 
import openfermion as of 
import numpy as np 
import time 

# Parameters 
alpha = 1 if len(sys.argv) < 2 else int(sys.argv[1])
vw = 0.05 if len(sys.argv) < 3 else float(sys.argv[2])
step = 1e-6 if len(sys.argv) < 4 else float(sys.argv[3])
mol = 'h2' if len(sys.argv) < 5 else sys.argv[4]

# Loading Hamiltonian. Get Htbt. 
Hf = sl.load_fermionic_hamiltonian(mol)
Htbt = feru.get_chemist_tbt(Hf)
n = of.count_qubits(Hf) // 2

# Loading tbt_ev/sq
tbtev_ln = sl.load_tbt_variance("ev", mol, geo=1.0, wfs_type='hf')
tbtev_sq = sl.load_tbt_variance("sq", mol, geo=1.0, wfs_type='hf', verbose=False)

# Randomize x 
x = np.random.randn(n**2 * alpha) * 2 * np.pi 

# Get Numerical gradient. Time cost function evaluation. 
cost_m = csau.var_decomp_cost(x, Htbt, alpha, vw, tbtev_ln, tbtev_sq)
num_start = time.time() 
num_grad_1 = np.full(x.shape, np.nan)
num_grad_2 = np.full(x.shape, np.nan)
for i in range(len(x)):
    x_r = np.copy(x); x_r[i] += step 
    x_l = np.copy(x); x_l[i] -= step 
    x_2r = np.copy(x); x_2r[i] += 2*step 
    x_2l = np.copy(x); x_2l[i] -= 2*step
    cost_r = csau.var_decomp_cost(x_r, Htbt, alpha, vw, tbtev_ln, tbtev_sq)
    cost_l = csau.var_decomp_cost(x_l, Htbt, alpha, vw, tbtev_ln, tbtev_sq)
    cost_2r = csau.var_decomp_cost(x_2r, Htbt, alpha, vw, tbtev_ln, tbtev_sq)
    cost_2l = csau.var_decomp_cost(x_2l, Htbt, alpha, vw, tbtev_ln, tbtev_sq)
    num_grad_1[i] = (cost_r - cost_l) / step / 2
    num_grad_2[i] = (-cost_2r + 8*cost_r - 8*cost_l + cost_2l) / step / 12
num_time = time.time() - num_start

# Get Analytical gradient. Time gradient evaluation. 
ana_start = time.time() 
ana_grad = csau.var_grad(x_r, Htbt, alpha, vw, tbtev_ln, tbtev_sq) 
ana_time = time.time() - ana_start

# Print 
print()
print("2/5 pts Numerical v.s. Analytical grad: \
    \n2 pts Num: {}\n5 pts Num: {}\nAnalytic : {}".format(num_grad_1, num_grad_2, ana_grad))
print("Norm grad: {}".format(csau.real_ct_grad(x, Htbt, alpha)))
print("Time it takes to obtain numerical gradient: {} s".format(round(num_time)))
print("Time it takes to obtain analytical gradient: {} s".format(round(ana_time)))
print("Time it takes compute vcsa cost function: {} s".format(round(num_time / 2 / len(x))))
