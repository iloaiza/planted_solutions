""" Testing CSA gradients. 
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
step = 1e-6 if len(sys.argv) < 4 else float(sys.argv[2])
mol = 'h2' if len(sys.argv) < 5 else sys.argv[3]

# Loading Hamiltonian. Get Htbt. 
Hf = sl.load_fermionic_hamiltonian(mol)
Htbt = feru.get_chemist_tbt(Hf)
n = of.count_qubits(Hf) // 2

# Randomize x 
x = np.random.randn(n**2 * alpha) * 2 * np.pi 

# Get Numerical gradient. Time cost function evaluation. 
cost_m = csau.ct_decomp_cost(x, Htbt, alpha, complex=False)
num_start = time.time() 
num_grad_1 = np.full(x.shape, np.nan)
num_grad_2 = np.full(x.shape, np.nan)
for i in range(len(x)):
    x_r = np.copy(x); x_r[i] += step 
    x_2r = np.copy(x); x_2r[i] += 2*step 
    x_l = np.copy(x); x_l[i] -= step 
    x_2l = np.copy(x); x_2l[i] -= 2*step 
    cost_r = csau.ct_decomp_cost(x_r, Htbt, alpha, complex=False)
    cost_l = csau.ct_decomp_cost(x_l, Htbt, alpha, complex=False)
    cost_2r = csau.ct_decomp_cost(x_2r, Htbt, alpha, complex=False)
    cost_2l = csau.ct_decomp_cost(x_2l, Htbt, alpha, complex=False)
    num_grad_1[i] = (cost_r - cost_l) / step / 2
    num_grad_2[i] = (-cost_2r + 8*cost_r - 8*cost_l + cost_2l) / step / 12
num_time = time.time() - num_start

# Get Analytical gradient. Time gradient evaluation. 
ana_start = time.time() 
ana_grad = csau.real_ct_grad(x, Htbt, alpha)
ana_time = time.time() - ana_start

# Print 
print()
print("1st/2nd ord Numerical v.s. Analytical grad: \
    \n1st ord Num: {}\n2nd ord Num: {}\nAna        : {}".format(num_grad_1, num_grad_2, ana_grad))
print("Time it takes to obtain numerical gradient: {} s".format(round(num_time)))
print("Time it takes to obtain analytical gradient: {} s".format(round(ana_time)))
print("Time it takes compute csa cost function: {} s".format(round(num_time / 2 / len(x))))
