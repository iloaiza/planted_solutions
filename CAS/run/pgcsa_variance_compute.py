"""Obtain the PGCSA fragments FCI variances. Use FCI wavefunction.
This also checks that all fragments sums up to the Hamiltonian as expected. 

Usage: python pgcsa_variance_compute.py (mol) (csa_nfrag) (use_precomputed T/F) (start_idx) (n_frag_compute)
"""
path_prefix = "../"
import sys
sys.path.append(path_prefix)
import numpy as np
import io
import time 

import saveload_utils as sl
import openfermion as of
import csa_utils as csau
import gc 

# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
csa_nfrag = 2 if len(sys.argv) < 3 else int(sys.argv[2])
use_precomputed = False if len(sys.argv) < 4 else sys.argv[3] == 'T'
start_idx = 0 if len(sys.argv) < 5 else int(sys.argv[4])
n_frag_compute = csa_nfrag+1 if len(sys.argv) < 6 else int(sys.argv[5])

wfs = 'fci'
method = 'PGCSA'
geo = 1.0
save = False
check_precomputed = True
verbose = True 
check_operator_sum = False 

# Prepare logging
if save:
    log_string = io.StringIO()
    sys.stdout = log_string

# Display run's details
print("Molecule: {}".format(mol))
print("Wavefunction: {}".format(wfs))
print("Geometries: {}".format(geo))
print("Method: {}".format(method))
print("Number of CSA fragments: {}".format(csa_nfrag))
print()

# Prep
pgcsa_sol = sl.load_csa_sols(mol, geo, method, csa_nfrag)
pgcsa_fragments = pgcsa_sol['grouping']

if use_precomputed: 
    pgcsa_tbts = csau.get_tbt_parts(pgcsa_sol['sol_array'],
                                    pgcsa_sol['n_qubits'] // 2, pgcsa_sol['csa_alpha'])
if check_operator_sum:
    h_ferm = sl.load_fermionic_hamiltonian(mol)
    h_reconstructed = of.FermionOperator.zero()
n_qubits = pgcsa_sol['n_qubits']
del pgcsa_sol
gc.collect()

# Get wfs
psi = sl.load_ground_state(mol, 'FM', path_prefix=path_prefix, verbose=True)

# Loading FCI tbtev_ln/sq
if use_precomputed:
    tbtev_ln = sl.load_tbt_variance("ev", mol, geo=1.0, wfs_type='fci')
    tbtev_sq = sl.load_tbt_variance("sq", mol, geo=1.0, wfs_type='fci', verbose=False)

# Compute FCI variance
group_vars = np.full(len(pgcsa_fragments), np.nan)
for idx in range(start_idx, start_idx+n_frag_compute):
    frag = pgcsa_fragments[idx]
    start = time.time() 
    if idx == 0 or not use_precomputed:
        frag_sp = of.get_sparse_operator(frag, n_qubits)
        ev = of.expectation(frag_sp, psi)
        frag_sq_sp = frag_sp @ frag_sp
        del frag_sp
        gc.collect()
        sq = of.expectation(frag_sq_sp, psi)
        del frag_sq_sp
        gc.collect()
        curvar = sq - ev**2 
        #2nd Implementation 
        #ev = of.expectation(of.get_sparse_operator(frag, n_qubits), psi)
        #gc.collect() 
        #sqfrag = frag * frag 
        #sq = of.expectation(of.get_sparse_operator(sqfrag, n_qubits), psi)
        #del sqfrag
        #gc.collect() 
        #curvar = sq - ev**2 
        #1st Implementation
        # curvar = of.variance(of.get_sparse_operator(frag, n_qubits), psi)
        if np.imag(curvar) > 1e-7:
            print("Current variance has non-zero complex part: {}".format(curvar))
        group_vars[idx] = np.real(curvar)
    else:
        group_vars[idx] = csau.get_precomputed_variance(pgcsa_tbts[idx - 1], tbtev_ln, tbtev_sq)
        if check_precomputed and idx < 3:
            print("Checking {}th fragment variance.".format(idx))
            curvar = of.variance(of.get_sparse_operator(frag, n_qubits), psi)
            assert np.isclose(curvar, group_vars[idx])
            print("Check passed.")
    if check_operator_sum:
        h_reconstructed += frag
    
    # Time printout 
    if verbose:
        print("{}th fragment done. Time elapsed: {}. Current var: {}".format(idx+1, round(time.time() - start), group_vars[idx]))
        if not save: sys.stdout.flush()
    del frag 
    pgcsa_fragments[idx] = None 
    gc.collect()

if check_operator_sum:
    print("Should be 0. H - \sum H_i: {}".format(of.normal_ordered(h_ferm - h_reconstructed)))
print("Optimal metric: {}".format(np.sum(group_vars**(1 / 2))**2))

if save:
    sl.save_variance_result(mol, wfs=wfs, method=method.lower(),
                            geometry=geo, groups=pgcsa_fragments, variances=group_vars)
    log_fpath = sl.get_logging_path_and_name(
        mol, geo, wfs, method.lower(), '../')
    with open(log_fpath, 'w') as f:
        print(log_string.getvalue(), file=f)
