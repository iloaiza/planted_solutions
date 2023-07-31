"""Testing the tbt obtain from usual code and tbt obtain from doing by parts
"""
path_prefix = "../"
import sys 
sys.path.append(path_prefix)

import saveload_utils as sl 
import numpy as np 

# Parameters 
mol = 'lih'
geo = 1.0
wfs = 'hf'
tsr_type = 'sq'
expected_post_fix = '_0-2'
received_post_fix = '_0-2-all'

# Get expected 
expected = sl.load_tbt_variance(tsr_type, mol, geo, wfs, path_prefix, file_post_fix=expected_post_fix)

# Get received 
received = sl.load_tbt_variance(tsr_type, mol, geo, wfs, path_prefix, file_post_fix=received_post_fix)

# Check same non-nan indices 
expected_non_nan = np.where(~np.isnan(expected))
received_non_nan = np.where(~np.isnan(received))

for i in range(8):
    assert((expected_non_nan[i] == received_non_nan[i]).all())

# Loop over indices. Check same value 
for i, j, k, l, a, b, c, d in zip(*received_non_nan):
    assert expected[i, j, k, l, a, b, c, d] == received[i, j, k, l, a, b, c, d]
