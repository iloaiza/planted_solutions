"""
Compare QWC, FC, SVD, CSA, GCSA methods using different metric 
Usage: python compute_metric (mol) (metric) (wfs). 
"""
import sys
sys.path.append("../")
import numpy as np

from saveload_utils import load_variance_result


def naive_metric(vs):
    M = len(vs)
    return M * np.sum(vs)


def optimal_metric(vs):
    return np.sum(vs ** (1 / 2)) ** 2


def l1_metric(vs, groups):
    def get_l1_norm(op):
        norm = 0
        for term, val in op.terms.items():
            if len(term) > 0: 
                norm += abs(val)
        return norm
    l1_norms = np.zeros(len(vs))
    for idx, group in enumerate(groups):
        l1_norms[idx] = get_l1_norm(group)
    l1_norms /= np.sum(l1_norms)

    metric_val = 0
    for idx, v in enumerate(vs):
        metric_val += v / l1_norms[idx]
    return metric_val


# Parameters
mol = 'h2' if len(sys.argv) < 2 else sys.argv[1]
metric = 'opt' if len(sys.argv) < 3 else sys.argv[2]
wfs = 'fci'
geo = 1.0

# Display parameters
print("Mol: {}".format(mol))
print("Metric: {}".format(metric.upper()))
print("Wavefunction: {}".format(wfs))

# Methods
methods = ['qwc', 'fc', 'gfc', 'gqwc', 'vgfc', 'svd', 'csa', 'pgcsa', 'gcsa', 'vgcsa', 'gcsa_svd', 'gcsa_fc']
methods_not_found = []

methods_vs = []  # List of list of fragment variances
methods_groups = [] # List of list of fragments 

for method in methods:
    try:
        cur_vs = load_variance_result(
            mol, geometry=geo, wfs=wfs, method=method, data_type='vars', path_prefix='../')
        cur_groups = load_variance_result(
            mol, geometry=geo, wfs=wfs, method=method, data_type='grps', path_prefix='../')
        methods_vs.append(cur_vs)
        methods_groups.append(cur_groups)
    except:
        methods_not_found.append(method)
        print("{} doesn't exist for {}".format(method, mol))
for method in methods_not_found:
    methods.remove(method)

print()  # Line break for results
for idx, method in enumerate(methods):
    if metric == 'dem':
        metric_val = naive_metric(methods_vs[idx])
    elif metric == 'opt':
        metric_val = optimal_metric(methods_vs[idx])
    else:
        metric_val = l1_metric(methods_vs[idx], methods_groups[idx])
    print("Method: {}. Fragments counts: {}. Metric value: {}".format(
        method, len(methods_vs[idx]), metric_val))
