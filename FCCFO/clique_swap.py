import openfermion as of
import numpy as np
from op_utils import *

def get_op_list(op):
    """
    Change the operator into a list of operators in chemist notation
    """
    lis = []
    if isinstance(op, of.FermionOperator):
        for i in op.terms:
            lis.append(of.transforms.chemist_ordered(of.FermionOperator(i, op.terms[i])))
    return lis
def count_commuting_terms(op, s):
    """
    Count the number of terms the operator op commmutes with the operators in the set s.
    in the form of sum of operator with its hermitian conjugate
    """
    return sum([check_pair_commuting(op + of.utils.hermitian_conjugated(op), 
                           i + of.utils.hermitian_conjugated(i)) for i in get_op_list(s)])

def is_num_op(op):
    """
    Check if an operator consists of only the number operators
    """
    o = of.transforms.chemist_ordered(op)
    for t in o.terms:
        if len(t) % 2 != 0:
            return False
        for i in range(len(t) // 2):
            if t[2*i][0] != t[2*i + 1][0]:
                return False
    return True

def swapping_cliques(FG: list):
    """
    Given a list of cliques of commuting Fermionic operators, the FG[0] is the clique of all
    number operators. From the rest of the cliques, swap some non-number operators into the
    first clique to make FG[0] a largest commuting clique with not all number operators
    """
    opt_size = 0
    opt_2norm = 0
    opt_clique = []
    for k in range(1, len(FG)):
    #     Loop through cliques of non-number operators
        excitation_ops = get_op_list(of.transforms.chemist_ordered(FG[k]))

    #     Loop through excitation opertors within the number operators
        for op in excitation_ops:
            commuting_num_ops = [l for l in get_op_list(FG[0]) if
                                 check_pair_commuting(op + of.utils.hermitian_conjugated(op), 
                                        l + of.utils.hermitian_conjugated(l))]


            count = 0
            cur_sub_clique = []
    #         Count the number of operators in the current clique which commutes with the number operators
    #         which commutes with the current operator
            for i in excitation_ops:
                if count_commuting_terms(i, sum(commuting_num_ops)) == len(commuting_num_ops):
                    count += 1
                    cur_sub_clique.append(i)

            if count >= 1 and count + len(commuting_num_ops) > opt_size:
                opt_clique = commuting_num_ops + cur_sub_clique
                opt_size = len(opt_clique) 

    zero = of.FermionOperator.zero()
    while zero in opt_clique:
        opt_clique.remove(zero)
        opt_size -= 1
    print("Optimum number of terms:{}".format(opt_size))

    opt_2norm = sum([np.abs(sum(opt_clique).terms[i]) for i in sum(opt_clique).terms])
    print("Optimum 2-norm of coefficients:{}".format(opt_2norm))
    return opt_clique, opt_2norm
