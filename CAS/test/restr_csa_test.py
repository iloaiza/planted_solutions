"""
Testing functions of restricted CSA. 
"""
import sys
sys.path.append("../")
import unittest

import saveload_utils as sl
import ferm_utils as feru
import csa_utils as csau
import matrix_utils as matu
import numpy as np
import openfermion as of


def rotate_fermion_op(fermion_op, O_params, n):
    """Rotate the given Fermionic operator given the orthogonal orbital rotation specified by O_params
    Following O a^+_p O = \sum_q O_pq a^+_q 

    Args: 
        fermion_op (FermionOperator):
        O_params (array): 

    Returns: 
        rotated_ops (FermionOperator)
    """
    rotated_ops = of.FermionOperator.zero()
    O = matu.construct_orthogonal(n, O_params)

    for fw, val in fermion_op.terms.items():
        rotated_fw = of.FermionOperator.identity() * val
        for fc in fw:
            spin_orb, cr = fc
            orb, spin = spin_orb // 2, spin_orb % 2

            rotated_fc = of.FermionOperator.zero()
            for q in range(len(O[orb, :])):
                if cr == 1:
                    rotated_fc += of.FermionOperator("{}^".format(2 * q + spin),
                                                     coefficient=O[orb, q])
                else:
                    rotated_fc += of.FermionOperator("{} ".format(2 * q + spin),
                                                     coefficient=O[orb, q])
            rotated_fw = rotated_fw * rotated_fc
        rotated_ops += rotated_fw
    return rotated_ops


class test_gmfr(unittest.TestCase):
    def get_h2_greedy(self):
        """Load restricted greedy csa's results. 
        """
        method = 'RESTR_GREEDY_CSA'
        restr_csa_sol = sl.load_restr_csa_sols(
            'h2', geo=1.0, method=method.lower(), alpha=4)
        return restr_csa_sol

    def test_summation(self):
        """Loading restricted greedy CSA results for H2 and test whether the fragments found sum up to the Hamiltonina, ignoring one-body terms.  
        The fragments are reconstructed as FermionOperator and rotated by code above using orbital rotation. 
        """
        # Parameters
        tol = 1e-6

        csa_sol = self.get_h2_greedy()
        sol_array = csa_sol['sol_array']
        cartans = csa_sol['cartans']

        h2_ferm = sl.load_fermionic_hamiltonian('h2')
        op_recon = of.FermionOperator.zero()
        for idx, cartan in enumerate(cartans):
            curop = of.FermionOperator.zero()
            cartan = cartan[0]
            i_indices, j_indices = np.nonzero(cartan)
            # Build corresponding Cartan operator for given cartan matrix
            for k in range(len(i_indices)):
                i, j = i_indices[k], j_indices[k]
                for a in range(2):
                    for b in range(2):
                        i_sp, j_sp = 2 * i + a, 2 * j + b
                        curop += of.FermionOperator(term="{}^ {} {}^ {}".format(
                            i_sp, i_sp, j_sp, j_sp), coefficient=cartan[i, j])

            # Multiply by coefficient & Rotate
            coeff, angle = sol_array[2 * idx], sol_array[2 * idx + 1]
            curop = rotate_fermion_op(coeff * curop, [angle], 2)
            op_recon += curop

        # Checking all two-body terms are reproduced up to tolerance.
        diff = of.normal_ordered(h2_ferm - op_recon)
        for term, val in diff.terms.items():
            if len(term) == 4:
                self.assertLess(np.abs(val), tol)


class test_operator_rotation(unittest.TestCase):
    def test_h2_csa(self):
        """This code test the operator's rotation code above by loading  the regular CSA results for H2 and check the cartan rotated by the code sums up to h2's hamiltonian correctly up to two body terms. 
        """
        # Load H2
        mol = 'h2'
        alpha = 2
        h2 = sl.load_fermionic_hamiltonian(mol)
        norb = of.count_qubits(h2) // 2
        verbose = False
        tol=1e-7

        # Load H2's CSA
        h2_x = np.load("../grad_res/" + mol + '/' + mol +
                       '_' + str(alpha) + '_True.npy')

        # Get operator form of one CSA.
        upnum, ctpnum, pnum = csau.get_param_num(norb, complex=False)
        reconstructed_op = of.FermionOperator.zero()
        for k in range(alpha):
            curp = h2_x[k * pnum: (k + 1) * pnum]
            ctvals = curp[:ctpnum]
            cartan = of.FermionOperator.zero()
            idx = 0
            for i in range(norb):
                for j in range(i, norb):
                    for spin_i in range(2):
                        for spin_j in range(2):
                            i_sporb, j_sporb = 2 * i + spin_i, 2 * j + spin_j
                            cartan += of.FermionOperator(term="{}^ {} {}^ {}".format(
                                i_sporb, i_sporb, j_sporb, j_sporb), coefficient=ctvals[idx])
                            if i != j:
                                cartan += of.FermionOperator(term="{}^ {} {}^ {}".format(
                                    j_sporb, j_sporb, i_sporb, i_sporb), coefficient=ctvals[idx])
                    idx += 1
            cartan_rotated = rotate_fermion_op(cartan, curp[ctpnum:], norb)
            reconstructed_op += cartan_rotated

        diff = of.normal_ordered(h2 - reconstructed_op)
        for term, val in diff.terms.items():
            if len(term) == 4:
                self.assertLess(np.abs(val), tol)
        if verbose:
            print("Diff: {}".format(diff))


class test_reflection_classes(unittest.TestCase):
    def test_reflections_without_spinorbitals(self):
        """
        """
        # Parameter
        norb = 4
        verbose = False

        for idx in range(9):
            reflection = csau.get_restr_reflections(idx, norb)[0]

            # Get the operator
            reflection_op = of.FermionOperator.zero()
            for i in range(norb):
                for j in range(norb):
                    reflection_op += of.FermionOperator(
                        term="{}^ {} {}^ {}".format(i, i, j, j), coefficient=reflection[i, j])

            # Print eigenvalues
            v, w = np.linalg.eigh(of.get_sparse_operator(
                reflection_op, n_qubits=norb).todense())
            if verbose:
                print("Eigvals: {}".format(v))


class test_restr_csa_gradient(unittest.TestCase):
    def test_all_gradients(self):
        # Parameters
        alpha = 2
        step = 1e-6

        # Get H2
        h2_ferm = sl.load_fermionic_hamiltonian('h2')
        h2_tbt = feru.get_chemist_tbt(h2_ferm)

        # Get random restr_csa coefficents with correct length.
        restr_csa_coeffs = np.random.rand(alpha * 4) * 2 * np.pi

        # Specify random cartans for the code.
        ct_idx_1, ct_idx_2 = np.random.choice(
            range(5)), np.random.choice(range(5))
        cartans = csau.get_restr_reflections(ct_idx_1, 2, 1)
        cartans.extend(csau.get_restr_reflections(ct_idx_2, 2, 1))

        # For each coefficients, check gradient
        received = csau.restr_cartan_gradient(
            restr_csa_coeffs, cartans, h2_tbt)

        cur_cost = csau.restr_csa_cost(restr_csa_coeffs, cartans, h2_tbt)
        for idx in range(len(restr_csa_coeffs)):
            tmp_coeffs = np.copy(restr_csa_coeffs)
            tmp_coeffs[idx] += step
            tmp_cost = csau.restr_csa_cost(tmp_coeffs, cartans, h2_tbt)

            cur_grad = (tmp_cost - cur_cost) / step
            self.assertAlmostEqual(cur_grad, received[idx], 3)
