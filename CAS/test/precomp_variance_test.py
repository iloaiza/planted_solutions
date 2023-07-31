"""Testing functions that used precomputed data to get variance. 
"""
import sys
sys.path.append("../")
import unittest

import ferm_utils as feru
import csa_utils as csau
from var_utils import get_system_details
import saveload_utils as sl
import openfermion as of
import numpy as np

runh2 = True 
runlih = False
runh2o = False
runbeh2 = False
runnh3 = False

class test_tbtev_and_tbtvar(unittest.TestCase):
    def ev_var_tester(self, mol, verbose=False):
        """Checking whether the expectation values and the variance agree between precomputed results and openfermion's. 
        """
        # Get two-body terms from H2.
        Hf = sl.load_fermionic_hamiltonian(mol)
        Htbt = feru.get_chemist_tbt(Hf)
        two_body = feru.get_ferm_op(Htbt)

        # Get HF.
        n_qubits = of.count_qubits(Hf)
        nelec, _ = get_system_details(mol)
        wfs = feru.get_openfermion_hf(n_qubits, nelec)

        # Make sure rest are one-body terms.
        diff = of.normal_ordered(Hf - two_body)
        for term, val in diff.terms.items():
            if len(term) == 4:
                self.assertLess(np.abs(val), 1e-8)

        # Compute expected from of.
        expected_ev = of.expectation(
            of.get_sparse_operator(two_body, n_qubits), wfs)
        expected_var = of.variance(
            of.get_sparse_operator(two_body, n_qubits), wfs)

        # Compute received from saved results.
        tbt_ev = sl.load_tbt_variance('ev', mol, 1.0, 'hf')
        tbt_sq = sl.load_tbt_variance('sq', mol, 1.0, 'hf')
        received_ev = np.einsum('ijkl,ijkl->', Htbt, tbt_ev)
        received_var = csau.get_precomputed_variance(Htbt, tbt_ev, tbt_sq)

        if verbose:
            print("EV Received: {}. Expected: {}".format(
                received_ev, expected_ev))
            print("VAR Received: {}. Expected: {}".format(
                received_var, expected_var))
        self.assertAlmostEqual(expected_ev, received_ev, delta=1e-7)
        self.assertAlmostEqual(expected_var, received_var, delta=1e-7)

    def test_h2(self):
        if runh2: 
            self.ev_var_tester('h2')

    def test_lih(self):
        if runlih: 
            self.ev_var_tester('lih')

    def test_h2o(self):
        if runh2o: 
            self.ev_var_tester('h2o')

    def test_beh2(self):
        if runbeh2: 
            self.ev_var_tester('beh2')

    def test_nh3(self):
        if runnh3:
            self.ev_var_tester('nh3')

    def test_random(self):
        """Checking expectation values and variance agreement using random two-body tensor 
        """
        # Get HF
        n_qubits = 4
        wfs = feru.get_openfermion_hf(n_qubits, 2)

        # Get random two-body tensor
        random_tbt = np.random.rand(2, 2, 2, 2)
        two_body = feru.get_ferm_op(random_tbt)

        # Compute expected from of.
        expected_ev = of.expectation(
            of.get_sparse_operator(two_body, n_qubits), wfs)
        expected_var = of.variance(
            of.get_sparse_operator(two_body, n_qubits), wfs)

        # Compute received from saved results.
        tbt_ev = sl.load_tbt_variance('ev', 'h2', 1.0, 'hf')
        tbt_sq = sl.load_tbt_variance('sq', 'h2', 1.0, 'hf')
        received_ev = np.einsum('ijkl,ijkl->', random_tbt, tbt_ev)
        received_var = csau.get_precomputed_variance(
            random_tbt, tbt_ev, tbt_sq)

        self.assertAlmostEqual(expected_ev, received_ev, delta=1e-7)
        self.assertAlmostEqual(expected_var, received_var, delta=1e-7)
