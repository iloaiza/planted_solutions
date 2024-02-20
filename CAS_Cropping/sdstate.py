"""Defines an implementation of the Slater Determinant states,with a dictionary to represent the occupied states with the corresponding constants.
"""
# Defines an implementation of the Slater Determinant states,with a dictionary to represent the occupied states with the corresponding constants.
import numpy as np
from itertools import product
import copy
from scipy.linalg import eigh_tridiagonal
from multiprocessing import Pool
import openfermion as of

class sdstate:
    eps = 1e-8
    dic = {}
    n_qubit = 0
    def __init__(self, s = None, coeff = 1, n_qubit = 0, eps = 1e-8):
        self.dic = {}
        self.eps = eps
        if n_qubit:
            self.n_qubit = n_qubit
        if s:
            self.dic[s] = coeff
            if not self.n_qubit:
                self.n_qubit = len(str(bin(s)))- 2
            
    def norm(self):
#         Return the norm of the current state
        return np.sqrt(self @ self)

    def normalize(self):
#         Normalize the current state
        n = self.norm()
        for i in self.dic:
            self.dic[i] /= n
        return None
    
    def __add__(self, other):
#         Expand the current Hilbert space if th new state is larger.
        self.n_qubit = max(other.n_qubit, self.n_qubit)
        result = copy.deepcopy(self)
#         s_in_result = set(result.dic.keys()).intersection(set(other.dic.keys()))
#         for s in s_in_result:
#             result.dic[s] += other.dic[s]
#         for s in (set(other.dic.keys()) - s_in_result):
#             result.dic[s] = other.dic[s]
        for s in other.dic:
            if s in self.dic:
                result.dic[s] += other.dic[s]
            else:
                result.dic[s] = other.dic[s]      
        return result
    
    def __sub__(self, other):
        return self + (-1) * other
    
    def __mul__(self, n):
#         Defines constant multiplication
        result = copy.deepcopy(self)
        for s in result.dic:
            result.dic[s] *= n
        return result
    
    def __rmul__(self, n):
        return self.__mul__(n)
    
    def __truediv__(self, n: int):
        return self.__mul__(1/n)
    
    def __matmul__(self, other):
        if isinstance(other, of.FermionOperator):
            return self.Hf_state(other)
        if self.n_qubit == 0 or other.n_qubit == 0:
            return 0
        count = 0
#         assert self.n_qubit == other.n_qubit, "qubit number mismatch"
        lis = list(set(list(self.dic.keys())) & set(list(other.dic.keys())))
        for s in lis:
            count += np.conjugate(self.dic[s]) * other.dic[s]
        return count 
    
    def __str__(self):
        return str({str(bin(k))[2:][::-1]: self.dic[k] for k in self.dic})
    
    def exp(self, Hf: of.FermionOperator):
#         Return the expectation value Hamiltonian on the current state with Hamiltonian in the two-body-tensor form
        if isinstance(Hf, of.FermionOperator):
            return np.real(self @ self.Hf_state(Hf))
        elif isinstance(Hf, np.ndarray):
            return np.real(self @ self.tensor_state(Hf))
        print("Invalid input of Hf")
        return -1
    
    def tensor_state(self, tbt):
        n = tbt.shape[0]
        assert len(tbt.shape) == 2 or len(tbt.shape) == 4, "Invalid tensor shape"
        re_state = sdstate(n_qubit = self.n_qubit)
        if len(tbt.shape) == 4:
            for p, q, r, s in product(range(n), repeat = 4):
                re_state += tbt[p, q, r, s] * sdstate.Epqrs(self, self.n_qubit - 1 - p, self.n_qubit - 1 -q, self.n_qubit - 1 - r, self.n_qubit - 1 -s)
        elif len(tbt.shape) == 2:
            for p, q in product(range(n), repeat = 2):
                re_state += tbt[p, q] * sdstate.Epq(self, self.n_qubit - 1 - p, self.n_qubit - 1 - q)
        return re_state
        
    def concatenate(self, st):
#         Return the direct product of two sdstates
        if len(self.dic) == 0:
            return st
        elif len(st.dic) == 0:
            return self
        n2 = st.n_qubit
        n = self.n_qubit + st.n_qubit
        tmp = sdstate(n_qubit = n)
        for s1 in self.dic:
            for s2 in st.dic:
                tmp += sdstate(s = s1 << n2 | s2, coeff = self.dic[s1] * st.dic[s2], n_qubit = n)
        return tmp
                
    def Epq(self, p, q):
        """
        Return the action of a_p^a_q on the current state.
        """
        tmp = sdstate(n_qubit = self.n_qubit)
        for n in self.dic:
            if actable_pq(n, p, q):
                t = n ^ (1 << p) ^ (1 << q)
                tmp += sdstate(t, self.dic[n] * (-1) ** parity_pq(n, p, q), n_qubit = self.n_qubit)
        return tmp
    
    def Epqrs(self, p, q, r, s):
#         To be changed or improved? Current implementation based on Epq
        """
        Return the action of a_p^a_q on the current state.
        """
        tmp = sdstate(n_qubit = self.n_qubit)
        for n in self.dic:
            if actable_pq(n, r, s):
                t = n ^ (1 << r) ^ (1 << s)
                if actable_pq(t, p, q):
                    k =  t ^ (1 << p) ^ (1 << q)
                    tmp += sdstate(k, self.dic[n] * (-1) ** (parity_pq(n, r, s) + parity_pq(k, p, q)), n_qubit = self.n_qubit)
        return tmp

        
    def Hf_state(self, Hf: of.FermionOperator, multiprocessing = False):
        """Apply a Hamiltonian in FermionOperator on the current state. multiprocessing can be used
        to parallelize the process of applying each Excitation operator in the Hamiltonian. The general
        cost is given by O(N^4M), for N as the qubit dimension and M as the size of the current state.
        """
        H = of.transforms.chemist_ordered(Hf)
        re_state = sdstate(n_qubit = self.n_qubit)
        if multiprocessing:
            with Pool() as pool:
                res = pool.starmap(self.op_state, [(t, H.terms[t]) for t in H.terms])
            for state in res:
                re_state += state
            return re_state
        for t in H.terms:
            re_state += self.op_state(t, H.terms[t])
        return re_state

    def op_state(self, t, coef):
#         if len(t) == 4:
#             return coef * self.Epqrs(t[0][0], t[1][0], t[2][0], t[3][0])
#         elif len(t) == 2:
#             return coef * self.Epq(t[0][0], t[1][0])
        if len(t) == 4:
            return coef * self.Epqrs(self.n_qubit - 1 - t[0][0], self.n_qubit - 1 - t[1][0],
                                     self.n_qubit - 1 - t[2][0], self.n_qubit - 1 - t[3][0])
        elif len(t) == 2:
            return coef * self.Epq(self.n_qubit - 1 - t[0][0], self.n_qubit - 1 - t[1][0])
        elif len(t) == 0:
            return coef * self
        return sdstate(n_qubit = self.n_qubit)
    
    def to_vec(self):
        vec = np.zeros(2 ** self.n_qubit)
        for i in self.dic:
            vec[i] = self.dic[i]
        return vec
    
def parity_pq(number: int, a, b):
    """Count the number of electrons between p and q bits (p+1, p+2, ... ,q-1), 
    return a binary number representing the parity of the substring in the binary representation of number
    """
    if abs(a - b) < 2:
        return 0
    p = min(a,b)
    q = max(a,b)
    # Create a mask with 1s between p/
    mask = ((1 << q) - 1)

    # Apply the mask to truncate the first q bits, and drop the last p bits.
    result = (number & mask ) >> (p + 1)
    # Compute the parity
    parity = 0
    while result:
        parity ^= 1
        result &= result - 1  # Drops the lowest set bit
    return parity

def actable_pq(n, p, q):
    """
    Determines if a_p^a_q annihilates the current state given by n, for n as an index in fock space
    """
    return (p == q and (n & 1 << q) != 0) or ((n & 1 << p) == 0 and (n & 1 << q) != 0)
    
    
def HF_energy(Hf, n, ne):
    """Find the energy of largest and smallest slater determinant states with Hf as Fermionic Hamiltonian and
     number of electrons as ne.
    """    
    lstate = sdstate((1 << ne) - 1, n_qubit = n)
#         <low|H|low>
    E_low = lstate.exp(Hf)
    hstate = sdstate(((1 << ne) - 1) << (n-ne), n_qubit = n)
#         <high|H|high>
    E_high = hstate.exp(Hf)
    return E_high, E_low

def HF_spectrum_range(Hf, multiprocessing = True):
    """Compute the Hatree-Fock energy range of the Hamiltonian 2e tensor Hf for all number of electrons.
    Multiprocessing parameter is set to parallelize computations for the states with different number of electrons
    """
    n = of.utils.count_qubits(Hf)
    if multiprocessing:
        with Pool() as pool:
            res = pool.starmap(HF_energy, [(Hf, n, ne) for ne in range(n)])
        low = 1e10
        low_state = ""
        high = -1e10
        high_state = ""
        for ne in range(len(res)):
            E_high = res[ne][0]
            E_low = res[ne][1]
            if E_low < low:
                low_state = (1 << ne) - 1
                low = E_low
            if E_high > high:
                high_state = ((1 << ne) - 1) << (n-ne)
                high = E_high
    else:
        low = 1e10
        low_state = ""
        high = -1e10
        high_state = ""
        for ne in range(n):
            low_int = (1 << ne) - 1
            lstate = sdstate(low_int, n_qubit = n)
    #         <low|H|low>
            E_low = lstate.exp(Hf)
            high_int = ((1 << ne) - 1) << (n-ne)
            hstate = sdstate(high_int, n_qubit = n)
    #         <high|H|high>
            E_high = hstate.exp(Hf)
            if E_low < low:
                low_state = low_int
                low = E_low
            if E_high > high:
                high_state = high_int
                high = E_high
    high_str = bin(high_state)[2:][::-1]
    low_str = bin(low_state)[2:]
    high_str = "0" * (n - len(high_str)) + high_str
    low_str += "0" * (n - len(low_str))
    print("HF E_max: {}".format(high))
    print("HF E_min: {}".format(low))
    return high_str, low_str, high, low


def lanczos(Hf: of.FermionOperator, steps, state = None, ne = None):
    """Applies lanczos iteration on the given 2e tensor Hf, with number of steps given by steps,
    with initial state as input or number of electrons as input ne.
    Returns normalized states in each iteration, and a tridiagonal matrix with main diagonal in A and sub-diagonal in B.
    """
    n_qubits = of.utils.count_qubits(Hf)
    if state == None:
        if ne == None:
            ne = n_qubits // 2
        state = sdstate(int("1"*ne + "0"*(n_qubits - ne), 2), n_qubit = n_qubits)
        
    tmp = state @ Hf
    ai = tmp @ state
    tmp -= ai * state
    A = [ai]
    B = []
    states = [state]
    vi = tmp
    for i in range(1,steps):
        bi = tmp.norm()
        if bi != 0:
            vi = tmp / bi
        tmp = vi @ Hf
        ai = vi @ tmp
        tmp -= ai * vi 
        tmp -= bi * states[i - 1]
        states.append(vi)
        A.append(ai)
        B.append(bi)
    return states, A, B

def lanczos_range(Hf, steps, state = None, ne = None):
    """ Returns the largest and the smallest eigenvalue from Lanczos iterations with given number of steps,
    number of electrons or initial state.
    """
    _, A, B = lanczos(Hf, steps = steps, state = state, ne = ne)
    eigs, _ = eigh_tridiagonal(A,B)
    return max(eigs), min(eigs)

def lanczos_total_range(Hf, steps, states = [], multiprocessing = True):
    """ Returns the largest and the smallest eigenvalue from Lanczos iterations with given number of steps,
    for all possible number of electrons. Multiprocessing will parallelize the computation for all possible 
    number of electrons. State is in the form of list of binary strings, indicating the maximum and minimum 
    HF energy states to start the iteration.
    """
    n = of.utils.count_qubits(Hf)
    if states != []:
        [max_str, min_str] = states
        max_state = sdstate(int(max_str[::-1], 2), n_qubit = n)
        min_state = sdstate(int(min_str[::-1], 2), n_qubit = n)
        E_max, _ = lanczos_range(Hf, steps = steps, state = max_state)
        _, E_min = lanczos_range(Hf, steps = steps, state = min_state)
    else:
        if multiprocessing:
            with Pool() as pool:
                res = pool.starmap(lanczos_range, [(Hf, steps, None, ne) for ne in range(n)])
            E_max = max([i[0] for i in res])
            E_min = min([i[1] for i in res])
        else:
            E_max = -1e10
            E_min = 1e10
            for ne in range(n):
                states, A, B = lanczos(Hf, steps = steps, ne = ne)
                eigs, _ = eigh_tridiagonal(A,B)
                E_max = max(max(eigs), E_max)
                E_min = min(min(eigs), E_min)
    return E_max, E_min