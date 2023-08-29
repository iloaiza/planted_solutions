"""
Containing project specific save/load functions. 

Variance:
    - Under variances_computed folder
    - Subfolders order: Molecule, wavefunction, partitioning method
    - File name: molecule_geometry_wavefunction_method 
    - Content: A dictionary with list of fragments and their variances. 

CSA solutions:
    - Under csa_computed folder
    - Subfolders order: Molecule, partitioning method 
    - File name: molecule_geometry_method_alpha
    - Content: A dictionary (Dict[str, *])
        'sol_array': A numpy array of csa solutions.
        'grouping': List of Fermionic Operators partitioned. They should sum up to H unless grouping_remains is non empty. 
            This includes one body term. . 
        'csa_alpha': Number of CSA fragments. 
        'n_qubits: Number of qubits.   
        'grouping_remains': List of QubitOperator partitioned in addition to FermionOperators 
        'num_fragments": Number of fragments. len(grouping) + len(grouping_remains)

Restricted CSA solutions:
    - Under restr_csa_computed folder
    - Subfolders order: Molecule, partitioning method 
    - File name: molecule_geometry_method_alpha
    - Content: A dictionary (Dict[str, *])
        'sol_array': A numpy array of restricted csa solutions.
        'cartans': A list of NxN matrix cartan indicating the type of cartans used. cartan[i, j] -> coefficients of ninj
        'grouping': List of Fermionic Operators partitioned. They should sum up to H unless grouping_remains is non empty. 
            This includes one body term.
        'csa_alpha': Number of CSA fragments. 
        'n_qubits: Number of qubits.
        'num_fragments": Number of fragments. len(grouping) + 1

Two-body tensor (tbt) variance:
    - Under scratch/tbt_variance/ folder 
    - Subfolders order: Molecule 
    - File name: molecule_geometry_wfs_ev or molecule_geometry_wfs_sq 
    - Content: 
        ev: A 4-rank tensor with elements indicating the expectation value of that corresponding spot. 
        sp: A 8-rank tensor with [i, j, k, l, a, b, c, d] indicating the expectation value of opeartors (tbt[i, j, k, l] * tbt[a, b, c, d]). 
"""
import os
import pickle
import numpy as np
import datetime


def get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y. %H:%M:%S")


def is_valid_method(method):
    """
    Checking whether the specified method exist
    Args:
        method: A str of method name
    Return:
        valid: whether the given method exists. 
    """
    valid = False
    valid_methods = ['qwc', 'fc', 'csa', 'svd', 'antic']
    if method in valid_methods:
        valid = True
    return valid


def is_valid_wfs(wfs):
    """
    Checking whether the specified wavefunction exists.
    Args:
        wfs: A str of wavefunction name
    Return:
        valid: whether the given wavefunction exists. 
    """
    valid = False
    valid_wfs = ['hf', 'fci']
    if wfs in valid_wfs:
        valid = True
    return valid


def get_variance_path(mol, wfs, method, prefix):
    """ 
    Write the path that contains the specified result 
    Args: 
        mol: Current molecule. 
        wfs: The wavefunction used. 
        method: A str of method name
        path_prefix: The prefix required to go to the top folder.
    Returns:
        path: The file path containing the specified result. 
    """
    folders = ["variances_computed", mol, wfs, method]
    path = ''
    for folder in folders:
        path = path + folder + '/'
    return prefix + path


def get_file_name(mol, geometry, wfs, method):
    fname = mol + '_' + str(float(geometry)) + '_' + wfs + '_' + method
    return fname


def prepare_path(mol, wfs, method, prefix):
    path = get_variance_path(mol, wfs, method, prefix)
    if not os.path.exists(path):
        os.makedirs(path)


def save_variance_result(mol, wfs, method, geometry, groups, variances, path_prefix="../"):
    """
    Saving the grouped term and their computed variances.

    Args:
        mol: Current molecule. 
        wfs: The wavefunction used. 
        method: A str of method name. 
        geometry: Geometry of the molecule.
        groups: list of (Fermion/Qubit)Operators that are grouped for measurement. 
        variances: the variance computed. 
        path_prefix: The prefix required to go to the top folder. 
    """
    if not is_valid_method(method):
        ValueError("Unrecognized method: {}".format(method))
    if not is_valid_wfs(wfs):
        ValueError("Unrecognized wavefunction: {}".format(wfs))

    # Making path
    path = get_variance_path(mol, wfs, method, path_prefix)
    fname = get_file_name(mol, geometry, wfs, method)
    prepare_path(mol, wfs, method, prefix='../')

    # Making dictionary to save
    result_dict = {}
    result_dict['vars'] = variances
    result_dict['grps'] = groups

    f = open(path + fname, 'wb')
    pickle.dump(result_dict, f)


def load_variance_result(mol, geometry, wfs, method, data_type, path_prefix="../"):
    """
    Loading the grouped term or their computed variances.

    Args:
        mol: Current molecule. 
        wfs: The wavefunction used. 
        method: A str of method name. 
        geometry: Geometry of the molecule.
        groups: list of (Fermion/Qubit)Operators that are grouped for measurement. 
        data_type (str): Either vars or grps. 
    Returns:
        data: Either list of variances or operators. 
    """
    if not is_valid_method(method):
        ValueError("Unrecognized method: {}".format(method))
    if not is_valid_wfs(wfs):
        ValueError("Unrecognized wavefunction: {}".format(wfs))
    if not data_type in ['vars', 'grps']:
        ValueError("Unrecognized data type: {}".format(data_type))

    # Making path
    path = get_variance_path(mol, wfs, method, prefix=path_prefix)
    fname = get_file_name(mol, geometry, wfs, method)
    if not os.path.exists(path):
        os.makedirs(path)

    # Obtaining dictionary
    f = open(path + fname, 'rb')
    result_dict = pickle.load(f)

    return result_dict[data_type]


def get_logging_path_and_name(mol, geometry, wfs, method, prefix):
    path = get_variance_path(mol, wfs, method, prefix)
    fname = get_file_name(mol, geometry, wfs, method)
    return path + fname + '.txt'


def load_fermionic_hamiltonian(mol, prefix="../"):
    with open(prefix + 'ham_lib/' + mol + '_fer.bin', 'rb') as f:
        Hf = pickle.load(f)
    return Hf


def load_interaction_hamiltonian(mol, prefix="../"):
    with open('../ham_lib/' + mol + '_int.bin', 'rb') as f:
        H = pickle.load(f)
    return H


def save_csa_sols(sol_array, grouping, n_qub, mol, geo, method, verbose=True,
                  grouping_remains=[], prefix='../', log_str_io=None, ext_dict=None):
    """ Save a CSA run. 

    Args: 
        sol_array: The array of converged CSA solutions. 
        grouping: The list of Fermionic Operators partitioned using CSA. 
        n_qub: number of qubits. 
        mol: Current molecule. 
        geo: The geometry of the molecule 
        method: The CSA method. CSA/GCSA/VCSA or others.   
        grouping_remains: The non-csa fragments needed to measured hamiltonian. 
            This can be SVD or FC fragments, usually indicated by method name. 
        prefix: The path that contains CSA solutions' folders
        log_str_io (io.StringIO): Contains log file from method get_value() 
        ext_dict (Dict[str, *]): Contains extra fields 
        verbose (bool): Whether path is printed out. 
    """
    # Making path & file name
    alpha = len(grouping) - 1
    fname = "{}_{}_{}_{}".format(mol, str(geo), method, alpha + 1)
    folder_path = prefix + "csa_computed/" + "{}/{}/".format(mol, method)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path_name = folder_path + fname

    if verbose:
        print("Date: {}".format(get_current_time()))
        print("Saving CSA solution to {}".format(file_path_name))

    # Build dictionary and save
    csa_dict = {
        'sol_array': sol_array,
        'grouping': grouping,
        'csa_alpha': alpha,
        'n_qubits': n_qub,
        'grouping_remains': grouping_remains,
        'num_fragments': alpha + len(grouping_remains) + 1
    }
    if ext_dict is not None:
        for key in ext_dict:
            csa_dict[key] = ext_dict[key]

    f = open(file_path_name, 'wb')
    pickle.dump(csa_dict, f)
    f.close()

    # Saving log file
    if log_str_io is not None:
        logf = open(file_path_name + ".txt", 'w')
        print(log_str_io.getvalue(), file=logf)
        logf.close()


def load_csa_sols(mol, geo, method, alpha, prefix='../', verbose=True):
    """ Load a CSA run. 

    Args:
        mol: Current molecule. 
        geo: The geometry of the molecule. 
        method: The CSA method. 
        alpha: The number of CSA fragments 
        verbose (bool): Whether path is printed out. 

    Returns:
        csa_dict: A dictionary containing details of converged runs. 
    """
    # Making path & file name
    fname = "{}_{}_{}_{}".format(mol, str(geo), method, alpha + 1)
    folder_path = prefix + "csa_computed/" + "{}/{}/".format(mol, method)
    file_path_name = folder_path + fname

    if verbose:
        print("Date: {}".format(get_current_time()))
        print("Loading CSA solution from {}".format(file_path_name))

    # Load and return.
    if not os.path.isfile(file_path_name):
        raise(ValueError("The file {} does not exist".format(file_path_name)))
    f = open(file_path_name, 'rb')
    csa_dict = pickle.load(f)
    f.close()
    return csa_dict


def save_restr_csa_sols(sol_array, cartans, grouping, n_qubits, mol, geo, method, path_prefix='../', log_string=None):
    """Save a restricted CSA run. 

    Args: 
        sol_array: The array of converged resritcted CSA solutions. 
        cartans: A list of NxN matrix indicating the type of cartans used. 
        grouping: The list of Fermionic Operators partitioned using CSA. 
        n_qub: number of qubits. 
        mol: Current molecule. 
        geo: The geometry of the molecule 
        method: The CSA method. CSA/GCSA/VCSA or others.   
        path_prefix: The path that contains CSA solutions' folders
        log_string (io.StringIO): Contains log file from method get_value() 
    """
    # Making path & file name
    alpha = len(grouping) - 1
    fname = "{}_{}_{}_{}".format(mol, str(geo), method, alpha + 1)
    folder_path = path_prefix + "restr_csa_computed/" + \
        "{}/{}/".format(mol, method)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Build dictionary and save
    restr_csa_dict = {
        'sol_array': sol_array,
        'cartans': cartans,
        'grouping': grouping,
        'csa_alpha': alpha,
        'n_qubits': n_qubits,
        'num_fragments': alpha + 1
    }
    f = open(folder_path + fname, 'wb')
    pickle.dump(restr_csa_dict, f)
    f.close()

    # Save log file
    if log_string is not None:
        logf = open(folder_path + fname + ".txt", 'w')
        print(log_string.getvalue(), file=logf)
        logf.close()


def load_restr_csa_sols(mol, geo, method, alpha, path_prefix='../'):
    """ Load a CSA run. 

    Args:
        mol: Current molecule. 
        geo: The geometry of the molecule. 
        method: The CSA method. 
        alpha: The number of CSA fragments 

    Returns:
        csa_dict: A dictionary containing details of converged runs. 
    """
    # Making path & file name
    fname = "{}_{}_{}_{}".format(mol, str(geo), method, alpha + 1)
    folder_path = path_prefix + "restr_csa_computed/" + \
        "{}/{}/".format(mol, method)
    file_path_name = folder_path + fname

    # Load and return.
    if not os.path.isfile(file_path_name):
        raise(ValueError("The corresponding file does not exist"))
    f = open(file_path_name, 'rb')
    csa_dict = pickle.load(f)
    f.close()
    return csa_dict


def save_tbt_variance(tensor, tensor_type, mol, geo, wfs_type, path_prefix="../", file_post_fix='',
                      log_string=None, verbose=True):
    """Save the tensor for two body tensor's variance computation. The saved tensor can be 4-rank expectation value tensor, or 8-rank squared expectation value tensor. 

    Args:
        tensor (ndarray N^4 or N^8): The computed expectation values for indicated type of tensor. 
            If type is "ev", the value of tensor[i, j, k, l] is the expectation value of 
            get_ferm_op(tbt) where tbt[i, j, k, l] = 1 with all other entries 0. 
            If type is "sq", the value of tensor[i, j, k, l, a, b, c, d] corresponds to the ev of 
            tbt[i, j, k, l] = 1 * tbt[a, b, c, d] = 1 
        tensor_type (str): Either "ev" or "sq", indicating the type of tensor saved. 
        mol (str): The type of molecule. 
        geo (float): The geometry of the molecule. 
        wfs (str): Type of wfs. Expect hf or fci. 
        path_prefix (str): The path precedes the folder of interest. 
        file_post_fix (str): The string following the file name (for splitting calculations)
        log_string (io.StringIO): Contains log file from method get_value() 
        verbose (bool): Whether path is printed out. 
    """
    # Check inputs
    if tensor_type != "ev" and tensor_type != "sq":
        raise(ValueError(
            "Incorrect type of tensor type specified. Got {}.".format(tensor_type)))

    # Make path & file name
    fname = "{}_{}_{}_{}".format(mol, float(
        geo), wfs_type, tensor_type) + file_post_fix
    folder_path = path_prefix + "scratch/tbt_variance/{}/".format(mol)
    file_path_name = folder_path + fname
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if verbose:
        print("Date: {}".format(get_current_time()))
        print("Saving TBT variance to {}".format(file_path_name))

    with open(file_path_name, 'wb') as f:
        np.save(f, tensor)

    # Save log file
    if log_string is not None:
        logf = open(folder_path + fname + ".txt", 'w')
        print(log_string.getvalue(), file=logf)
        logf.close()


def load_tbt_variance(tensor_type, mol, geo, wfs_type, path_prefix="../", file_post_fix='', verbose=True):
    """Load the tensor for two body tensor's variance computation. 

    Args:
        tensor_type (str): Either "ev" or "sq", indicating the type of tensor saved. 
        mol (str): The type of molecule. 
        geo (float): The geometry of the molecule. 
        wfs (str): Type of wfs. Expect hf or fci. 
        path_prefix (str): The path precedes the folder of interest.
        file_post_fix (str): The string following the file name (for splitting calculations)
        verbose (bool): Whether path is printed out. 

    Returns:
        tensor (ndarray N^4 or N^8): The computed expectation values for indicated type of tensor. 
    """
    # Check inputs
    if tensor_type != "ev" and tensor_type != "sq":
        raise(ValueError(
            "Incorrect type of tensor type specified. Got {}.".format(tensor_type)))

    # Make path & file name
    fname = "{}_{}_{}_{}".format(mol, float(
        geo), wfs_type, tensor_type) + file_post_fix
    folder_path = path_prefix + "scratch/tbt_variance/{}/".format(mol)
    file_path_name = folder_path + fname
    if not os.path.exists(folder_path) or not os.path.isfile(file_path_name):
        raise(ValueError(
            "The file '{}' does not exist. There's no results to load from. ".format(file_path_name)))

    if verbose:
        print("Date: {}".format(get_current_time()))
        print("Loading TBT variance from {}".format(file_path_name))

    # Load from tensor
    with open(file_path_name, 'rb') as f:
        tensor = np.load(f)
    return tensor


def save_ground_state(gs, mol, ham_form, path_prefix="../", log_str_io=None, verbose=True):
    """Save the ground state of Hamiltonians 
    Args: 
        gs (sparse or vector): The ground state obtained from openfermion 
        mol (str): The type of molecule. 
        ham_form (str): The form of the Hamiltonian from which ground state is obtained. BK or FM. 
        path_prefix (str): The path precedes the folder of interest.
        log_str_io (io.StringIO): Contains log file from method get_value() 
        verbose (bool): Whether path & date is printed out. 
    """
    fname = "{}_{}".format(mol, ham_form)
    folder_path = path_prefix + "scratch/ground_states/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path_name = folder_path + fname

    if verbose:
        print("Date: {}".format(get_current_time()))
        print("Saving ground state to {}".format(file_path_name))

    with open(file_path_name, "wb") as f:
        pickle.dump(gs, f)

    # Save log file
    if log_str_io is not None:
        with open(file_path_name + ".txt", 'w') as logf:
            print(log_str_io.getvalue(), file=logf)


def load_ground_state(mol, ham_form, path_prefix="../", verbose=True):
    """Save the ground state of Hamiltonians 
    Args: 
        mol (str): The type of molecule. 
        ham_form (str): The form of the Hamiltonian from which ground state is obtained. BK or FM. 
        path_prefix (str): The path precedes the folder of interest.
        verbose (bool): Whether path & date is printed out. 
    Returns:
        gs (sparse or vector): The ground state obtained from openfermion 
    """
    fname = "{}_{}".format(mol, ham_form)
    folder_path = path_prefix + "scratch/ground_states/"
    file_path_name = folder_path + fname
    if not os.path.exists(folder_path) or not os.path.isfile(file_path_name):
        raise(ValueError("The file '{}' does not exist. ".format(file_path_name)))

    if verbose:
        print("Date: {}".format(get_current_time()))
        print("Loading ground state from {}".format(file_path_name))

    with open(file_path_name, 'rb') as f:
        gs = pickle.load(f)
    return gs


def save_pauliword_ev(ev_dict, mol, tf, wfs_type, geo=1.0, prefix='../', log_string=None, verbose=True):
    """Save the dictionary that maps tuple of pauli-words to their expectation given molecule and wavefunction type. 
    Args:
        ev_dict (Dict[tuple, float]): The dictionary where ev_dict[pw_tuple] = expectation(pw) given specified wavefunction. 
        mol (str): The molecule's name. 
        tf (str): The transformation used. bk for bravyi-kitaev and jw for jordan-wigner 
        wfs_type (str): The type of wavefunction. fci or hf. 
        geo (float): The geometry of the molecule. 
        prefix (str): The path that contains scratch/ folder
        log_string (io.StringIO): Contains log file from method get_value() 
        verbose (bool): Whether path is printed out. 
    """
    # Making path & file name
    fname = "{}_{}_{}_{}".format(mol, geo, tf, wfs_type)
    folder_path = prefix + "scratch/pw_expectation/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path_name = folder_path + fname

    if verbose:
        print("Date: {}".format(get_current_time()))
        print("Saving expectation values to {}".format(file_path_name))

    # Save the dictionary
    with open(file_path_name, 'wb') as f:
        pickle.dump(ev_dict, f)

    # Save log file
    if log_string is not None:
        with open(file_path_name + ".txt", 'w') as f:
            print(log_string.getvalue(), file=f)


def load_pauliword_ev(mol, tf, wfs_type, geo=1.0, prefix='../', verbose=True):
    """Load the dictionary that maps tuple of pauli-words to their expectation given molecule and wavefunction type. 
    Args:
        mol (str): The molecule's name. 
        tf (str): The transformation used. bk for bravyi-kitaev and jw for jordan-wigner 
        wfs_type (str): The type of wavefunction. fci or hf. 
        geo (float): The geometry of the molecule. 
        prefix (str): The path that contains scratch/ folder
        verbose (bool): Whether path is printed out. 

    Returns:
        ev_dict (Dict[tuple, float]): The dictionary where ev_dict[pw_tuple] = expectation(pw) given specified wavefunction. 
    """
    # Making path & file name
    fname = "{}_{}_{}_{}".format(mol, geo, tf, wfs_type)
    folder_path = prefix + "scratch/pw_expectation/"
    file_path_name = folder_path + fname

    if verbose:
        print("Date: {}".format(get_current_time()))
        print("Loading expectation values from {}".format(file_path_name))

    # Load and return.
    if not os.path.isfile(file_path_name):
        raise(ValueError("The file {} does not exist".format(file_path_name)))
    with open(file_path_name, 'rb') as f:
        ev_dict = pickle.load(f)
    return ev_dict
