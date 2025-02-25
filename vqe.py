import anderson_impurity_model as aim
from anderson_impurity_model import AndersonImpurityModel
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import binom
from openfermion.utils import count_qubits
from openfermion.ops import QubitOperator
from qulacs import QuantumState, GeneralQuantumOperator
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from qulacs.state import inner_product
from qulacs.gate import P0, P1, X, Z


def continued_fraction(
        w: NDArray[np.complex128],
        a: NDArray[np.complex128],
        b: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """
    Given a series of a/b Krylov basis coefficients and a linear spaced array of complex values, recursively construct and return
    a portion of the complex-valued continued fraction representation Green's function.
    :param w: Complex, linearly-spaced frequency array containing broadening term.
    :param a: Krylov basis coefficients - Diagonal entries in the tridiagonalized Hamiltonian.
    :param b: Krylov basis coefficients - Off-diagonal entries in the tridiagonalized Hamiltonian.
    :return: Complex-valued Green's function component.
    """

    assert len(a) == len(b)
    if len(a) == 1:
        return 1. / (w - a[0])
    else:
        return 1. / ((w - a[0]) - b[1] ** 2 * continued_fraction(w, a[1:], b[1:]))


def lanczos_cost_function(
    params: NDArray,
    h: GeneralQuantumOperator,
    ut: list[QuantumState],
    b: list[np.complex128],
    i: int,
    n_layers: int,
    initial_occupations_indices: list[int],
    connected_graphs: dict,
    compilation: str,
) -> float:
    """
    Given specified parameters, construct a variational ansatz and return the value of the cost function with e_n contributions.
    :param params: VQE angles
    :param h: Qulacs Hamiltonian
    :param ut: List of qulacs Krylov vectors.
    :param b: Array of Krylov b coefficients.
    :param i: Lanczos iteration index - always starts at 1.
    :param n_layers: Number of layers of the ansatz.
    :param initial_occupations_indices: Indices to place initial 'ones' (X gates) in ansatz.
    :param connected_graphs: Dictionary of graph objects used for representing AIM.
    :param compilation: Should mostly just be "generic".
    :return: Value of the cost function with e_n contributions.
    """
    symm_qulacs = aim.SymmQulacsVqeEmulator(h)
    n_qubits = h.get_qubit_count()
    ws = QuantumState(n_qubits)
    qc_ansatz = symm_qulacs.all_symmetry_ansatzae(
        theta=params, n_layers=n_layers, initial_occupations_indices=initial_occupations_indices,
        connected_graphs=connected_graphs, compilation=compilation)
    ws.set_zero_state()
    qc_ansatz.update_quantum_state(ws)
    # First term in the cost function will be to enforce that b[i] is the off-diagonal term in the tridiagonalized Hamiltonian.
    e1 = abs(h.get_transition_amplitude(ws, ut[i - 1]) - b[i])

    # e2, e3, e4 are all constructed by enforcing that the overlap of the trial wavefunction ws has zero overlap
    # with the previously constructed Krylov basis states.
    e2 = abs(inner_product(ws, ut[i - 1]))
    if i > 1:
        e3 = abs(inner_product(ws, ut[i - 2]))
    else:
        e3 = 0
    # Create additional term that enforces orthogonality for each new Krylov vector - overlap for each j < i = 0
    # Note how e4 can be comprised of multiple terms when i > 4
    e4 = 0
    # Avoid overlap vector with initial Krylov vector ut[0]
    if i == 4:
        e4 += abs(inner_product(ws, ut[1])) ** 2
    if i >= 5:
        # Sliding scale - calculate just 3 overlap terms for each new iteration
        for j in range(3, 6):  # Note: Change the range to (3, i+1) for full range implementation
            k_overlap = abs(inner_product(ws, ut[i - j])) ** 2
            e4 += k_overlap

    return e1 ** 2 + (e2 ** 2 + e3 ** 2) + e4


def create_qulacs_hamiltonian(
        qubit_hamiltonian: QubitOperator,
        vqe_gs_energy: float
) -> tuple[GeneralQuantumOperator, GeneralQuantumOperator, int]:
    """
    Create a qulacs Hamiltonian for the variational states.
    :param qubit_hamiltonian: Hamiltonian in the form of an OpenFermion QubitOperator.
    :param vqe_gs_energy: Ground state energy as found by VQE.
    :return: Hamiltonian and Hamiltonian squared as qulacs quantum operators and the number of qubits.
    """
    qubit_hamiltonian_shifted = qubit_hamiltonian - vqe_gs_energy  # Shifts ground state to zero for Hamiltonian
    qulacs_ham = create_quantum_operator_from_openfermion_text(f"{qubit_hamiltonian_shifted}")
    qulacs_ham_2 = create_quantum_operator_from_openfermion_text(f"{qubit_hamiltonian_shifted ** 2}")
    n_qubits = count_qubits(qubit_hamiltonian_shifted)

    return qulacs_ham, qulacs_ham_2, n_qubits


def solve_vqe(
        test_model: AndersonImpurityModel,
        vqe_depth: int,
        optimizer: str,
        gs_gtol: float, 
        checkpoint_file: str, 
        results_file: str,
        display: bool,
) -> tuple[float, int, int, NDArray, dict]:
    """
    Solves for ground state using VQE over all possible spin/charge sectors.
    :param test_model: AndersonImpurityModel
    :param vqe_depth: Number of layers for ground state optimizer and ansatz.
    :param optimizer: Type of optimizer for VQE.
    :param gs_gtol: Gradient norm tolerance for the ground state optimizer.
    :return: Ground state energy, charge and spin as ints, final parameters from running VQE, and dictionary of selected results.
    """
    # TODO: Have dynamic setting of display instead of hardcoding it to False
    symm_qulacs = aim.SymmQulacsVqeEmulator(test_model)
    symmetric_ansatz_test_results = aim.symmetric_ansatz_test(symm_qulacs, model_hamiltonian=test_model, n_layers=vqe_depth,
                                                              optimizer=optimizer, display=display, gtol=gs_gtol, checkpoint_file=checkpoint_file, 
                                                              results_file=results_file)
    minimum_sector = symmetric_ansatz_test_results["minimum_sector"]
    minimum_energy = symmetric_ansatz_test_results["minimum_energy"]
    minimum_result = symmetric_ansatz_test_results["minimum_result"]
    min_sector_grad_history = symmetric_ansatz_test_results["min_sector_grad_history"]
    min_sector_cost_history = symmetric_ansatz_test_results["min_sector_cost_history"]
    nit_total = symmetric_ansatz_test_results["nit_total"]
    nfev_total = symmetric_ansatz_test_results["nfev_total"]
    njev_total = symmetric_ansatz_test_results["njev_total"]

    # Unpack VQE Results
    vqe_spin = int(np.round(minimum_sector[0]))
    vqe_charge = int(np.round(minimum_sector[1]))
    vqe_nu = (vqe_charge + vqe_spin) // 2
    vqe_nd = (vqe_charge - vqe_spin) // 2
    minimum_angles = minimum_result.x

    ###
    record_keeping = {}
    record_keeping["local_vqe_success"] = minimum_result["success"]
    record_keeping["local_vqe_nparams"] = len(minimum_angles)
    # Unpack OptimizeResults from local vqe and save
    record_keeping["local_vqe_nfev"] = minimum_result["nfev"]
    record_keeping["local_vqe_njev"] = minimum_result["njev"]
    record_keeping["local_vqe_nit"] = minimum_result["nit"]
    # Save total nit, nfev, njev for the total over each spin/charge sector
    record_keeping["local_vqe_nfev_total"] = nfev_total
    record_keeping["local_vqe_njev_total"] = njev_total
    record_keeping["local_vqe_nit_total"] = nit_total
    record_keeping["min_sector_grad_history"] = str(min_sector_grad_history)
    record_keeping["min_sector_cost_history"] = str(min_sector_cost_history)
    record_keeping["vqe_gs_energy"] = minimum_energy
    record_keeping["vqe_charge"] = vqe_charge
    record_keeping["vqe_spin"] = vqe_spin
    record_keeping["vqe_nu"] = vqe_nu
    record_keeping["vqe_nd"] = vqe_nd
    ###

    return minimum_energy, vqe_charge, vqe_spin, minimum_angles, record_keeping


def qulacs_to_python_ordering(
        qulacs_state: NDArray[np.complex128],
        n_qubits: int
) -> NDArray[np.complex128]:
    """
    Given a qulacs state as an NDArray and number of qubits, convert the endianness and return a new NDArray such that 
    an overlap calculation can be performed.
    :param qulacs_state: Numpy array converted from a qulacs quantum state.
    :param n_qubits: Number of qubits generated by the AIM.
    :return:
    """
    new_state = np.zeros(qulacs_state.shape, dtype=complex)
    for ii in range(qulacs_state.shape[0]):
        format_string = "{0:0" + str(n_qubits) + "b}"
        b_string = format_string.format(ii)
        b_reversed = b_string[::-1]
        new_ii = int(b_reversed, 2)
        new_state[new_ii] = qulacs_state[ii]

    return new_state


def get_initial_occupations_indices(
        up_qubit_indices: list[int],
        down_qubit_indices: list[int],
        n_up: int,
        n_down: int
) -> list[int]:
    """
    Given params, return a list of equispaced indices on which to apply initial bit flips in constructing the SPA.
    *Assumes a linear chain for each register.
    :param up_qubit_indices: List of spin-up register qubit indices.
    :param down_qubit_indices: List of spin-down register qubit indices.
    :param n_up: Up register excitation number.
    :param n_down: Down register excitation number.
    :return: List of indices on which to apply bit flips.
    """
    most_equispaced_indices = np.linspace(0, len(up_qubit_indices) - 1, n_up, dtype=int).tolist()
    initial_occupations_indices = [up_qubit_indices[i] for i in most_equispaced_indices]

    # Next two lines take care of spin down sector
    most_equispaced_indices = np.linspace(0, len(down_qubit_indices) - 1, n_down, dtype=int).tolist()
    initial_occupations_indices.extend([down_qubit_indices[i] for i in most_equispaced_indices])
    initial_occupations_indices.sort()

    return initial_occupations_indices


def construct_vqe_gs(
    n_qubits: int,
    qulacs_ham: GeneralQuantumOperator,
    up_qubit_indices: list[int],
    down_qubit_indices: list[int],
    vqe_depth: int,
    connected_graphs: dict,
    minimum_angles: NDArray,
    vqe_nu: int,
    vqe_nd: int,
    **kwargs,
) -> tuple[QuantumState, NDArray[np.complex128]]:
    """
    Constructs the variationally found ground state from the minimum angles previously found in VQE. This is done such 
    that the full wavefuncion need not be stored in memory throughout the experiment, rather just the list of minimum 
    angles for the ansatz circuit. Returns the QuantumState qulacs representation and the NDArray representation.
    :param test_model: AndersonImpurityModel
    :param up_qubit_indices: List of spin-up register qubit indices.
    :param down_qubit_indices: List of spin-down register qubit indices.
    :param vqe_depth: Number of layers for ansatz.
    :param connected_graphs: Dictionary of graph objects used for representing AIM.
    :param minimum_angles: The final parameters from running VQE.
    :param vqe_nu: Up register excitation number.
    :param vqe_nd: Down register excitation number.
    :param vqe_gs_energy: Ground state energy as found by VQE.
    :return: Ground state
    """
    # Create the ground state from VQE
    ws = QuantumState(n_qubits)
    ws.set_zero_state()
    symm_qulacs = aim.SymmQulacsVqeEmulator(qulacs_ham)

    # Get initial occupations indices for the SPA construction
    initial_occupations_indices = get_initial_occupations_indices(up_qubit_indices, down_qubit_indices, vqe_nu, vqe_nd)
    qc_ansatz = symm_qulacs.all_symmetry_ansatzae(theta=minimum_angles, n_layers=vqe_depth,
                                                  initial_occupations_indices=initial_occupations_indices,
                                                  connected_graphs=connected_graphs)
    qc_ansatz.update_quantum_state(ws)

    # numpy vector of qulacs wavefunction
    approx_gs = ws.get_vector()

    new_approx_gs = qulacs_to_python_ordering(approx_gs, n_qubits)
    del approx_gs

    return ws, new_approx_gs


def vqe_gf_pm(
        g_vqe: NDArray[np.complex128],
        phi_norm: float,
        w: NDArray[np.complex128],
        **phi_lanczos_kwargs
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], dict, NDArray[np.complex128]]:
    """
    Find the Green's function based on variational Lanczos iterations. The pm appended is to denote it can be used both 
    for the phi_plus and phi_minus calculations.
    :param g_vqe: Green's function numpy array, initially np.zeroes with the same length as w.
    :param phi_norm: Norm of phi minus or phi plus.
    :param w: Complex, linearly-spaced frequency array containing broadening term.
    :param phi_lanczos_kwargs: phi plus and minus kwargs for Lanczos iterations.
    :return: ..., ..., dictionary with OptimizeResult data for each Lanczos iteration, and GF based on vqe.
    """
    # If the norm of the associated initial Krylov state is 0 then it has no contribution in the Green's function
    if not np.isclose(phi_norm, 0, ):
        a_vqe, b_vqe, lanczos_iterations_results = vqe_ideal_lanczos_iterations(**phi_lanczos_kwargs)
        # NOTE because we save full 'a' and 'b' Lanczos arrays, we can call 'continued faction' as a numerical fun.
        # w/o re-computing each time (just compute a,b once)
        g_vqe += phi_norm ** 2 * continued_fraction(w, a_vqe, b_vqe)
    else:
        a_vqe = np.zeros((phi_lanczos_kwargs["niter"] + 1), dtype=np.complex128)
        b_vqe = np.zeros((phi_lanczos_kwargs["niter"] + 1), dtype=np.complex128)
        lanczos_iterations_results = {}

    return a_vqe, b_vqe, lanczos_iterations_results, g_vqe


def vqe_ideal_lanczos_iterations(
    niter: int,
    u: QuantumState,
    h: GeneralQuantumOperator,
    h2: GeneralQuantumOperator,
    charge_sec: int,
    spin_sec: int,
    n_layers: int,
    up_qubit_indices: list[int],
    down_qubit_indices: list[int],
    connected_graphs: dict,
    conv_tol: float = 1e-6,
    optimizer: str = "COBYLA",
    maxiter: int = 1e6,
    gf_gtol: float = 5e-5,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], dict]:
    """
    Given parameters specified, run noiseless simulation of VQE returning the Krylov basis coefficients computed by the
    Lanczos iterations.
    :param niter: The total number of Lanczos iterations to perform given that there is no early termination.
    :param u: Initial Krylov vector.
    :param h: Hamiltonian qulacs quantum operator.
    :param h2: Hamiltonian squared qulacs quantum operator.
    :param charge_sec: The charge sector the ground state resides in as found by VQE.
    :param spin_sec: The spin sector the ground state resides in as found by VQE.
    :param n_layers: Number of layers for the lanczos vectors optimizer and ansatz.
    :param up_qubit_indices: List of spin-up register qubit indices.
    :param down_qubit_indices: List of spin-down register qubit indices.
    :param connected_graphs: Dictionary of graph objects used for representing AIM.
    :param conv_tol: Broadly defined convergence tolerance for scipy optimizers.
    :param optimizer: Type of optimizer for VQE.
    :param maxiter: Maximum number of iterations performed by scipy optimizer (1 iteration may contain multiple function
    evaluations).
    :param gtol: Gradient norm tolerance for the VQE optimizer.
    :return: Krylov basis coefficients a and b within NDArrays and dictionary with OptimizeResult data for each Lanczos
    iteration.
    """
    n_qubits = h.get_qubit_count()
    n_edges = connected_graphs["graph_full"].number_of_edges()
    n_up = (charge_sec + spin_sec) // 2
    n_down = (charge_sec - spin_sec) // 2
    initial_occupations_indices = get_initial_occupations_indices(up_qubit_indices, down_qubit_indices, n_up, n_down)
    compilation = "generic"

    a = np.zeros((niter + 1), dtype=np.complex128)  # Holds a coefficients
    b = np.zeros((niter + 1), dtype=np.complex128)  # Holds b coefficients
    ut = [u]  # Stores first Krylov vector in new list
    b[0] = 0 + 0j  # First b coefficient always 0
    a[0] = h.get_expectation_value(ut[0])  # Initial a coefficient

    # Key/value will be iteration number for Lanczos iteration/another dictionary containing results from the optimizer
    lanczos_iterations_results_dict = {}
    for i in range(1, niter + 1):
        # First calculate b[i]
        if i == 1:
            b[i] = np.sqrt(h2.get_expectation_value(ut[i - 1]) - (a[i - 1] ** 2))
        else:
            b[i] = np.sqrt(h2.get_expectation_value(ut[i - 1]) - (a[i - 1] ** 2) - (b[i - 1] ** 2))
        if np.isclose(b[i], 0. + 0.j):
            break

        # Next calculate new Krylov vector. This is where VQE is used.
        opt_args = (h, ut, b, i, n_layers, initial_occupations_indices, connected_graphs, compilation)
        n_params = n_layers * (n_edges + n_qubits)
        lanczos_results = {}
        cost_histories = []
        grad_norms = []

        # For Rzz, Givens, and Rzs the domain for theta should be [0,2*np.pi)
        theta_0 = np.random.uniform(low=0, high=2 * np.pi, size=n_params)

        # Grab the first cost function w/the original params
        cost_histories.append(
            lanczos_cost_function(theta_0, h, ut, b, i, n_layers, initial_occupations_indices, connected_graphs, compilation))

        # Perform an optimization to find a new basis vector
        lanczos_result = minimize(
            lanczos_cost_function, theta_0, args=opt_args, method=optimizer, tol=conv_tol,
            options={"maxiter": maxiter, "gtol": gf_gtol, "eps": 1e-7})

        # dict where key/value pair are the minimum value of the cost function/OptimizeResult of scipy.optimize
        lanczos_results[lanczos_result.fun] = lanczos_result

        # Retry optimization upon optimizer failure
        retries = 5
        if not lanczos_result.success:
            for retry in range(retries):
                # reseed starting point and re-initialize cost history/grad norms
                cost_histories = []
                grad_norms = []
                theta_0 = np.random.uniform(low=0, high=2 * np.pi, size=n_params)
                lanczos_result = minimize(lanczos_cost_function, theta_0, args=opt_args, method=optimizer, tol=conv_tol,
                                          options={"maxiter": maxiter, "gtol": gf_gtol, "eps": 1e-7})
                lanczos_results[lanczos_result.fun] = lanczos_result
                if lanczos_result.success:
                    break
                # Of the re-tried failed optimizations, propagate forward the result with the lowest cost function value
                else:
                    min_lanczos_fun = min(lanczos_results.keys())
                    min_lanczos_result = lanczos_results[min_lanczos_fun]
                    lanczos_result = min_lanczos_result

        # After each iteration, create and save the newly found Krylov Vector
        symm_qulacs = aim.SymmQulacsVqeEmulator(h)
        ws = QuantumState(n_qubits)
        qc_ansatz = symm_qulacs.all_symmetry_ansatzae(
            theta=lanczos_result.x, n_layers=n_layers, initial_occupations_indices=initial_occupations_indices,
            connected_graphs=connected_graphs, compilation=compilation)
        ws.set_zero_state()
        qc_ansatz.update_quantum_state(ws)

        # Normalize the state:
        squared_norm = ws.get_squared_norm()
        ws.normalize(squared_norm)
        ut.append(ws.copy())

        # Next calculate <xi|H|xi>
        a[i] = h.get_expectation_value(ut[i])

        lanczos_iterations_results_dict[i] = {
            "message": lanczos_result.message,
            "success": lanczos_result.success,
            "status": lanczos_result.status,
            "fun": lanczos_result.fun,
            "cost_funs": cost_histories,
            "grad_norms": grad_norms,
        }
    return a, b, lanczos_iterations_results_dict


def vqe_krylov_zero_state(
        vqe_gs: QuantumState,
        impurity_orbital: int,
        n_qubits: int
) -> tuple[QuantumState, NDArray[np.complex128], float, QuantumState, NDArray[np.complex128], float, dict]:
    """
    Create and return the initial Krylov vectors to start the Lanczos Iterations.
    :param vqe_gs: Ground state as found by VQE.
    :param impurity_orbital: Index of the impurity orbital.
    :param n_qubits: Number of qubits generated by the AIM.
    :return: Initial Krylov vectors and norm and dictionary of selected results.
    """
    # Phi minus
    phi_minus = vqe_gs.copy()

    # Loop deals with -W parities
    for ii in range(0, impurity_orbital):
        Z(ii).update_quantum_state(phi_minus)
    P1(impurity_orbital).update_quantum_state(phi_minus)
    X(impurity_orbital).update_quantum_state(phi_minus)
    phi_minus_norm = np.sqrt(phi_minus.get_squared_norm())
    phi_minus.normalize(phi_minus.get_squared_norm())

    # Cast to python ordering to check overlap
    new_phi_minus = qulacs_to_python_ordering(phi_minus.get_vector(), n_qubits)

    # Phi plus
    phi_plus = vqe_gs.copy()

    # Loop deals with J-W parities
    for ii in range(0, impurity_orbital):
        Z(ii).update_quantum_state(phi_plus)
    P0(impurity_orbital).update_quantum_state(phi_plus)
    X(impurity_orbital).update_quantum_state(phi_plus)
    phi_plus_norm = np.sqrt(phi_plus.get_squared_norm())
    phi_plus.normalize(phi_plus.get_squared_norm())

    # Cast to python ordering to check overlap
    new_phi_plus = qulacs_to_python_ordering(phi_plus.get_vector(), n_qubits)

    ###
    record_keeping = {}
    record_keeping["phi_plus_norm"] = phi_plus_norm
    record_keeping["phi_plus_normalized"] = np.sqrt(phi_plus.get_squared_norm())
    record_keeping["phi_minus_norm"] = phi_minus_norm
    record_keeping["phi_minus_normalized"] = np.sqrt(phi_minus.get_squared_norm())
    ###

    return phi_minus, new_phi_minus, phi_minus_norm, phi_plus, new_phi_plus, phi_plus_norm, record_keeping


def solve_vqe_gs(
    test_model: AndersonImpurityModel,
    qubit_hamiltonian: QubitOperator,
    connected_graphs: dict,
    up_qubit_indices: list[int],
    down_qubit_indices: list[int],
    vqe_depth: int,
    optimizer: str,
    gs_gtol: float,
    display: bool,
    **kwargs,
) -> tuple[float, NDArray[np.complex128], QuantumState, NDArray, dict]:
    """
    Given an AndersonImpurityModel, solve for the ground state energy, quantum state, and wavefunction as well as the 
    charge and spin sector that it resides in. Calls solve_vqe() as a subroutine.
    :param test_model: AndersonImpurityModel
    :param qubit_hamiltonian: Hamiltonian in the form of an OpenFermion QubitOperator.
    :param connected_graphs: Dictionary of graph objects used for representing AIM.
    :param up_qubit_indices: List of spin-up register qubit indices.
    :param down_qubit_indices: List of spin-down register qubit indices.
    :param vqe_depth: Number of layers for ground state optimizer and ansatz.
    :param optimizer: Type of optimizer for VQE.
    :param gs_gtol: Gradient norm tolerance for the ground state optimizer.
    :param display: Bool to display progression of algorithm information. 
    :return: Ground state energy, quantum state, and wavefunction, charge and spin, Hamiltonian and Hamiltonian squared quantum
    operators, number of qubits, and dictionary of selected results.
    """
    checkpoint_file = kwargs['checkpoint_file']
    results_file = kwargs['results_file']
    vqe_gs_energy, _, _, minimum_angles, record_keeping = solve_vqe(test_model, vqe_depth, optimizer, gs_gtol, checkpoint_file, results_file, display)
    del test_model

    qulacs_ham, _, n_qubits = create_qulacs_hamiltonian(qubit_hamiltonian, vqe_gs_energy)
    vqe_gs_qs, vqe_gs = construct_vqe_gs(
        n_qubits, qulacs_ham, up_qubit_indices, down_qubit_indices, vqe_depth, connected_graphs, minimum_angles, **record_keeping)

    return vqe_gs_energy, vqe_gs, vqe_gs_qs, minimum_angles, record_keeping


def construct_vqe_krylov_dimensions(
        up_idx: list,
        impurity_orbital: int,
        n_orbitals: int,
        vqe_charge: int,
        vqe_spin: int
) -> tuple[int, int, int, int, int, int, int, int, int, int, dict]:
    """
    Create and return the Krylov dimensions for both cases of phi plus and phi minus for the variational case.
    :param up_idx: Spin-up register indices.
    :param impurity_orbital: Index of the impurity orbital.
    :param n_orbitals: Number of orbitals generated by the AIM.
    :param vqe_charge: Charge eigenvalue for the VQE ground state.
    :param vqe_spin: Spin eigenvalue for the VQE ground state.
    :return: VQE charge-spin plus/minus sectors, VQE charge plus/minus sectors, VQE spin plus/minus sectors,
    ideal plus/minus Krylov dimensions, and dictionary of selected results.
    """
    # Phi minus sector and Krylov dimension
    vqe_charge_minus = vqe_charge - 1
    if impurity_orbital in up_idx:
        vqe_spin_minus = vqe_spin - 1
    # clause left in for sym. breaking
    else:
        vqe_spin_minus = vqe_spin + 1

    vqe_nu_minus = (vqe_charge_minus + vqe_spin_minus) // 2
    vqe_nd_minus = (vqe_charge_minus - vqe_spin_minus) // 2
    vqe_krylov_ideal_dim_minus = int(np.round(binom(n_orbitals, vqe_nu_minus) * binom(n_orbitals, vqe_nd_minus)))

    # Phi plus sector and Krylov dimension
    vqe_charge_plus = vqe_charge + 1
    if impurity_orbital in up_idx:
        vqe_spin_plus = vqe_spin + 1
    # clause left in for sym. breaking
    else:
        vqe_spin_plus = vqe_spin - 1
    vqe_nu_plus = (vqe_charge_plus + vqe_spin_plus) // 2
    vqe_nd_plus = (vqe_charge_plus - vqe_spin_plus) // 2
    vqe_krylov_ideal_dim_plus = int(np.round(binom(n_orbitals, vqe_nu_plus) * binom(n_orbitals, vqe_nd_plus)))

    ###
    record_keeping = {}
    record_keeping["vqe_krylov_ideal_dim_minus"] = vqe_krylov_ideal_dim_minus
    record_keeping["vqe_nu_minus"] = vqe_nu_minus
    record_keeping["vqe_nd_minus"] = vqe_nd_minus
    record_keeping["vqe_krylov_ideal_dim_plus"] = vqe_krylov_ideal_dim_plus
    record_keeping["vqe_nu_plus"] = vqe_nu_plus
    record_keeping["vqe_nd_plus"] = vqe_nd_plus
    ###

    return (vqe_nu_minus, vqe_nd_minus, vqe_charge_minus, vqe_spin_minus, vqe_krylov_ideal_dim_minus,
            vqe_nu_plus, vqe_nd_plus, vqe_charge_plus, vqe_spin_plus, vqe_krylov_ideal_dim_plus, record_keeping)


def calculate_gf_vqe(
    qulacs_ham: GeneralQuantumOperator,
    qulacs_ham_2: GeneralQuantumOperator,
    phi_minus: QuantumState,
    phi_plus: QuantumState,
    vqe_charge_minus: int,
    vqe_charge_plus: int,
    vqe_spin_minus: int,
    vqe_spin_plus: int,
    up_qubit_indices: list[int],
    down_qubit_indices: list[int],
    connected_graphs: dict,
    w: NDArray[np.complex128],
    vqe_krylov_ideal_dim_minus: int,
    vqe_krylov_ideal_dim_plus: int,
    phi_minus_norm: float,
    phi_plus_norm: float,
    vqe_depth: int,
    optimizer: str,
    conv_tol: float,
    maxiters: int,
    gf_gtol: float,
    **kwargs,
) -> tuple[NDArray[np.complex128], dict]:
    """
    Find the Green's function based on variational Lanczos iterations for phi plus or minus.
    :param qulacs_ham: Hamiltonian qulacs quantum operator.
    :param qulacs_ham_2: Hamiltonian squared qulacs quantum operator.
    :param phi_minus: Initial Krylov Vector for phi minus.
    :param phi_plus: Initial Krylov Vector for phi plus.
    :param vqe_charge_minus: The charge sector the ground state resides in as found by VQE for phi minus.
    :param vqe_charge_plus: The charge sector the ground state resides in as found by VQE for phi plus.
    :param vqe_spin_minus: The spin sector the ground state resides in as found by VQE for phi minus.
    :param vqe_spin_plus: The spin sector the ground state resides in as found by VQE for phi plus.
    :param up_qubit_indices: List of spin-up register qubit indices.
    :param down_qubit_indices: List of spin-down register qubit indices.
    :param connected_graphs: Dictionary of graph objects used for representing AIM.
    :param w: Complex, linearly-spaced frequency array containing broadening term.
    :param vqe_krylov_ideal_dim_minus: Ideal Krylov dimension for phi minus.
    :param vqe_krylov_ideal_dim_plus: Ideal Krylov dimension for phi plus.
    :param phi_minus_norm: Norm of phi minus.
    :param phi_plus_norm: Norm of phi plus.
    :param vqe_depth: Number of layers for the lanczos vector optimizer and ansatz.
    :param optimizer: Type of optimizer for VQE.
    :param conv_tol: Broadly defined convergence tolerance for scipy optimizers.
    :param maxiters: Max iteration number used as stopping criteria for optimizers.
    :param gf_gtol: Gradient norm tolerance for the VQE optimizer.
    :return: Green's function generated by VQE and dictionary of selected results.
    """
    # Since most of the Lanczos keyword arguments are shared betwen the +/- states, it seems prudent to keep them together
    lanczos_iteration_minus_kwargs = {
        "h": qulacs_ham,
        "h2": qulacs_ham_2,
        "n_layers": vqe_depth,
        "up_qubit_indices": up_qubit_indices,
        "down_qubit_indices": down_qubit_indices,
        "connected_graphs": connected_graphs,
        "conv_tol": conv_tol,
        "optimizer": optimizer,
        "maxiter": maxiters,
        "gf_gtol": gf_gtol,
    }
    lanczos_iteration_plus_kwargs = {
        "h": qulacs_ham,
        "h2": qulacs_ham_2,
        "n_layers": vqe_depth,
        "up_qubit_indices": up_qubit_indices,
        "down_qubit_indices": down_qubit_indices,
        "connected_graphs": connected_graphs,
        "conv_tol": conv_tol,
        "optimizer": optimizer,
        "maxiter": maxiters,
        "gf_gtol": gf_gtol,
    }

    phi_minus_lanczos_kwargs = dict({"niter": vqe_krylov_ideal_dim_minus, "u": phi_minus, "charge_sec": vqe_charge_minus,
                                     "spin_sec": vqe_spin_minus}, **lanczos_iteration_minus_kwargs)
    phi_plus_lanczos_kwargs = dict({"niter": vqe_krylov_ideal_dim_plus, "u": phi_plus, "charge_sec": vqe_charge_plus,
                                    "spin_sec": vqe_spin_plus}, **lanczos_iteration_plus_kwargs)

    g_vqe_plus = np.zeros(len(w), dtype=np.complex128)
    g_vqe_minus = np.zeros(len(w), dtype=np.complex128)
    a_minus_vqe, b_minus_vqe, lanczos_iterations_results_minus, g_vqe_minus = vqe_gf_pm(g_vqe_minus, phi_minus_norm, -w,
                                                                                        **phi_minus_lanczos_kwargs)
    a_plus_vqe, b_plus_vqe, lanczos_iterations_results_plus, g_vqe_plus = vqe_gf_pm(g_vqe_plus, phi_plus_norm, w,
                                                                                    **phi_plus_lanczos_kwargs)

    # Combine the two contributions of the VQE Green's function
    g_vqe = g_vqe_plus - g_vqe_minus

    ###
    record_keeping = {}
    record_keeping["lanczos_iteration_results_minus_vqe"] = lanczos_iterations_results_minus
    record_keeping["a_minus_vqe_real"] = [a.real for a in a_minus_vqe]
    record_keeping["a_minus_vqe_imag"] = [a.imag for a in a_minus_vqe]
    record_keeping["b_minus_vqe_real"] = [b.real for b in b_minus_vqe]
    record_keeping["b_minus_vqe_imag"] = [b.imag for b in b_minus_vqe]
    record_keeping["lanczos_iteration_results_plus_vqe"] = lanczos_iterations_results_plus
    record_keeping["a_plus_vqe_real"] = [a.real for a in a_plus_vqe]
    record_keeping["a_plus_vqe_imag"] = [a.imag for a in a_plus_vqe]
    record_keeping["b_plus_vqe_real"] = [b.real for b in b_plus_vqe]
    record_keeping["b_plus_vqe_imag"] = [b.imag for b in b_plus_vqe]
    # Total Green's function record keeping
    #record_keeping["g_vqe"] = g_vqe
    record_keeping["g_vqe_real"] = [g.real for g in g_vqe]
    record_keeping["g_vqe_imag"] = [g.imag for g in g_vqe]
    ###

    return g_vqe, record_keeping
