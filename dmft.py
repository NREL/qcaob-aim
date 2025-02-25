import anderson_impurity_model as aim
from anderson_impurity_model import AndersonImpurityModel
import exact
import vqe
from n_site_graph_creation import AIMSiteModelsEnum, create_connected_graphs
import numpy as np
from openfermion.ops import QubitOperator
from scipy import sparse
import matplotlib.pyplot as plt
from qulacs import QuantumState
import os
import argparse
import time
import json
from numpy.typing import NDArray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-size", "--system_size", help="System size for experiment.", type=int)
    parser.add_argument("-s", "--seed", help="Seed number for VQE run.", default=0, type=int)
    parser.add_argument("-t", "--target_error", help="Target error for scaling.", type=float)
    parser.add_argument("-m", "--maxiters", help="Maxiters for the phi minus and plus VQE optimizations.", default=1e6, type=int)
    parser.add_argument("-g", "--gtol", help="Gradient norm tolerance for the VQE optimizer.", default=5e-5,
                        type=float)
    parser.add_argument("-gs_gtol", "--gs_gtol", help="Initial gradient norm tolerance for the ground state optimizer.",
                        default=5e-4, type=float)
    parser.add_argument("-pel", "--pre_empt_layers", help="VQE layers to iterate through when running scaling.", default=1,
                        type=int)
    parser.add_argument("-svd", "--starting_vqe_depth", help="Initial number of layers for the SciPy optimizers and ansatz.",
                        default=1, type=int)
    parser.add_argument("-gs", "--ground_state", action=argparse.BooleanOptionalAction, default=False,
                        help="Flag that will just run the ground state determination, not generating a Green's function via "
                        "Lanczos iterations.")
    parser.add_argument("-d", "--display", action=argparse.BooleanOptionalAction, default=False,
                        help="Flag to display algorithm information for each Green's function calculation.")
    parser.add_argument("-p", "--plot", action=argparse.BooleanOptionalAction, default=False,
                        help="Flag to plot Green's function (exact and VQE) for each Green's function calculation.")

    args = parser.parse_args()

    if args.starting_vqe_depth > args.pre_empt_layers:
        raise ValueError('starting_vqe_depth must not be greater than pre_empt_layers.')

    run_gs_error_experiment(
        system_size=args.system_size,
        seed=args.seed,
        target_err=args.target_error,
        maxiters=args.maxiters,
        gf_gtol=args.gtol,
        gs_gtol=args.gs_gtol,
        pre_empt_layers=args.pre_empt_layers,
        starting_depth=args.starting_vqe_depth,
        gs=args.ground_state,
        display=args.display,
        plot=args.plot,
    )

    return


def construct_aim(hamiltonian_parameters: tuple[NDArray, NDArray, NDArray, NDArray]) -> tuple[AndersonImpurityModel, int]:
    """
    Instantiates an AndersonImpurityModel object and returns the number of orbitals.
    :param hamiltonian_parameters: Matrix elements as constructed by the randomized seeding.
    :return: AndersonImpurityModel of test model and number of orbitals.
    """
    test_model = aim.AndersonImpurityModel(*hamiltonian_parameters)
    number_orbitals = (test_model.himp.shape[0] + test_model.ebath.shape[0]) // 2

    return test_model, number_orbitals


def initialize_system(
        system_size: int,
        seed: int
) -> tuple[int, AndersonImpurityModel, int, list[int], list[int], dict, QubitOperator]:
    """
    Intantiates an AndersonImpurityModel and corresponding number of orbitals, spin-up/-down indices, connected graph, and qubit
    Hamiltonian.
    :param system_size: System size for experiment.
    :param seed: Seed number for VQE run.
    :return: Frequency array with imaginary broadening term, AndersonImpurityModel of the experiment, number of orbitals,
    spin-up/-down qubit indices, connected graph, index of the impurity orbital, and qubit mapped Hamiltonian.
    """
    n_bath_n_imp_tup = (system_size - 1, 1)
    connected_graphs = create_connected_graphs(AIMSiteModelsEnum(n_bath_n_imp_tup).create_n_site_model_idx(),
                                               show_full_plot=False, show_sub_graphs=False)
    impurity_orbital = sorted([idx for idx, attributes in connected_graphs["graph_up"].nodes(data=True)
                               if attributes["type"] == "I"])[0]

    model_params = aim.random_n_site_up_down_symmetric_example(system_size, seed)
    test_model, n_orbitals = construct_aim(model_params)

    up_qubit_indices, down_qubit_indices = test_model.return_up_and_down_indices()

    qubit_hamiltonian = test_model.construct_qubit_hamiltonian()

    return impurity_orbital, test_model, n_orbitals, up_qubit_indices, down_qubit_indices, connected_graphs, qubit_hamiltonian


def compare_gs(
        exact_gs_energy: float,
        vqe_gs_energy: float,
        exact_gs: sparse.csc_array,
        vqe_gs: QuantumState
) -> dict:
    """
    Compare the ground state and ground state energy found exactly and by VQE.
    :param exact_gs_energy: The exact ground state energy.
    :param vqe_gs_energy: Ground state energy as found by VQE.
    :param exact_gs: Exactly solved ground state.
    :param vqe_gs: Ground state as found by VQE.
    :return: Dictionary of selected results.
    """
    # Relative error for exact ground state energy
    gs_energy_rel_error = np.abs(exact_gs_energy - vqe_gs_energy) / np.abs(exact_gs_energy)

    # Error rate for ground state overlap
    gs_overlap = np.abs(vqe_gs.T.conj().dot(exact_gs))
    gs_overlap_error = 1 - gs_overlap

    del vqe_gs
    del exact_gs

    ###
    record_keeping = {}
    record_keeping["gs_overlap"] = gs_overlap
    record_keeping["gs_energy_rel_error"] = gs_energy_rel_error
    record_keeping["gs_error"] = gs_overlap_error
    ###

    return record_keeping


def calculate_rel_errors(
        g_vqe: NDArray[np.complex128],
        g_exact: NDArray[np.complex128]
) -> tuple[float, dict]:
    """
    Calculates the relative error between Green's function found exactly and by VQE.
    :param g_vqe: Green's function generated by VQE
    :param g_exact: Green's function generated by exact solution
    :return: Relative error and dictionary of selected results.
    """
    avg_rel_diff = 0
    rel_diffs = []
    for z, gz_vqe in enumerate(g_vqe):
        avg_rel_diff += np.sqrt((gz_vqe.real - g_exact[z].real) ** 2 + (gz_vqe.imag - g_exact[z].imag) ** 2)
        rel_diffs.append(np.sqrt((gz_vqe.real - g_exact[z].real) ** 2 + (gz_vqe.imag - g_exact[z].imag) ** 2))
    avg_rel_diff /= len(g_vqe)

    numerator = np.linalg.norm(np.subtract(g_vqe, g_exact))
    denominator = np.linalg.norm(g_exact)
    rel_error = numerator / denominator

    ###
    record_keeping = {}
    record_keeping["g_numerator"] = numerator
    record_keeping["g_denominator"] = denominator
    record_keeping["g_rel_error"] = rel_error
    record_keeping["g_avg_rel_diff"] = avg_rel_diff
    ###

    return rel_error, record_keeping


def run_gs_error_experiment(
    system_size: int,
    seed: int,
    target_err: float,
    maxiters: int,
    gs_gtol: float,
    gf_gtol: float,
    pre_empt_layers: int,
    starting_depth: int,
    gs: bool,
    display: bool,
    plot: bool,
) -> None:
    """
    Run calculate_gs() as a subroutine in order to hit target threshold ground state energy relative error. If error
    threshold is met and gs flag is set to false, also call calculate_gf() to find the Green's function.
    :param system_size: System size for experiment.
    :param seed: This seed is used in the random creation of the Hamiltonian parameters for the AIM.
    :param target_err: The relative error in the Green's function used as a threshold for re-running with an additional VQE layer.
    :param maxiters: Maximum iteration number for ground state and Lanczos iteration optimizes.
    :param gf_gtol: Gradient norm tolerance for the gf optimizer.
    :param gs_gtol: Gradient norm tolerance for the ground state optimizer.
    :param pre_empt_layers: VQE layers to iterate through when running scaling.
    :param starting_depth: Starting VQE depth.
    :param gs: Flag that will just run the ground state determination, not generating a Green's function via Lanczos iterations.
    :param display: Boolean that decides whether or not to display algorithm information as it runs.
    :param plot:
    :return: None
    """

    t0 = time.time()

    impurity_orbital, test_model, n_orbitals, up_qubit_indices, down_qubit_indices, \
        connected_graphs, qubit_hamiltonian = initialize_system(system_size, seed)
    if display:
        print("qubit Hamiltonian:\n", str(qubit_hamiltonian))

    optimizer = "BFGS"
    conv_tol = 1e-6
    w = np.linspace(-25, 25, 1000, dtype=np.complex128) + 1j * 0.1

    req_layers = None
    req_gs_gtol = None
    prev_gs_error = 1
    convergence_success = False
    degenerate_bin = set([])

    pre_empt_layers_arr = np.arange(starting_depth, pre_empt_layers + 1, step=1, dtype=int)

    for n_layers in pre_empt_layers_arr:
        # Generate the directory name here and save it within record dict as we'll want it for the checkpointing 
        dir_name = "gs_se_data_target_error={:.0e}/n={}_pe_layers={}".format(target_err, system_size, pre_empt_layers)
        f_name = "sites={}_seed={}_layers={}_gs_gtol={:.2e}".format(system_size, seed, n_layers, gs_gtol)
        # Create results dir 
        #checkpoint_dir = dir_name + "/checkpoint/"
        results_dir = dir_name + "/results/"

        # Checkpoint dir is used to save results mid-optimization, whereas results dir saves the progress at the end of 
        # each vqe run (one per spin, charge sector search)
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        #checkpoint_file = checkpoint_dir + f_name + ".pkl"
        checkpoint_file=""
        results_file = results_dir + f_name + ".pkl"
        #record["checkpoint_file"] = checkpoint_file

        if display:
            print("-" * 75)
            print(f"Starting run with {n_layers} layers on {system_size}-site case with seed {seed}, gs gtol {gs_gtol:.3e}")

        record_gs, exact_gs, min_vqe_angles = calculate_gs(impurity_orbital, test_model, up_qubit_indices, down_qubit_indices, connected_graphs,
                                                        qubit_hamiltonian, n_layers, gs_gtol, optimizer, gs_gtol, maxiters, conv_tol,
                                                        display=display, gs=gs, checkpoint_file=checkpoint_file, results_file=results_file)
        t1 = time.time()

        ###
        record_gs["wall_clock_time"] = t1 - t0
        record_gs["n_sites"] = system_size
        record_gs["seed"] = seed
        record_gs["target_error"] = target_err
        record_gs["qubit_hamiltonian"] = str(qubit_hamiltonian)
        ###

        prev_gs_error = record_gs["gs_error"]
        convergence_success = record_gs["local_vqe_success"]
        if (prev_gs_error < target_err) and (convergence_success is True):
            record_gs["overall_success"] = True
        else:
            record_gs["overall_success"] = False

        save_to_file(target_err, pre_empt_layers, system_size, seed, n_layers, gs_gtol, record_gs, type='gs')

        if record_gs["degenerate"] == True:
            degenerate_bin.add((system_size, seed))
            req_layers = n_layers
            req_gs_gtol = gs_gtol
            break

        if prev_gs_error < target_err:
            req_layers = n_layers
            req_gs_gtol = gs_gtol
            break

    # After obtaining the desired accuracy for the variationally prepared gs + gs is set to False, find the green's function
    if (record_gs["overall_success"] == True) & (gs == False):
        record_gf = calculate_gf(impurity_orbital, test_model, n_orbitals, up_qubit_indices, down_qubit_indices, connected_graphs,
                              qubit_hamiltonian, n_layers, w, optimizer, gf_gtol, vqe_gs_energy=record_gs['vqe_gs_energy'],
                              vqe_spin=record_gs['vqe_spin'], vqe_charge=record_gs['vqe_charge'], exact_gs=exact_gs,
                              exact_spin=record_gs['exact_spin'], exact_charge=record_gs['exact_charge'], minimum_angles=min_vqe_angles,
                              record_keeping=record_gs, display=display, plot=plot)
        save_to_file(target_err, pre_empt_layers, system_size, seed, n_layers, gs_gtol, record_gf, type='gf')

    if display:
        if record_gs["degenerate"] is True:
            print("DEGENERATE CASE")
        if record_gs["overall_success"] is True:
            print("{}-layers at {:.2e} gs gtol needed for {}-site {}-seed case".format(req_layers, req_gs_gtol, system_size, seed))
        else:
            print("Experiment failed with {}-layers trying to achieve {:.1e} ground state target error ".format(n_layers, target_err))
            if not gs:
                print("Green's function calculation not run as ground state preparation failed.")
        print("{:.4e} ground state error, {:.1e} ground state target error".format(prev_gs_error, target_err))
        print("convergence success message: {}".format(convergence_success))
        print("degenerate runs:", degenerate_bin)

    return None


def calculate_gs(
        impurity_orbital: int,
        test_model: AndersonImpurityModel,
        up_qubit_indices: list[int],
        down_qubit_indices: list[int],
        connected_graphs: dict,
        qubit_hamiltonian: QubitOperator,
        vqe_depth: int,
        gs_gtol: float,
        optimizer: str,
        gtol: float,
        maxiters: int,
        conv_tol: float,
        display: bool = False,
        gs: bool = False, 
        checkpoint_file: str= '',
        results_file: str = '',
) -> tuple[dict, NDArray]:
    """
    Given params, compute exact ground state solution by exact diagonalization. Repeat the process for the same
    Hamiltonian but instead by minimizing the energy over VQE run on individual spin-charge sectors.
    :param impurity_orbital: Index of the impurity orbital.
    :param test_model: AndersonImpurityModel
    :param up_qubit_indices: List of spin-up register qubit indices.
    :param down_qubit_indices: List of spin-down register qubit indices.
    :param connected_graphs: Dictionary of graph objects used for representing AIM.
    :param qubit_hamiltonian: Hamiltonian in the form of an OpenFermion QubitOperator.
    :param vqe_depth: Number of layers for the SciPy optimizers and ansatz.
    :param gs_gtol: Gradient norm tolerance for the ground state optimizer.
    :param optimizer: Type of optimizer for VQE.
    :param gtol: Gradient norm tolerance for the VQE optimizer.
    :param maxiters: Max iteration number used as stopping criteria for optimizers.
    :param conv_tol: Broadly defined convergence tolerance for scipy optimizers.
    :param display: Boolean that decides whether or not to display algorithm information as it runs.
    :param gs: Flag that will just run the ground state determination, not generating a Green's function via Lanczos iterations..
    :return: Dictionary of selected results.
    """

    record_keeping = {"impurity_orbital": impurity_orbital, "vqe_depth": vqe_depth, "optimizer": optimizer,
                      "conv_tol": conv_tol, "maxiters": maxiters, "gtol": gtol, "gs_gtol": gs_gtol, "gs": gs, 
                      "checkpoint_file": checkpoint_file, "results_file": results_file}

    ###
    # 1 Solve for ground state and ground state energy
    # 1.1 Exact gs from exact diagonalization
    exact_gs_energy, exact_gs, _, _, degenerate, record_exact_gs = exact.solve_exact_gs(test_model)
    record_keeping.update(record_exact_gs)

    # Terminate the script and return record in degenerate case
    if degenerate:
        if display:
            print_record(record_keeping, degenerate=True)
        return record_keeping, _, _

    # 1.2 VQE
    vqe_gs_energy, vqe_gs, _, minimum_angles, \
        record_vqe_gs = vqe.solve_vqe_gs(test_model, qubit_hamiltonian, connected_graphs, up_qubit_indices, down_qubit_indices, display=display, **record_keeping)
    record_keeping.update(record_vqe_gs)
    ###

    ###
    # 2 Compare exact ground state and VQE ground state
    record_compare_gs = compare_gs(exact_gs_energy, vqe_gs_energy, exact_gs, vqe_gs)
    record_keeping.update(record_compare_gs)
    ###

    if display:
        print_record(record_keeping, gs=True)

    return record_keeping, exact_gs, minimum_angles


def calculate_gf(
    impurity_orbital: int,
    test_model: AndersonImpurityModel,
    n_orbitals: int,
    up_qubit_indices: list[int],
    down_qubit_indices: list[int],
    connected_graphs: dict,
    qubit_hamiltonian: QubitOperator,
    vqe_depth: int,
    w: NDArray[np.complex128],
    optimizer: str,
    gf_gtol: float,
    vqe_gs_energy: float,
    vqe_spin: float,
    vqe_charge: float,
    exact_gs: NDArray,
    exact_spin: int,
    exact_charge: int,
    minimum_angles: NDArray,
    record_keeping: dict,
    display: bool = False,
    plot: bool = False,
) -> dict:

    """
    Given parameters, calculate the Green's function for a seeded Hamiltonian corresponding to an AndersonImpurityModel
    at a specified system size. Compute exact Green's function by classical Lanczos iterations.
    Repeat the process for the variationally found ground state. From this variational ground state, perform Lanczos
    iterations using VQE to construct the Krylov basis. Construct Green's functions from both methodologies and
    calculate relative error of the variational approach relative to the exact solution.
    :param test_model: AndersonImpurityModel
    :param n_orbitals: Number of orbitals generated by the AIM.
    :param up_qubit_indices: List of spin-up register qubit indices.
    :param down_qubit_indices: List of spin-down register qubit indices.
    :param connected_graphs: Dictionary of graph objects used for representing AIM.
    :param qubit_hamiltonian: Hamiltonian in the form of an OpenFermion QubitOperator.
    :param vqe_depth: Number of layers for the SciPy optimizers and ansatz.
    :param w: Complex, linearly-spaced frequency array containing broadening term.
    :param optimizer: Type of optimizer for VQE.
    :param gtol: Gradient norm tolerance for the VQE optimizer.
    :param vqe_gs_energy: Ground state energy as found by VQE ground state.
    :param vqe_spin: Spin sector as found by variational ground state.
    :param vqe_charge: Charge sector as found by variational ground state.
    :param exact_gs: Array containing the exact ground state.
    :param exact_spin: Spin sector as found by exact ground state.
    :param exact_charge: Charge sector as found by exact ground state.
    :param minimum_angles: Minimum angles for the variational ansatz.
    :param record_keeping: Dictionary containing results from the previous calculate_gs() run.
    :param display: Boolean that decides whether or not to display algorithm information as it runs.
    :param plot: Boolean that specifies if the Green's functions (exact and VQE) should be plotted at the end.
    :return: Dictionary of selected results.
    """

    ###
    # 3 Get Krylov dimensions
    # 3.1 Exact
    _, _, _, _, _, _, record_exact_krylov_dims = exact.construct_exact_krylov_dimensions(
        up_qubit_indices, impurity_orbital, n_orbitals, exact_charge, exact_spin)
    record_keeping.update(record_exact_krylov_dims)

    # 3.2 VQE

    # Parameter intialization - recreating qulacs gs for green's function determination
    qulacs_ham, qulacs_ham_2, n_qubits = vqe.create_qulacs_hamiltonian(qubit_hamiltonian, vqe_gs_energy)
    vqe_gs_qs, _ = vqe.construct_vqe_gs(
        n_qubits, qulacs_ham, up_qubit_indices, down_qubit_indices, vqe_depth, connected_graphs, minimum_angles,
        vqe_nu=record_keeping['vqe_nu'],
        vqe_nd=record_keeping['vqe_nd'])
    _, _, vqe_charge_minus, vqe_spin_minus, _, \
        _, _, vqe_charge_plus, vqe_spin_plus, _, record_vqe_krylov_dims = vqe.construct_vqe_krylov_dimensions(
            up_qubit_indices, impurity_orbital, n_orbitals, vqe_charge, vqe_spin)
    record_keeping.update(record_vqe_krylov_dims)

    # Create initial Krylov states
    phi_minus, new_phi_minus, _, phi_plus, new_phi_plus, _, record_vqe_krylov_zero_state = vqe.vqe_krylov_zero_state(
        vqe_gs=vqe_gs_qs, impurity_orbital=impurity_orbital, n_qubits=n_qubits)
    del vqe_gs_qs
    record_keeping.update(record_vqe_krylov_zero_state)
    ###

    ###
    # 4 Green's Function
    # 4.1 Exact
    g_exact, record_exact_gf = exact.calculate_gf_exact(exact_gs, test_model, new_phi_plus, new_phi_minus, n_orbitals, w,
                                                        **record_keeping)
    record_keeping.update(record_exact_gf)
    del test_model

    # 4.2 VQE
    g_vqe, record_vqe_gf = vqe.calculate_gf_vqe(
        qulacs_ham, qulacs_ham_2, phi_minus, phi_plus, vqe_charge_minus, vqe_charge_plus, vqe_spin_minus, vqe_spin_plus,
        up_qubit_indices, down_qubit_indices, connected_graphs, w, gf_gtol=gf_gtol, **record_keeping)
    record_keeping.update(record_vqe_gf)
    ###

    ###
    # 5 Average relative difference
    rel_error, record_rel_errors = calculate_rel_errors(g_vqe, g_exact)
    record_keeping.update(record_rel_errors)
    ###

    if display:
        print_record(record_keeping, gf=True)

    if plot:
        plot_gfs(
            w=w,
            g_vqe=g_vqe,
            g_exact=g_exact,
            impurity_orbital=impurity_orbital,
            vqe_depth=vqe_depth,
            gtol=gf_gtol,
            optimizer=optimizer,
            relative_error=rel_error
            )

    return record_keeping


def print_record(
        record_keeping: dict,
        degenerate: bool = False,
        gs: bool = False,
        gf: bool = False
) -> None:
    """
    Prints values of selected keywords from record_keeping.
    :param record_keeping: Dictionary of selected results.
    :param degenerate: Boolean to check if case was degenerate.
    :param gs: Boolean to check if only the ground state determination was run.
    :return: None
    """

    if gs:
        # Ground state and ground state energy - Exact
        print("exact charge:", record_keeping["exact_charge"])
        print("exact spin:", record_keeping["exact_spin"])
        print("exact ground state energy:", record_keeping["exact_gs_energy"])

        if degenerate:
            return None

        # Ground state and ground state energy - VQE
        print("VQE charge:", record_keeping["vqe_charge"])
        print("VQE spin:", record_keeping["vqe_spin"])
        print("VQE ground state energy:", record_keeping["vqe_gs_energy"])
        print("nparams:", record_keeping["local_vqe_nparams"])
        print("VQE nu:", record_keeping["vqe_nu"])
        print("VQE nd:", record_keeping["vqe_nd"])

        # Compare exact ground state and VQE ground state
        print("ground state overlap:", record_keeping["gs_overlap"])
        print("ground state error:", record_keeping["gs_error"])
        print("ground state energy relative error: {:.4e}".format(record_keeping["gs_energy_rel_error"]))

    if gf:
        # Krylov dimenstions - Exact
        print("exact nu minus:", record_keeping["exact_nu_minus"])
        print("exact nd minus:", record_keeping["exact_nd_minus"])
        print("exact Krylov minus dimension:", record_keeping["exact_krylov_dim_minus"])
        print("exact nu plus:", record_keeping["exact_nu_plus"])
        print("exact nd plus:", record_keeping["exact_nd_plus"])
        print("exact Krylov plus dimension:", record_keeping["exact_krylov_dim_plus"])

        # Krylov dimenstions - VQE
        print("VQE nu minus:", record_keeping["vqe_nu_minus"])
        print("VQE nd minus:", record_keeping["vqe_nd_minus"])
        print("ideal VQE Krylov minus dimension:", record_keeping["vqe_krylov_ideal_dim_minus"])
        print("VQE nu plus:", record_keeping["vqe_nu_plus"])
        print("VQE nd plus:", record_keeping["vqe_nd_plus"])
        print("ideal VQE Krylov plus dimension:", record_keeping["vqe_krylov_ideal_dim_plus"])
        print("VQE phi_m norm:", record_keeping["phi_minus_norm"])
        print("VQE normalized phi_m norm:", record_keeping["phi_minus_normalized"])
        print("VQE phi_p norm:", record_keeping["phi_plus_norm"])
        print("VQE normalized phi_p norm:", record_keeping["phi_plus_normalized"])

        # Green's Function - Exact
        print("norm cp:", record_keeping["norm_cp"])
        print("phi plus overlap:", record_keeping["phi_plus_overlap"])
        print("phi plus overlap check:", record_keeping["phi_plus_overlap_sum"])
        print("norm cm:", record_keeping["norm_cm"])
        print("phi minus overlap:", record_keeping["phi_minus_overlap"])
        print("phi minus overlap check:", record_keeping["phi_minus_overlap_sum"])

        # Green's Function - VQE
        print("VQE a_minus_real:\n", record_keeping["a_minus_vqe_real"])
        print("VQE b_minus_real:\n", record_keeping["b_minus_vqe_real"])
        print("VQE a_plus_real:\n", record_keeping["a_plus_vqe_real"])
        print("VQE b_plus_real:\n", record_keeping["b_plus_vqe_real"])

        # Relative Errors
        print("GF numerator:", record_keeping["g_numerator"])
        print("GF denominator:", record_keeping["g_denominator"])
        print("GF relative error:", record_keeping["g_rel_error"])
        print("GF avg. relative diff.:", record_keeping["g_avg_rel_diff"])

    return None


def save_to_file(
        target_err: float,
        pre_empt_layers: int,
        system_size: int,
        seed: int,
        vqe_depth: int,
        gs_gtol: float,
        record: dict,
        type: str
) -> None:
    """
    Saves all the data from the record dictionary to a json file. Filenmae is based on the input paramters of the experiment
    and iteration of layers and tolerance.
    :param target_err: The relative error in the Green's function used as a threshold for re-running with an additional VQE layer.
    :param pre_empt_layers: VQE layers to iterate through when running scaling.
    :param system_size: System size for experiment.
    :param seed: Seed number for VQE run.
    :param vqe_depth: The number of layers for the SciPy optimizers and ansatz.
    :param gs_gtol: Gradient norm tolerance for the ground state optimizer.
    :param record: Dictionary of selected results.
    :return: None
    """
    dir_name = "last_run_gs_se_data_target_error={:.0e}/n={}_pe_layers={}".format(target_err, system_size, pre_empt_layers)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    if type == 'gs':
        fname = "gs_sites={}_seed={}_layers={}_gs_gtol={:.2e}.json".format(system_size, seed, vqe_depth, gs_gtol)

    if type == 'gf':
        fname = "gf_sites={}_seed={}_gs_gtol={:.2e}.json".format(system_size, seed, gs_gtol)

    dest = dir_name + "/" + fname

    with open(dest, "w") as f:
        json.dump(record, f, default=int, indent=4)

    return None


def plot_gfs(
    w: NDArray[np.complex128],
    g_vqe: NDArray[np.complex128],
    g_exact: NDArray[np.complex128],
    impurity_orbital: int,
    vqe_depth: int,
    gtol: float,
    optimizer: str,
    relative_error: float,
) -> None:
    """
    Given kwargs including the calculated Green's function from both methodologies, plot the real and imaginary parts.
    :param w: Complex, linearly-spaced frequency array containing broadening term.
    :param g_vqe: Green's function generated by VQE
    :param g_exact: Green's function generated by exact solution
    :param impurity_orbital: Index of the impurity orbital.
    :param vqe_depth: Number of layers for the SciPy optimizers and ansatz.
    :param gtol: Gradient norm tolerance for the VQE optimizer.
    :param optimizer: Type of optimizer for VQE.
    :param relative_error: Relative error in the Green's function for the two methodologies.
    :return: None
    """

    plt.plot(w.real, g_vqe.real, "b-", label="VQE real")
    plt.plot(w.real, -g_vqe.imag / np.pi, "g-", label="=VQE spectral")
    plt.plot(w.real, g_exact.real, color="tab:blue", linestyle="--", label="exact real")
    plt.plot(w.real, -g_exact.imag / np.pi, color="tab:green", linestyle="--", label="exact spectral")
    plt.legend()
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$G^{ret}_%s(\omega)$" % impurity_orbital)
    plt.title(r"$n_{layers}=%s, gtol=%s$, %s, GF rel err=%.3f" % (vqe_depth, gtol, optimizer, relative_error))
    plt.show()

    return None


if __name__ == "__main__":
    main()
