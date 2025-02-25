# This file has been modified for the AIM-QC code base. Original: https://github.com/SRI-International/QC-App-Oriented-Benchmarks/blob/master/vqe/qiskit/vqe_benchmark.py
    
"""
Variational Quantum Eigensolver Benchmark Program - Qiskit
"""

import time
import argparse
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp, Statevector

import os
import sys

from _common.qiskit import execute
from _common import metrics

from operator import itemgetter

sys.path.append(os.path.dirname(os.getcwd()))
from dmft import initialize_system

# Benchmark Name
benchmark_name = "AIM SPA VQE Simulation"

verbose = False

# saved circuit for display
QC_ = None


################### Circuit Definition #######################################

# Construct a Qiskit circuit for VQE Energy evaluation with Symmetric-Preserving Ansatz
# param: n_spin_orbs - The number of spin orbitals.
# return: return a Qiskit circuit for this VQE ansatz
def VQEEnergy(n_spin_orbs, n_up, n_down, seed, n_layers, method=1):

    # number of sites
    system_size = int(n_spin_orbs / 2)

    # allocate qubits
    n_qubits = n_spin_orbs

    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)

    _, _, _, up_qubit_indices, down_qubit_indices, connected_graphs, hamiltonian_qubitop = initialize_system(system_size, seed)

    hamiltonian_sparsepauliop = qubitop_to_sparsepauliop(hamiltonian_qubitop, n_qubits)

    # Next two lines take care of spin up sector
    most_equispaced_indices = np.linspace(0, len(up_qubit_indices) - 1, n_up, dtype=int).tolist()
    initial_occupations_indices = [up_qubit_indices[i] for i in most_equispaced_indices]
    # Next two lines take care of spin down sector
    most_equispaced_indices = np.linspace(0, len(down_qubit_indices) - 1, n_down, dtype=int).tolist()
    initial_occupations_indices.extend([down_qubit_indices[i] for i in most_equispaced_indices])
    initial_occupations_indices.sort()

    # generate n random parameters
    graph_up, graph_down, graph_stitch = itemgetter("graph_up", "graph_down", "graph_stitch")(connected_graphs)
    n_edges = sum([len(connected_graphs[key].edges()) for key in ["graph_up", "graph_down", "graph_stitch"]])
    n_params = n_layers * (n_edges + n_qubits)  # This factor of two accounts for additional cphase
    theta = np.random.uniform(low=0, high=2 * np.pi, size=n_params)

    # add symmetric-preserving ansatz to circuit
    qc_ansatz = ansatz(qc, theta, initial_occupations_indices, n_layers, graph_up, graph_down, graph_stitch, n_qubits)
    qc = qc.compose(qc_ansatz)

    # method 1, only compute one term in the Hamiltonian
    if method == 1:
        # last term in Hamiltonian
        qc_with_mea, is_diag = ExpectationCircuit(qc, hamiltonian_sparsepauliop[1], n_qubits)

        # create sample circuit
        if QC_ is None and n_qubits == 4 and n_layers < 3:
            sample_circuit(initial_occupations_indices, n_layers, theta, graph_up, graph_down, graph_stitch, n_qubits, hamiltonian_sparsepauliop[1])

        # return the circuit
        return qc_with_mea

    # now we need to add the measurement parts to the circuit
    # circuit list
    qc_list = []
    diag = []
    off_diag = []
    global normalization
    normalization = 0.0

    # add the first non-identity term
    identity_qc = qc.copy()
    identity_qc.measure_all()
    qc_list.append(identity_qc)  # add to circuit list
    diag.append(hamiltonian_sparsepauliop[1])
    normalization += abs(hamiltonian_sparsepauliop[1].coeffs[0])  # add to normalization factor
    diag_coeff = abs(hamiltonian_sparsepauliop[1].coeffs[0])  # add to coefficients of diagonal terms

    # if method = 2, loop over the rest of the terms in the Hamiltonian
    for _, p in enumerate(hamiltonian_sparsepauliop[2:]):

        # get the circuit with expectation measurements
        qc_with_mea, is_diag = ExpectationCircuit(qc, p, n_qubits)

        # accumulate normalization
        normalization += abs(p.coeffs[0])

        # add to circuit list if non-diagonal
        if not is_diag:
            qc_with_mea.name = qc_with_mea.name + " " + str(seed)
            qc_list.append(qc_with_mea)
        else:
            diag_coeff += abs(p.coeffs[0])

        # diagonal term
        if is_diag:
            diag.append(p)
        # off-diagonal term
        else:
            off_diag.append(p)

    # if method = 2, create sample circuit for second to last term in the Hamiltonian
    if QC_ is None and n_qubits == 4 and n_layers < 3:
        sample_circuit(initial_occupations_indices, n_layers, theta, graph_up, graph_down, graph_stitch, n_qubits, hamiltonian_sparsepauliop[-2])

    # modify the name of diagonal circuit
    qc_list[0].name = hamiltonian_sparsepauliop[1].to_list()[0][0] + " " + str(np.real(diag_coeff)) + " " + str(seed)
    normalization /= len(qc_list)

    return qc_list


# Function that adds expectation measurements to the raw circuits
def ExpectationCircuit(qc, pauli, nqubit):

    # copy the unrotated circuit
    raw_qc = qc.copy()

    # whether this term is diagonal
    is_diag = True

    # primitive Pauli string
    PauliString = pauli.to_list()[0][0]

    # coefficient
    coeff = pauli.coeffs[0]

    # basis rotation
    for i, p in enumerate(PauliString):

        target_qubit = nqubit - i - 1
        if (p == "X"):
            is_diag = False
            raw_qc.h(target_qubit)
        elif (p == "Y"):
            raw_qc.sdg(target_qubit)
            raw_qc.h(target_qubit)
            is_diag = False

    # perform measurements
    raw_qc.measure_all()

    # name of this circuit
    raw_qc.name = PauliString + " " + str(np.real(coeff))

    return raw_qc, is_diag


################ Helper Functions ################

def qubitop_to_sparsepauliop(qubit_op, n_qubits):

    pauli_list = []

    for term, coefficient in qubit_op.terms.items():  # .terms changes the order of the pauli terms
        pauli_string = ['I'] * n_qubits
        for indexed_pauli_gate in term:
            index = indexed_pauli_gate[0]
            pauli_gate = indexed_pauli_gate[1]
            pauli_string[index] = pauli_gate
        pauli_list.append((''.join(pauli_string), coefficient.real))

    sparse_pauli_op = SparsePauliOp.from_list(pauli_list)

    return sparse_pauli_op


# submit circuit for execution on statevector simulator
def run_statevector_sim(qc):

    # remove measurements
    qc.remove_final_measurements()
    statevector = Statevector(qc)

    return statevector


################ Ansatz ################

def ansatz(qc, theta, initial_occupations_indices, n_layers, graph_up, graph_down, graph_stitch, n_qubits):

    # add initial state to circuit
    qc.x(initial_occupations_indices)

    k = 0  # This index keeps track of theta parameter angles

    for layer in range(n_layers):

        # Block below this comment iterates through 2d NN connections on spin up qubits
        for edge in graph_up.edges():
            qubit_a = edge[0]
            qubit_b = edge[1]
            # This block of code constitutes one Givens/SO(4) rotation gate
            qc.s(qubit_a)
            qc.s(qubit_b)
            qc.h(qubit_a)
            qc.cx(qubit_a, qubit_b)
            qc.ry(theta[k], qubit_a)
            qc.ry(theta[k], qubit_b)
            k += 1  # Increments theta counter
            qc.cx(qubit_a, qubit_b)
            qc.h(qubit_a)
            qc.sdg(qubit_a)
            qc.sdg(qubit_b)

        # Block below iterates through all 2d NN connections on spin down qubits
        for edge in graph_down.edges():
            qubit_a = edge[0]
            qubit_b = edge[1]
            # This block of code constitutes one Givens/SO(4) rotation gate
            qc.s(qubit_a)
            qc.s(qubit_b)
            qc.h(qubit_a)
            qc.cx(qubit_a, qubit_b)
            qc.ry(theta[k], qubit_a)
            qc.ry(theta[k], qubit_b)
            k += 1  # Increments theta counter
            qc.cx(qubit_a, qubit_b)
            qc.h(qubit_a)
            qc.sdg(qubit_a)
            qc.sdg(qubit_b)

        # Connect qubits across up/down patches
        for edge in graph_stitch.edges():
            qubit_a = edge[0]  # up qubit
            qubit_b = edge[1]  # down qubit
            qc.cx(qubit_a, qubit_b)
            qc.rz(theta[k], qubit_b)
            qc.cx(qubit_a, qubit_b)
            k += 1

        for q in range(0, n_qubits):
            qc.rz(theta[k], q)
            k += 1

    return qc


def sample_circuit(initial_occupations_indices, n_layers, theta, graph_up, graph_down, graph_stitch, n_qubits, pauli):

    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)

    qc.x(initial_occupations_indices)
    qc.barrier()

    k = 0  # This index keeps track of theta parameter angles

    for layer in range(n_layers):

        # Block below this comment iterates through 2d NN connections on spin up qubits
        for edge in graph_up.edges():
            qubit_a = edge[0]
            qubit_b = edge[1]
            # This block of code constitutes one Givens/SO(4) rotation gate
            qc.s(qubit_a)
            qc.s(qubit_b)
            qc.h(qubit_a)
            qc.cx(qubit_a, qubit_b)
            qc.ry(theta[k], qubit_a)
            qc.ry(theta[k], qubit_b)
            k += 1  # Increments theta counter
            qc.cx(qubit_a, qubit_b)
            qc.h(qubit_a)
            qc.sdg(qubit_a)
            qc.sdg(qubit_b)
            qc.barrier()

        # Block below iterates through all 2d NN connections on spin down qubits
        for edge in graph_down.edges():
            qubit_a = edge[0]
            qubit_b = edge[1]
            # This block of code constitutes one Givens/SO(4) rotation gate
            qc.s(qubit_a)
            qc.s(qubit_b)
            qc.h(qubit_a)
            qc.cx(qubit_a, qubit_b)
            qc.ry(theta[k], qubit_a)
            qc.ry(theta[k], qubit_b)
            k += 1  # Increments theta counter
            qc.cx(qubit_a, qubit_b)
            qc.h(qubit_a)
            qc.sdg(qubit_a)
            qc.sdg(qubit_b)
            qc.barrier()

        # Connect qubits across up/down patches
        for edge in graph_stitch.edges():
            qubit_a = edge[0]  # up qubit
            qubit_b = edge[1]  # down qubit
            qc.cx(qubit_a, qubit_b)
            qc.rz(theta[k], qubit_b)
            qc.cx(qubit_a, qubit_b)
            k += 1
            qc.barrier()

        for q in range(0, n_qubits):
            qc.rz(theta[k], q)
            k += 1

        qc.barrier()

    # primitive Pauli string
    PauliString = pauli.to_list()[0][0]

    # basis rotation
    for i, p in enumerate(PauliString):
        target_qubit = n_qubits - i - 1
        if (p == "X"):
            qc.h(target_qubit)
        elif (p == "Y"):
            qc.sdg(target_qubit)
            qc.h(target_qubit)

    # perform measurements
    qc.measure_all()

    # name circuit
    qc.name = "sample-circuit-aim-spa-vqe-ansatz-" + str(n_layers) + "layers" + str(n_qubits) + "qubits-" + PauliString

    # save circuit
    global QC_
    QC_ = qc

    return None


################ Result Data Analysis ################

# Analyze and print measured results
# Compute the quality of the result based on measured probability distribution for each state
def analyze_and_print_result(qc, result, statevector, n_shots, method):

    # total circuit name (pauli string + coefficient)
    total_name = qc.name

    # get results counts
    actual_counts = result.get_counts(qc)

    # get the probability distribution
    expected_counts = statevector.sample_counts(n_shots, qargs=None)

    # compute fidelity
    fidelity = metrics.polarization_fidelity(actual_counts, expected_counts)

    if verbose:
        print(f"... fidelity = {fidelity}")

    # modify fidelity based on the coefficient (only for method 2)
    if method == 2:
        coefficient = abs(float(total_name.split()[1])) / normalization
        fidelity = {f: v * coefficient for f, v in fidelity.items()}
        if verbose:
            print(f"... total_name={total_name}, coefficient={coefficient}, product_fidelity={fidelity}")

    if verbose:
        print(f"... total fidelity = {fidelity}")

    return fidelity


################ Benchmark Loop ################

# Max qubits must be 12 since the referenced files only go to 12 qubits
MAX_QUBITS = 12


# Execute program with default parameters
def run(min_qubits=4, max_qubits=8, skip_qubits=1,
        max_circuits=3, num_shots=4092, seed=0, n_layers=1, method=1,
        backend_id=None, provider_backend=None,
        hub="ibm-q", group="open", project="main", noise_model=None,
        exec_options=None, context=None):

    print(f"{benchmark_name} ({method}) Benchmark Program - Qiskit")

    max_qubits = max(max_qubits, min_qubits)  # max must be >= min

    # validate parameters (smallest circuit is 4 qubits and largest is 10 qubitts)
    max_qubits = min(max_qubits, MAX_QUBITS)
    min_qubits = min(max(4, min_qubits), max_qubits)
    if min_qubits % 2 == 1: min_qubits += 1  # min_qubits must be even
    skip_qubits = max(1, skip_qubits)

    if method == 2: max_circuits = 1

    if max_qubits < 4:
        print(f"Max number of qubits {max_qubits} is too low to run method {method} of VQE algorithm")
        return

    # create context identifier
    if context is None: context = f"{benchmark_name} ({method}) Benchmark"

    ##########

    # Initialize the metrics module
    metrics.init_metrics()

    # Define custom result handler
    def execution_handler(qc, result, n_qubits, type, n_shots):

        statevector_results = run_statevector_sim(qc)

        fidelity = analyze_and_print_result(qc, result, statevector_results, n_shots, method)

        if method == 2:
            qc_name_list = qc.name.split()
            circuit_id = str(qc_name_list[0]) + str(qc_name_list[2])
            metrics.store_metric(n_qubits, circuit_id, 'fidelity', fidelity)
        else:
            metrics.store_metric(n_qubits, qc.name.split()[2], 'fidelity', fidelity)

    # Initialize execution module using the execution result handler above and specified backend_id
    execute.init_execution(execution_handler)
    execute.set_execution_target(backend_id, provider_backend=provider_backend,
                                 hub=hub, group=group, project=project,
                                 noise_model=noise_model, exec_options=exec_options,
                                 context=context)

    ##########

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for input_size in range(min_qubits, max_qubits + 1, 2):

        # determine the number of circuits to execute for this group
        num_circuits = min(3, max_circuits)

        n_qubits = input_size

        # decides number of electrons, total spin is always 0
        n_up = int(n_qubits / 4)
        n_down = int(n_qubits / 4)

        # create the circuit for given qubit size and simulation parameters, store time metric
        ts = time.time()

        # circuit list
        qc_list = []

        # Method 1 (default)
        if method == 1:
            # loop over circuits
            for circuit in range(num_circuits):

                # construct circuit
                qc_single = VQEEnergy(n_qubits, n_up, n_down, seed, n_layers, method)
                circuit_id = str(circuit) + str(seed)
                qc_single.name = qc_single.name + " " + str(circuit_id)

                # add to list
                qc_list.append(qc_single)
        # method 2
        elif method == 2:

            # construct all circuits
            qc_list_tmp = VQEEnergy(n_qubits, n_up, n_down, seed, n_layers, method)

            # add to list
            qc_list.extend(qc_list_tmp)

        print(f"************\nExecuting [{len(qc_list)}] circuits with n_qubits = {n_qubits}")

        for qc in qc_list:

            qc_name_list = qc.name.split()

            # get circuit id
            if method == 1:
                circuit_id = qc_name_list[2]
            else:
                circuit_id = str(qc_name_list[0]) + str(qc_name_list[2])

            # record creation time
            metrics.store_metric(input_size, circuit_id, 'create_time', time.time() - ts)

            # collapse the sub-circuits used in this benchmark (for qiskit)
            qc2 = qc.decompose()

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            execute.submit_circuit(qc2, input_size, circuit_id, num_shots)

        # Wait for some active circuits to complete; report metrics when group complete
        execute.throttle_execution(metrics.finalize_group)

    # Wait for all active circuits to complete; report metrics when groups complete
    execute.finalize_execution(metrics.finalize_group)

    ##########

    # print a sample circuit
    if QC_ is not None:
        print(QC_.name)
        print(QC_)

    # Plot metrics for all circuit sizes
    metrics.plot_metrics(f"Benchmark Results - {benchmark_name} ({method}) - Qiskit")


################# MAIN #######################

def get_args():
    parser = argparse.ArgumentParser(description="Variational Quantum Eigensolver Benchmark")
    parser.add_argument("--backend_id", "-b", default=None, help="Backend Identifier", type=str)
    parser.add_argument("--n_qubits", "-n", default=0, help="Number of qubits (min = max = N)", type=int)
    parser.add_argument("--min_qubits", "-min", default=4, help="Minimum number of qubits", type=int)
    parser.add_argument("--max_qubits", "-max", default=8, help="Maximum number of qubits", type=int)
    parser.add_argument("--skip_qubits", "-k", default=1, help="Number of qubits to skip", type=int)
    parser.add_argument("--max_circuits", "-c", default=3, help="Maximum circuit repetitions", type=int)
    parser.add_argument("--num_shots", "-s", default=4092, help="Number of shots", type=int)
    parser.add_argument("--seed", "-e", default=0, help="Seed number for VQE run.", type=int)
    parser.add_argument("--n_layers", "-l", default=1, help="Number of layers", type=int)
    parser.add_argument("--method", "-m", default=1, help="Algorithm Method", type=int)
    parser.add_argument("--noise_model", "-u", help="Noise model", type=str)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    # special argument handling
    execute.verbose = args.verbose
    verbose = args.verbose

    if args.n_qubits > 0: args.min_qubits = args.max_qubits = args.n_qubits

    # execute benchmark program
    run(min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        skip_qubits=args.skip_qubits,
        max_circuits=args.max_circuits,
        num_shots=args.num_shots,
        seed=args.seed,
        n_layers=args.n_layers,
        method=args.method,
        backend_id=args.backend_id,
        noise_model=args.noise_model,
        exec_options={} if args.noise_model else {"noise_model": None}
        )
