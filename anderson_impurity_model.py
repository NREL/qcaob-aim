import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, brute, Bounds
from itertools import product
from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev, normal_ordered
from openfermion.linalg import get_sparse_operator, get_ground_state, expectation, get_gap
from openfermion.utils import count_qubits, commutator
from openfermion.measurements import group_into_tensor_product_basis_sets
from qulacs import ParametricQuantumCircuit, QuantumState
from qulacs.quantum_operator import create_quantum_operator_from_openfermion_text
from qulacsvis import circuit_drawer
from n_site_graph_creation import create_connected_graphs, AIMSiteModelsEnum
from numpy import Inf
import sys
import os
from scipy.optimize import approx_fprime, OptimizeResult
from operator import itemgetter
from qulacsvis import circuit_drawer
import pickle


def main2():
    fp_relative_minima = []
    shots_relative_minima = []
    fp_happy_counter = 0
    fp_super_happy_counter = 0
    shots_happy_counter = 0
    shots_super_happy_counter = 0
    zero_results_seeds = []
    n_seeds = 10
    for i in range(0, n_seeds):
        h, u, v, e = random_two_site_up_down_symmetric_example(seed=i)
        test_model = AndersonImpurityModel(h, u, v, e)

        blockprint()
        exact_gs, exact_charge, exact_spin = test_model.exact_diagonalization()
        exact_gs = exact_gs[0]  # Gets just energy
        if np.abs(exact_gs) < 1e-12:
            zero_results_seeds.append(i)
        minimum_sector, shots_minimum_sector, minimum_energy, shots_minimum_energy, minimum_nfev, shots_minimum_nfev = \
            symmetric_ansatz_test(model_hamiltonian=test_model, n_layers=1, shots=2_000, optimizer="COBYLA")
        enableprint()

        fp_relative_minimum = np.abs((minimum_energy - exact_gs) / exact_gs)
        shots_relative_minimum = np.abs((shots_minimum_energy - exact_gs) / exact_gs)
        fp_relative_minima.append(fp_relative_minimum)
        shots_relative_minima.append(shots_relative_minimum)

        if fp_relative_minimum < .1:
            fp_happy_counter += 1
        if fp_relative_minimum < .03:
            fp_super_happy_counter += 1
        if shots_relative_minimum < .1:
            shots_happy_counter += 1
        if shots_relative_minimum < .03:
            shots_super_happy_counter += 1

        print("SEED:", i, "Rel. Min. < 0.03?", shots_relative_minimum < 0.03)
        print("EXACT GS, CHARGE, SPIN:", exact_gs, exact_charge, exact_spin)
        print("FP GS, CHARGE, SPIN:", minimum_energy, minimum_sector, "\t \t", "Relative Minimum:",
              fp_relative_minimum, "\t \t", "nfev:", minimum_nfev)
        print("SHOTS GS, CHARGE, SPIN:", shots_minimum_energy, shots_minimum_sector, "\t \t", "Relative Minimum:",
              shots_relative_minimum, "\t \t", "nfev:", shots_minimum_nfev)
        print("\n")

    print("Zero result seeds:", zero_results_seeds)
    n_good_denoms = n_seeds - len(zero_results_seeds)
    print("Number of good denominator results:", n_good_denoms)
    print("Number, fraction of FP VQE GS below .1 error:", fp_happy_counter, fp_happy_counter / n_good_denoms)
    print("Number, fraction of FP VQE GS below .03 error:", fp_super_happy_counter,
          fp_super_happy_counter / n_good_denoms)
    print("Number, fraction of SHOTS VQE GS below .1 error:", shots_happy_counter,
          shots_happy_counter / n_good_denoms)
    print("Number, fraction of SHOTS VQE GS below .03 error:", shots_super_happy_counter,
          shots_super_happy_counter / n_good_denoms)

    fig = plt.figure()
    plt.plot([i for i in range(0, n_seeds)], fp_relative_minima, marker="o", linestyle="", color="red", label="FP")
    plt.plot([i for i in range(0, n_seeds)], shots_relative_minima, marker="o", linestyle="", color="blue",
             label="Shots")
    plt.axhline(0.10, xmin=0, xmax=100, linestyle="dashed", color="grey")
    plt.axhline(0.03, xmin=0, xmax=100, linestyle="dashed", color="black")
    plt.legend()
    plt.yscale("log")
    plt.savefig("AIM_stats.pdf")
    plt.close(fig)
    return

def main():
    # h, u, v, e = random_two_impurity_example(seed=100)
    # h, u, v, e = francois_example_inputs()
    h, u, v, e = random_two_site_up_down_symmetric_example(seed=9340)  # Bunch of testing done on 9340

    print("Imp. Hopping:", h)
    print("Imp. Coulomb:", u)
    print("Hybridization:", v)
    print("Bath Hopping", e)
    print("\n")

    test_model = AndersonImpurityModel(h, u, v, e)

    print("Fermionic Hamiltonian")
    fermionic_hamiltonian = test_model.construct_fermionic_hamiltonian()
    print(fermionic_hamiltonian)
    print("\n")

    print("Qubit Hamiltonian")
    qubit_ham = test_model.construct_qubit_hamiltonian(method="jordan-wigner")
    print(qubit_ham)
    print("\n")

    print("Sparse Ham. and GS")
    exact_gs = test_model.exact_diagonalization()
    exact_gs = exact_gs[0]  # Gets just energy
    print(exact_gs)
    print("\n")

    print("qulacs Ham.")
    qulacs_ham = create_quantum_operator_from_openfermion_text(f"{qubit_ham}")
    print(qulacs_ham.get_qubit_count(), "|", qulacs_ham)
    print("\n")

    # Qulacs Mean Field VQE results
    mf_res, mf_qgs = MeanFieldQulacsVqeEmulator(qulacs_ham).solve_ground_state_local(optimizer="BFGS")
    amp_tol = 1e-4
    n_qubits = qulacs_ham.get_qubit_count()
    print("Number of states in superposition:", len(mf_qgs.get_vector()))
    print("QULACS MF GS: BASIS STATE \t AMPLITUDE \t PROBABILITY")
    for ii, amplitude in enumerate(mf_qgs.get_vector()):
        if (amplitude * amplitude.conjugate()).real > amp_tol:
            format_string = '{0:0' + str(n_qubits) + 'b}'
            b_string = format_string.format(ii)
            print(b_string, amplitude, (amplitude * amplitude.conjugate()).real)
    vqe_gs_energy = mf_res.fun
    print("MF VQE GS Energy:", vqe_gs_energy)
    print("MF Energy Diff.:", vqe_gs_energy - exact_gs)
    # Check action under charge, spin ops
    charge, spin = test_model.construct_charge_and_spin_operators()
    comm = normal_ordered(commutator(charge, fermionic_hamiltonian))  # This is so cool/useful!
    print("Charge comm:", comm)
    charge = create_quantum_operator_from_openfermion_text(f"{jordan_wigner(charge)}")
    spin = create_quantum_operator_from_openfermion_text(f"{jordan_wigner(spin)}")
    # qgs.set_zero_state()
    print(f'MF N = {charge.get_expectation_value(mf_qgs).real:1.5f}')
    print(f'MF S = {spin.get_expectation_value(mf_qgs).real:1.5f}')
    res_circuit = MeanFieldQulacsVqeEmulator(qulacs_ham).spin_coherent_ansatz(theta=mf_res.x,
                                                                              compilation="standard")
    print("MF Qulacs Min. Angles:", mf_res.x)
    circuit_drawer(res_circuit, "mpl")
    plt.show()
    print("\n")

    return


# Printing utilities
# Disable
def blockprint():
    sys.stdout = open(os.devnull, 'w')

# Save the OptimizeResults upon convergence
def save_result(spin_charge, optimize_result, qulacs_spin_charge, results_file):    
    # Use 'ab' as this will be a pickle file with many optimize results, one for each spin_charge sector
    with open(results_file, 'ab') as f:
        pickle.dump({spin_charge: (optimize_result, qulacs_spin_charge)}, f)
    return None

# Load all results
def load_all_results(filename):
    if os.path.exists(filename):
        try: 
            with open(filename, "rb") as f:
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break
        except FileNotFoundError:
            return None

# Checkpoint functionality for optimization in case script terminates from hpc - callback for scipy.optimize
def save_checkpoint(x, fun, iteration, spin_charge, checkpoint_file):    
    # Use 'wb' as we just delete the last checkpoint, no need to carry around a ton of results
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({spin_charge: {iteration: {'x': x, 'fun': fun} } }, f)
        # TODO could always delete everything but the last saved checkpoint as needed
    return None

# Load checkpoint upon restarting a run
def load_checkpoint(checkpoint_file: str, sector: tuple):
    "Given a checkpoint file name and the charge/spin sector as strings, return the latest relevant OptimizeResult data."
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
                # The checkpoint file may exist but not the specific charge/sector data we want in the loop
                try: 
                    checkpoint_data = data[sector]
                    iteration_checkpoint = list(checkpoint_data.keys())[0]
                    # we just want the latest result
                    # iteration_checkpoint = max([it for it in checkpoint_data.keys()])
                except KeyError: 
                    # If a checkpoint exists but not for the sector of interest
                    return 0, None, None
                except EOFError:
                    pass
            return iteration_checkpoint, checkpoint_data[iteration_checkpoint]['x'],checkpoint_data[iteration_checkpoint]['fun']
        except FileNotFoundError:
            return 0, None, None
    else: 
        # If no checkpoint file exists at all 
        return 0, None, None


# Restore
def enableprint():
    sys.stdout = sys.__stdout__


def fit_func(x, a, b):
    return a * np.power(x, b)


def random_two_impurity_example(seed):
    random.seed(seed)
    np.random.seed(seed)

    n_imp_orbs = 2
    n_bath_orbs = 2
    n_imp_spin_orbs = 2 * n_imp_orbs
    n_bath_spin_orbs = 2 * n_bath_orbs

    # Impurity hopping
    # vec = np.random.choice([-5, -3, -1, 1, 3, 5], n_imp_spin_orbs)
    # himp_symm = np.diag(vec) / 1.
    sub_himp = np.random.randint(-10, 11, size=(n_imp_orbs, n_imp_orbs))
    sub_himp_symm = (sub_himp + sub_himp.T) / 2
    himp_symm = np.kron(np.eye(2, dtype=int), sub_himp_symm)  # 2 is number of repeats

    # Coulomb terms
    coulomb_1 = random.randrange(1, 11, 1)  # usually goes to ~10. Repulsive interaction
    coulomb_2 = coulomb_1  # Keep both impurity interaction terms same for now.
    uimp = np.zeros((n_imp_spin_orbs, n_imp_spin_orbs, n_imp_spin_orbs, n_imp_spin_orbs))
    uimp[0, 2, 0, 2] = - coulomb_1  # These should be the only two non-zero elements per spin sector
    uimp[2, 0, 2, 0] = - coulomb_1  # Note the relative minus sign versus Francois' Coulomb terms
    uimp[1, 3, 1, 3] = - coulomb_2
    uimp[3, 1, 3, 1] = - coulomb_2  # Use same coulomb for both sites for now. Should be correct.

    # Hybridization
    sub_vhyb = np.random.randint(1, 4, size=(n_bath_orbs, n_imp_orbs))
    vhyb = np.kron(np.eye(2), sub_vhyb)  # 2 is number of repeats

    # Bath hopping
    # ebath = np.random.randint(-10, 10, (n_bath_spin_orbs, n_bath_spin_orbs))
    # ebath_symm = (ebath + ebath.T) / 2
    ebath_symm = np.zeros((n_bath_spin_orbs, n_bath_spin_orbs))
    for jj in range(0, n_bath_orbs):
        bath_energy = random.randrange(-3, 3, 1)
        while bath_energy == 0:
            bath_energy = random.randrange(-3, 3, 1)
        ebath_symm[jj, jj] = bath_energy
        ebath_symm[jj + n_bath_orbs, jj + n_bath_orbs] = bath_energy

    return himp_symm, uimp, vhyb, ebath_symm


def random_n_site_up_down_symmetric_example(n, seed):
    ''' Create a random up-down symmetric Hamtiltonian for one impurity 
    '''
    random.seed(seed)
    np.random.seed(seed)

    n_imp_orbs = 1
    n_bath_orbs = n - 1
    n_imp_spin_orbs = 2 * n_imp_orbs
    n_bath_spin_orbs = 2 * n_bath_orbs


    # Coulomb terms
    coulomb = random.uniform(1,10)
    uimp = np.zeros((n_imp_spin_orbs, n_imp_spin_orbs, n_imp_spin_orbs, n_imp_spin_orbs))
    uimp[0, 1, 0, 1] = - coulomb  # These should be the only two non-zero elements
    uimp[1, 0, 1, 0] = - coulomb  # Note the relative minus sign versus Francois' Coulomb terms

    # Impurity hopping
    impurity_energy = random.uniform(-5,5)
    # Bind impurity energies to the selection of coulombic energy
    #bounded_choice_array = [coulomb/2-2, coulomb/2, -coulomb/2+2]
    #impurity_energy = np.random.choice(a=bounded_choice_array, size=1)
    himp_symm = np.zeros((n_imp_spin_orbs, n_imp_spin_orbs))
    for i in range(n_imp_spin_orbs):
        himp_symm[i, i] = impurity_energy

    # Hybridization
    vhyb = np.zeros((n_bath_spin_orbs, n_imp_spin_orbs))
    v = [random.uniform(-5, 5) for _ in range(n_bath_orbs)]
    for b in range(n_bath_orbs):
        # spin up register
        vhyb[b, 0] = v[b]
        # spin down register
        vhyb[n_bath_orbs + b, 1] = v[b]

    # Bath hopping
    ebath_symm = np.zeros((n_bath_spin_orbs, n_bath_spin_orbs))
    bath_energy = [random.uniform(-5, 5) for _ in range(n_bath_orbs)]
    while np.any(bath_energy == 0.):
        bath_energy = [random.uniform(-5, 5) for _ in range(n_bath_orbs)]
    for i in range(n_bath_spin_orbs):
        ebath_symm[i, i] = bath_energy[i % n_bath_orbs]

    return himp_symm, uimp, vhyb, ebath_symm


def random_two_site_up_down_symmetric_example(seed):
    random.seed(seed)
    np.random.seed(seed)

    n_imp_orbs = 1
    n_bath_orbs = 1
    n_imp_spin_orbs = 2 * n_imp_orbs
    n_bath_spin_orbs = 2 * n_bath_orbs

    # Impurity hopping
    impurity_energy = random.randrange(-5, 5, 1)
    himp_symm = np.zeros((n_imp_spin_orbs, n_imp_spin_orbs))
    himp_symm[0, 0] = impurity_energy
    himp_symm[1, 1] = impurity_energy

    # Coulomb terms
    coulomb = random.randrange(1, 10, 1)  # usually goes to 10
    uimp = np.zeros((n_imp_spin_orbs, n_imp_spin_orbs, n_imp_spin_orbs, n_imp_spin_orbs))
    uimp[0, 1, 0, 1] = - coulomb  # These should be the only two non-zero elements
    uimp[1, 0, 1, 0] = - coulomb  # Note the relative minus sign versus Francois' Coulomb terms

    # Hybridization
    vhyb = np.zeros((n_bath_spin_orbs, n_imp_spin_orbs))
    v1 = random.randrange(-5, 5, 1)
    vhyb[0, 0] = v1  # In its current form, this says that hybridization is symmetric between spin up and down compts.
    vhyb[1, 1] = v1

    # Bath hopping
    ebath_symm = np.zeros((n_bath_spin_orbs, n_bath_spin_orbs))
    bath_energy = random.randrange(-5, 5, 1)
    while bath_energy == 0:
        bath_energy = random.randrange(-5, 5, 1)
    ebath_symm[0, 0] = bath_energy
    ebath_symm[1, 1] = bath_energy

    return himp_symm, uimp, vhyb, ebath_symm


def francois_example_inputs():
    """
    norb = 3  # number of orbitals (impurity + bath)
    nq = 2 * norb  # number of qubits

    Uimp = 10.  # local coulomb interaction on the impurity
    eimp = np.zeros((norb, norb))  # diagonal element are the local energies of the impurity and bath, off diagonal elements are the hopping between the differents sites
    eimp[0, 0] = -Uimp / 2  # half filling

    # For the rest of the element, here we choose random elements, but in principle it should come from a bath fitting (See presentation)
    v1 = 2
    v2 = 1
    eimp[0, 1] = v1
    eimp[1, 0] = v1
    eimp[0, 2] = v2
    eimp[2, 0] = v2
    eimp[1, 1] = -1
    eimp[2, 2] = -3
    """

    # My re-writing. In each object, all indices with move through up spins and then down.
    Uimp = 10.
    v1 = 2.
    v2 = 1.

    n_imp_orbs = 1
    n_imp_spin_orbs = 2 * n_imp_orbs

    n_bath_orbs = 2
    n_bath_spin_orbs = 2 * n_bath_orbs

    # Impurity hopping
    himp = np.zeros((n_imp_spin_orbs, n_imp_spin_orbs))  # No hopping between up and down states. Half-filling
    himp[0, 0] = -Uimp / 2.
    himp[1, 1] = -Uimp / 2.

    # Coulomb terms
    uimp = np.zeros((n_imp_spin_orbs, n_imp_spin_orbs, n_imp_spin_orbs, n_imp_spin_orbs))
    uimp[0, 1, 0, 1] = - Uimp  # These should be the only two non-zero elements
    uimp[1, 0, 1, 0] = - Uimp  # Note the relative minus sign versus Francois' Coulomb terms

    # Hybridization
    vhyb = np.zeros((n_bath_spin_orbs, n_imp_spin_orbs))  # Multiplies terms which annihilate impurity and create bath
    vhyb[0, 0] = v1
    vhyb[1, 0] = v2
    vhyb[2, 1] = v1
    vhyb[3, 1] = v2

    # Bath hopping
    ebath = np.zeros((n_bath_spin_orbs, n_bath_spin_orbs))
    ebath[0, 0] = -1
    ebath[1, 1] = -3
    ebath[2, 2] = -1
    ebath[3, 3] = -3
    return himp, uimp, vhyb, ebath

# TODO: Create parent class such that classes below can inherit shared methods (avoid repetitive definitions)
class AndersonImpurityModel:
    """
    This class specifies an Anderson Impurity Model and its associated methods. In order to instantiate you must
    specify four inputs:

    himp is a matrix containing the single-particle impurity Hamiltonian terms
    uimp is a rank 4 tensor containing the Coulomb interactions of the impurity
    vhyb are the hybridization terms between the impurity and bath
    ebath are the single-particle terms of the bath

    Note that himp and ebath should be Hermitian, uimp also, but with additional permutational symmetries. In general,
    all four inputs can be complex-valued.
    """

    def __init__(self, himp, uimp, vhyb, ebath):
        self.himp = himp
        self.uimp = uimp
        self.vhyb = vhyb
        self.ebath = ebath
        # Should put checks on Hermiticity and permutational symmetry of inputs

    def return_n_bath_n_imp_orb(self):
        n_bath_orb = self.ebath.shape[0] // 2
        n_imp_orb = self.himp.shape[0] // 2
        n_bath_n_imp_orb = (n_bath_orb, n_imp_orb)
        return n_bath_n_imp_orb

    def return_n_bath_n_imp_so(self):
        n_bath_so = self.ebath.shape[0]
        n_imp_so = self.himp.shape[0]
        n_bath_n_imp_so = (n_bath_so, n_imp_so)
        return n_bath_n_imp_so

    def create_aim_graphs(self):
        n_bath_n_imp_orb_tup = self.return_n_bath_n_imp_orb()
        n_site_model_idx = AIMSiteModelsEnum(n_bath_n_imp_orb_tup).create_n_site_model_idx()
        aim_graphs = create_connected_graphs(n_site_model_idx=n_site_model_idx,
                                                                        show_sub_graphs=False,
                                                                        show_full_plot=False)
        return aim_graphs

    def orbital_maps(self, imp_middle=True):
        # This utility will create two lists. Indices of the lists are impurity and orbital 1-body indices.
        # List values are the logical qubits they map to.
        # TODO: Note that currently imp_middle has no functionality anymore
        total_graph = self.create_aim_graphs()["graph_full"]
        # Return the indices of all the bath and imps - note you must construct the full graphs instead of just
        # instantiating the NSiteModelIdx because of the re-indexing that occurs after taking out vacant sites
        bath_map = sorted([idx for idx, attributes in total_graph.nodes(data=True) if attributes['type'] == 'B'])
        imp_map = sorted([idx for idx, attributes in total_graph.nodes(data=True) if attributes['type'] == 'I'])

        return imp_map, bath_map

    def get_first_imp_orbital_idx(self):
        """Returns the first impurity orbital index. Note this must be done after the full graph is constructed."""
        total_graph = self.create_aim_graphs()["graph_full"]
        imp_orb_idx = [sorted([idx for idx, attributes in total_graph.nodes(data=True) if attributes['type'] == 'I'])[0]]
        return imp_orb_idx

    # AIM instance method to construct the Fermionic Hamiltonian
    def construct_fermionic_hamiltonian(self):
        n_bath_so = self.return_n_bath_n_imp_so()[0]
        n_imp_so = self.return_n_bath_n_imp_so()[1]

        # Find orbital maps
        imp_map, bath_map = self.orbital_maps()

        # Instantiate Fermionic Hamiltonian
        h_fermionic = FermionOperator()
        # Add terms from impurity hopping (note for one impurity in the model this will only be the self-energy)
        for a, b in product(range(n_imp_so), range(n_imp_so)):
            h_fermionic += FermionOperator(f"{imp_map[a]}^ {imp_map[b]}", self.himp[a, b])

        # Add terms from bath hopping
        for i, j in product(range(n_bath_so), range(n_bath_so)):
            h_fermionic += FermionOperator(f"{bath_map[i]}^ {bath_map[j]}", self.ebath[i, j])

        # Add terms from hybridization
        for i, a in product(range(n_bath_so), range(n_imp_so) ):
            h_fermionic += FermionOperator(f"{bath_map[i]}^ {imp_map[a]}", self.vhyb[i, a])
            h_fermionic += FermionOperator(f"{imp_map[a]}^ {bath_map[i]}", np.conj(self.vhyb[i, a]))  # Hermit conjugate

        # Need to add terms for Coulomb interaction
        for a in range(n_imp_so):
            for b in range(n_imp_so):
                for c in range(n_imp_so):
                    for d in range(n_imp_so):
                        h_fermionic += 0.5 * FermionOperator(f"{imp_map[a]}^ {imp_map[b]}^ {imp_map[c]} {imp_map[d]}",
                                                             self.uimp[a, b, c, d])
        return h_fermionic

    # AIM instance method to construct charge and spin operators
    def construct_charge_and_spin_operators(self):
        n_bath_so = self.return_n_bath_n_imp_so()[0]
        n_imp_so = self.return_n_bath_n_imp_so()[1]
        charge = 0
        spin = 0
        # TODO: Replace below with reference to graph
        for i in range(n_imp_so + n_bath_so):
            charge += FermionOperator(f"{i}^ {i} ", 1.0)
            if i < (n_imp_so + n_bath_so) // 2:
                spin += FermionOperator(f"{i}^ {i} ", 1.0)
            else:
                spin += FermionOperator(f"{i}^ {i} ", -1.0)
        return charge, spin

    def return_up_and_down_indices(self):
        up_graph, down_graph = itemgetter("graph_up", "graph_down")(self.create_aim_graphs())
        up_indices = sorted(up_graph)
        down_indices = sorted(down_graph)
        return up_indices, down_indices

    # AIM instance method to obtain qubit-transformed Hamiltonian, e.g., Jordan-Wigner or Bravyi-Kitaev
    def construct_qubit_hamiltonian(self, method="jordan-wigner"):
        h_fermionic = self.construct_fermionic_hamiltonian()
        if method == "jordan-wigner":
            h_qubit = jordan_wigner(h_fermionic)
        elif method == "bravyi-kitaev":
            h_qubit = bravyi_kitaev(h_fermionic)
        # Later will put in binary encodings
        else:
            raise ValueError("Not a valid transform.")
        return h_qubit

    # AIM instance method to obtain GS or low-lying spectrum of Hamiltonian
    def exact_diagonalization(self, quantity="gs"):
        h_qubit = self.construct_qubit_hamiltonian()
        h_sparse = get_sparse_operator(h_qubit)
        # print("Sparse Ham:", h_sparse)
        if quantity == "gs":
            ground_state = get_ground_state(h_sparse)
        else:
            raise ValueError("Not a valid quantity.")

        # Check charge/spin sectors of exact ground state
        charge_ferm, spin_ferm = self.construct_charge_and_spin_operators()  # These are fermion operators
        # Translate to qubit operators via J-W
        charge_qubit = jordan_wigner(charge_ferm)
        spin_qubit = jordan_wigner(spin_ferm)
        # Translate to sparse matrix operators
        charge_sparse = get_sparse_operator(charge_qubit)
        spin_sparse = get_sparse_operator(spin_qubit)
        # Expectation values
        charge_exp_val = round(expectation(charge_sparse, ground_state[1]).real, 3)
        spin_exp_val = round(expectation(spin_sparse, ground_state[1]).real, 3)
        gap = get_gap(h_sparse)
        return ground_state, charge_exp_val, spin_exp_val, gap


class MeanFieldQulacsVqeEmulator:
    """
    This class emulates the hardware efficient variational quantum eigensolver. Initially it will be built just
    to find ground states of a Hamiltonian (or cost function represented there-as). The Hamiltonian can be constructed
    first in OpenFermion but should be cast to a qulacs object first before class instantiation.
    """

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian

    def spin_coherent_ansatz(self, theta, compilation="standard"):
        n_qubits = self.hamiltonian.get_qubit_count()
        qc = ParametricQuantumCircuit(n_qubits)

        if compilation == "standard":
            k = 0

            for i in range(n_qubits):
                qc.add_parametric_RY_gate(i, theta[k])
                k += 1
                qc.add_parametric_RZ_gate(i, theta[k])
                k += 1
        else:
            raise ValueError("Not a valid compilation.")
        return qc

    def expectation_value(self, params):
        n_qubits = self.hamiltonian.get_qubit_count()
        ws = QuantumState(n_qubits)
        ws.set_zero_state()
        qc_ansatz = self.spin_coherent_ansatz(theta=params)
        ws.set_zero_state()
        qc_ansatz.update_quantum_state(ws)
        return self.hamiltonian.get_expectation_value(ws).real

    def solve_ground_state_local(self, optimizer='BFGS'):
        """
        In order to solve GS using local optimization, just need Hamiltonian specified in init, a number of layers,
        and a local optimizer.
        :param n_layers:
        :param optimizer:
        :return:
        """
        n_qubits = self.hamiltonian.get_qubit_count()
        n_params = n_qubits * 2
        theta_0 = np.random.random(n_params)

        if optimizer == "diff_ev":
            maxiter = 1000
            res = differential_evolution(self.expectation_value, bounds=[(0, 2 * np.pi) for i in range(n_params)])
        elif optimizer == "BFGS":
            res = minimize(self.expectation_value, theta_0, method=optimizer, tol=1e-9,
                           options={'maxiter': 1e9})
        elif optimizer == "COBYLA":
            res = minimize(self.expectation_value, theta_0, method=optimizer, tol=1e-9,
                           options={'maxiter': 1e9})
        else:
            raise ValueError("Not a valid optimizer")

        gs_energy = self.expectation_value(params=res.x)
        # print("Qulacs GS Energy Check:", gs_energy, res.fun)
        # Let's calculate explicit statevector info too
        qgs = QuantumState(n_qubits)
        qgs.set_zero_state()
        self.spin_coherent_ansatz(theta=res.x).update_quantum_state(qgs)
        return res, qgs

    def expectation_value(self, params, n_layers):
        n_qubits = self.hamiltonian.get_qubit_count()
        ws = QuantumState(n_qubits)
        ws.set_zero_state()
        qc_ansatz = self.hardware_efficient_ansatz(theta=params, n_layers=n_layers)
        ws.set_zero_state()
        qc_ansatz.update_quantum_state(ws)
        return self.hamiltonian.get_expectation_value(ws).real

    def solve_ground_state_local(self, n_layers=1, optimizer='BFGS'):
        """
        In order to solve GS using local optimization, just need Hamiltonian specified in init, a number of layers,
        and a local optimizer.
        :param n_layers:
        :param optimizer:
        :return:
        """
        n_qubits = self.hamiltonian.get_qubit_count()
        n_params = n_qubits + n_layers * (2 * n_qubits - 2)
        theta_0 = np.random.random(n_params)

        if optimizer == "diff_ev":
            maxiter = 1000
            res = differential_evolution(self.expectation_value, bounds=[(0, 2 * np.pi) for i in range(n_params)],
                                         args=(n_layers,))  # , maxiter=maxiter, tol=1.e-10)  # np.sqrt(shots))
        elif optimizer == "BFGS":
            res = minimize(self.expectation_value, theta_0, args=n_layers, method=optimizer, tol=1e-7,
                           options={'maxiter': 1e7})
        elif optimizer == "COBYLA":
            res = minimize(self.expectation_value, theta_0, args=n_layers, method=optimizer, tol=1e-7,
                           options={'maxiter': 1e7})
        else:
            raise ValueError("Not a valid optimizer")

        gs_energy = self.expectation_value(params=res.x, n_layers=n_layers)
        # print("Qulacs GS Energy Check:", gs_energy, res.fun)
        # Let's calculate explicit statevector info too
        qgs = QuantumState(n_qubits)
        qgs.set_zero_state()
        self.hardware_efficient_ansatz(theta=res.x, n_layers=n_layers).update_quantum_state(qgs)
        return res, qgs

class SymmQulacsVqeEmulator:
    """
    This class emulates the hardware efficient variational quantum eigensolver. Initially it will be built just
    to find ground states of a Hamiltonian (or cost function represented there-as). The Hamiltonian can be constructed
    first in OpenFermion but should be cast to a qulacs object first before class instantiation.
    """

    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian
        #self.checkpoint_file = ''
        #self.spin_charge = None
    
    def checkpoint(self, checkpoint_file):
        # Bind the name of the checkpoint file to the class instance
        self.checkpoint_file = checkpoint_file

    def change_sector(self, spin_charge):
        # Bind the current spin/charge sector to the class instance
        self.spin_charge = spin_charge

    def all_symmetry_ansatzae(self, theta, n_layers, initial_occupations_indices,
                              connected_graphs: dict[str, nx.Graph],
                              compilation="generic"):
        n_qubits = self.hamiltonian.get_qubit_count()
        qc = ParametricQuantumCircuit(n_qubits)
        graph_up, graph_down, graph_stitch = itemgetter("graph_up", "graph_down", "graph_stitch")(connected_graphs)
        if compilation == "generic":
            # Create spin-delineated interaction graphs and reference mappings
            # Instantiates which charge/spin sector we're in
            for inds in initial_occupations_indices:
                qc.add_X_gate(inds)

            k = 0  # This index keeps track of theta parameter angles
            # Block below iterates through requisite number of layers
            for layer in range(n_layers):

                # Block below this comment iterates through 2d NN connections on spin up qubits
                for edge in graph_up.edges():
                    qubit_a = edge[0]
                    qubit_b = edge[1]
                    # This block of code constitutes one Givens/SO(4) rotation gate
                    # qc.barrier()
                    qc.add_S_gate(qubit_a)
                    qc.add_S_gate(qubit_b)
                    qc.add_H_gate(qubit_a)
                    qc.add_CNOT_gate(qubit_a, qubit_b)
                    # qc.ry(theta[k]/2., qubit_a)
                    # qc.ry(theta[k]/2., qubit_b)
                    # print("k:", k)
                    qc.add_parametric_RY_gate(qubit_a, theta[k])
                    qc.add_parametric_RY_gate(qubit_b, theta[k])
                    k += 1  # Increments theta counter
                    qc.add_CNOT_gate(qubit_a, qubit_b)
                    qc.add_H_gate(qubit_a)
                    qc.add_Sdag_gate(qubit_a)
                    qc.add_Sdag_gate(qubit_b)

                    # qc.add_parametric_multi_Pauli_rotation_gate([qubit_a, qubit_b], [3, 3], theta[k])
                    # k += 1

                # Block below iterates through all 2d NN connections on spin down qubits
                for edge in graph_down.edges():
                    qubit_a = edge[0]
                    qubit_b = edge[1]
                    # This block of code constitutes one Givens/SO(4) rotation gate
                    # qc.barrier()
                    qc.add_S_gate(qubit_a)
                    qc.add_S_gate(qubit_b)
                    qc.add_H_gate(qubit_a)
                    qc.add_CNOT_gate(qubit_a, qubit_b)
                    # qc.ry(theta[k]/2., qubit_a)
                    # qc.ry(theta[k]/2., qubit_b)
                    # print("k:", k)
                    qc.add_parametric_RY_gate(qubit_a, theta[k])
                    qc.add_parametric_RY_gate(qubit_b, theta[k])
                    k += 1  # Increments theta counter
                    qc.add_CNOT_gate(qubit_a, qubit_b)
                    qc.add_H_gate(qubit_a)
                    qc.add_Sdag_gate(qubit_a)
                    qc.add_Sdag_gate(qubit_b)

                # Connect qubits across up/down patches
                for edge in graph_stitch.edges():
                    # TODO: Check that this is always faithfully assigning up/down qubits properly
                    qubit_a = edge[0]  # up qubit
                    qubit_b = edge[1]  # down qubit
                    qc.add_CNOT_gate(qubit_a, qubit_b)
                    qc.add_parametric_RZ_gate(qubit_b, theta[k])
                    qc.add_CNOT_gate(qubit_a, qubit_b)
                    # This decomposition should be equivalent to commented line below
                    # qc.add_parametric_multi_Pauli_rotation_gate([qubit_a, qubit_b], [3, 3], theta[k])

                    k += 1

                for q in range(0, n_qubits):
                    qc.add_parametric_RZ_gate(q, theta[k])
                    k += 1

        elif compilation == "with_phases":
            # These four lines create spin-delineated interaction graphs and reference mappings
            # Instantiates which charge/spin sector we're in
            for inds in initial_occupations_indices:
                qc.add_X_gate(inds)

            k = 0  # This index keeps track of theta parameter angles
            # Block below iterates through requisite number of layers
            for layer in range(n_layers):
                # Block below this comment iterates through 2d NN connections on spin up qubits

                for edge in graph_up.edges():
                    qubit_a = edge[0]
                    qubit_b = edge[1]

                    qc.add_CNOT_gate(qubit_b, qubit_a)
                    alpha = theta[k]
                    k += 1
                    beta = theta[k]
                    k += 1
                    qc.add_parametric_RZ_gate(qubit_b, -beta - np.pi)
                    qc.add_parametric_RY_gate(qubit_b, -alpha - np.pi / 2.)
                    qc.add_CNOT_gate(qubit_a, qubit_b)
                    qc.add_parametric_RY_gate(qubit_b, alpha + np.pi / 2.)
                    qc.add_parametric_RZ_gate(qubit_b, beta + np.pi)
                    qc.add_CNOT_gate(qubit_b, qubit_a)

                for edge in graph_down.edges():
                    qubit_a = edge[0]
                    qubit_b = edge[1]

                    qc.add_CNOT_gate(qubit_b, qubit_a)
                    alpha = theta[k]
                    k += 1
                    beta = theta[k]
                    k += 1
                    qc.add_parametric_RZ_gate(qubit_b, -beta - np.pi)
                    qc.add_parametric_RY_gate(qubit_b, -alpha - np.pi / 2.)
                    qc.add_CNOT_gate(qubit_a, qubit_b)
                    qc.add_parametric_RY_gate(qubit_b, alpha + np.pi / 2.)
                    qc.add_parametric_RZ_gate(qubit_b, beta + np.pi)
                    qc.add_CNOT_gate(qubit_b, qubit_a)
        else:
            raise ValueError("Not a valid compilation.")
        return qc

    def expectation_value(self, params, n_layers, initial_occupations_indices,
                          connected_graphs: dict[str, nx.Graph],
                          compilation="generic"):
        n_qubits = self.hamiltonian.get_qubit_count()
        ws = QuantumState(n_qubits)
        qc_ansatz = self.all_symmetry_ansatzae(theta=params, n_layers=n_layers,
                                               initial_occupations_indices=initial_occupations_indices,
                                               connected_graphs=connected_graphs,
                                               compilation=compilation)
        ws.set_zero_state()
        qc_ansatz.update_quantum_state(ws)
        return self.hamiltonian.get_expectation_value(ws).real

    def solve_ground_state_local(self, n_layers, optimizer, up_qubit_indices, down_qubit_indices, impurity_orbital_idx,
                                 initial_occupations_indices, gtol, compilation="generic", checkpoint_data=(), 
                                 spin_charge=(), checkpoint_file='', spin_op=None, charge_op=None):
        """
        In order to solve GS using local optimization, just need Hamiltonian specified in init, a number of layers,
        and a local optimizer.
        :param compilation:
        :param initial_occupations_indices:
        :param up_qubit_indices:
        :param down_qubit_indices:
        :param n_layers:
        :param optimizer:
        :return:
        """
        setattr(self, 'nit', 0)
        setattr(self, 'spin_charge', spin_charge)
        setattr(self, 'checkpoint_file', checkpoint_file)
        n_qubits = self.hamiltonian.get_qubit_count()
        # Create graphs wth new functionality:
        n_bath_n_imp_tup = ((len(up_qubit_indices) - len(impurity_orbital_idx)), len(impurity_orbital_idx))
        n_site_model_idx = AIMSiteModelsEnum(n_bath_n_imp_tup).create_n_site_model_idx()
        connected_graphs = create_connected_graphs(n_site_model_idx=n_site_model_idx,
                                                             show_sub_graphs=False,
                                                             show_full_plot=False)

        n_edges = sum([len(connected_graphs[key].edges()) for key in ["graph_up", "graph_down", "graph_stitch"]])

        if compilation == "generic":
            n_params = n_layers * (n_edges + n_qubits)  # This factor of two accounts for additional cphase
            # n_params = n_layers * n_edges  # Includes layer of Rzs after Givens rotations
        elif compilation == "with_phases":
            n_params = n_layers * n_edges * 2
        else:
            raise ValueError("Not a valid ansatz.")
        theta_0 = np.random.uniform(low=0, high=2 * np.pi, size=n_params)
        #opt_args = (n_layers, initial_occupations_indices, connected_graphs, compilation, spin_charge, checkpoint_file)
        opt_args = (n_layers, initial_occupations_indices, connected_graphs, compilation) #
        gradients = []
        grad_norms = []
        cost_histories = []
        nm_theta_bounds = Bounds(lb=0, ub=2*np.pi)
        # Currently depreciated since we are not using this data, speed up runtime a bit by not tracking (note this tracks cost histories, etc):
        # def callback(theta):
        #     # Add values to the jacobian array every iteration
        #     grad_it = approx_fprime(theta, self.expectation_value, np.sqrt(np.finfo(float).eps), n_layers,
        #                             initial_occupations_indices, connected_graphs,
        #                             compilation)
        #     gradients.append(list(grad_it))  # when we go to dump this to a json it needs to be a list, not ndarray
        #     # Calculate L_inf norm (as is done in scipy optimize)
        #     grad_norm = np.linalg.norm(x=grad_it, ord=Inf)
        #     grad_norms.append(grad_norm)
        #     # Add values to the cost history array every iteration
        #     cost_histories.append(
        #         self.expectation_value(theta, n_layers, initial_occupations_indices,
        #                                connected_graphs,
        #                                compilation))
        def callback(intermediate_result: OptimizeResult):
        # Checkpointing callback function - save state
            iteration_save_w = 100
            nit = getattr(self, 'nit')
            if nit % iteration_save_w == 0:  # Checkpoint every iteration_save_w iterations
                save_checkpoint(intermediate_result.x, intermediate_result.fun, nit, self.spin_charge, self.checkpoint_file)
            setattr(self, 'nit', nit+1)

        if optimizer == "diff_ev":
            maxiter = 1000
            res = differential_evolution(self.expectation_value, bounds=[(0, 2 * np.pi) for i in range(n_params)],
                                         args=opt_args, maxiter=maxiter, tol=.01) 
        elif optimizer == "BFGS":
                res = minimize(fun=self.expectation_value, x0=theta_0, args=opt_args, method=optimizer,
                options={'maxiter': 1e9, 'gtol': gtol})
                # Call save one more time to save the very last iteration data on completion
                #save_checkpoint(res.x, res.fun, res.nit, self.spin_charge, self.checkpoint_file)
            #If data exists for this spin/charge sector, load and continue, else start from scratch - first element is the iteration number
            #if checkpoint_data[0]!= 0 : 
            # Three different cases here - (1) where there is no data, (2) resume iteration at a certain checkpoint, (3) prev. checkpoint was converged (treat 2/3 the same as it's one iteration) 
            # resuming from checkpoint
            # if isinstance(checkpoint_data[1], np.ndarray):  
            #     last_iter = checkpoint_data[0]
            #     setattr(self, 'nit', last_iter)
            #     xn = checkpoint_data[1]
            #     res = minimize(fun=self.expectation_value, x0=xn, args=opt_args, method=optimizer,
            #     options={'maxiter': 1e9-last_iter, 'gtol': gtol}, callback=callback)

            # Starting for the first time
            # else: 
            #     # reset Nfeval to 0 upon a new optimization routine
            #     setattr(self, 'nit', 0)
            #     res = minimize(fun=self.expectation_value, x0=theta_0, args=opt_args, method=optimizer,
            #     options={'maxiter': 1e9, 'gtol': gtol}, callback=callback)
            #     # Call save one more time to save the very last iteration data on completion
            #     save_checkpoint(res.x, res.fun, res.nit, self.spin_charge, self.checkpoint_file)
        elif optimizer == "Nelder-Mead":
            res = minimize(self.expectation_value, theta_0, args=opt_args, method=optimizer, tol=1e-9, bounds=nm_theta_bounds,
                           options={'maxiter': 1e9})
        elif optimizer == "COBYLA":
            res = minimize(self.expectation_value, theta_0, args=opt_args, method=optimizer, tol=1e-4,
                           options={'maxiter': 1e9})
        else:
            raise ValueError("Not a valid optimizer")
        # Let's calculate explicit statevector info too
        qgs = QuantumState(n_qubits)
        qgs.set_zero_state()
        self.all_symmetry_ansatzae(theta=res.x, n_layers=n_layers,
                                   initial_occupations_indices=initial_occupations_indices,
                                   connected_graphs=connected_graphs,
                                   compilation=compilation).update_quantum_state(qgs)

        # Calculate qulacs data here instead of carrying QuantumState vectors around everywhere
        qulacs_charge = round(charge_op.get_expectation_value(qgs).real, 3)
        qulacs_spin = round(spin_op.get_expectation_value(qgs).real, 3)
        qulacs_spin_charge = (qulacs_spin, qulacs_charge)

        return res, qulacs_spin_charge, cost_histories, grad_norms

def symmetric_ansatz_test(self, model_hamiltonian, n_layers, optimizer, gtol,
                          is_noise=False, compilation="generic", display=False, checkpoint_file="", results_file="") -> dict:
    up_qubit_indices, down_qubit_indices = model_hamiltonian.return_up_and_down_indices()
    impurity_orbital_idx = model_hamiltonian.get_first_imp_orbital_idx()
    qubit_ham = model_hamiltonian.construct_qubit_hamiltonian(method="jordan-wigner")
    qulacs_ham = create_quantum_operator_from_openfermion_text(f"{qubit_ham}")
    qulacs_solver = SymmQulacsVqeEmulator(qulacs_ham)
    # Now need to iterate through different charge and spin sectors
    n_qubits = count_qubits(qubit_ham)  # Generally, I think we can assume n_qubits is even since = 2 * n_orbs
    sector_to_energy_dict = {}
    sector_to_nfev_dict = {}
    sector_to_result_dict = {}
    shots_sector_to_energy_dict = {}
    shots_sector_to_nfev_dict = {}
    shots_sector_to_result_dict = {}
    sector_to_history_dict = {}
    sector_to_opt_history_dict = {}
    sector_to_nit_dict = {}
    sector_to_njev_dict = {}

    # Try and load any results from the checkpoint file here 
    all_res = load_all_results(results_file)
    completed_sectors = [list(dict.keys())[0] for dict in  all_res]
    print(completed_sectors)

    # Calculating the total number of spin/charge sectors (without the redundant cases where S_z>0)
    num_secs = ((n_qubits//2 + 1) * (n_qubits//2 + 2))//2

    ###
    if display:
        if len(completed_sectors) == 0:
            print("No previous checkpoint data loaded, starting complete spin, charge sector search from start.")
        else:
            print("Previous checkpoint data loaded.")
            # Catching the case where the checkpoint data is a complete set
            if len(completed_sectors) == num_secs:
                print("All spin, charge sector searches complete from previous checkpoint data.")
            else: 
                print(f"{len(completed_sectors)} / {num_secs} VQE runs performed.")
                print(f"Restarting search at spin, charge sector = ({completed_sectors[-1]}) ")
    ###

    # Charge Sector (N) < N_q//2 
    for n_electrons in range(0, n_qubits // 2 + 1): # includes all-zeros state
    #for n_electrons in range(1, n_qubits // 2 + 1):  # Excludes trivial all-zeros state
        # continue
        ###
        if display:
            print("Particle_number:", n_electrons)
        ###
        for z_spin in range(-n_electrons, n_electrons + 2, 2):
            if z_spin > 0: # This should be correct for up-down symmetric case (Just searching over S_z <= 0)
                continue  
            ###
            if display:
                print("\t Z-spin eigenvalue:", z_spin)
            ###
            n_up = (n_electrons + z_spin) // 2
            n_down = (n_electrons - z_spin) // 2
            # Next two lines take care of spin up sector
            most_equispaced_indices = np.linspace(0, len(up_qubit_indices) - 1, n_up, dtype=int).tolist()
            initial_occupations_indices = [up_qubit_indices[i] for i in most_equispaced_indices]
            # Next two lines take care of spin down sector
            most_equispaced_indices = np.linspace(0, len(down_qubit_indices) - 1, n_down, dtype=int).tolist()
            initial_occupations_indices.extend([down_qubit_indices[i] for i in most_equispaced_indices])
            initial_occupations_indices.sort()
            ###
            if display:
                print("\t Init. Occ. Inds:", initial_occupations_indices)
            ###
            
            # First check if the spin/charge sector already exists within the results file
            spin_charge = (z_spin, n_electrons)
            if spin_charge in completed_sectors: 
                continue 
            else: 
            #Check if there is already a checkpoint for the given spin/charge sector, if so then resume from that point 
                setattr(self, 'spin_charge', spin_charge)
                setattr(self, 'checkpoint_file', checkpoint_file)
                # We have the AIM object within this scope so use it to construct charge/spin ops
                charge, spin = model_hamiltonian.construct_charge_and_spin_operators()
                charge_op = create_quantum_operator_from_openfermion_text(f"{jordan_wigner(charge)}")
                spin_op = create_quantum_operator_from_openfermion_text(f"{jordan_wigner(spin)}")
        
                #c_data = load_checkpoint(checkpoint_file=checkpoint_file, sector=spin_charge)
                c_data = None
                # c_data could be 0,None,None if not loaded 
                result, qulacs_spin_charge, local_cost_histories, local_grad_norms = qulacs_solver.solve_ground_state_local(n_layers, optimizer, up_qubit_indices,
                                                    down_qubit_indices, impurity_orbital_idx,
                                                    initial_occupations_indices,
                                                    compilation=compilation,
                                                    gtol=gtol,
                                                    checkpoint_data=c_data,
                                                    spin_charge=spin_charge, 
                                                    checkpoint_file=checkpoint_file, 
                                                    spin_op=spin_op,
                                                    charge_op=charge_op,
                                                    )
                del spin_op
                del charge_op
                # Pickle successful OptimizeResult
                # First remove the hess_inv and jac from each OptimizeResult
                delattr(result, 'hess_inv')
                delattr(result, 'jac')
                save_result(spin_charge, result, qulacs_spin_charge, results_file)

                ###
                if display:
                    print("\t Local VQE Results:")
                    print("\t FP GS Energy:", result.fun)  # .fun
                    #print("\t FP GS Energy Diff:", result.fun - exact_gs)  # .fun
                    print(f"\t local cost histories {local_cost_histories} ")
                    print(f"\t local grad norms {local_grad_norms}")
                    #print(result)

    # Charge Sector (N) > N_q//2 
    for n_electrons in range(n_qubits // 2 + 1, n_qubits + 1):  # For full sector search want to end at n_qubits + 1
    #for n_electrons in range(n_qubits // 2 + 1, n_qubits):  # Excludes trivial all-ones state
        ###
        if display:
            print("Particle_number:", n_electrons)
        ###
        n_mirror = n_qubits - n_electrons
        for z_spin in range(-n_mirror, n_mirror + 1, 2):
            if z_spin > 0: # This should be correct for up-down symmetric case (Just searching over S_z <= 0)
                continue
            ###
            if display:
                print("\t Z-spin eigenvalue:", z_spin)
            ###
            n_up = (n_electrons + z_spin) // 2
            n_down = (n_electrons - z_spin) // 2
            # Next two lines take care of spin up sector
            most_equispaced_indices = np.linspace(0, len(up_qubit_indices) - 1, n_up, dtype=int).tolist()
            initial_occupations_indices = [up_qubit_indices[i] for i in most_equispaced_indices]
            # Next two lines take care of spin down sector
            most_equispaced_indices = np.linspace(0, len(down_qubit_indices) - 1, n_down, dtype=int).tolist()
            initial_occupations_indices.extend([down_qubit_indices[i] for i in most_equispaced_indices])
            initial_occupations_indices.sort()
            ###
            if display:
                print("\t Init. Occ. Inds:", initial_occupations_indices)
            ###
 
            spin_charge = (z_spin, n_electrons)
            if spin_charge in completed_sectors: 
                continue 
            else: 
                # Check if there is already a checkpoint for the given spin/charge sector, if so then resume from that point 
                setattr(self, 'spin_charge', spin_charge)
                setattr(self, 'checkpoint_file', checkpoint_file)

                #c_data = load_checkpoint(checkpoint_file=checkpoint_file, sector=spin_charge)
                # c_data could be 0,None,None if not loaded 
                c_data = None
                charge, spin = model_hamiltonian.construct_charge_and_spin_operators()
                charge_op = create_quantum_operator_from_openfermion_text(f"{jordan_wigner(charge)}")
                spin_op = create_quantum_operator_from_openfermion_text(f"{jordan_wigner(spin)}")
        
                result, qulacs_spin_charge, local_cost_histories, local_grad_norms = qulacs_solver.solve_ground_state_local(n_layers, optimizer, up_qubit_indices,
                                                    down_qubit_indices, impurity_orbital_idx,
                                                    initial_occupations_indices,
                                                    compilation=compilation,
                                                    gtol=gtol,
                                                    checkpoint_data=c_data,
                                                    spin_charge=spin_charge, 
                                                    checkpoint_file=checkpoint_file, 
                                                    spin_op=spin_op,
                                                    charge_op=charge_op,
                                                    )
                del spin_op
                del charge_op
                # Pickle successful OptimizeResult
                # First remove the hess_inv and jac from each OptimizeResult
                delattr(result, 'hess_inv')
                delattr(result, 'jac')
                save_result(spin_charge, result, qulacs_spin_charge, results_file)
            ###
            if display:
                print("\t Local VQE Results:")
                print("\t FP GS Energy:", result.fun)  # .fum
                #print("\t FP GS Energy Diff:", result.fun - exact_gs)  # .fum
                print(f"\t FP Local Grad Norm histories {local_grad_norms}")
                #print("\t", result)
    #         ###

    #         ###
    #         if display:
    #             print("\t FP GS Charge, Spin Check:", charge, spin)
    #             print("\n")
    #         ###
    ###
                    
    # Load in data from all the pickling and then fill back in the necessary dictionaries
    all_res = load_all_results(results_file)
    # all_res is a generator, let's rangle it into a list
    all_results_list = [dict for dict in all_res]
    
    for dict in all_results_list:
        spin_charge = list(dict.keys())[0]
        spin = spin_charge[0]
        charge = spin_charge[1]
        result = dict[spin_charge][0]
        qulacs_spin_charge = dict[spin_charge][1]
        qulacs_spin = qulacs_spin_charge[0]
        qulacs_charge = qulacs_spin_charge[1]
        # for the below block we need spin, charge, qgs and result
        sector_to_energy_dict[(qulacs_spin, qulacs_charge)] = result.fun  
        sector_to_nfev_dict[(qulacs_spin, qulacs_charge)] = result.nfev
        sector_to_nit_dict[(qulacs_spin, qulacs_charge)] = result.nit
        sector_to_njev_dict[(qulacs_spin, qulacs_charge)] = result.njev
        sector_to_result_dict[(qulacs_spin, qulacs_charge)] = result
        # Garbage to make sure rest of code works 
        local_cost_histories = []
        local_grad_norms = []
        sector_to_opt_history_dict[(qulacs_spin, qulacs_charge)] = [local_cost_histories, local_grad_norms]
    if display:
        print("Shots Sector to Energies:", shots_sector_to_energy_dict)
        print("FP Sector to Energies:", sector_to_energy_dict)
    

    ###
    # Sum up optimizer stats and find the min sector
    nit_total = sum(list((sector_to_nit_dict.values())))
    nfev_total = sum(list((sector_to_nfev_dict.values())))
    njev_total = sum(list((sector_to_njev_dict.values())))
    minimum_sector = min(sector_to_energy_dict, key=sector_to_energy_dict.get)
    ###
    if display:
        print(f"Min Sector Gradient History: {sector_to_opt_history_dict[minimum_sector]}")
    ###
    symmetric_ansatz_test_result = {  # This dictionary should make it easier to handle this function than the tuple it returned previously
        "minimum_sector": minimum_sector,
        "shots_minimum_sector": None,  # No longer a relevant value
        "minimum_energy": sector_to_energy_dict[minimum_sector],
        "shots_minimum_energy": None,  # No longer a relevant value
        "minimum_nfev": sector_to_nfev_dict[minimum_sector],
        "shots_minimum_nfev": None,  # No longer a relevant value
        "minimum_result": sector_to_result_dict[minimum_sector],
        "shots_minimum_result": None,  # No longer a relevant value
        "min_sector_grad_history": sector_to_opt_history_dict[minimum_sector][0],
        "min_sector_cost_history": sector_to_opt_history_dict[minimum_sector][1],
        "nit_total": nit_total,
        "nfev_total": nfev_total,
        "njev_total": njev_total
    }
    return symmetric_ansatz_test_result

if __name__ == "__main__":
    main2()
