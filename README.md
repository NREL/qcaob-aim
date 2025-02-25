# qcaob-aim 
For now, this repository hosts prototype code to solve the Anderson impurity model in a single script:
anderson_impurity_model.py
Note: qcaob-aim stands for quantum computation application oriented benchmark - anderson impurity model.

To run the code, a few dependencies need to be installed. Some of the main libraries include:

- Python packages (Conda is a good way to manage these as well as the additional installations described below): python, numpy, scipy, matplotlib, networkx
- OpenFermion: https://quantumai.google/openfermion
- Qulacs: https://github.com/qulacs/qulacs
- Qulacsvis: https://pypi.org/project/qulacsvis/
- Noisyopt: https://noisyopt.readthedocs.io/en/latest/

## Creating virtualenv
In order to use pip/virtualenv to create the environment needed from a list of dependencies use the commands below. You must have python version 3.11.4, pip, virtualenv, and pipenv to begin.
We choose pip as it has quicker adoption to software packages, and also we should avoid using conda and pip together as 
it can lead to inconsistent states with conflicting package dependencies. We utilize pipenv to create and manage virtualenvs for this repo. 
We provide two ways to insatll the environment, one with pipenv to manage the virtual environment [1] [RECOMMENDED] or just with pip and virtualenv [2]

```

#0 - Install or upgrade pip, pipenv, and virtualenv:
python3 -m pip install --upgrade pip
pip install virtualenv
pip install pipenv
which virtualenv # ensure that virtualenv was installed properly
which pipenv

#1 - Create environment using pipfile.lock and pipenv: 
cd <aim_directory>
pipenv install 
pipenv shell

#2 - Create a virtual environment and install dependencies with pip and only pip - DO NOT USE IF USING PIPENV: 
which python # This will print out the path to the python we are using (use this in <path_to_python_bin> below)
virtualenv -p <path_to_python_bin> <venv-name> # Create virtualenv using the version of python we have installed at the path dir
source <venv-name>/bin/activate # Activate the virtual environment we created
cd <path_to_aim_repo> # Navigate to the aim-repo
pip install -r requirements.txt # Install packages within the virtual environment


# Run dmft.py to ensure environment is working properly: 
python dmft.py -s 0 -size 2 -l 2 -t 1e-4 -gs -d

# Run scaling experiment on seed 0 with 3 sites with a target error of 1e-4 in the ground state overlap, displaying results,
# and setting maximum vqe layers at 3
python dmft.py -s 0 -size 3 -pel 3 -t 1e-4 -gs -d


# Deactivate virtualenv environment
deactivate
 
```

Because the code is very much still in development, I will try and provide a cursory description of the functionality 
below. With the dependencies above installed, it should more or less run out of the box.

MAIN TEST FUNCTIONS:
- Function: main2()-- Ignore this function for now. Being written as a test.
- Function: main()-- Main function where all current testing occurs. This function is currently called when the script
is run. To understand usage of various classes and methods look at this function.
- Function: fit_func(x, a, b)-- A fit function used somewhere else in the code. (Not really a main function.)
- Function: symmetric_ansatz_test(model_hamiltonian)-- This is a wrapper function to test the various performance 
aspects of the symmetry-preserving ansatz. Need to provide a Hamiltonian for it to test on.

FUNCTIONS TO CONSTRUCT TEST AIMS:
- Function: random_two_impurity_example(seed)-- This function takes a random seed and generates a random AIM with two
impurities and two bath sites. It's intended to explore the viability of doing cluster dynamical mean-field theory.
- Function: random_two_site_example(seed)-- Creates a random AIM with one impurity site and one bath site. Corresponds
to four qubits. (Values may be hard-coded/not random at the moment).
- Function: francois_example_inputs()-- Generates an AIM with one impurity site and two bath sites. AIM parameters are
hard-coded.

CLASSES (CORE SCRIPT FUNCTIONALITY):
- Class: AndersonImpurityModel-- This class takes Hamiltonian parameters to instantiate. Includes methods to create
Fermionic and qubit Hamiltonian representations as well as to perform exact diagonalization.
- Class: MeanFieldQulacsVqeEmulator-- This is a class to solve a given AIM via VQE in the qubit mean-field limit using
the Qulacs emulator. Floating-point precision in VQE angles, i.e., no shot noise.
- Class: SymmQulacsVqeEmulator-- This is a class whose methods allow for the solution of a given AIM using VQE. It uses
what I am calling the "2D Interacting, Symmetry-Preserving Ansatz" and Qulacs as the emulator. 
Floating-point precision in VQE angles, i.e., no shot noise.
