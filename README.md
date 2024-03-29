# Tensor-Networks <a href="https://zenodo.org/badge/latestdoi/461973238"><img src="https://zenodo.org/badge/461973238.svg" alt="DOI"></a>

Collection of codes with Tensors Networks. (work in progress)

Author: Vittorio Vitale 

Acknowledgements: Alessandro Santini (https://github.com/alessandro-santini)

Packages Needed:
- Distributed (for parallel computing)
- TensorOperations
- KrylovKit
- LinearAlgebra
- LsqFit for Long-range Hamiltonians
- RandomMatrices for Haar random unitaries
- JLD2, FileIO and NPZ for handling file saving/reading 
- ITensors for AutoMPO

Running "julia setup.jl" installs all the packages needed.

# mps.jl contains the basics for mps
- Particular types of MPS:
  - Random
  - GHZ
  - Null
- Functions for contractions of mps
- Functions for reshaping the mps moving the orthogonality center
- Functions for constructing reduced density matrices from mps

Tutorial:
- A=MPS() initializes an empty A MPS "object".
- Initialize!(A,*args) builds the particular MPS of choice. 
  - Ex: Initialize!(A,d,D,L) inizializes an empty MPS with 
    - bond dimension D, 
    - local dimension d,
    - length L.
  - Equivalently one can call MPS(*args)

# mpo.jl contains several MPOs and routines to handle them
- Transverse Field Ising Chain
- XXZ for spin S=1/2
- XXZ for spin S=1
- Cluster Ising Model (XZX +YY)
- Long Range Ising model
- Modelization of [Brydges et al., Science (2019)] experiment 
- J1-J2 model on a square lattice

Tutorial:
- W=MPO() initializes an empty W MPO "object".
- Initialize!(W,*args) builds the particular MPO of choice. 
  - Ex: Initialize!("XXZ S=1",W,J,h,L); inizializes an MPO with for the XXZ chain of length L with spin S=1:
    - J and h are two parameters in the MPO,
    - bond dimension and local dimension are automatically handled.
  - Equivalently one can call MPO(s::String,*args)

# dmrg.jl contains algorithm for the diagonalization of mpos
- one site dmrg
- two sites dmrg
- contraction routines needed

Tutorial:
- two_sites_dmrg!(A,W,sweeps;chimax=2048,tol=1e-15) calls the two-sites dmrg routine
  - A is an initial mps that can be just called as A=MPS(),
  - W is the MPO one wants to diagonalize,
  - sweeps set the number of sweeps (back and forth) along the mps in the optimization,
  - chimax set the maximum bond dimension if needed, default is chimax=128.


# time_evo.jl contains all the necessary for doing a time-dependent variational principle algorithm or a trotterized evolution
- tdvp routines and contractions
- trotterized evolution given two qubits gates

# RUC.jl contains all the necessary for simulation of Random Unitary Circuits with local measurements
- evolution with two qubits random unitaries 
- measurement depending on the local magnetization

# traj_utils.jl contains all the necessary for doing open dynamics with trajectories
- routines for handling the evolution together with the tdvp in tdvp.jl
- routines for calculating observables on different trajectories in parallel


