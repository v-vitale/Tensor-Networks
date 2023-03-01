# Tensor-Networks
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

# mps.jl containes the basics for mps
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

# mpo.jl contains several MPOs and routines to handle them
- Transverse Field Ising Chain
- XXZ for spin S=1/2
- XXZ for spin S=1
- Cluster Ising Model (XZX +YY)
- Long Range Ising model
- Modelization of [Brydges et al., Science (2019)] experiment 

Tutorial:
- W=MPO() initializes an empty W MPO "object".
- Initialize!(W,*args) builds the particular MPO of choice. 
  - Ex: Initialize!("XXZ S=1",W,J,h,L); inizializes an MPO with for the XXZ chain of length L with spin S=1:
    - J and h are two parameters in the MPO,
    - bond dimension and local dimension are automatically handled.

# dmrg.jl contains algorithm for the diagonalization of mpos
- one site dmrg
- two sites dmrg
- contraction routines needed

Tutorial:
- two_sites_dmrg!(A,W,sweeps,chimax) calls the two-sites dmrg routine
  - A is an initial mps that can be just called as A=MPS(),
  - W is the MPO one wants to diagonalize,
  - sweeps set the number of sweeps (back and forth) along the mps in the optimization,
  - chimax set the maximum bond dimension if needed, default is chimax=128.


# tdvp.jl contains all the necessary for doing a time-dependent variational principle algorithm
- tdvp routines and contractions

# traj_utils.jl contains all the necessary for doing open dynamics with trajectories
- routines for handling the evolution together with the tdvp in tdvp.jl
- routines for calculating observables on different trajectories in parallel


