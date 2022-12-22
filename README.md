# Tensor-Networks
Collection of codes with Tensors Networks. (work in progress)

Author: Vittorio Vitale 

Needed:
- TensorOperations
- KrylovKit
- LinearAlgebra
- LsqFit for Long-range Hamiltonians

# mps.jl containes the basics for mps
- Functions for contractions of mps
- Functions for reshaping the mps moving the orthogonality center
- Functions for constructing reduced density matrices from mps

# mpo.jl contains several MPOs and routines to handle them
- Transverse Field Ising Chain
- XXZ for spin S=1/2
- XXZ for spin S=1
- Cluster Ising Model (XZX +YY)
- Long Range Ising model
- Modelization of [Brydges et al., Science (2019)] experiment 

# dmrg.jl contains algorithm for the diagonalization of mpos
- one site dmrg
- two sites dmrg
- contraction routines needed

# tdvp.jl contains all the necessary for doing a time-dependent variational principle algorithm
- tdvp routines and contractions

# traj_utils.jl contains all the necessary for doing open dynamics with trajectories
- routines for handling the evolution together with the tdvp in tdvp.jl
- routines for calculating observables on different trajectories in parallel


