using TensorOperations
using KrylovKit
using LinearAlgebra
using Random

abstract type AbstractTN end

"""
    length(::MPS/MPO)
The number of sites of an MPS/MPO.
"""
length(m::AbstractTN) = m.N

data(m::AbstractTN) = m.data

size(m::AbstractTN) = size(data(m))

dims(m::AbstractTN) = dims(data(m))