#   Author: V. Vitale
#   Feb 2022

using TensorOperations
using KrylovKit
using LinearAlgebra
using Random
using RandomMatrices
using JLD2, FileIO
using NPZ
abstract type AbstractTN end

"""
    length(::MPS/MPO)
The number of sites of an MPS/MPO.
"""

Base.:length(m::AbstractTN) = m.N

data(m::AbstractTN) = m.data

Base.:size(m::AbstractTN) = m.N

dims(m::AbstractTN) = Dict([key=>Base.size(m.data[key]) for key in keys(data(m))])

function writeTN(filename::String,m::AbstractTN)
    jldsave(filename*".jld2", data=m)
end

function readTN(filename::String)
    m=jldopen(filename*".jld2")["data"]
    return m
end
    
    
function copy(A::AbstractTN)
    N=length(A)
    if typeof(A)==MPS
        temp=MPS()
        temp.b=A.b
    elseif typeof(A)==MPO
        temp=MPO()
    end
    temp.N=N
    for i in 1:N
        temp.data[i]=Base.copy(A.data[i])
    end

    return temp
end

++(A::AbstractArray, B::AbstractArray) = cat(A, B,dims=(1,2))
const ⨁=++

function direct_sum(A::AbstractTN, B::AbstractTN)
    N=length(A)
    temp=copy(A)
    for i in 1:L
        temp.data[i]=++(A.data[i],B.data[i])
    end
    return temp
end

function direct_sub(A::AbstractTN, B::AbstractTN)
    N=length(A)
    temp=copy(A)
    println(typeof(temp))
    for i in 1:L
        temp.data[i]=++(A.data[i],-B.data[i])
    end
    return temp
end


Base.:+(A::AbstractTN, B::AbstractTN) = direct_sum(A, B)
Base.:-(A::AbstractTN, B::AbstractTN) = direct_sub(A, B)
