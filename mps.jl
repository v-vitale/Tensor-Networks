#   Author: V. Vitale
#   Feb 2022

include("abstractTN.jl")


# MPS A-matrix is a 3-index tensor, A[i,s,j]
#    s
#    |
# i -A- j
#
# [s] acts on the local Hilbert space
# [i,j] act on the virtual bonds

mutable struct MPS <: AbstractTN
  data::Dict
  N::Int
end

MPS() = MPS(Dict(), 0)

function MPS_dot(A::MPS,B::MPS)
    E=ones(1,1)
    for i in 1:A.N
        @tensor temp[:] := E[-1,1]*A.data[i][1,-2,-3] 
        @tensor E[:] := temp[1,2,-1] * conj( B.data[i][1,2,-2] ) 
        #E = contract_from_left_MPS( A.data[i] , E , B.data[i] )
    end
    return E[1]
end

Base.:*(A::MPS,B::MPS)=MPS_dot(A,B)

function Normalize!(A::MPS)
    norm = MPS_dot(A,A)
    for i in 1:A.N
        A.data[i] = A.data[i]/norm^(1/(2*A.N))
    end
end


function Initialize!(A::MPS,d::Int,chi::Int,N::Int)
    MPS=Dict()
    MPS[1] = im*rand(1,d,chi)
    for i in 2:N-1
        MPS[i]= im*rand(chi,d,chi)
    end
    MPS[N] = im*rand(chi,d,1)
    A.N=N
    A.data=MPS
end

