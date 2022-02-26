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
        #@tensor E[:] := temp[1,2,-1] * conj( B.data[i][1,2,-2] )
        @tensor E[:] := temp[1,2,-2] * conj( B.data[i][1,2,-1] )
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
    temp=Dict()
    temp[1] = im*rand(1,d,chi)
    for i in 2:N-1
        temp[i]= im*rand(chi,d,chi)
    end
    temp[N] = im*rand(chi,d,1)
    A.N=N
    A.data=temp
end

function right_normalize!(A::MPS)
    for i in A.N:-1:1
        sA = size(A.data[i])
        U, S, V = svd(reshape(A.data[i],sA[1], sA[2]*sA[3]), full=false)
        V=V'
        S /= norm(S)
        A.data[i] = reshape(V,(:, sA[2], sA[3]))
        if i>1
            S=diagm(S)
            @tensor A.data[i-1][:] := A.data[i-1][-1,-2,2] * U[ 2,3 ] * S[ 3,-3 ]  
        end
    end  
end
    
function left_normalize!(A::MPS)
    for i in 1:A.N
        sA = size(A.data[i])
        U,S,V = svd(reshape(A.data[i],(sA[1]*sA[2],sA[3])),full=false)
        S /= norm(S)
        V=V'  
        A.data[i] = reshape( U,( sA[1], sA[2], :)) 
        if i<A.N
            S=diagm(S)
            @tensor A.data[i+1][:] := S[-1,1 ] * V[ 1,2 ] * A.data[i+1][2,-2,-3] 
        end
    end    
end