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

function move_orthogonality_center!(A::MPS,b::Int)
    right_normalize!(psi)
    for i in 1:b
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

    
function compute_entropy(A::MPS)
    M=copy(A)
    Sent = zeros(M.N)
    for i in 1:M.N
        sM = size(M.data[i])
        U,S,V = svd(reshape(M.data[i],(sM[1]*sM[2],sM[3])),full=false)
        S /= norm(S)
        V=V'  
        M.data[i] = reshape( U,( sM[1], sM[2], :)) 
        if i<M.N
            @tensor M.data[i+1][:] := diagm(S)[-1,1 ] * V[ 1,2 ] * M.data[i+1][2,-2,-3] 
        end
        Sent[i] = sum(-dot(S.^2,log.(S.^2)))
    end 

    
    return Sent
end

function compute_Renyi2(A::MPS)
    M=copy(A)
    Sent = zeros(M.N)
    for i in 1:M.N
        sM = size(M.data[i])
        U,S,V = svd(reshape(M.data[i],(sM[1]*sM[2],sM[3])),full=false)
        S /= norm(S)
        V=V'  
        M.data[i] = reshape( U,( sM[1], sM[2], :)) 
        if i<M.N
            @tensor M.data[i+1][:] := diagm(S)[-1,1 ] * V[ 1,2 ] * M.data[i+1][2,-2,-3] 
        end
        Sent[i] = -log(sum(S.^4))
    end   
    
    return Sent
end


function purity(A::MPS)
    M=copy(A)
    Sent = zeros(M.N)
    for i in 1:M.N
        sM = size(M.data[i])
        U,S,V = svd(reshape(M.data[i],(sM[1]*sM[2],sM[3])),full=false)
        S /= norm(S)
        V=V'  
        M.data[i] = reshape( U,( sM[1], sM[2], :)) 
        if i<M.N
            @tensor M.data[i+1][:] := diagm(S)[-1,1 ] * V[ 1,2 ] * M.data[i+1][2,-2,-3] 
        end
        Sent[i] = sum(S.^4)
    end   
    
    return Sent
end

function to_dm(A::MPS)
    M=MPS()
    M.N=A.N
    for i in 1:M.N
        sA=size(A.data[i]) 
        @tensor M.data[i][:] := A.data[i][-1,-3,-5]*conj(A.data[i][-2,-4,-6])
        M.data[i]= reshape(M.data[i],(sA[1]*sA[1],sA[2]*sA[2],sA[3]*sA[3]))
    end   
    return M
end
