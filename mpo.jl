#   Author: V. Vitale
#   Feb 2022

include("mps.jl")

# MPO W-matrix is a 4-index tensor, W[i,j,s,t]
#     s
#     |
#  i -W- j
#     |
#     t
#
# [s,t] act on the local Hilbert space,
# [i,j] act on the virtual bonds

mutable struct MPO <: AbstractTN
  data::Dict
  N::Int
end

MPO() = MPO(Dict(), 0)

function MPO_dot(W::MPO,Q::MPO)
    if isempty(W.data) || isempty(Q.data)
        @warn "Empty MPO."
        return 0
    end
    temp=Dict()
    for i in 1:W.N
        sW=size(W.data[i])
        sQ=size(Q.data[i])
        @tensor temp[i][:] := W.data[i][-1,-3,-5,1]*Q.data[i][-2,-4,1,-6]
        temp[i]=reshape(temp[i],sW[1]*sQ[1],sW[2]*sQ[2],sW[3],sQ[4])
    end
    return temp
end

Base.:*(W::MPO,Q::MPO)=MPO_dot(W,Q)

function MPO_dot(W::MPO,Q::MPS)
    if isempty(W.data) || isempty(Q.data)
        @warn "Empty MPO."
        return 0
    end
    temp=MPO()
    temp.N=W.N
    for i in 1:W.N
        sW=size(W.data[i])
        sQ=size(Q.data[i])
        @tensor temp.data[i][:] := W.data[i][-1,-4,-3,1]*Q.data[i][-2,-5,1]
        temp.data[i]=reshape(temp.data[i],sW[1]*sQ[1],sW[3],sW[2]*sQ[3])
    end
    return temp
end

Base.:*(W::MPO,A::MPS)=MPO_dot(W,A)

function trace_MPO(A::MPO)
    if isempty(W.data)
        @warn "Empty MPO."
        return 0
    end
    temp=Base.copy(A.data[1])
    for i in 1:A.N-1
        @tensor temp[-1,-2,-3,-4] := temp[-1,2,1,1]*A.data[i+1][2,-2,-3,-4]
    end
    @tensor result=temp[1,1,2,2]
    return result
end

LinearAlgebra.:tr(A::MPO)= trace_MPO(A)

function Initialize!(s::String,W::MPO,h::Float64,J::Float64,N::Int)
    if s=="Ising"
        d=2
        D=3
        id = [ 1 0 ; 0 1 ]
        sp = [ 0 1 ; 0 0 ]
        sm = [ 0 0 ; 1 0 ]
        sz = [ 1 0 ; 0 -1 ]
        sx = [ 0 1 ; 1 0 ]
        id = [ 1 0 ; 0 1 ]
        Wt = im *  zeros(D,D,d,d)
        Wt1 = im *  zeros(1,D,d,d)
        Wt2 = im *  zeros(D,1,d,d)
        Wt[1,1,:,:]=id
        Wt[2,1,:,:]=sz
        Wt[3,1,:,:]=-h*sx
        Wt[3,2,:,:]=-J*sz
        Wt[3,3,:,:]=id;

        Wt1[1,1,:,:]=-h*sx
        Wt1[1,2,:,:]=-J*sz
        Wt1[1,3,:,:]=id

        Wt2[1,1,:,:]=id
        Wt2[2,1,:,:]=sz
        Wt2[3,1,:,:]=-h*sx


        MPO = Dict()
        MPO[1] = Wt1
        for i in 2:(N-1)
            MPO[i] = Wt
        end
        MPO[N] = Wt2
        W.data=MPO
        W.N=N
        return "TFIM MPO"
    else
        @warn "Wrong parameters"
    end
end

function Initialize!(s::String,W::MPO,N::Int)
    if s=="Magnetization"
    chi=2
        d=2

        σz = 0.5*[1 0; 0 -1]
        Id2= [1 0; 0 1]
        O2 = [0 0; 0 0]

        W = im *  zeros(chi,chi,d,d)
        W1 = im *  zeros(1,chi,d,d)
        W2 = im *  zeros(chi,1,d,d)

        W1[1,1,:,:] = σz 
        W1[1,2,:,:] = Id2
        W[1,1,:,:]= Id2
        W[1,2,:,:]= O2
        W[2,1,:,:]= σz 
        W[2,2,:,:]= Id2
        W2[1,1,:,:] = Id2
        W2[2,1,:,:] = σz 

        MPO = Dict()
        MPO[1] = W1
        for i in 2:L-1
            MPO[i] = W
        end
        MPO[L] = W2
        return "Magnetization MPO"
    else
        @warn "Wrong parameters"
    end
end

function Initialize!(s::String,W::MPO,d::Int,chi::Int,N::Int)
    if s=="Zeros"
        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)

        MPO = Dict()
        MPO[1] = Wt1
        for i in 2:(N-1)
            MPO[i] = Wt
        end
        MPO[N] = Wt2
        W.data=MPO
        W.N=N
        return "Null MPO"
    elseif s=="Random"
        Wt = im *  rand(chi,chi,d,d)
        Wt1 = im *  rand(1,chi,d,d)
        Wt2 = im *  rand(chi,1,d,d)

        MPO = Dict()
        MPO[1] = Wt1
        for i in 2:(N-1)
            MPO[i] = Wt
        end
        MPO[N] = Wt2
        W.data=MPO
        W.N=N
        return "Random MPO"
    else
        @warn "Wrong parameters"
    end
end


function truncate_MPO(W::MPO,tol::Float64)
    N=length(W)
    for i in 1:N-1
        temp=permutedims(W.data[i],(1,3,4,2))
        sW= size(temp)
        F = qr(reshape(temp,(sW[1]*sW[2]*sW[3],sW[4])))
        q=convert(Matrix{Complex{Float64}}, F.Q)
        r=convert(Matrix{Complex{Float64}}, F.R)
        q = permutedims(reshape(q,(sW[1],sW[2],sW[3],:)),(1, 4, 2, 3))
        W.data[i] = q
        @tensor W.data[i+1][:]:= r[-1,1]*W.data[i+1][1,-2,-3,-4]
    end
    
    for i in N:-1:2
        temp = permutedims(W.data[i],(1,3,4,2))
        sW = size(temp)
        U,S,V = svd(reshape(temp,(sW[1],sW[2]*sW[3]*sW[4])),full=false)
        s_norm = norm(S)
        S=S/norm(S)
        indices = findall(1 .-cumsum(S.^2) .< tol)
        if length(indices)>0
            chi = indices[1]+1
        else
            chi = size(S)[1]
        end
        if size(S)[1] > chi
            U = U[:,1:chi]
            S = S[1:chi]
            V =  V[:,1:chi]
        end
        S = S/norm(S)
        S *= s_norm
        
        W.data[i] = permutedims(reshape(V,(:,sW[2],sW[3],sW[4])),(1,4,2,3))
        temp=U*diagm(S)
        @tensor W.data[i-1][:] := W.data[i-1][-1,1,-3,-4]*temp[1,-2]
    end
    return W
end

