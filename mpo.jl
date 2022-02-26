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
    temp=MPO()
    temp.N=W.N
    for i in 1:W.N
        sW=size(W.data[i])
        sQ=size(Q.data[i])
        @tensor temp.data[i][:] := W.data[i][-1,-3,-5,1]*Q.data[i][-2,-4,1,-6]
        temp.data[i]=reshape(temp.data[i],sW[1]*sQ[1],sW[2]*sQ[2],sW[3],sQ[4])
    end
    return temp
end

Base.:*(W::MPO,Q::MPO)=MPO_dot(W,Q)

function MPO_MPS_dot(W::MPO,Q::MPS)
    if isempty(W.data) || isempty(Q.data)
        @warn "Empty MPO."
        return 0
    end

    temp=MPS()
    temp.N=W.N
    for i in 1:W.N
        sW=size(W.data[i])
        sQ=size(Q.data[i])
        @tensor temp.data[i][:] := W.data[i][-1,-4,-3,1]*Q.data[i][-2,1,-5]
        temp.data[i]=reshape(temp.data[i],sW[1]*sQ[1],sW[3],sW[2]*sQ[3])
    end
    return temp
end

Base.:*(W::MPO,A::MPS)=MPO_MPS_dot(W,A)

function trace_MPO(A::MPO)
    if isempty(A.data)
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

function adjoint(A::MPO)
    dagA=copy(A)
    for i in 1:dagA.N
       @tensor dagA.data[i][:] := conj(dagA.data[i][-1,-2,-4,-3])
    end
    return dagA   
end



function Initialize!(s::String,M::MPO,J::Float64,h::Float64,N::Int)
    if s=="Ising"
        d=2
        D=3
        id = [ 1 0 ; 0 1 ]
        sp = [ 0 1 ; 0 0 ]
        sm = [ 0 0 ; 1 0 ]
        sz = [ 1 0 ; 0 -1 ]
        sx = [ 0 1 ; 1 0 ]
        id = [ 1 0 ; 0 1 ]
        W = im *  zeros(D,D,d,d)
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        W[1,1,:,:]=id
        W[2,1,:,:]=sz
        W[3,1,:,:]=-h*sx
        W[3,2,:,:]=-J*sz
        W[3,3,:,:]=id

        W1[1,1,:,:]=-h*sx
        W1[1,2,:,:]=-J*sz
        W1[1,3,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=sz
        W2[3,1,:,:]=-h*sx

    
        M.data[1] = W1
        for i in 2:(N-1)
            M.data[i] = W
        end
        M.data[N] = W2
        M.N=N
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

        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)

        Wt1[1,1,:,:] = σz 
        Wt1[1,2,:,:] = Id2
        Wt[1,1,:,:]= Id2
        Wt[1,2,:,:]= O2
        Wt[2,1,:,:]= σz 
        Wt[2,2,:,:]= Id2
        Wt2[1,1,:,:] = Id2
        Wt2[2,1,:,:] = σz 

        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
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

        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
        return "Null MPO"
    elseif s=="Random"
        Wt = im *  rand(chi,chi,d,d)
        Wt1 = im *  rand(1,chi,d,d)
        Wt2 = im *  rand(chi,1,d,d)

        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
        return "Random MPO"
    else
        @warn "Wrong parameters"
    end
end


function Initialize!(s::String,W::MPO,N::Int)
    if s=="Local_Haar"
        chi=1
        d=2
        dist = Haar(d)
        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)
        Wt[1,1,:,:] =rand(dist, d)
        Wt1[1,1,:,:] = rand(dist, d)
        Wt2[1,1,:,:] = rand(dist, d)
        W.N=N
        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
    elseif s=="Id"
        chi=1
        d=2
        id= [1 0; 0 1]
        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)
        Wt[1,1,:,:] = id
        Wt1[1,1,:,:] = id
        Wt2[1,1,:,:] = id
        W.N=N
        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
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
        V=V'
        S /= norm(S)
        S *= s_norm
        
        W.data[i] = permutedims(reshape(V,(:,sW[2],sW[3],sW[4])),(1,4,2,3))
        temp=U*diagm(S)
        @tensor W.data[i-1][:] := W.data[i-1][-1,1,-3,-4]*temp[1,-2]
    end
    return W
end