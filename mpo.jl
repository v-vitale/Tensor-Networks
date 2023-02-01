#   Author: V. Vitale
#   Feb 2022

include("mps.jl")
using LsqFit
using ITensors

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

++(A::AbstractArray, B::AbstractArray)=cat(A, B,dims=(1,2))
const ⊕ = ++

function sum_MPO(W::MPO,Q::MPO)
    if isempty(W.data)
        @warn "Empty MPO."
        #return 0
    end
    if  isempty(Q.data)
        @warn "Empty MPO."
        #return 0
    end
    temp=MPO()
    temp.N=W.N
    for i in 1:W.N
        tempsum=W.data[i]⊕Q.data[i]
        temp.data[i] =  tempsum
    end
    return temp
end

Base.:+(A::MPO, B::MPO)=sum_MPO(A,B)

function MPO_MPS_dot(W::MPO,Q::MPS)
    if isempty(W.data)
        @warn "Empty MPO."
        #return 0
    end
    if  isempty(Q.data)
        @warn "Empty MPS."
        #return 0
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

function MPS_MPO_dot(Q::MPS,W::MPO)
    if isempty(W.data)
        @warn "Empty MPO."
        #return 0
    end
    if  isempty(Q.data)
        @warn "Empty MPS."
        #return 0
    end
    
    temp=MPS()
    temp.N=W.N
    for i in 1:W.N
        sW=size(W.data[i])
        sQ=size(Q.data[i])
        @tensor temp.data[i][:] := W.data[i][-1,-4,1,-3]*Q.data[i][-2,1,-5]
        temp.data[i]=reshape(temp.data[i],sW[1]*sQ[1],sW[3],sW[2]*sQ[3])
    end
    return temp
end


Base.:*(W::MPO,A::MPS)=MPO_MPS_dot(W,A)
Base.:*(A::MPS,W::MPO)=MPS_MPO_dot(A,W)

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
    dagA=MPO()
    dagA.N=A.N
    for i in 1:dagA.N
       @tensor dagA.data[i][:] := conj(A.data[i][-1,-2,-4,-3])
    end
    return dagA   
end

function Initialize!(s::String,W::MPO,J::Float64,Jz::Float64,hz2::Float64,hz::Float64,N::Int)
    if s=="Kennedy-Tasaki"
        d=3
        D=5
        id = [ 1 0 0 ; 0 1 0 ; 0 0 1]
        sx = 1/sqrt(2)*[ 0 1 0 ; 1 0 1; 0 1 0]
        sy = 1/sqrt(2)*[ 0 -1im 0 ; 1im 0 -1im ; 0 1im 0]
        sz = [ 1 0 0 ; 0 0 0; 0 0 -1 ]
        Wt = im *  zeros(D,D,d,d)
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        Wt[1,1,:,:]=id
        Wt[2,1,:,:]=sx
        Wt[3,1,:,:]=sy
        Wt[4,1,:,:]=sz
        Wt[5,1,:,:]=hz2*sz*sz+hz*sz
        
        Wt[5,2,:,:]=J*sx
        Wt[5,3,:,:]=J*sy
        Wt[5,4,:,:]=Jz*sz
        Wt[5,5,:,:]=id

        W1[1,1,:,:]=hz2*sz*sz+hz*sz
        W1[1,2,:,:]=J*sx
        W1[1,3,:,:]=J*sy
        W1[1,4,:,:]=Jz*sz
        W1[1,5,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=sx
        W2[3,1,:,:]=sy
        W2[4,1,:,:]=sz
        W2[5,1,:,:]=hz2*sz*sz+hz*sz
        
    
        W.data[1] = Base.copy(W1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(W2)
        W.N=N
        return "Kennedy-Tasaki S=1 MPO"
    else
        @warn "Wrong parameters"
    end
end


function Initialize!(s::String,W::MPO,J::Float64,h::Float64,hz::Float64,N::Int)
    if s=="Cluster_Ising"
        d=2
        D=5
        id = [ 1 0 ; 0 1 ]
        sp = [ 0 1 ; 0 0 ]
        sm = [ 0 0 ; 1 0 ]
        sz = [ 1 0 ; 0 -1 ]
        sx = [ 0 1 ; 1 0 ]
        sy = im *[ 0 -1 ; 1 0 ]
        Wt = im *  zeros(D,D,d,d)
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        
        Wt[1,1,:,:]=id
        Wt[2,1,:,:]=sx
        Wt[4,1,:,:]=sy
        Wt[5,1,:,:]=hz*sz
        Wt[3,2,:,:]=-J*sz
        Wt[5,3,:,:]=sx
        Wt[5,4,:,:]=h*sy
        Wt[5,5,:,:]=id

        W1[1,1,:,:]=hz*sz
        W1[1,3,:,:]=sx
        W1[1,4,:,:]=h*sy
        W1[1,5,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=sx
        W2[4,1,:,:]=sy
        W2[5,1,:,:]=hz*sz
    
        W.data[1] = Base.copy(W1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(W2)
        W.N=N
        return "Cluster Ising MPO"
    elseif s=="XXZ"
        d=2
        D=5
        id = [ 1 0 ; 0 1 ]
        sp = [ 0 1 ; 0 0 ]
        sm = [ 0 0 ; 1 0 ]
        sz = [ 1 0 ; 0 -1 ]
        Wt = im *  zeros(D,D,d,d)
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        Wt[1,1,:,:]=id
        Wt[2,1,:,:]=sp
        Wt[3,1,:,:]=sm
        Wt[4,1,:,:]=sz
        Wt[5,1,:,:]=hz*sz
        
        Wt[5,2,:,:]=J*sm
        Wt[5,3,:,:]=J*sp
        Wt[5,4,:,:]=0.5*h*sz
        Wt[5,5,:,:]=id

        W1[1,1,:,:]=hz*sz
        W1[1,2,:,:]=J*sm
        W1[1,3,:,:]=J*sp
        W1[1,4,:,:]=0.5*h*sz
        W1[1,5,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=sp
        W2[3,1,:,:]=sm
        W2[4,1,:,:]=sz
        W2[5,1,:,:]=hz*sz
    
        W.data[1] = Base.copy(W1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(W2)
        W.N=N
        return "XXZ MPO"
    elseif s=="XXZ S=1"
        d=3
        D=5
        id = [ 1 0 0 ; 0 1 0 ; 0 0 1]
        sx = 1/sqrt(2)*[ 0 1 0 ; 1 0 1; 0 1 0]
        sy = 1/sqrt(2)*[ 0 -1im 0 ; 1im 0 -1im ; 0 1im 0]
        sz = [ 1 0 0 ; 0 0 0; 0 0 -1 ]
        Wt = im *  zeros(D,D,d,d)
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        Wt[1,1,:,:]=id
        Wt[2,1,:,:]=sx
        Wt[3,1,:,:]=sy
        Wt[4,1,:,:]=sz
        Wt[5,1,:,:]=hz*sz
        
        Wt[5,2,:,:]=J*sx
        Wt[5,3,:,:]=J*sy
        Wt[5,4,:,:]=h*sz
        Wt[5,5,:,:]=id

        W1[1,1,:,:]=hz*sz
        W1[1,2,:,:]=J*sx
        W1[1,3,:,:]=J*sy
        W1[1,4,:,:]=h*sz
        W1[1,5,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=sx
        W2[3,1,:,:]=sy
        W2[4,1,:,:]=sz
        W2[5,1,:,:]=hz*sz
    
        W.data[1] = Base.copy(W1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(W2)
        W.N=N
        return "XXZ S=1 MPO"
    else
        @warn "Wrong parameters"
    end
end


function Initialize!(s::String,W::MPO,J::Float64,h::Float64,N::Int)
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
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        Wt[1,1,:,:]=id
        Wt[2,1,:,:]=sz
        Wt[3,1,:,:]=-h*sx
        Wt[3,2,:,:]=-J*sz
        Wt[3,3,:,:]=id

        W1[1,1,:,:]=-h*sx
        W1[1,2,:,:]=-J*sz
        W1[1,3,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=sz
        W2[3,1,:,:]=-h*sx

    
        W.data[1] = Base.copy(W1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(W2)
        W.N=N
        return "TFIM MPO"
    elseif s=="Cluster_Ising"
        d=2
        D=5
        id = [ 1 0 ; 0 1 ]
        sp = [ 0 1 ; 0 0 ]
        sm = [ 0 0 ; 1 0 ]
        sz = [ 1 0 ; 0 -1 ]
        sx = [ 0 1 ; 1 0 ]
        sy = im *[ 0 -1 ; 1 0 ]
        Wt = im *  zeros(D,D,d,d)
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        Wt[1,1,:,:]=id
        Wt[2,1,:,:]=sx
        Wt[4,1,:,:]=sy
        Wt[3,2,:,:]=-J*sz
        Wt[5,3,:,:]=sx
        Wt[5,4,:,:]=h*sy
        Wt[5,5,:,:]=id

        W1[1,3,:,:]=sx
        W1[1,4,:,:]=h*sy
        W1[1,5,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=sx
        W2[4,1,:,:]=sy

    
        W.data[1] = Base.copy(W1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(W2)
        W.N=N
        return "Cluster Ising MPO"
    elseif s=="XXZ"
        d=2
        D=5
        id = [ 1 0 ; 0 1 ]
        sp = [ 0 1 ; 0 0 ]
        sm = [ 0 0 ; 1 0 ]
        sz = [ 1 0 ; 0 -1 ]
        Wt = im *  zeros(D,D,d,d)
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        Wt[1,1,:,:]=id
        Wt[2,1,:,:]=sp
        Wt[3,1,:,:]=sm
        Wt[4,1,:,:]=sz
        
        Wt[5,2,:,:]=J*sm
        Wt[5,3,:,:]=J*sp
        Wt[5,4,:,:]=0.5*h*sz
        Wt[5,5,:,:]=id

        W1[1,2,:,:]=J*sm
        W1[1,3,:,:]=J*sp
        W1[1,4,:,:]=0.5*h*sz
        W1[1,5,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=sp
        W2[3,1,:,:]=sm
        W2[4,1,:,:]=sz
        
    
        W.data[1] = Base.copy(W1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(W2)
        W.N=N
        return "XXZ MPO"
    elseif s=="XXZ S=1"
        d=3
        D=5
        id = [ 1 0 0 ; 0 1 0 ; 0 0 1]
        sx = 1/sqrt(2)*[ 0 1 0 ; 1 0 1; 0 1 0]
        sy = 1/sqrt(2)*[ 0 -1im 0 ; 1im 0 -1im ; 0 1im 0]
        sz = [ 1 0 0 ; 0 0 0; 0 0 -1 ]
        Wt = im *  zeros(D,D,d,d)
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        Wt[1,1,:,:]=id
        Wt[2,1,:,:]=sx
        Wt[3,1,:,:]=sy
        Wt[4,1,:,:]=sz
        
        Wt[5,2,:,:]=J*sx
        Wt[5,3,:,:]=J*sy
        Wt[5,4,:,:]=h*sz
        Wt[5,5,:,:]=id

        W1[1,2,:,:]=J*sx
        W1[1,3,:,:]=J*sy
        W1[1,4,:,:]=h*sz
        W1[1,5,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=sx
        W2[3,1,:,:]=sy
        W2[4,1,:,:]=sz
        
    
        W.data[1] = Base.copy(W1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(W2)
        W.N=N
        return "XXZ S=1 MPO"
    else
        @warn "Wrong parameters"
    end
end

function Initialize!(s::String,W::MPO,config::Array,N::Int)
    if length(config)!= N
        @warn "Wrong length"
    end
    if s=="proj"
        op=[[1 0 ; 0 0],[0 0;0 1]] 
        chi=1
        d=2
        Wt = im *  zeros(chi,chi,d,d)  
        W.N=N
        for i in 1:N
            Wt[1,1,:,:] = op[config[i]+1]
            W.data[i] = Base.copy(Wt)
        end
    end
end


function Initialize!(s::String,W::MPO,config::Array,subsystem::Array,N::Int)
    if length(config)!= length(subsystem)
        @warn "Wrong lengths"
    end
    if s=="proj"
        op = [[1 0 ; 0 0],[0 0;0 1]] 
        id = [1 0; 0 1]
        chi=1
        d=2
        W.N=N
        Wt = im *  zeros(chi,chi,d,d)  
        Wt[1,1,:,:] = id
        
        for i in 1:N
            W.data[i]= Base.copy(Wt)
        end
        for (i,j) in enumerate(subsystem)
            Wt[1,1,:,:] = op[config[i]+1]
            W.data[j]= Base.copy(Wt)
        end
    elseif s=="proj S=1"
        op = [[1 0 0; 0 0 0; 0 0 0],[0 0 0;0 1 0; 0 0 0],[0 0 0; 0 0 0; 0 0 1]] 
        id = [1 0 0; 0 1 0; 0 0 1]
        chi=1
        d=3
        W.N=N
        Wt = im *  zeros(chi,chi,d,d)  
        Wt[1,1,:,:] = id
        
        for i in 1:N
            W.data[i]= Base.copy(Wt)
        end
        for (i,j) in enumerate(subsystem)
            Wt[1,1,:,:] = op[2-config[i]]
            W.data[j]= Base.copy(Wt)
        end
        
    end
end



function Initialize!(s::String,W::MPO,N::Int)
    if s=="Magnetization"
        chi=2
        d=2

        σz = [1 0; 0 -1]
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

        W.N=N
        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
        return "Magnetization MPO"
    elseif s=="Neel"
        h=100
        d=2
        D=2
        id = [ 1 0 ; 0 1 ]
        sz = [ 1 0 ; 0 -1 ]
        Wp = im *  zeros(D,D,d,d)
        Wd = im *  zeros(D,D,d,d)
        W1 = im *  zeros(1,D,d,d)
        W2 = im *  zeros(D,1,d,d)
        
        Wp[1,1,:,:]=id
        Wp[2,1,:,:]=-h*sz
        Wp[2,2,:,:]=id
        Wd[1,1,:,:]=id
        Wd[2,1,:,:]=h*sz
        Wd[2,2,:,:]=id

        W1[1,1,:,:]=-h*sz
        W1[1,2,:,:]=id

        W2[1,1,:,:]=id
        W2[2,1,:,:]=h*sz

        W.data[1] = Base.copy(W1)
        for i in 2:2:(N-1)
            W.data[i] = Base.copy(Wd)
            W.data[i+1] = Base.copy(Wp)
        end
        W.data[N] = Base.copy(W2)
        W.N=N
        return "Neel"
    elseif s=="Local_Haar"
        function CUE(nh)
            U = (randn(nh,nh)+im*randn(nh,nh))/sqrt(2)
            q,r = qr(U)
            d = r[diagind(r)]
            ph = d./abs.(d)
            U = ph.*q
            return U
        end

        chi=1
        d=2
        dist = Haar(d)
        Wt = im *  zeros(chi,chi,d,d)
        W.N=N
        for i in 1:N
            Wt[1,1,:,:] = rand(dist,d)#CUE(2)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt)
        return "Local Haar MPO"
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
        return "Identity"
    elseif s=="Rx"
        chi=1
        d=2
        id= [1/sqrt(2) -1im/sqrt(2); -1im/sqrt(2) 1/sqrt(2)]
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
        return "Rx"
    elseif s=="Ry"
        chi=1
        d=2
        id= [1/sqrt(2) -1/sqrt(2); 1/sqrt(2) 1/sqrt(2)]
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
        return "Ry"
    elseif s=="Rx S=1"
        chi=1
        d=3
        id= [1 0 0;  0 0 -1; 0 1 0]
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
        return "Rx S=1"
    elseif s=="Ry S=1"
        chi=1
        d=3
        id= [0 0 1; 0 1 0; -1 0 0]
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
        return "Ry S=1"
    else
        @warn "Wrong parameters"
    end
end

function Initialize!(s::String,W::MPO,d::Int,chi::Int,N::Int)
    if s=="Zeros"
        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)

        W.N=N
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

        W.N=N
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


function truncate_MPO!(W::MPO;tol=1e-12)
    N=length(W)
    for i in 1:N-1
        temp=permutedims(W.data[i],(1,3,4,2))
        sW= size(temp)
        F = qr(reshape(temp,(sW[1]*sW[2]*sW[3],sW[4])))
        q=convert(Matrix{Complex{Float64}}, F.Q)
        r=convert(Matrix{Complex{Float64}}, F.R)
        q = permutedims(reshape(q,(sW[1],sW[2],sW[3],size(q)[2])),(1, 4, 2, 3))
        W.data[i] = q
        @tensor W.data[i+1][:]:= r[-1,1]*W.data[i+1][1,-2,-3,-4]
    end
    
    for i in N:-1:2
        temp = permutedims(W.data[i],(1,3,4,2))
        sW = size(temp)
        U,S,V = svd(reshape(temp,(sW[1],sW[2]*sW[3]*sW[4])),full=false)
        V=V'
        s_norm = norm(S)
        S=S/norm(S)
        
        indices = findall(1 .-cumsum(S.^2) .< tol)
        if length(indices)>0
            chi = indices[1]+1
        else
            chi = size(S)[1]
        end
        if chi>chimax
            chi=chimax
        end
        if size(S)[1] > chi
            U = U[:,1:chi]
            S = S[1:chi]
            V =  V[1:chi,:]
        end
        S = S/norm(S)
        S = S*s_norm
        
        W.data[i] = permutedims(reshape(V,(:,sW[2],sW[3],sW[4])),(1,4,2,3))
        temp=U*diagm(S)
        @tensor W.data[i-1][:] := W.data[i-1][-1,1,-3,-4]*temp[1,-2]
    end
    return W
end


function Initialize!(s::String,W::MPO,alpha::Float64,J::Float64,hz::Float64,k::Int,N::Int)
    chi=k+2
    if s=="LongRangeIsing"
        function expfit(x,p)
            res=0
            for i in 1:div(length(p),2)
                res=res.+p[2*i-1]*exp.(-x./p[2*i])
            end
            return res
        end
        model(x, p) = expfit(x,p)
        xdata=Array(range(1, stop=N,length=2048))
        ydata=xdata.^(-alpha);
        fit =curve_fit(model,xdata,ydata,rand(2*k),lower=1e-9zeros(2*k))
        c = coef(fit)[1:2:end]; λ = coef(fit)[2:2:end]
        
        σx = [0 1; 1 0]
        σz = [1 0; 0 -1]
        Id2= [1 0; 0 1]
        
        Nalpha = 0
        for i in 1:N
            for j in i+1:N
                Nalpha += 1/(j-i)^alpha
            end
        end
        J = J/(N-1)*Nalpha
                
        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)

        
        Wt[1, 1, :, :] = Id2
        for i in 1:k
            Wt[1+i, 1, :, :] = σx
            Wt[1+i,1+i,:, :] = exp(-1/λ[i])*Id2
            Wt[end, 1+i,:, :] = -J*exp(-1/λ[i])*c[i]*σx
        end
        Wt[end, 1, :, :] = -hz*σz
        Wt[end,end,:,:] = Id2
        Wt1  = reshape(Wt[end,:,:,:],(1,chi,2,2))
        Wt2 = reshape(Wt[:,1,:,:],(chi,1,2,2))
    
        W.N=N
        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
        return "Long Range MPO"  
    else
        @warn "Wrong parameters"
    end
end


function Initialize!(s::String,W::MPO,alpha::Float64,J::Float64,hz::Float64,γp::Float64,γm::Float64,γz::Float64,k::Int,N::Int)
    chi=2*k+2
    d=4
    if s=="OpenLongRangeIsing"
        function expfit(x,p)
            res=0
            for i in 1:div(length(p),2)
                res=res.+p[2*i-1]*exp.(-x./p[2*i])
            end
            return res
        end
        model(x, p) = expfit(x,p)
        xdata=Array(range(1, stop=N,length=2048))
        ydata=xdata.^(-alpha);
        fit =curve_fit(model,xdata,ydata,ones(2*k),lower=1e-9zeros(2*k))
        c = coef(fit)[1:2:end]; λ = coef(fit)[2:2:end]
        println(coef(fit))
        σx = [0 1; 1 0]
        σz = [1 0; 0 -1]
        Id2= [1 0; 0 1]
        Id4= kron(Id2,Id2)
        σx1 = kron(σx,Id2)
        σx2 = kron(Id2,σx)
        σz1 = kron(σz,Id2)
        σz2 = kron(Id2,σz)
        Diss= [-(γm+γp)/2 0 0 γp ; 0 -(γp+2*γz) 0 0; 0 0 -(γm +2*γz) 0; γm 0 0 -(γm+γp)/2]
        
        Nalpha = 0
        for i in 1:N
            for j in i+1:N
                Nalpha += 1/(j-i)^alpha
            end
        end
        J = J/(N-1)*Nalpha
                
        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)

        
        Wt[1, 1, :, :] = Id4
        for i in 1:k
            Wt[1+i, 1, :, :] = σx1
            Wt[1+i,1+i,:, :] = exp(-1/λ[i])*Id4
            Wt[end, 1+i,:, :] = -J*exp(-1/λ[i])*c[i]*σx1
            Wt[1+i+k, 1, :, :] = σx2
            Wt[1+i+k,1+i+k,:, :] = exp(-1/λ[i])*Id4
            Wt[end,1+i+k,:, :] = -J*exp(-1/λ[i])*c[i]*σx2
        end

        Wt[end, 1, :, :] += -hz*σz1-hz*σz2+im*Diss
        Wt[end,end,:,:] = Id4
        
        Wt1  = reshape(Wt[end,:,:,:],(1,chi,d,d))
        Wt2 = reshape(Wt[:,1,:,:],(chi,1,d,d))
  
        W.N=N
        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
        return "Open Long Range MPO"  
    else
        @warn "Wrong parameters"
    end
end

function Initialize_Brydges!(s::String,W::MPO,N::Int) 
    #if N==20
    J=[0.001*223.394 0.778458; 0.001*292.381 0.539682; 0.001*208.695 0.182969]
    γm=(1/1.17)*0.001
    γx=0.69*0.001
    #elseif N==10
    #    J=[1.29066 0.325936; 1.09155 0.0349184; 0.771306 0.640559]
    #    γm=(1/1.17)*0.001
    #    γx=0.69*0.001
    #else 
    #    @warn "Wrong N"
    #end
    if s=="Closed"
        d=2
        k=3
        chi=2*k+2
        hz=0.
        c = J[:,1]
        λ = J[:,2]
        
        σp = [0 1; 0 0]
        σm = [0 0; 1 0]
        σz = [1 0; 0 -1]
        Id2= [1 0; 0 1]
                
        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)
        
        Wt[1, 1, :, :] = Id2
        for i in 1:k
            Wt[1+i, 1, :, :] = σp
            Wt[1+i,1+i,:, :] = λ[i]*Id2
            Wt[end, 1+i,:, :] = -λ[i]*c[i]*σm
            
            Wt[k+1+i, 1, :, :] = σm
            Wt[k+1+i,k+1+i,:, :] = λ[i]*Id2
            Wt[end, k+1+i,:, :] = -λ[i]*c[i]*σp
        end
        Wt[end, 1, :, :] = -hz*σz
        Wt[end,end,:,:] = Id2
        Wt1  = reshape(Wt[end,:,:,:],(1,chi,d,d))
        Wt2 = reshape(Wt[:,1,:,:],(chi,1,d,d))
    
        W.N=N
        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
        return "Brydges MPO"  
    elseif s=="Open"
        k=3
        chi=4*k+2
        hz=0.
        d=4
        
        c = J[:,1]
        λ = J[:,2]
        σp = [0 1; 0 0]
        σm = [0 0; 1 0]
        σx = [0 1; 1 0]
        σz = [1 0; 0 -1]
        Id2= [1 0; 0 1]

        Id4= kron(Id2,Id2)
        σp1 = kron(σp,Id2)
        σp2 = kron(Id2,σp)
        σm1 = kron(σm,Id2)
        σm2 = kron(Id2,σm)
        σz1 = kron(σz,Id2)
        σz2 = kron(Id2,σz)
        Diss= γm*kron(σm,σm)-0.5*γm*kron(σp*σm,Id2)-0.5*γm*kron(Id2,σp*σm)+γx*kron(σx,σx)-γx*Id4
        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)

        Wt[1, 1, :, :] = Id4
        for i in 1:k
            Wt[1+i, 1, :, :] = σp1
            Wt[1+i,1+i,:, :] = λ[i]*Id4
            Wt[end, 1+i,:, :] = -λ[i]*c[i]*σm1

            Wt[k+1+i, 1, :, :] = σm1
            Wt[k+1+i,k+1+i,:, :] = λ[i]*Id4
            Wt[end, k+1+i,:, :] = -λ[i]*c[i]*σp1

            Wt[2*k+1+i, 1, :, :] = σp2
            Wt[2*k+1+i,2*k+1+i,:, :] = λ[i]*Id4
            Wt[end, 2*k+1+i,:, :] = λ[i]*c[i]*σm2

            Wt[3*k+1+i, 1, :, :] = σm2
            Wt[3*k+1+i,3*k+1+i,:, :] = λ[i]*Id4
            Wt[end, 3*k+1+i,:, :] = λ[i]*c[i]*σp2
        end
        Wt[end, 1, :, :] = im*Diss
        Wt[end,end,:,:] = Id4
        Wt1  = reshape(Wt[end,:,:,:],(1,chi,d,d))
        Wt2 = reshape(Wt[:,1,:,:],(chi,1,d,d))

        W.N=N
        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
        return "Open Brydges MPO"  
    elseif s=="Trajectories"
        d=2
        k=3
        chi=2*k+2
        hz=0.
        c = J[:,1]
        λ = J[:,2]
        
        σp = [0 1; 0 0]
        σm = [0 0; 1 0]
        σz = [1 0; 0 -1]
        Id2= [1 0; 0 1]
                
        Wt = im *  zeros(chi,chi,d,d)
        Wt1 = im *  zeros(1,chi,d,d)
        Wt2 = im *  zeros(chi,1,d,d)
        
        Wt[1, 1, :, :] = Id2
        for i in 1:k
            Wt[1+i, 1, :, :] = σp
            Wt[1+i,1+i,:, :] = λ[i]*Id2
            Wt[end, 1+i,:, :] = -λ[i]*c[i]*σm
            
            Wt[k+1+i, 1, :, :] = σm
            Wt[k+1+i,k+1+i,:, :] = λ[i]*Id2
            Wt[end, k+1+i,:, :] = -λ[i]*c[i]*σp
        end
        Wt[end, 1, :, :] = -hz*σz-0.5*im*γm*σp*σm
        Wt[end,end,:,:] = Id2
        Wt1  = reshape(Wt[end,:,:,:],(1,chi,d,d))
        Wt2 = reshape(Wt[:,1,:,:],(chi,1,d,d))
    
        W.N=N
        W.data[1] = Base.copy(Wt1)
        for i in 2:(N-1)
            W.data[i] = Base.copy(Wt)
        end
        W.data[N] = Base.copy(Wt2)
        return "Brydges MPO"    
    else
        @warn "Wrong parameters"
    end
end


function Initialize!(s::String,W::MPO,J1::Float64,J2::Float64,cols::Int,rows::Int,config::Array)
    if s=="J1-J2_ITensors"
        N=rows*cols
        nneigh=Dict()
        nnneigh=Dict()
        for x in 1:cols*rows
            idx=findall(y->y==x,config)[1]
            i = idx[1]; j = idx[2]
            nneigh[x]=[]
            if i !=1
                push!(nneigh[x],config[i-1,j])
            end
            if j !=1
                push!(nneigh[x],config[i,j-1])
            end
            if i !=rows
                push!(nneigh[x],config[i+1,j])
            end
            if j !=cols
                push!(nneigh[x],config[i,j+1])
            end
            nneigh[x]=nneigh[x][nneigh[x].>x]
            nnneigh[x]=[]
            if i !=1 && j !=1
                push!(nnneigh[x],config[i-1,j-1])
            end
            if i !=1 && j !=rows
                push!(nnneigh[x],config[i-1,j+1])
            end
            if i !=rows && j !=1
                push!(nnneigh[x],config[i+1,j-1])
            end
            if i !=cols && j != rows
                push!(nnneigh[x],config[i+1,j+1])
            end
            nnneigh[x]=nnneigh[x][nnneigh[x].>x]
        end

        sites = siteinds("S=1/2",N)
        ampo = OpSum()
        for site1 in 1:N
            for site2 in nneigh[site1]
                ampo += (J1,"Z",site1,"Z",site2)
                ampo += (J1,"X",site1,"X",site2)
                ampo += (J1,"Y",site1,"Y",site2)
            end
            for site2 in nnneigh[site1]
                ampo += (J2,"Z",site1,"Z",site2)
                ampo += (J2,"X",site1,"X",site2)
                ampo += (J2,"Y",site1,"Y",site2)
            end
        end

        H=ITensors.MPO(ampo,sites)
        
        W.N=N
        W1=Array(H[1],inds(H[1])...)
        s1=size(W1)
        W.data[1]=Base.copy(reshape(W1,(1,s1...)))
        for i in 2:N-1
            W.data[i]=Base.copy(Array(H[i],inds(H[i])...))
        end
        W2=Array(H[N],inds(H[N])...)
        s2=size(W2)
        W.data[N]=Base.copy(reshape(W2,(s2[1],1,s2[2],s2[3])))
    elseif s=="J1-J2"
        function dim_MPO(cols::Int,rows::Int,config::Array)
            dim=[]
            nearest=[]
            nnearest=[]
            for x in 1:cols*rows
                idx=findall(y->y==x,config)[1]
                i = idx[1]; j = idx[2]
                nn = []
                if i !=1
                    push!(nn,config[i-1,j])
                end
                if j !=1
                    push!(nn,config[i,j-1])
                end
                if i !=rows
                    push!(nn,config[i+1,j])
                end
                if j !=cols
                    push!(nn,config[i,j+1])
                end
                push!(nearest,nn[nn.>x].-x)

                nnn = []
                if i !=1 && j !=1
                    push!(nnn,config[i-1,j-1])
                end
                if i !=1 && j !=rows
                    push!(nnn,config[i-1,j+1])
                end
                if i !=rows && j !=1
                    push!(nnn,config[i+1,j-1])
                end
                if i !=cols && j != rows
                    push!(nnn,config[i+1,j+1])
                end
                push!(nnearest,nnn[nnn.>x].-x)

                new_dim=3*maximum(vcat(nn.-x,nnn.-x))+2
                if x > 1
                    if new_dim < dim[x-1][2]
                        push!(dim,(dim[x-1][2],dim[x-1][2]-1,2,2))
                    else
                        push!(dim,(dim[x-1][2],new_dim,2,2))
                    end
                else
                    push!(dim,(1,new_dim,2,2))
                end
            end
            dim[end] = (dim[end-1][2],1,2,2)
            return dim,nearest,nnearest
        end

        dimW, NN, NNN = dim_MPO(cols,rows,config)

        id = [ 1 0 ; 0 1 ]
        sz = [ 1 0 ; 0 -1 ]
        sx = [ 0 1 ; 1 0 ]
        sy = im *[ 0 -1 ; 1 0 ]
        Wt=[]
        for i in 1:cols*rows
            push!(Wt,im * zeros(dimW[i]))
        end

        for k in 3*NN[1].+1
            Wt[1][1,k-2,:,:] = J1*sz
            Wt[1][1,k-1,:,:] = J1*sx
            Wt[1][1,k,:,:] = J1*sy
        end

        for k in 3*NNN[1].+1
            Wt[1][1,k-2,:,:] = J2*sz
            Wt[1][1,k-1,:,:] = J2*sx
            Wt[1][1,k,:,:] = J2*sy
        end

        Wt[1][1,end,:,:] = id

        Wt[end][1,1,:,:] = id
        Wt[end][2,1,:,:] = sz
        Wt[end][3,1,:,:] = sx
        Wt[end][4,1,:,:] = sy

        for i in 2:cols*rows-1
            Wt[i][1,1,:,:]   = id
            Wt[i][end,end,:,:] = id
            Wt[i][2,1,:,:]   = sz
            Wt[i][3,1,:,:]   = sx
            Wt[i][4,1,:,:]   = sy
            for k in 3*NN[i].+1
                Wt[i][end,k-2,:,:] = J1*sz
                Wt[i][end,k-1,:,:] = J1*sx
                Wt[i][end,k,:,:] = J1*sy
            end
            for k in 3*NNN[i].+1
                Wt[i][end,k-2,:,:] = J2*sz
                Wt[i][end,k-1,:,:] = J2*sx
                Wt[i][end,k,:,:] = J2*sy
            end
            for k in 5:size(Wt[i])[1]-1
                Wt[i][k,k-3,:,:] = id
            end
        end

        for i in 1:cols*rows
            W.data[i] = Base.copy(Wt[i])
        end
        W.N=cols*rows
    else
        @warn "Wrong parameters"
    end
end

function draw(s::String,M::MPO,J1::Float64,J2::Float64,site::Int)
    if s=="J1-J2"
        for i in 1:dims(M)[site][1]
            for j in 1:dims(M)[site][2]
                if M.data[site][i,j,:,:]==J1*sz
                    print("  σz  ")
                elseif M.data[site][i,j,:,:]==J1*sx
                    print("  σx  ")
                elseif M.data[site][i,j,:,:]==J1*sy
                    print("  σy  ")
                elseif M.data[site][i,j,:,:]==J2*sz
                    print(" J2σz ")
                elseif M.data[site][i,j,:,:]==J2*sx
                    print(" J2σx ")
                elseif M.data[site][i,j,:,:]==J2*sy
                    print(" J2σy ")
                elseif M.data[site][i,j,:,:]==id
                    print("  id  ")
                else
                    print("  //  ")
                end
            end
            println()
        end
    elseif s=="Ising"
        for i in 1:dims(M)[site][1]
            for j in 1:dims(M)[site][2]
                if M.data[site][i,j,:,:]==J1*sz
                    print("  Jσz  ")
                elseif M.data[site][i,j,:,:]==J2*sx
                    print("  hσx  ")
                elseif M.data[site][i,j,:,:]==id
                    print("  id  ")
                else
                    print("  //  ")
                end
            end
            println()
        end
    else
        @warn "No method for drawing this MPO"
    end
end