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
    temp=Dict()
    for i in 1:W.N
        sW=size(W.data[i])
        sQ=size(Q.data[i])
        @tensor temp[i][:] := W.data[i][-1,-4,-3,1]*Q.data[i][-2,-5,1]
        temp[i]=reshape(temp[i],sW[1]*sQ[1],sW[3],sW[2]*sQ[3])
    end
    return temp
end

Base.:*(W::MPO,A::MPS)=MPO_dot(W,A)

function trace_MPO(A::MPO)
    temp=copy(A.data[1])
    for i in 1:A.N-1
        @tensor temp[-1,-2,-3,-4] := temp[-1,2,1,1]*A.data[i+1][2,-2,-3,-4]
    end
    @tensor result=temp[1,1,2,2]
    return result
end

tr(A::MPO)= trace_MPO(A)

function Initialize!(s::String,W::MPO,h::Float64,J::Float64,N::Float64)
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
    else
        println("Wrong parameters")
    end
end
