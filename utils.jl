using TensorOperations
using KrylovKit
using LinearAlgebra
using Random

# MPS A-matrix is a 3-index tensor, A[s,i,j]
#    s
#    |
# i -A- j
#
# [s] acts on the local Hilbert space
# [i,j] act on the virtual bonds

# MPO W-matrix is a 4-index tensor, W[i,j,s,t]
#     s
#     |
#  i -W- j
#     |
#     t
#
# [s,t] act on the local Hilbert space,
# [i,j] act on the virtual bonds



function initial_L(W,N)
    sW=size(W)
    L = ones(sW[1],1,1)#im*zeros(sW[1],1,1)
    #L[1] = 1
    #println(size(L))
    return L
end

function initial_R(W,N)
    sW=size(W)
    R = ones(sW[2],1,1)#im*zeros(sW[2],1,1)
    #R[end] = 1
    #println(size(R))
    return R
end

function construct_R(MPS, MPO, Blist, N)
    R = Dict()
    R[N] = initial_R(MPO[N],N)
    for i in N:-1:2
        R[i-1] = contract_from_right(R[i], MPS[i], MPO[i])
    end
    return R
end

function construct_L(Alist, MPO, Blist,N)
    L = Dict()
    L[1] = initial_L(MPO[1],N)
    return L
end

function init_random_MPS(d,m,N)

    MPS=Dict()
    MPS[1] = im*rand(d,1,m)
    for i in 2:N-1
        MPS[i]= im*rand(d,m,m)
    end
    MPS[N] = im*rand(d,m,1)
    return MPS
end

function Normalize(MPS)
    norm = MPS_dot(MPS,MPS)
    for i in 1:length(MPS)
        MPS[i] = MPS[i]/norm^(1/(2*length(MPS)))
    end
    return MPS
end


function MPS_dot(A,B)
    E=ones(1,1)
    for i in 1:length(A)
        E = contract_from_left_MPS( A[i] , E , B[i] )
    end
    return E[1]
end

function Construct_Ising_MPO(h,J,N)
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
    W[3,3,:,:]=id;

    W1[1,1,:,:]=-h*sx
    W1[1,2,:,:]=-J*sz
    W1[1,3,:,:]=id

    W2[1,1,:,:]=id
    W2[2,1,:,:]=sz
    W2[3,1,:,:]=-h*sx


    MPO = Dict()
    MPO[1] = W1
    for i in 2:(N-1)
        MPO[i] = W
    end
    MPO[N] = W2

    return MPO
end


