using TensorOperations
using KrylovKit
using Printf
using Random
using LinearAlgebra
using TSVD

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


function contract_from_left_MPS(A,E,B)
    @tensor temp_1[:] := E[-1,1] * A[-2,1,-3] 
    @tensor temp[:] := temp_1[1,2,-1] * conj( B[2,1,-2] ) 
    return temp
end

function contract_from_right_MPS(A,F,B)
    @tensor R_1[ e I ; c ] :=  A[ e I ; a ]  * F[ a ; c ]
    @tensor R[ e ; d ] := R_1[ e I ; c ] * conj( B[ d I ; c ] ) 
    return R
end

## initial E and F matrices for the left and right vacuum states
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

## tensor contraction from the left hand side
## +-    +--A-
## |     |  |
## L' =  L--R-
## |     |  |
## +-    +--B-  

function contract_from_left(E_L,X,W)
    @tensor temp_1[:] := E_L[1,-3,-4]  * X[-2,1,-1] 
    @tensor temp_2[:] := temp_1[-1, -2, -5, 1 ]* conj( X[-4, 1,-3] ) 
    @tensor temp[:] := temp_2[ -1, 2, -3, 3, 1 ] * W[1,-2, 2, 3]
    return temp
end


## tensor contraction from the right hand side
##  -+     -A--+
##   |      |  |
##  -R' =  -W--R
##   |      |  |
##  -+     -B--+
function contract_from_right(E_R,Y,W)
    @tensor temp_1[:] :=  Y[-2,-1,1]  * E_R[ 1, -3,-4]
    @tensor temp_2[:] := temp_1[ -1, -2, -5,  1] * conj( Y[-4,-3,1] ) 
    @tensor temp[:] := temp_2[ -1, 1, -3, 2, 6] * W[ -2 6 1 2 ]  
    return temp
end


function one_site_dmrg( MPS::Dict,
                        MPO::Dict,
                        chi::Int64,
                        sweeps::Int64,
                        N::Int64)

    d = 2

    #MPS=init_random_MPS(d,chi,N)
    MPS=Normalize(MPS)
    
    L = construct_L(MPS, MPO, MPS, N)
    R = construct_R(MPS, MPO, MPS, N)

    global MPS,L,R

    for sweep in 1:Int(sweeps/2)
        for i in 1:N-1
            println(i)
            Energy,MPS[i],MPS[i+1] = optimize_one_site( MPS[i], MPS[i+1], MPO[i], L[i], R[i], "right", chi)
            println("Sweep ",sweep*2," Sites ",i," ",i+1," Energy ",Energy)
            L[i+1] = contract_from_left(L[i], MPS[i], MPO[i])
        end

        for i in N:-1:2
            Energy,MPS[i-1],MPS[i] = optimize_one_site( MPS[i-1], MPS[i], MPO[i], L[i], R[i], "left", chi)
            println("Sweep ",sweep*2,"Sites ",i," ",i+1,"Energy %-16.15f\n",Energy)
            R[i-1] = contract_from_right(R[i], MPS[i], MPO[i])
        end

    end

    return MPS

end


function optimize_one_site(A, B, W, E, F, dir, chi )
    function H_lin(v)
        ##     +--A--+
        ##     |  |  |
        ##     L--W--R
        ##     |  |  |
        #println("E ",size(E))
        #println("MPS ",size(v))
        #println("F ",size(F))
        #println("W ",size(W))
        @tensor temp_1[ :] := E[ 1, -2, -3] *  v[ -1, 1, -4]  
        @tensor temp_2[:] :=  temp_1[-1,-2,-3,1] * F[1,-4,-5] 
        @tensor temp[:] := temp_2[1, 2, -2, 3,-3] * W[ 2 3 ; 1 -1 ]
        return temp
    end


    if (dir == "right" )
        sA=size(A)
        
        en,V,info = eigsolve( H_lin , A ,  1  , :SR ; issymmetric = true  )
        en = en[1]
        V = V[1]
        sV = size(V)
        
        V=reshape(V,(sV[1]*sV[2],sV[3]))#s i j -> i*s, j
        A,S,V = svd(V,full=false)
        V=V'
        A = reshape( A , ( sA[1], sA[2], :) )
        #"ij,jl,slk->sik"
        S=diagm(S)
        @tensor B[:] := S[-2,1 ] * V[ 1,2 ] * B[-1,2,-3] 
    else
        sB=size(B)
        print(sB)
        en,V,info = eigsolve( H_lin , B ,  1  , :SR ; issymmetric = true  )
        en = en[1]
        V = V[1]
        
        V=permutedims(V,(2,1,3))
        sV=size(V)
        V=reshape(V,(sV[1],sV[2]*sV[3]))  #s i j -> i s j -> i, s*j
        
        U,S,B = svd(V,full=false)
        B=B'
        B = permutedims(reshape(B,(:,sB[1],sB[3])),(2,1,3))
        #"sij,jk,kl->sil"
        S=diagm(S)
        @tensor A[:] := A[-1,-2,2] * U[ 2,3 ] * S[ 3,-3 ]  
    end

    return en, A, B

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


