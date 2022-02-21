using TensorOperations
using KrylovKit
using LinearAlgebra

include("mps.jl")
include("mpo.jl")


## initial E and F matrices for the left and right vacuum states
function initial_L(W::Array)
    sW=size(W)
    L = ones(sW[1],1,1)#im*zeros(sW[1],1,1)
    return L
end

function initial_R(W::Array)
    sW=size(W)
    R = ones(sW[2],1,1)
    return R
end

function construct_R(A::MPS, W::MPO, B::MPO)
    R = Dict()
    R[A.N] = initial_R(W.data[A.N])
    for i in A.N:-1:2
        R[i-1] = contract_from_right(R[i], A.data[i], B.data[i])
    end
    return R
end

function construct_L(A::MPS, W::MPO)
    L = Dict()
    L[1] = initial_L(W.data[1],N)
    return L
end


function one_site_dmrg( C::MPS,
                        W::MPO,
                        sweeps::Int64)

    d = size(C.data[2])[2]
    chi= size(C.data[2])[1]
    
    L = construct_L(C, W, C)
    R = construct_R(C, W, C)

    global C,L,R

    for sweep in 1:Int(sweeps/2)
        for i in 1:N-1
            println(i)
            Energy,C.data[i],C.data[i+1] = optimize_one_site( C.data[i], C.data[i+1], W.data[i], L[i], R[i], "right", chi)
            println("Sweep ",sweep*2," Sites ",i," ",i+1," Energy ",Energy)
            L[i+1] = contract_from_left(L[i], C.data[i], W.data[i])
        end

        for i in N:-1:2
            Energy,C.data[i-1],C.data[i] = optimize_one_site( C.data[i-1], C.data[i], W.data[i], L[i], R[i], "left", chi)
            println("Sweep ",sweep*2,"Sites ",i," ",i+1,"Energy %-16.15f\n",Energy)
            R[i-1] = contract_from_right(R[i], C.data[i], W.data[i])
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
        @tensor temp_1[ :] := E[ 1, -2, -3] *  v[ 1, -1, -4]  
        @tensor temp_2[:] :=  temp_1[-1,-2,-3,1] * F[1,-4,-5] 
        @tensor temp[:] := temp_2[1, 2, -1, 3,-3] * W[ 2 3 ; 1 -2 ]
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
        @tensor B[:] := S[-1,1 ] * V[ 1,2 ] * B[2,-2,-3] 
    else
        sB=size(B)
        print(sB)
        en,V,info = eigsolve( H_lin , B ,  1  , :SR ; issymmetric = true  )
        en = en[1]
        V = V[1]
        
        sV=size(V)
        V=reshape(V,(sV[1],sV[2]*sV[3]))  #s i j -> i s j -> i, s*j
        
        U,S,B = svd(V,full=false)
        B=B'
        B = reshape(B,(:,sB[2],sB[3]))
        #"sij,jk,kl->sil"
        S=diagm(S)
        @tensor A[:] := A[-1,-2,2] * U[ 2,3 ] * S[ 3,-3 ]  
    end

    return en, A, B

end