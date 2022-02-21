using TensorOperations
using KrylovKit
using LinearAlgebra


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