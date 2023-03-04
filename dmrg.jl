#   Author: V. Vitale
#   Feb 2022
#   1 site and 2 sites DMRG

include("mps.jl")
include("mpo.jl")
include("contractions.jl")
using TensorOperations
using KrylovKit
using LinearAlgebra


function one_site_dmrg!(psi::MPS,
                        W::MPO,
                        sweeps::Int64)

    right_normalize!(psi)

    d = dims(psi)[2][2]
    chi = dims(psi)[2][1]
    L = construct_L(psi, W)
    R = construct_R(psi, W)

    #global psi,L,R
    Energy=0
    for sweep in 1:Int(sweeps/2)
        for i in 1:psi.N-1
            Energy,psi.data[i],psi.data[i+1] = optimize_one_site(psi.data[i], psi.data[i+1], W.data[i], L[i], R[i], "right", chi)
            L[i+1] = contract_from_left(L[i], psi.data[i], W.data[i])
        end

        for i in psi.N:-1:2
            Energy,psi.data[i-1],psi.data[i] = optimize_one_site( psi.data[i-1],psi.data[i], W.data[i], L[i], R[i], "left", chi)
            R[i-1] = contract_from_right(R[i], psi.data[i], W.data[i])
        end
    end
    println("Done! Energy= ",real(Energy),"; Variance: ",real(psi*(W*(W*psi))-(psi*(W*psi))^2))
end



function optimize_one_site(A::Array, B::Array, W::Array, E::Array, F::Array, dir::String, chi::Int )
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

        en,V,info = eigsolve( H_lin , A ,  1  , :SR ; ishermitian = true  )
        en = en[1]
        V = V[1]
        sV = size(V)

        V=reshape(V,(sV[1]*sV[2],sV[3]))#s i j -> i*s, j
        
        A,S,V = svd(V,full=false,alg=LinearAlgebra.QRIteration())

        V=V'
        A = reshape( A , ( sA[1], sA[2], :) )
        #"ij,jl,slk->sik"
        S=diagm(S)
        @tensor B[:] := S[-1,1 ] * V[ 1,2 ] * B[2,-2,-3] 
    else
        sB=size(B)
        en,V,info = eigsolve( H_lin , B ,  1  , :SR ; ishermitian = true  )
        en = en[1]
        V = V[1]

        sV=size(V)
        V=reshape(V,(sV[1],sV[2]*sV[3]))  #s i j -> i s j -> i, s*j
        
        U,S,B = svd(V,full=false,alg=LinearAlgebra.QRIteration())

        B=B'
        B = reshape(B,(:,sB[2],sB[3]))
        #"sij,jk,kl->sil"
        S=diagm(S)
        @tensor A[:] := A[-1,-2,2] * U[ 2,3 ] * S[ 3,-3 ]  
    end

    return en, A, B

end



function two_sites_dmrg!(psi::MPS,
                        W::MPO,
                        sweeps::Int;
                        chimax=2048,
                        tol=1e-15,
                        verbose=false)

    right_normalize!(psi)

    d = dims(psi)[2][2]
    L = construct_L(psi, W)
    R = construct_R(psi, W)

    #global psi,L,R
    Energy=0
    for sweep in 1:Int(sweeps)
        if verbose==true
          println("Sweep: ",sweep)
        end
        #println(dims(psi))
        for i in 1:psi.N-1
            Energy,psi.data[i],psi.data[i+1] = two_sites_swipe_right( psi.data[i],
                                                                        psi.data[i+1],
                                                                        W.data[i],                              
                                                                        W.data[i+1],
                                                                        L[i],
                                                                        R[i+1],
                                                                        chimax,
                                                                        tol)
            L[i+1] = contract_from_left(L[i], psi.data[i], W.data[i])
        end

        for i in psi.N:-1:2
            Energy,psi.data[i-1],psi.data[i] = two_sites_swipe_left(  psi.data[i-1],
                                                                        psi.data[i], 
                                                                        W.data[i-1], 
                                                                        W.data[i], 
                                                                        L[i-1], 
                                                                        R[i],
                                                                        chimax,
                                                                        tol)
            R[i-1] = contract_from_right(R[i], psi.data[i], W.data[i])
        end
        if verbose==true
          println("Done! Energy= ",real(Energy),"; Variance: ",real(average(ψ,M*M)-average(ψ,M)^2))    
        end
    end
end

function two_sites_swipe_right(AL::Array, AR::Array, WL::Array, WR::Array, E::Array, F::Array, chimax::Int, tol::Float64)
    #tol=1e-15

    sAL = size(AL)
    sAR = size(AR)
    @tensor A[:] := AL[-1,-2,1]*AR[1,-3,-4]
    A = reshape(A,(sAL[1],sAL[2]*sAR[2],sAR[3]))
    sWL = size(WL)
    sWR = size(WR)

    @tensor M[:] := WL[-1,1,-3,-5]*WR[1,-2,-4,-6]
    M = reshape(M,(sWL[1],sWR[2],sWL[3]*sWR[3],sWL[4]*sWR[4]))
    function H_lin(v)
        @tensor temp_1[ :] := E[ 1, -2, -3] *  v[ 1, -1, -4]  
        @tensor temp_2[:] :=  temp_1[-1,-2,-3,1] * F[1,-4,-5] 
        @tensor temp[:] := temp_2[1, 2, -1, 3,-3] * M[ 2 3 ; 1 -2 ]
        return temp
    end

    en,V,info = eigsolve( H_lin , A ,  1  , :SR ; ishermitian = true  )
    en = en[1]
    V = V[1]

    V=reshape(V,(sAL[1]*sAL[2],sAR[2]*sAR[3]))
    
    U,S,V = svd(V,full=false,alg=LinearAlgebra.QRIteration())

        
    V=V'
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

    
    S /= norm(S)

    AL = reshape( U , ( sAL[1], sAL[2], :) )
    #"ij,jl,slk->sik"
    S=diagm(S)
    @tensor AR[:] := S[-1,1 ] * V[ 1,-2 ] 
    AR=reshape(AR,(:,sAR[2],sAR[3]))
    return en, AL, AR
end

function two_sites_swipe_left(AL::Array, AR::Array, WL::Array, WR::Array, E::Array, F::Array , chimax::Int,tol::Float64)
    #tol=1e-15

    sAL = size(AL)
    sAR = size(AR)

    @tensor A[:] :=AL[-1,-2,1]*AR[1,-3,-4]
    A = reshape(A,(sAL[1],sAL[2]*sAR[2],sAR[3]))

    sWL = size(WL)
    sWR = size(WR)


    @tensor M[:] := WL[-1,1,-3,-5]*WR[1,-2,-4,-6]
    M = reshape(M,(sWL[1],sWR[2],sWL[3]*sWR[3],sWL[4]*sWR[4]))

    function H_lin(v)
            @tensor temp_1[ :] := E[ 1, -2, -3] *  v[ 1, -1, -4]  
            @tensor temp_2[:] :=  temp_1[-1,-2,-3,1] * F[1,-4,-5] 
            @tensor temp[:] := temp_2[1, 2, -1, 3,-3] * M[ 2 3 ; 1 -2 ]
        return temp
    end

    en,V,info = eigsolve( H_lin , A ,  1  , :SR ; ishermitian = true  )
    en = en[1]
    V = V[1]
    sV=size(V)
    V=reshape(V,(sAL[1]*sAL[2],sAR[2]*sAR[3]))
    
    U,S,V = svd(V,full=false,alg=LinearAlgebra.QRIteration())

    V=V'
    
    S /=norm(S)
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

    S /= norm(S)

    AR = reshape(V,(:,sAR[2],sAR[3]))

    S=diagm(S)
    @tensor AL[:] :=  U[ -1,1 ] * S[ 1,-2 ]  
    AL = reshape( AL , ( sAL[1], sAL[2], :) )
    return en, AL, AR
end
