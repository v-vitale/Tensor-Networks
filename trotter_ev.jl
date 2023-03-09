#   Author: V. Vitale
#   Feb 2022
#   1 site and 2 sites DMRG

include("mps.jl")
include("mpo.jl")
include("contractions.jl")
using TensorOperations
using KrylovKit
using LinearAlgebra


function trotter_ev!(psi::MPS,
                     gates::Array,
                     sweeps::Int;
                     chimax=2048,
                     tol=1e-15,
                     verbose=false)

    right_normalize!(psi)

    d = dims(psi)[2][2]

    for sweep in 1:Int(sweeps)
        if verbose==true
            println("Sweep: ",sweep)
        end
        for i in 1:psi.N-1
            psi.data[i],psi.data[i+1] = trotter_swipe_right(psi.data[i],
                                                            psi.data[i+1],
                                                            gates[i],  
                                                            chimax,
                                                            tol)
        end

        for i in psi.N:-1:2
            psi.data[i-1],psi.data[i] = trotter_swipe_left(psi.data[i-1],
                                                           psi.data[i], 
                                                           gates[i-1], 
                                                           chimax,
                                                           tol)
        end
    end
end

function trotter_swipe_right(AL::Array, AR::Array, M::Array, chimax::Int, tol::Float64)
    #tol=1e-15

    sAL = size(AL)
    sAR = size(AR)
    @tensor A[:] := AL[-1,-2,1]*AR[1,-3,-4]
    @tensor theta[:]:= A[-1,1,2,-4]*M[1,2,-2,-3]

    theta=reshape(theta,(sAL[1]*sAL[2],sAR[2]*sAR[3]))
    
    U,S,V = svd(theta,full=false,alg=LinearAlgebra.QRIteration())
        
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
    return AL, AR
end

function trotter_swipe_left(AL::Array, AR::Array, M::Array, chimax::Int,tol::Float64)
    #tol=1e-15

    sAL = size(AL)
    sAR = size(AR)

    @tensor A[:] := AL[-1,-2,1]*AR[1,-3,-4]
    @tensor theta[:]:= A[-1,1,2,-4]*M[1,2,-2,-3]

    theta=reshape(theta,(sAL[1]*sAL[2],sAR[2]*sAR[3]))
    
    U,S,V = svd(theta,full=false,alg=LinearAlgebra.QRIteration())

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
    return AL, AR
end
