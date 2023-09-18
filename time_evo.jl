#   Author: V. Vitale
#   Feb 2022
#   TDVP
#   Trotter Evolution


include("base_utils.jl")
using TensorOperations
using KrylovKit
using LinearAlgebra


function trotter_evolution!(psi::MPS,
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

        for i in psi.N:-1
            psi.data[i-1],psi.data[i] = trotter_swipe_left(psi.data[i-1],
                                                           psi.data[i], 
                                                           gates[i-1], 
                                                           chimax,
                                                           tol)
        end
    end
    right_normalize!(psi)
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
    #S=S/norm(S)
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

    
    #S /= norm(S)

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

    #S /= norm(S)

    AR = reshape(V,(:,sAR[2],sAR[3]))

    S=diagm(S)
    @tensor AL[:] :=  U[ -1,1 ] * S[ 1,-2 ]  
    AL = reshape( AL , ( sAL[1], sAL[2], :) )
    return AL, AR
end




function tdvp!(psi::MPS, W::MPO,dt::Complex,is_hermitian::Bool; krylovdim=10,sweeps=1,tol=1e-15,chimax=128)

    right_orthogonalize!(psi)

    d = dims(psi)[2][2]
    L = construct_L(psi, W)
    R = construct_R(psi, W)

    #global psi,L,R
    Energy=0
    for sweep in 1:Int(sweeps)
        for i in 1:psi.N-1
            psi.data[i],psi.data[i+1] = evolve_right(psi.data[i],psi.data[i+1],W.data[i],W.data[i+1],
                                                     L[i], R[i+1], dt/2, krylovdim,is_hermitian,tol,chimax)
            if i!=psi.N-1
                L[i+1] = contract_from_left(L[i], psi.data[i], W.data[i])
                psi.data[i+1] = local_step(psi.data[i+1], W.data[i+1],
                                           L[i+1], R[i+1],dt/2 ,krylovdim,is_hermitian,tol)
            end
        end
        for i in psi.N:-1:2
            psi.data[i-1],psi.data[i] = evolve_left(psi.data[i-1], psi.data[i], W.data[i-1],  W.data[i],
                                                    L[i-1], R[i], dt/2, krylovdim,is_hermitian,tol,chimax)
            if i!=2
                R[i-1] = contract_from_right(R[i], psi.data[i], W.data[i])
                psi.data[i-1] = local_step(psi.data[i-1], W.data[i-1],
                                           L[i-1], R[i-1],dt/2,krylovdim,is_hermitian,tol)
            end
        end
    end
    right_orthogonalize!(psi)
    return psi
end

function evolve_right(AL::Array, AR::Array, WL::Array, WR::Array, E::Array, F::Array, dt::Complex, krylovdim::Int,is_hermitian::Bool,tol::Float64,chimax::Int)

    
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

    #en,V,info = eigsolve( H_lin , A ,  1  , :SR ; issymmetric = true  )
    V,info = exponentiate( H_lin , -dt ,  A ; ishermitian = is_hermitian, tol=tol ,krylovdim=krylovdim)

    V=reshape(V,(sAL[1]*sAL[2],sAR[2]*sAR[3]))
    U,S,V = svd(V,full=false)
    V=V'
    #S=S/norm(S)
    indices = findall(norm(S) .-cumsum(S.^2) .< tol)
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


    #S /= norm(S)

    AL = reshape( U , ( sAL[1], sAL[2], :) )
    #"ij,jl,slk->sik"
    S=diagm(S)
    @tensor AR[:] := S[-1,1 ] * V[ 1,-2 ] 
    AR=reshape(AR,(:,sAR[2],sAR[3]))
    return AL, AR
end


function evolve_left(AL::Array, AR::Array, WL::Array, WR::Array, E::Array, F::Array, dt::Complex, krylovdim::Int,is_hermitian::Bool,tol::Float64,chimax::Int)


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

    V,info = exponentiate( H_lin , -dt ,  A ; ishermitian = is_hermitian, tol=tol ,krylovdim=krylovdim)
    
    sV=size(V)
    V=reshape(V,(sAL[1]*sAL[2],sAR[2]*sAR[3]))

    U,S,V = svd(V,full=false)
    V=V'
    #S /=norm(S)
    indices = findall(norm(S) .-cumsum(S.^2) .< tol)
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


    #S /= norm(S)

    AR = reshape(V,(:,sAR[2],sAR[3]))

    S=diagm(S)
    @tensor AL[:] :=  U[ -1,1 ] * S[ 1,-2 ]  
    AL = reshape( AL , ( sAL[1], sAL[2], :) )
    return AL, AR
end
    
function local_step(A::Array, M::Array, E::Array, F::Array, dt::Complex, krylovdim::Int,is_hermitian::Bool,tol::Float64)

    sA = size(A)
   
        function H_lin(v)
        @tensor temp_1[ :] := E[ 1, -2, -3] *  v[ 1, -1, -4]  
        @tensor temp_2[:] :=  temp_1[-1,-2,-3,1] * F[1,-4,-5] 
        @tensor temp[:] := temp_2[1, 2, -1, 3,-3] * M[ 2 3 ; 1 -2 ]
        return temp
    end

    V,info = exponentiate( H_lin , dt ,  A ; ishermitian = is_hermitian, tol=tol ,krylovdim=krylovdim)

    A=reshape(V,(sA[1],sA[2],sA[3]))
    return A
end
