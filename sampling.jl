#   Author: V. Vitale
#   Feb 2022
  
include("mps.jl")
include("mpo.jl")
using StatsBase

all_perm(xs, n) = vec(map(collect, Iterators.product(ntuple(_ -> xs, n)...)))

function save_p_configs(base_states::Array,A::MPS)
    right_normalize!(A)
    p=zeros(2^(A.N))
    for (step,state) in enumerate(base_states)
        W=MPO()
        Initialize!("proj",W,state)
        p[step]=real(A*W*A)
    end
    if abs(sum(p)-1.)>1e-12
        @warn "Probability not summing to 1: "*string(sum(p))
    end
    return p
end

function sample_shadows(A::MPS,nu::Int,base_states::Array)
    shadows=[]
    proj=[[1. 0. ; 0. 0.],[0. 0.; 0. 1.]]
    for r in 1:nu
        W=MPO()
        Initialize!("Local_Haar",W,A.N)
        
        state=copy(W*A)
        
        p_state=save_p_configs(base_states,state)
        for (step,conf) in enumerate(StatsBase.sample(1:2^(A.N),ProbabilityWeights(p_state),1))
            config=base_states[conf]
            shadow=MPO()
            Initialize!("Zeros",shadow,2,1,A.N)

            for j in 1:A.N
                shadow.data[j][1,1,:,:]=3*proj[config[j]+1]-1.0*I(2) 
            end
            push!(shadows,adjoint(W)*shadow*W)
        end
    end
    return shadows
end


function calc_dist(rhos::Array,nu::Int,nm::Int)

    distrhos=zeros(nu*nm,nu*nm)
    idx=zeros(Int,nu*nm,nu*nm)

    for i in 1:nu*nm
        for j in i:nu*nm
            distrhos[i,j]=real(tr(rhos[i]*adjoint(rhos[i]))-
                tr(rhos[i]*adjoint(rhos[j]))-
                tr(rhos[j]*adjoint(rhos[i]))+
                tr(rhos[j]*adjoint(rhos[j])))^(1/2)
        end
    end
    distrhos=distrhos+distrhos'
    for i in 1:length(rhos)
        idx[i,:]=sortperm(distrhos[i,:])
        distrhos[i,:]=distrhos[i,idx[i,:]]
    end

    return distrhos
end


function sample_configs(ψ,Nr)
    ψ0=copy(ψ)
    right_normalize!(ψ0)
    P=MPO()
    configs=zeros(ψ.N,Nr)
    for rep in 1:Nr
        ψ=copy(ψ0)
        config=[]
        for site in 1:ψ.N
            @tensor temp[:]:=ψ.data[site][1,-1,2]*conj(ψ.data[site][1,-2,2])
            rho=temp/tr(temp)
            p=real(rho[1,1])
            if rand() < p 
                push!(config,0)
            else
                push!(config,1)
            end
            Initialize!("proj",P,[config[site]],[site],ψ.N)
            ψ=P*ψ
            
            sA = size(ψ.data[site])
            U,S,V = svd(reshape(ψ.data[site],(sA[1]*sA[2],sA[3])),full=false,alg=LinearAlgebra.QRIteration())
            V=V'  
            ψ.data[site] = reshape( U,( sA[1], sA[2], :)) 
            if site<ψ.N
                S=diagm(S)
                @tensor ψ.data[site+1][:] := S[-1,1 ] * V[ 1,2 ] * ψ.data[site+1][2,-2,-3] 
            end
        end
    end
    return configs
end

