#   Author: V. Vitale
#   Feb 2022
  
include("mps.jl")
include("mpo.jl")
include("tdvp.jl");

function prob_jump(A::MPS,op::Array,γ::Float64,dt::Float64)
    right_normalize!(A)
    p=zeros(A.N)
    
    @tensor temp =γ*op[4,3]*A.data[1][1,3,2]*conj(A.data[1][1,4,2])*dt
    p[1]=real(temp)
    for i in 1:A.N
        sA = size(A.data[i])
        U,S,V = svd(reshape(A.data[i],(sA[1]*sA[2],sA[3])),full=false)
        V=V'  
        A.data[i] = reshape( U,( sA[1], sA[2], :)) 
        if i<A.N
            S=diagm(S)
            @tensor A.data[i+1][:] := S[-1,1 ] * V[ 1,2 ] * A.data[i+1][2,-2,-3]
            @tensor temp =γ*op[4,3]*A.data[i+1][1,3,2]*conj(A.data[i+1][1,4,2])*dt
            p[i+1]=real(temp)
        else
            @tensor temp =γ*op[4,3]*A.data[i][1,3,2]*conj(A.data[i][1,4,2])*dt
            p[i]=real(temp)
        end
    end   
    return p
end

function jump_MPO(p::BitArray,op::Array,γ::Float64)
    N=length(p)
    d=2
    W=MPO()
    id=[1 0 ; 0 1]
    Wop = im *  zeros(1,1,d,d)
    WId = im *  zeros(1,1,d,d)
    Wop[1,1,:,:] = sqrt(γ)*op
    WId[1,1,:,:] = id

    W.N=N
    for i in 1:N
        if p[i]==true
            W.data[i] = Wop
        else
            W.data[i] = WId
        end
    end
    
    return W
end

function state_preparation(ψ::MPS)
    σp = [0 1; 0 0];
    σm = [0 0; 1 0];
    σx = [0 1; 1 0];
    Id2= [1 0; 0 1];
    γ=0.008326
    pj_list=prob_jump(ψ,σm*σp,γ,1.)
    p_extract=rand(ψ.N)
    pj_bool=p_extract.<pj_list
    jumpo=jump_MPO(pj_bool,σp,γ)
    ψ=jumpo*ψ
    right_normalize!(ψ)
    return ψ
end

function apply_jump(ψ::MPS,dt::Complex)
    σp = [0 1; 0 0];
    σm = [0 0; 1 0];
    σx = [0 1; 1 0];
    Id2= [1 0; 0 1];
    γm=(1/1.17)*0.001;
    pj_list=prob_jump(ψ,σp*σm,γm,imag(dt))
    p_extract=rand(ψ.N)
    pj_bool=p_extract.<pj_list
    jumpo=jump_MPO(pj_bool,σm,γm)
    ψ=jumpo*ψ
    right_normalize!(ψ)

    γx=0.69*0.001;
    pj_list=prob_jump(ψ,σx*σx,γx,imag(dt))
    p_extract=rand(ψ.N)
    pj_bool=p_extract.<pj_list
    jumpo=jump_MPO(pj_bool,σx,γx)
    ψ=jumpo*ψ
    right_normalize!(ψ)
    return ψ
end

function traj_evolution(ψ0::MPS,
                    M::MPO,
                    N::Int,
                    dt::Complex,
                    steps::Int;
                    fs=1,
                    ls=N,
                    sweeps=1,
                    krylovdim=10,
                    chimax=64,
                    is_hermitian=false,
                    ntraj=1,
                    dir="./",
                    save_step=1)
    
    println("# sites: ",N)
    println("krylovdim: ",krylovdim)
    println("max bond dimension: ",chimax)
    println("sweeps per timestep: ",sweeps)
    println(steps," steps with ",imag(sweeps*dt)," timestep; divided in ", sweeps," sweeps")
    sd=ls-fs
    println("Calculate rdm of sites [",fs,",",fs+sd,"]; saving every ",save_step," steps")
    rdm=Dict()
    ρ=Dict()

    ψtlist=pmap(k->state_preparation(ψ0),1:ntraj)
    println(1," ")
    ρ[Array(fs:fs+sd)]=pmap(k->rdm_from_state(ψtlist[k],Array(fs:fs+sd)),1:ntraj)
    rdm[Array(fs:fs+sd)]=sum([ρ[Array(fs:fs+sd)][k] for k in 1:ntraj])/ntraj
        
    npzwrite(dir*"rhoA_"*string(Array(fs:fs+sd))*"_N=$N"*"_steps=$steps"*"_chi=$chimax"*"_ts=1"*".npy",rdm[Array(fs:fs+sd)])
    
    for i in 2:steps
        println(i," ")
        ψtlist=pmap(k->tdvp!(ψtlist[k],M,dt,is_hermitian; tol=1e-12,chimax=chimax, sweeps=sweeps),1:ntraj)
        ψtlist=pmap(k->apply_jump(ψtlist[k],dt),1:ntraj)
        if mod(steps,save_step)==0
            ρ[Array(fs:fs+sd)]=pmap(k->rdm_from_state(ψtlist[k],Array(fs:fs+sd)),1:ntraj)
            rdm[Array(fs:fs+sd)]=sum([ρ[Array(fs:fs+sd)][k] for k in 1:ntraj])/ntraj
            npzwrite(dir*"data/rhoA_"*string(Array(fs:fs+sd))*"_N=$N"*"_steps=$steps"*"_chi=$chimax"*"_ts=$i"*".npy",
                rdm[Array(fs:fs+sd)])
        end
    end
    return rdm
end


function single_traj_evolution( ψ0::MPS,
                    M::MPO,
                    N::Int,
                    dt::Complex,
                    steps::Int;
                    fs=1,
                    ls=N,
                    sweeps=1,
                    krylovdim=10,
                    chimax=64,
                    is_hermitian=false,
                    traj_idx=1,
                    dir="./",
                    save_step=1)
    
    println("# sites: ",N)
    println("krylovdim: ",krylovdim)
    println("max bond dimension: ",chimax)
    println("sweeps per timestep: ",sweeps)
    println(steps," steps with ",imag(sweeps*dt)," timestep; divided in ", sweeps," sweeps")
    sd=ls-fs
    println("Calculate rdm of sites [",fs,",",fs+sd,"]; saving every ",save_step," steps")

    ψt=state_preparation(ψ0)
    println(1," ")
    ρ=rdm_from_state(ψt,Array(fs:fs+sd))
    npzwrite(dir*"rhoA_"*string(Array(fs:fs+sd))*"_N=$N"*"_steps=$steps"*"_chi=$chimax"*"_ts=1_ntraj=$traj_idx"*".npz",ρ )
    
    for i in 2:steps
        println(i," ")
        ψt=tdvp!(ψt,M,dt,is_hermitian; tol=1e-12,chimax=chimax,sweeps=sweeps)
        ψt=apply_jump(ψt,sweeps*dt)
        if mod(steps,save_step)==0
            ρ=rdm_from_state(ψt,Array(fs:fs+sd))
            npzwrite(dir*"rhoA_"*string(Array(fs:fs+sd))*"_N=$N"*"_steps=$steps"*"_chi=$chimax"*"_ts=$i"*"_ntraj=$traj_idx"*".npz",ρ)
        end
    end
    return "End"
end