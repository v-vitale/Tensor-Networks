#   Author: V. Vitale
#   Feb 2022
  
include("mps.jl")
include("mpo.jl")


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