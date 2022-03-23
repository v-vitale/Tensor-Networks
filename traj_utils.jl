#   Author: V. Vitale
#   Feb 2022
  
include("mps.jl")
include("mpo.jl")


function prob_jump(A::MPS,op::Array,γ::Float64)
    right_normalize!(A)
    p=zeros(A.N)
    
    @tensor temp[:]:=γ*op[-1,3]*A.data[1][1,3,2]*conj(A.data[1][1,-2,2])
    p[1]=real(temp[1,1])
    for i in 1:A.N
        sA = size(A.data[i])
        U,S,V = svd(reshape(A.data[i],(sA[1]*sA[2],sA[3])),full=false)
        V=V'  
        A.data[i] = reshape( U,( sA[1], sA[2], :)) 
        if i<A.N
            S=diagm(S)
            @tensor A.data[i+1][:] := S[-1,1 ] * V[ 1,2 ] * A.data[i+1][2,-2,-3] 
            @tensor temp[:]:=γ*op[-1,3]*A.data[i+1][1,3,2]*conj(A.data[i+1][1,-2,2])
            p[i+1]=real(temp[1,1])
        else
            @tensor temp[:]:=γ*op[-1,3]*A.data[i][1,3,2]*conj(A.data[i][1,-2,2])
            p[i]=real(temp[1,1])
        end
        
    end   
    return p
end

function jump_MPO(p::Array,op::Array,γ::Float64)
    chi=1
    d=2
    W=MPO()
    
    Wop = im *  zeros(chi,chi,d,d)
    Wop1 = im *  zeros(1,chi,d,d)
    Wop2 = im *  zeros(chi,1,d,d)
    WId = im *  zeros(chi,chi,d,d)
    WId1 = im *  zeros(1,chi,d,d)
    WId2 = im *  zeros(chi,1,d,d)
    Wop[1,1,:,:] = op
    Wop1[1,1,:,:] = op
    Wop2[1,1,:,:] = op
    WId[1,1,:,:] = id
    WId1[1,1,:,:] = id
    WId2[1,1,:,:] = id
        
    
    W.N=N
    if p[1]==true
        W.data[1] = Wop1
    else
         W.data[1] = WId1
    end
    for i in 2:(N-1)
        if p[i]==true
            W.data[1] = Wop
        else
            W.data[1] = WId
        end
    end
    if p[end]==true
        W.data[length(p)] = Wop2
    else
        W.data[length(p)] = WId2
    end
    
    return W
end