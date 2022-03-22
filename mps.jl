#   Author: V. Vitale
#   Feb 2022

include("abstractTN.jl")


# MPS A-matrix is a 3-index tensor, A[i,s,j]
#    s
#    |
# i -A- j
#
# [s] acts on the local Hilbert space
# [i,j] act on the virtual bonds

mutable struct MPS <: AbstractTN
  data::Dict
  N::Int
end

MPS() = MPS(Dict(), 0)

function MPS_dot(A::MPS,B::MPS)
    E=ones(1,1)
    for i in 1:A.N
        @tensor temp[:] := E[-1,1]*A.data[i][1,-2,-3] 
        #@tensor E[:] := temp[1,2,-1] * conj( B.data[i][1,2,-2] )
        @tensor E[:] := temp[1,2,-2] * conj( B.data[i][1,2,-1] )
    end
    return E[1]
end

Base.:*(A::MPS,B::MPS)=MPS_dot(A,B)

function Normalize!(A::MPS)
    norm = MPS_dot(A,A)
    for i in 1:A.N
        A.data[i] = A.data[i]/norm^(1/(2*A.N))
    end
end

function Initialize!(A::MPS,d::Int,chi::Int,N::Int)
    temp=Dict()
    temp[1] = im*rand(1,d,chi)
    for i in 2:N-1
        temp[i]= im*rand(chi,d,chi)
    end
    temp[N] = im*rand(chi,d,1)
    A.N=N
    A.data=temp
end

function Initialize!(s::String,A::MPS,N::Int)
    if s=="Brydges_Closed"
        chi=1
        d=2
        temp=Dict()
        temp[1] = im*zeros(1,d,chi)
        for i in 2:N-1
            temp[i]= im*zeros(chi,d,chi)
        end
        temp[N] = im*zeros(chi,d,1)
        
        temp[1][1,:,1] = [0 1]
        for i in 2:2:N-1
            temp[i][1,:,1]= [1 0]
            temp[i+1][1,:,1]= [0 1]
        end
        temp[N][1,:,1] = [1 0]
        A.N=N
        A.data=temp
    elseif s=="Brydges_Open"
        chi=1
        d=4
        p=0.008326
        temp=Dict()
        temp[1] = im*zeros(1,d,chi)
        for i in 2:N-1
            temp[i]= im*zeros(chi,d,chi)
        end
        temp[N] = im*zeros(chi,d,1)
        
        temp[1][1,:,1] = reshape([p 0 ; 0 1-p],d)
        for i in 2:2:N-1
            temp[i][1,:,1]= reshape([ 1 0 ; 0 0],d)
            temp[i+1][1,:,1]= reshape([p 0 ; 0 1-p],d)
        end
        temp[N][1,:,1] = reshape([ 1 0 ; 0 0],d)
        A.N=N
        A.data=temp
    end
end

function right_normalize!(A::MPS)
    for i in A.N:-1:1
        sA = size(A.data[i])
        U, S, V = svd(reshape(A.data[i],sA[1], sA[2]*sA[3]), full=false)
        V=V'
        S /= norm(S)
        A.data[i] = reshape(V,(:, sA[2], sA[3]))
        if i>1
            S=diagm(S)
            @tensor A.data[i-1][:] := A.data[i-1][-1,-2,2] * U[ 2,3 ] * S[ 3,-3 ]  
        end
    end  
end
   
function right_orthogonalize!(A::MPS)
    for i in A.N:-1:1
        sA = size(A.data[i])
        U, S, V = svd(reshape(A.data[i],sA[1], sA[2]*sA[3]), full=false)
        V=V'
        A.data[i] = reshape(V,(:, sA[2], sA[3]))
        if i>1
            S=diagm(S)
            @tensor A.data[i-1][:] := A.data[i-1][-1,-2,2] * U[ 2,3 ] * S[ 3,-3 ]  
        end
    end  
end

function left_normalize!(A::MPS)
    for i in 1:A.N
        sA = size(A.data[i])
        U,S,V = svd(reshape(A.data[i],(sA[1]*sA[2],sA[3])),full=false)
        S /= norm(S)
        V=V'  
        A.data[i] = reshape( U,( sA[1], sA[2], :)) 
        if i<A.N
            S=diagm(S)
            @tensor A.data[i+1][:] := S[-1,1 ] * V[ 1,2 ] * A.data[i+1][2,-2,-3] 
        end
    end    
end

function left_orthogonalize!(A::MPS)
    for i in 1:A.N
        sA = size(A.data[i])
        U,S,V = svd(reshape(A.data[i],(sA[1]*sA[2],sA[3])),full=false)
        V=V'  
        A.data[i] = reshape( U,( sA[1], sA[2], :)) 
        if i<A.N
            S=diagm(S)
            @tensor A.data[i+1][:] := S[-1,1 ] * V[ 1,2 ] * A.data[i+1][2,-2,-3] 
        end
    end    
end

function move_orthogonality_center!(A::MPS,b::Int)
    right_orthogonalize!(A)
    for i in 1:b
        sA = size(A.data[i])
        U,S,V = svd(reshape(A.data[i],(sA[1]*sA[2],sA[3])),full=false)
        #S /= norm(S)
        V=V'  
        A.data[i] = reshape( U,( sA[1], sA[2], :)) 
        if i<A.N
            S=diagm(S)
            @tensor A.data[i+1][:] := S[-1,1 ] * V[ 1,2 ] * A.data[i+1][2,-2,-3] 
        end
    end    
end

    
function calc_entropy(A::MPS)
    M=copy(A)
    Sent = zeros(M.N)
    for i in 1:M.N
        sM = size(M.data[i])
        U,S,V = svd(reshape(M.data[i],(sM[1]*sM[2],sM[3])),full=false)
        #S /= norm(S)
        V=V'  
        M.data[i] = reshape( U,( sM[1], sM[2], :)) 
        if i<M.N
            @tensor M.data[i+1][:] := diagm(S)[-1,1 ] * V[ 1,2 ] * M.data[i+1][2,-2,-3] 
        end
        Sent[i] = sum(-dot(S.^2,log.(S.^2)))
    end 

    
    return Sent
end

function calc_Renyi2(A::MPS)
    M=copy(A)
    Sent = zeros(M.N)
    for i in 1:M.N
        sM = size(M.data[i])
        U,S,V = svd(reshape(M.data[i],(sM[1]*sM[2],sM[3])),full=false)
        #S /= norm(S)
        V=V'  
        M.data[i] = reshape( U,( sM[1], sM[2], :)) 
        if i<M.N
            @tensor M.data[i+1][:] := diagm(S)[-1,1 ] * V[ 1,2 ] * M.data[i+1][2,-2,-3] 
        end
        Sent[i] = -log(sum(S.^4))
    end   
    
    return Sent
end


function calc_purity(A::MPS)
    M=copy(A)
    Sent = zeros(M.N)
    for i in 1:M.N
        sM = size(M.data[i])
        U,S,V = svd(reshape(M.data[i],(sM[1]*sM[2],sM[3])),full=false)
        #S /= norm(S)
        V=V'  
        M.data[i] = reshape( U,( sM[1], sM[2], :)) 
        if i<M.N
            @tensor M.data[i+1][:] := diagm(S)[-1,1 ] * V[ 1,2 ] * M.data[i+1][2,-2,-3] 
        end
        Sent[i] = sum(S.^4)
    end   
    
    return Sent
end

function calc_trace(A::MPS)
    M=copy(A)
    Sent = zeros(M.N)
    for i in 1:M.N
        sM = size(M.data[i])
        U,S,V = svd(reshape(M.data[i],(sM[1]*sM[2],sM[3])),full=false)
        #S /= norm(S)
        V=V'  
        M.data[i] = reshape( U,( sM[1], sM[2], :)) 
        if i<M.N
            @tensor M.data[i+1][:] := diagm(S)[-1,1 ] * V[ 1,2 ] * M.data[i+1][2,-2,-3] 
        end
        Sent[i] = sum(S.^2)
    end   
    
    return Sent
end

function to_dm_MPS(A::MPS)
    M=MPS()
    M.N=A.N
    for i in 1:M.N
        sA=size(A.data[i]) 
        @tensor M.data[i][:] := A.data[i][-1,-3,-5]*conj(A.data[i][-2,-4,-6])#*conj(A.data[i][-2,-4,-6])
        M.data[i]= reshape(M.data[i],(sA[1]*sA[1],sA[2]*sA[2],sA[3]*sA[3]))
    end   
    return M
end

"
function rdm_list(A::MPS,r::Array)
    #move_orthogonality_center!(A,1)
    sA=size(A.data[1]) 
    @tensor rd_rho[:] := A.data[1][-1,-3,-2]
    rd_rho=reshape(rd_rho,(sA[1],sA[3],isqrt(sA[2]),isqrt(sA[2])))
    for j in 2:A.N
        sA=size(A.data[j])
        @tensor M[:] := A.data[j][-1,-3,-2]
        M=reshape(M,(sA[1],sA[3],isqrt(sA[2]),isqrt(sA[2])))
        if j ∈ r
            st=size(rd_rho)
            sM=size(M)
            @tensor rd_rho[:] :=rd_rho[-1,1,-3,-5]*M[1,-2,-4,-6]
            rd_rho=reshape(rd_rho,(st[1],sM[2],st[3]*sM[3],st[4]*sM[4]))
        elseif j ∉ r
            @tensor rd_rho[:] :=rd_rho[-1,1,-3,-4]*M[1,-2,2,2]
        end
    end 
    @tensor rd_rho[:] :=rd_rho[1,1,-1,-2]
    return rd_rho
end
"

function rdm_rho(A::MPS,r::Array)
    #move_orthogonality_center!(A,1)
    sA=size(A.data[r[1]]) 
    rd_rho=reshape(A.data[r[1]],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
    for j in r[2:end]
        sA=size(A.data[j])
        M=reshape(A.data[j],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
        st=size(rd_rho)
        sM=size(M)
        @tensor rd_rho[:] :=rd_rho[-1,-2,-4,1]*M[1,-3,-5,-6]
        rd_rho=reshape(rd_rho,(st[1],st[2]*sM[2],st[3]*sM[3],sM[4]))
    end 
    for j in r[1]-1:-1:1
        sA=size(A.data[j])
        M=reshape(A.data[j],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
        st=size(rd_rho)
        sM=size(M)
        @tensor rd_rho[:] :=M[-1,2,2,1]*rd_rho[1,-2,-3,-4]
    end
    for j in r[end]+1:A.N
        sA=size(A.data[j])
        M=reshape(A.data[j],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
        st=size(rd_rho)
        sM=size(M)
        @tensor rd_rho[:] :=rd_rho[-1,-2,-3,1]*M[1,2,2,-4]
    end
    @tensor rd_rho[:] :=rd_rho[1,-1,-2,1]
    return rd_rho
end


function rdm_from_state(A::MPS,r::Array)
    L=ones(1,1)
    for j in 1:r[1]-1
        @tensor L[:] :=L[1,2]*A.data[j][1,3,-3]*conj(A.data[j])[2,3,-4]
    end
    R=ones(1,1)
    for j in A.N:-1:r[end]+1
        @tensor R[:] :=A.data[j][-1,3,1]*conj(A.data[j])[-2,3,2]*R[1,2]
    end
    @tensor L[:] := L[1,2]*A.data[r[1]][1,-1,-3]*conj(A.data[r[1]])[2,-2,-4]
    @tensor R[:] := A.data[r[end]][-1,-3,1]*conj(A.data[r[end]])[-2,-4,2]*R[1,2];

    j=r[1]
    while A.N-2*j>0
        sA=size(A.data[j+1])
        sL=size(L)
        @tensor L[:] := L[-1,-3,1,2]*A.data[j+1][1,-2,-5]*conj(A.data[j+1])[2,-4,-6]
        L=reshape(L,(sL[1]*sA[2],sL[2]*sA[2],sA[3],sA[3]))
        sA=size(A.data[A.N-j])
        sR=size(R)
        @tensor R[:] := A.data[A.N-j][-1,-3,1]*conj(A.data[A.N-j])[-2,-5,2]*R[1,2,-4,-6]
        R=reshape(R,(sA[1],sA[1],sR[3]*sA[2],sR[4]*sA[2]))
        j+=1
    end
    @tensor rd_rho[:] := L[-1,-3,1,2]*R[1,2,-2,-4]
    return reshape(rd_rho,(2^length(r),2^length(r)))
end

function rdm_from_dm(A::MPS,r::Array)

    L=ones(1)
    for j in 1:r[1]-1
        sA=size(A.data[j])
        M=reshape(A.data[j],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
        @tensor L[:] :=L[1]*M[1,2,2,-1]
    end
    
    R=ones(1)
    for j in A.N:-1:r[end]+1
        sA=size(A.data[j])
        M=reshape(A.data[j],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
        @tensor R[:] :=M[-1,2,2,1]*R[1]
    end

    sA=size(A.data[r[1]])
    M=reshape(A.data[r[1]],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
        
    @tensor L[:] := L[1]*M[1,-1,-2,-3]
    sA=size(A.data[r[end]])
    M=reshape(A.data[r[end]],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
    @tensor R[:] := M[-1,-2,-3,1]*R[1];
    
    j=r[1]
    while A.N-2*j>0
        sA=size(A.data[j+1])
        M=reshape(A.data[j+1],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
        sL=size(L)
        @tensor L[:] := L[-1,-3,1]*M[1,-2,-4,-5]
        L=reshape(L,(sL[1]*isqrt(sA[2]),sL[2]*isqrt(sA[2]),sL[3]))

        sR=size(A.data[A.N-j])
        M=reshape(A.data[A.N-j],(sA[1],isqrt(sA[2]),isqrt(sA[2]),sA[3]))
        sR=size(R)

        @tensor R[:] := M[-1,-3,-5,1]*R[1,-2,-4]
        R=reshape(R,(sA[1],sR[2]*isqrt(sA[2]),sR[3]*isqrt(sA[2])))
        j+=1
    end
    @tensor rd_rho[:] := L[-1,-3,1]*R[1,-2,-4]
    return reshape(rd_rho,(2^length(r),2^length(r)))
end


