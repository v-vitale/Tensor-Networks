#   Author: V. Vitale
#   Feb 2022
#   Conctraction utilities

## initial E and F matrices for the left and right vacuum states
function initial_L(W::Array)
    sW=size(W)
    L = ones(1,sW[1],1)#ones(sW[1],1,1)
    return L
end

function initial_R(W::Array)
    sW=size(W)
    R = ones(1,sW[2],1)#ones(sW[2],1,1)
    return R
end

function construct_R(A::MPS, W::MPO)
    R = Dict()
    R[A.N] = initial_R(W.data[A.N])
    for i in A.N:-1:1
        R[i-1] = contract_from_right(R[i], A.data[i], W.data[i])
    end
    return R
end

function construct_L(A::MPS, W::MPO)
    L = Dict()
    L[1] = initial_L(W.data[1])
    return L
end


## tensor contraction from the left hand side
## +-    +--X-
## |     |  |
## L' =  L--W-
## |     |  |
## +-    +--X-  

function contract_from_left(E_L::Array,X::Array,W::Array)
    @tensor temp_1[:] := E_L[1,-3,-4]  * X[1,-2,-1] 
    @tensor temp_2[:] := temp_1[-1, -2, -5, 1 ]* conj( X[1, -4,-3] ) 
    @tensor temp[:] := temp_2[ -1, 2, -3, 3, 1 ] * W[1,-2, 2, 3]
    return temp
end

## tensor contraction from the right hand side
##  -+     -Y--+
##   |      |  |
##  -R' =  -W--R
##   |      |  |
##  -+     -Y--+
function contract_from_right(E_R::Array,Y::Array,W::Array)
    @tensor temp_1[:] :=  Y[-1,-2,1]  * E_R[ 1, -3,-4]
    @tensor temp_2[:] := temp_1[ -1, -2, -5,  1] * conj( Y[-3,-4,1] ) 
    @tensor temp[:] := temp_2[ -1, 1, -3, 2, 6] * W[ -2 6 1 2 ]  
    return temp
end


function average(psi::MPS,W::MPO)
    sW=size(W.data[1])
    L = ones(1,sW[1],1)#ones(sW[1],1,1)
    
    right_normalize!(psi)
    
    for i in 1:psi.N
         @tensor L[:] := L[1,2,3]*psi.data[i][1,4,-1]*W.data[i][2,-2,4,5]*conj(psi.data[i][3,5,-3]) 
    end
    return L[1,1,1]
end



function MPS_dot(A::MPS,B::MPS)
    E=ones(1,1)
    for i in 1:A.N
        @tensor temp[:] := E[-1,1]*A.data[i][1,-2,-3] 
        @tensor E[:] := temp[1,2,-2] * conj( B.data[i][1,2,-1] )
    end
    return E[1]
end

Base.:*(A::MPS,B::MPS)=MPS_dot(A,B)


function MPO_dot(W::MPO,Q::MPO)
    if isempty(W.data) || isempty(Q.data)
        @warn "Empty MPO."
        return 0
    end
    temp=MPO()
    temp.N=W.N
    for i in 1:W.N
        sW=size(W.data[i])
        sQ=size(Q.data[i])
        @tensor temp.data[i][:] := W.data[i][-1,-3,-5,1]*Q.data[i][-2,-4,1,-6]
        temp.data[i]=reshape(temp.data[i],sW[1]*sQ[1],sW[2]*sQ[2],sW[3],sQ[4])
    end
    return temp
end

Base.:*(W::MPO,Q::MPO)=MPO_dot(W,Q)



function MPO_MPS_dot(W::MPO,Q::MPS)
    if isempty(W.data)
        @warn "Empty MPO."
        #return 0
    end
    if  isempty(Q.data)
        @warn "Empty MPS."
        #return 0
    end
    
    temp=MPS()
    temp.N=W.N
    for i in 1:W.N
        sW=size(W.data[i])
        sQ=size(Q.data[i])
        @tensor temp.data[i][:] := W.data[i][-1,-4,-3,1]*Q.data[i][-2,1,-5]
        temp.data[i]=reshape(temp.data[i],sW[1]*sQ[1],sW[3],sW[2]*sQ[3])
    end
    return temp
end

function MPS_MPO_dot(Q::MPS,W::MPO)
    if isempty(W.data)
        @warn "Empty MPO."
        #return 0
    end
    if  isempty(Q.data)
        @warn "Empty MPS."
        #return 0
    end
    
    temp=MPS()
    temp.N=W.N
    for i in 1:W.N
        sW=size(W.data[i])
        sQ=size(Q.data[i])
        @tensor temp.data[i][:] := W.data[i][-1,-4,1,-3]*Q.data[i][-2,1,-5]
        temp.data[i]=reshape(temp.data[i],sW[1]*sQ[1],sW[3],sW[2]*sQ[3])
    end
    return temp
end


Base.:*(W::MPO,A::MPS)=MPO_MPS_dot(W,A)
Base.:*(A::MPS,W::MPO)=MPS_MPO_dot(A,W)
