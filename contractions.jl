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
