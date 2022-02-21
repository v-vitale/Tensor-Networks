using TensorOperations
using KrylovKit
using LinearAlgebra


# MPS A-matrix is a 3-index tensor, A[s,i,j]
#    s
#    |
# i -A- j
#
# [s] acts on the local Hilbert space
# [i,j] act on the virtual bonds

# MPO W-matrix is a 4-index tensor, W[i,j,s,t]
#     s
#     |
#  i -W- j
#     |
#     t
#
# [s,t] act on the local Hilbert space,
# [i,j] act on the virtual bonds

function contract_from_left_MPS(A,E,B)
    @tensor temp_1[:] := E[-1,1] * A[-2,1,-3] 
    @tensor temp[:] := temp_1[1,2,-1] * conj( B[2,1,-2] ) 
    return temp
end


## tensor contraction from the left hand side
## +-    +--A-
## |     |  |
## L' =  L--R-
## |     |  |
## +-    +--B-  

function contract_from_left(E_L,X,W)
    @tensor temp_1[:] := E_L[1,-3,-4]  * X[-2,1,-1] 
    @tensor temp_2[:] := temp_1[-1, -2, -5, 1 ]* conj( X[-4, 1,-3] ) 
    @tensor temp[:] := temp_2[ -1, 2, -3, 3, 1 ] * W[1,-2, 2, 3]
    return temp
end


## tensor contraction from the right hand side
##  -+     -A--+
##   |      |  |
##  -R' =  -W--R
##   |      |  |
##  -+     -B--+
function contract_from_right(E_R,Y,W)
    @tensor temp_1[:] :=  Y[-2,-1,1]  * E_R[ 1, -3,-4]
    @tensor temp_2[:] := temp_1[ -1, -2, -5,  1] * conj( Y[-4,-3,1] ) 
    @tensor temp[:] := temp_2[ -1, 1, -3, 2, 6] * W[ -2 6 1 2 ]  
    return temp
end

