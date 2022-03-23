#   Author: V. Vitale
#   Feb 2022
  
include("mps.jl")
include("mpo.jl")


function save_rhoi(A::MPS)
    right_normalize!(A)
    p=zeros(A.N)
    @tensor temp[:]:=A.data[1][1,-1,2]*conj(A.data[1][1,-2,2])
    p[1]=real(temp[1,1])
    for i in 1:A.N
        sA = size(A.data[i])
        U,S,V = svd(reshape(A.data[i],(sA[1]*sA[2],sA[3])),full=false)
        V=V'  
        A.data[i] = reshape( U,( sA[1], sA[2], :)) 
        if i<A.N
            S=diagm(S)
            @tensor A.data[i+1][:] := S[-1,1 ] * V[ 1,2 ] * A.data[i+1][2,-2,-3] 
            @tensor temp[:]:=A.data[i+1][1,-1,2]*conj(A.data[i+1][1,-2,2])
            p[i+1]=real(temp[1,1])
        else
            @tensor temp[:]:=A.data[i][1,-1,2]*conj(A.data[i][1,-2,2])
            p[i]=real(temp[1,1])
        end
        
    end   
    return p
end



function sample_shadows(A::MPS,nu::Int,nm::Int,lA::Int)
    if nm==1
        shadows=[]
        ket=[[1 ; 0],[0 ; 1]]
        for r in 1:nu
            WlA=MPO()
            Initialize!("Local_Haar",WlA,lA)

            if lA<A.N
                W=MPO()
                Initialize!("Id",W,A.N)
                for i in 1:lA
                    W.data[i]=Base.copy(WlA.data[i])
                end
            else
                W=copy(WlA)
            end 

            state=W*A
            p_state=save_rhoi(state)[1:lA]
            for m in 1:nm        
                shadow=MPO()
                Initialize!("Zeros",shadow,2,1,lA)
                k=zeros(Int,lA)
                p_extract=rand(lA)
                k[p_extract.>p_state].=1
                for (qubit,spin) in enumerate(k)
                    shadow.data[qubit][1,1,:,:]=3*kron(ket[spin+1],ket[spin+1]')-1.0*I(2) 
                end
                push!(shadows,(WlA*shadow)*adjoint(WlA))
            end
        end
    else
        shadows=[]
        ket=[[1 ; 0],[0 ; 1]]
        for r in 1:nu
            WlA=MPO()
            Initialize!("Local_Haar",WlA,lA)

            if lA<A.N
                W=MPO()
                Initialize!("Id",W,A.N)
                for i in 1:lA
                    W.data[i]=Base.copy(WlA.data[i])
                end
            else
                W=copy(WlA)
            end 

            state=W*A
            p_state=save_rhoi(state)[1:lA]
            for m in 1:nm        
                shadow=MPO()
                Initialize!("Zeros",shadow,2,1,lA)
                k=zeros(Int,lA)
                p_extract=rand(lA)
                k[p_extract.>p_state].=1
                for (qubit,spin) in enumerate(k)
                    shadow.data[qubit][1,1,:,:]=3*kron(ket[spin+1],ket[spin+1]')-1.0*I(2) 
                end
                push!(shadows,(WlA*shadow)*adjoint(WlA))
            end
        end
    end
    return shadows
end

function calculate_shadows_MPO(nu::Int,nm::Int,lU::Array,lstr::Array,qubit_set::Array)
    ket=[[1 ; 0],[0 ; 1]]
    chi=1
    d=2
    rho=Dict()
    for r in 1:nu
        counts=lstr[r]
        rho[r]=MPO()
        Initialize!("Zeros",rho[r],d,chi,length(qubit_set))
        for (step_m,m) in enumerate(keys(counts))
            str=string(m-1,base=2,pad=length(qubit_set))
            for (step,j) in enumerate(qubit_set)
                kj = parse(Int32,str[j])+1
                temp=Array{Complex{Float64}}(undef,chi,chi,d,d)  
                temp[chi,chi,:,:]=3*(lU[r][j]')*kron(ket[kj],ket[kj]')*lU[r][j]-1.0*I(2)
                
                if step_m==1
                    rho[r].data[step]=temp*((counts[m]/nm)^(1/length(qubit_set)))
                else
                    rho[r].data[step]=++(rho[r].data[step],temp*((counts[m]/nm)^(1/length(qubit_set))))
                end
            end
        end
    end
    return rho
end


function calc_dist(rhos::Array,nu::Int,nm::Int)

    distrhos=zeros(nu*nm,nu*nm)
    idx=zeros(Int,nu*nm,nu*nm)

    for i in 1:nu*nm
        for j in 1:nu*nm
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
