#   Author: V. Vitale
#   Feb 2022
  
include("mps.jl")
include("mpo.jl")


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


function calc_dist(rhos::Dict)
    distrhos=zeros(length(rhos),length(rhos))
    idx=zeros(Int,length(rhos),length(rhos))
    
    for i in 1:length(rhos)
        for j in i+1:length(rhos)
            println(typeof(rhos[i]-rhos[j])*(rhos[i]-rhos[j]))
            distrhos[i,j]=real(tr((rhos[i]-rhos[j])*(rhos[i]-rhos[j]))^(1/2))
        end
        distrhos=distrhos+distrhos'
        idx[i,:]=sortperm(distrhos[i,:])
        distrhos[i,:]=distrhos[i,idx[i,:]]
    end
    return distrhos
end