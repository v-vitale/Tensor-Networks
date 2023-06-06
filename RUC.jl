include("base_utils.jl")
include("time_evo.jl")

function RUC_layer(N::Int)  
  function CUE(nh)
      U = (randn(nh,nh)+im*randn(nh,nh))/sqrt(2)
      q,r = qr(U)
      d = r[diagind(r)]
      ph = d./abs.(d)
      U = ph.*q
      return U
  end
  #dist=Haar(2)
  return [reshape(CUE(4),(2,2,2,2)) for i in 1:N-1]#[reshape(rand(dist,4),(2,2,2,2)) for i in 1:N-1]
end 


function RUC_evolution!(psi::MPS,
                     sweeps::Int;
                     chimax=2048,
                     tol=1e-15,
                     verbose=false)
                     
    move_orthogonality_center!(psi,1)
    for sweep in 1:Int(sweeps)
  	
        if verbose==true
            println("Sweep: ",sweep)
        end
        gates=RUC_layer(psi.N);
            
        for i in 1:psi.N-1
            psi.data[i],psi.data[i+1] = trotter_swipe_right(psi.data[i],
                                                            psi.data[i+1],
                                                            gates[i],  
                                                            chimax,
                                                            tol)
            psi.b=i+1
        end
    end
    move_orthogonality_center!(psi,1)
end


function RUC_measure!(ψ::MPS,p::Float64; keep_track=false)
    if keep_track==true
    	outcomes=Dict()
    end	
    for site in 1:ψ.N
        if rand() < p
     	    move_orthogonality_center!(ψ,site)
            @tensor temp[:]:=ψ.data[site][1,-1,2]*conj(ψ.data[site][1,-2,2])
            rho=temp/tr(temp)
            pup=real(rho[1,1])
            if rand() < pup
    	        if keep_track==true
		    outcomes[site]=0
     	        end	
                P=MPO("proj",[0],[site],ψ.N)
            else
	        if keep_track==true
		    outcomes[site]=1
     	        end
                P=MPO("proj",[1],[site],ψ.N)
            end
            ψ=P*ψ
            sA = size(ψ.data[site])
	    #U,S,V = svd(reshape(ψ.data[site],(sA[1]*sA[2],sA[3])),full=false,alg=LinearAlgebra.QRIteration())
	    F=SVD{ComplexF64, Float64, Matrix{ComplexF64}}
	    try
		F = svd(reshape(ψ.data[site],(sA[1]*sA[2],sA[3])),full=false,alg=LinearAlgebra.DivideAndConquer())
	    catch e
		F = svd(reshape(ψ.data[site],(sA[1]*sA[2],sA[3])),full=false,alg=LinearAlgebra.QRIteration())
            end
   	    U=F.U
	    S=F.S
	    V=F.V
	    
	    V=V'  
	    S/=norm(S)
	    ψ.data[site] = reshape( U,( sA[1], sA[2], :)) 
	    if site<ψ.N
	        S=diagm(S)
	        @tensor ψ.data[site+1][:] := S[-1,1 ] * V[ 1,2 ] * ψ.data[site+1][2,-2,-3] 
	        ψ.b=site+1
	    end
	    
        end
    end
    move_orthogonality_center!(ψ,1)
    if keep_track==true
        return (ψ,outcomes)
    else
    	return ψ
    end
end
