include("mps.jl")
include("mpo.jl")

function RUC_layer(N::Int)  
  function CUE(nh)
      U = (randn(nh,nh)+im*randn(nh,nh))/sqrt(2)
      q,r = qr(U)
      d = r[diagind(r)]
      ph = d./abs.(d)
      U = ph.*q
      return U
  end
  return [reshape(CUE(4),(2,2,2,2)) for i in 1:N-1]
end 

function RUC_measure!(ψt::MPS,p::Float64)
    for site in 1:ψt.N
        if rand() < p
            move_orthogonality_center!(ψt,site)
            @tensor temp[:]:=ψt.data[site][1,-1,2]*conj(ψt.data[site][1,-2,2])
            rho=temp/tr(temp)
            pup=real(rho[1,1])
            P=MPO()
            if rand() < pup
                Initialize!("proj",P,[0],[site],ψt.N)
            else
                Initialize!("proj",P,[1],[site],ψt.N)
            end
            ψt=P*ψt
            right_normalize!(ψt)
        end
    end
end