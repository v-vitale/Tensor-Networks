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
  return [reshape(CUE(4),(2,2,2,2)) for i in 1:Int(N/2)]
end 

      
