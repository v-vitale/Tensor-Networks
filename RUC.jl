include("mps.jl")
include("mpo.jl")

function RUC_layer(s::String,N::Int)  
  function CUE(nh)
      U = (randn(nh,nh)+im*randn(nh,nh))/sqrt(2)
      q,r = qr(U)
      d = r[diagind(r)]
      ph = d./abs.(d)
      U = ph.*q
      return U
  end

  
  if mod(N,2) != 0
    @warn "Odd number of sites"
  end
  if s=="odd"
        return [reshape(CUE(4),(2,2,2,2)) for i in 1:(Int(N/2)-1)]
  elseif s=="even"
        return [reshape(CUE(4),(2,2,2,2)) for i in 1:Int(N/2)]
  end
end 

      
