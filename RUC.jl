include("mps.jl")
include("mpo.jl")

function RUC_layer(s::String,N::Int)  
  if mod(N,2) != 0
    @warn "Odd number of sites"
  end
  dist = Haar(4)
  if s=="odd"
        return [reshape(rand(dist,d),(2,2,2,2)) for i in 1:(Int(N/2)-1)]
  elseif s=="even"
        return [reshape(rand(dist,d),(2,2,2,2)) for i in 1:Int(N/2)]
  end
end 
