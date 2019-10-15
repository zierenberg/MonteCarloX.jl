module MonteCarloX

greet() = print("Hello World!")

#todo: sort code according to classic, canonical, ..? 

include("Metropolis.jl")
include("Gillespie.jl")
include("ClusterWolff.jl")


end # module
