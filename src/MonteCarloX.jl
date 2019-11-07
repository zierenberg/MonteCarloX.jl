module MonteCarloX

greet() = print("Hello World!")

#todo: sort code according to classic, canonical, ..? 

include("Metropolis.jl")
include("ClusterWolff.jl")


include("Gillespie.jl")
include("InhomogeneousPoissonProcess.jl")

end # module
