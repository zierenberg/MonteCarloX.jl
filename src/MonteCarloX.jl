module MonteCarloX

greet() = print("Loading MonteCarloX...")

#include("Basic_functions.jl")

#todo: sort code according to classic, canonical, ..? 

include("Metropolis.jl")
include("ClusterWolff.jl")


include("Gillespie.jl")
include("InhomogeneousPoissonProcess.jl")

include("Reweighting.jl")

end # module
