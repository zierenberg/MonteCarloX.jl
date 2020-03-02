module MonteCarloX
#dependencies
using Random
#do we need this? use sampling instead of own random_element... if we also can use Distributions than it may be worth while
using StatsBase
using LinearAlgebra

#exports that are relevant to run simulations in MonteCarloX
export Random


#TODO: look at Distributions.jl -> one big namespace ... We should consider making less namespaces here, i.e., less modules
#TODO: look at Distances.jl -> concept of singletons

include("Utils.jl")
#include("Histograms.jl")
include("EventHandler.jl")

#Importance sampling
include("Metropolis.jl")
include("Reweighting.jl")

#Non-equilibrium 
include("KineticMonteCarlo.jl")
include("InhomogeneousPoissonProcess.jl")
include("Gillespie.jl")


#TODO: move to external SpinSystems.jl module
include("ClusterWolff.jl")

#reexport Random, StatsBase
export Random, 
       StatsBase

export  add!
#export what I want API to be 

# Check georges 2nd workshop notebook on github

#    i::Int = StatsBase.binindex(d.h, x) -> remember this for custom things similar to EmpiricalDistributions.jl (not well documented though)

 
end # module

#Maybe embedd this into StatisticalPhysics.jl the including SpinSystems.jl PolymerSystems.jl etc ;)
#TODO: external modules obviously in external modules
#("DirectedPercolation.jl") [inlcuding ContactProcess, CellularAutomatoa, etc but not as modules but as models]
#("NeuralNetworks.jl) -> maybe NeuralSystems.jl?
#("ComplexNetworks.jl)
