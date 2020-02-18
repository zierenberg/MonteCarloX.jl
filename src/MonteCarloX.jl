module MonteCarloX
#Maybe embedd this into StatisticalPhysics.jl the including SpinSystems.jl PolymerSystems.jl etc ;)
#

greet() = print("Loading MonteCarloX...")

#TODO: look at Distributions.jl -> one big namespace ... We should consider making less namespaces here, i.e., less modules

include("Utils.jl")
include("Histograms.jl")
include("EventHandler.jl")

#todo: sort code according to classic, canonical, ..? 
#todo: implement function integration as test and prime example!!!

#Importance sampling
include("Metropolis.jl")
include("Reweighting.jl")

#Non-equilibrium 
include("KineticMonteCarlo.jl")
include("InhomogeneousPoissonProcess.jl")
include("Gillespie.jl")


#TODO: move to external SpinSystems.jl module
include("ClusterWolff.jl")

#TODO: external modules obviously in external modules
#("ContactProcess.jl")
#("CellularAutomata.jl)
#("Networks.jl)
#("NeuralNetworks.jl)

#TODO: exports
  
 
end # module
