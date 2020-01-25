module MonteCarloX
#Maybe embedd this into StatisticalPhysics.jl the including SpinSystems.jl PolymerSystems.jl etc ;)
#

greet() = print("Loading MonteCarloX...")

include("Utils.jl")
include("Histograms.jl")

#todo: sort code according to classic, canonical, ..? 
#todo: implement function integration as test and prime example!!!

#Importance sampling
include("Metropolis.jl")

#Non-equilibrium 
include("KineticMonteCarlo.jl")
include("InhomogeneousPoissonProcess.jl")
include("Gillespie.jl")

include("Reweighting.jl")

#TODO: move to external SpinSystems.jl module
include("ClusterWolff.jl")

#TODO: external modules obviously in external modules
#("ContactProcess.jl")
#("CellularAutomata.jl)
#("Networks.jl)
#("NeuralNetworks.jl)


end # module
