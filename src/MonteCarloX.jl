module MonteCarloX
#Maybe embedd this into StatisticalPhysics.jl the including SpinSystems.jl PolymerSystems.jl etc ;)
#

greet() = print("Loading MonteCarloX...")

include("Utils.jl")

#todo: sort code according to classic, canonical, ..? 
#todo: implement function integration as test and prime example!!!

#TODO: Metropolis.sweep (or MonteCarloX.sweep? I guess this is well suited there
include("Metropolis.jl")
#TODO: Move this to SpinSystems.Updates
include("ClusterWolff.jl")

#TODO: Gillespie.advance(T)
include("Gillespie.jl")
include("KineticMonteCarlo.jl")
include("InhomogeneousPoissonProcess.jl")

include("Histograms.jl")
include("Reweighting.jl")



#TODO: how to sort these?
#obviously in external modules
#("ContactProcess.jl")
#("CellularAutomata.jl)

#TODO: NeuralNetworks.jl (including Networks.jl?)

end # module
