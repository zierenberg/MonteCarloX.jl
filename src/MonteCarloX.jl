module MonteCarloX

greet() = print("Loading MonteCarloX...")

#include("Utils.jl")
include("Histograms.jl")

#todo: sort code according to classic, canonical, ..? 
#todo: implement function integration as test and prime example!!!

include("Metropolis.jl")
include("ClusterWolff.jl")

#Todo: implemeent a sweep by passing array of possible updates that are then performed with probability
#include("Sweep.jl")

include("Gillespie.jl")
include("InhomogeneousPoissonProcess.jl")

include("Reweighting.jl")
include("EventHandler.jl")

function sweep(list_updates, list_probabilities, rng; number_updates=1)
  #TODO: find out if this is a speed problem
  @assert length(list_updates) == length(list_probabilities)
  @assert sum(list_probabilities) == 1

  for i in 1:number_updates
    id = EventHandler.next_event_probability(list_probabilities,rng)
    #calls metropolis itself
    list_updates[id]()
  end
end

end # module
