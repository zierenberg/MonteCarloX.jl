# Next things to do
- sweep interface (think about how to add list of updates with probabilities - similar to non-equilibrium case, i.e., make a new function for this in a more general module?
- how to call update with different number of arguments?
- EventHandler module -> push, pop, get next event, get event probability

- StatisticalMechanics.jl StatisticalPhysics.jl? ->SpinSystems.jl (for all the Ising stuff)


# API notes

# equilibrium part
READY for advanced updates: pass energy new, energy old 
* metropolis calculates dE on its own,
* muca gets weight function, 
* parallel tempering needs also temperatures .... but only in additional swap move..)
* population annealing also uses metropolis but additional resampling step
* cluster update? -> try to solve this today

# what I learned

## add dependencies
-> cd(".julia/dev/MonteCarloX")
-> ] # Pkg
-> activate .
-> add Random
