# 2020/04/23

- Metropolis and such updates in 2 flavors, one with flag that update is already performed, then do undo function :)
- last question is about heatbath like updates ...

# Next things to do
- sweep interface (think about how to add list of updates with probabilities - similar to non-equilibrium case, i.e., make a new function for this in a more general module?
- how to call update with different number of arguments?

#- StatisticalMechanics.jl StatisticalPhysics.jl? ->SpinSystems.jl (for all the Ising stuff!!!
TODO: 
- namespaces
- modular structure on which level 


# EventHandler 
TOOD:
- sort
- test
- documentation

# API notes
- reduce modules!!!

# equilibrium part
READY for advanced updates: pass energy new, energy old 
* metropolis calculates dE on its own,
* muca gets weight function, 
* parallel tempering needs also temperatures .... but only in additional swap move..)
* population annealing also uses metropolis but additional resampling step

# what I learned
- create explicit arrays with test = [1 2 3; 4 5 6; 7 8 9]


## add dependencies
-> cd(".julia/dev/MonteCarloX")
-> ] # Pkg
-> activate .
-> add Random
