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
