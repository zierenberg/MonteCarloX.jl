"""
    SpinSystems

Module for spin system models (Ising, Blume-Capel, etc.).

This module is designed to be used with MonteCarloX.jl and provides
concrete implementations of AbstractSystem for various spin models.
"""
module SpinSystems

# Import required types from MonteCarloX
using MonteCarloX: AbstractSystem, AbstractImportanceSampling, accept!

export AbstractSpinSystem,
       Ising,
    IsingGraph,
    IsingMatrix,
    IsingLatticeOptim,
       BlumeCapel,
       # Initialization
       init!,
       # Observables
       energy,
       magnetization,
       delta_energy,
       # Updates
       spin_flip!,
       modify!,
       # Utilities
       pick_site,
       local_pair_interactions

include("abstractions.jl")
include("ising.jl")
include("blume_capel.jl")

end # module SpinSystems
