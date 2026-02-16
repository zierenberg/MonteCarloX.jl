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
       Ising_2Dgrid_optim,
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
       pick_site

include("abstractions.jl")
include("ising.jl")
include("blume_capel.jl")

end # module SpinSystems
