"""
    SpinSystems

Module for spin system models (Ising, Blume-Capel, etc.).

This module is designed to be used with MonteCarloX.jl and provides
concrete implementations of AbstractSystem for various spin models.
"""
module SpinSystems

# using for type definitions and utilities
using MonteCarloX: AbstractSystem,
                   AbstractImportanceSampling,
                   AbstractMetropolis,
                   AbstractHeatBath,
                   Multicanonical,
                   BinnedObject,
                   accept!,
                   logistic

# import for extensions of MonteCarloX functions
import MonteCarloX

export AbstractSpinSystem,
       Ising,
       IsingGraph,
       IsingMatrix,
       IsingLatticeOptim,
       BlumeCapel,
       logdos_exact_ising2D,
       # Initialization
       init!,
       # Observables
       energy,
       magnetization,
       delta_energy,
       # Updates
       spin_flip!,
       # Utilities
       pick_site,
       local_pair_interactions

include("abstractions.jl")
include("ising.jl")
include("blume_capel.jl")
include("ising2d_exact.jl")

end # module SpinSystems
