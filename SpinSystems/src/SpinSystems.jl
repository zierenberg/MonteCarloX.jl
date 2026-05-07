"""
    SpinSystems

Spin system models (Ising, Blume-Capel) for use with MonteCarloX.jl.

Three backends per model:
- `Lattice`: D-dimensional periodic hypercubic, NTuple neighbors (fastest)
- `Graph`: arbitrary graph, Vector neighbors, uniform J
- `Matrix`: sparse J_{ij}, arbitrary topology

# Constructors
    Ising([L,L]; J=1, h=0)                  # -> IsingLattice (fast path)
    Ising([L,L]; J=1, periodic=false)       # -> IsingGraph
    Ising(graph, J; h=0)                    # -> IsingGraph
    Ising(J_sparse; h=0)                    # -> IsingMatrix
    BlumeCapel([L,L]; J=1, D=0.5, h=0)      # -> BlumeCapelLattice

# Interface (required for spin_flip!)
    propose_state(rng, sys, i) -> new spin state
    delta_sys(sys, i, s_new)   -> local move payload (optional; defaults to s_new)
    delta_energy(sys, i, dsys) -> energy change
    modify!(sys, i, dsys)      -> apply change

# Updates
`spin_flip!`

# Observables
`energy`, `magnetization`, `delta_energy`, `local_pair_interactions`
"""
module SpinSystems

using Random
using Graphs
using SparseArrays: SparseMatrixCSC, sparse
using MonteCarloX: AbstractSystem,
                   AbstractImportanceSampling,
                   AbstractMetropolis,
                   AbstractHeatBath,
                   BinnedObject,
                   accept!,
                   logistic

import MonteCarloX
import MonteCarloX: modify!

# -- Abstractions ------------------------------------------------------------

include("abstractions.jl")
export  AbstractSpinSystem,
        NoField,
        pick_site,
        local_pair_interactions,
        propose_state

# -- Systems ------------------------------------------------------------------

include("systems/ising.jl")
export  Ising,
        IsingLattice,
        IsingGraph,
        IsingMatrix,
        init!,
        energy,
        magnetization,
        delta_energy

include("systems/blume_capel.jl")
export  BlumeCapel,
        BlumeCapelLattice,
        BlumeCapelGraph,
        BlumeCapelMatrix

# -- Updates ------------------------------------------------------------------

include("updates/spin_flip.jl")
export  spin_flip!

# -- Exact results ------------------------------------------------------------

include("exact_solutions/ising2d_exact.jl")
export  logdos_exact_ising2D,
        distribution_exact_ising2D

end # module SpinSystems
