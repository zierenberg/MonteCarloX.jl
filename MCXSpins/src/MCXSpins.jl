"""
    MCXSpins

Spin systems for use with MonteCarloX.jl.

A system is a spin type (local degree of freedom) plus a tuple of interaction terms:

    SpinSystem(Spin(1), (PairInteraction(J, partners), CrystalField(Δ)))

Spin types: `Spin(S)` (discrete, σ-convention), `XYSpin()`, `HeisenbergSpin()`.
Interactions: `PairInteraction` (uniform J, local neighborhood), `PairInteractionMatrix`
(sparse J_ij), `ExternalField`, `CrystalField`, `VisionConeInteraction` (nonreciprocal).
The classic models are one-liners; the topology argument is a dims vector (periodic
lattice), a `SimpleGraph`, or a sparse J_ij matrix:

    IsingSystem([L,L]; J=1, h=0)            BlumeCapelSystem([L,L]; J=1, D=0.5, h=0)
    IsingSystem(graph; J, h)                XYSystem([L,L]; J, h)
    IsingSystem(J_sparse; h)                HeisenbergSystem([L,L]; J)
    VisionConeIsingSystem([L,L]; κ)         VisionConeBlumeCapelSystem([L,L]; κ, D)
    HopfieldSystem(patterns)                EdwardsAndersonSystem(dims; rng, dist)

# Interface (required for spin_flip!)
    propose_state(rng, sys, i, params...) -> new spin state
    delta_sys(sys, i, s_new)            -> delta payload (optional; defaults to nothing)
    delta_energy(sys, i, s_new, δs)     -> energy change
    modify!(sys, i, s_new, δs)          -> apply change

# Updates
`spin_flip!` (Metropolis/Glauber/heat bath), `cluster_update!` (`Wolff`, `SwendsenWang`),
`spin_exchange!` (Kawasaki); the composed systems implement the core local-states interface,
so `SiteEvents(sys, NFoldRates(β=β))` under `Gillespie` gives the n-fold way out of the box.

# Observables
`energy`, `magnetization`, `hamiltonian_energy`, `structure_factor`, `correlation_length`
"""
module MCXSpins

using Random
using Graphs
using SparseArrays: SparseMatrixCSC, sparse
using LinearAlgebra: issymmetric, dot
using StaticArrays: SVector
using MonteCarloX: AbstractSystem,
                   MetropolisHastingsAlgorithm,
                   HeatBathAlgorithm,
                   BoltzmannEnsemble,
                   BinnedObject,
                   accept!,
                   resample!,
                   ensemble,
                   logweight,
                   linear_logweight,
                   assert_linear_ensemble,
                   logistic

import MonteCarloX
import MonteCarloX: modify!, init!, delta_energy, partners, local_states, nsites

# -- Abstractions ---------------------------------------------------------------

include("abstractions.jl")
export  AbstractSpinSystem,
        pick_site,
        propose_state

# -- Systems: spin types, geometries, SpinSystem, models -------------------------

include("systems/spin_types.jl")
export  SpinType,
        Spin,
        XYSpin,
        HeisenbergSpin,
        states

include("systems/geometries.jl")
export  lattice_random_J

# -- Interactions -----------------------------------------------------------------

include("interactions/interactions.jl")      # protocol: delta/commit!/energy + tuple walkers
include("interactions/pairwise.jl")          # PairInteraction (uniform J, local neighborhood)
include("interactions/matrix.jl")            # PairInteractionMatrix (sparse J_ij)
include("interactions/field.jl")             # ExternalField (uniform / site-dependent h)
include("interactions/crystal_field.jl")     # CrystalField (+Δ Σσ²)
include("interactions/vision_cone.jl")       # VisionConeInteraction (nonreciprocal)
export  AbstractInteraction,
        PairInteraction,
        PairInteractionMatrix,
        ExternalField,
        CrystalField,
        VisionConeInteraction,
        SymmetricCoupling,
        AsymmetricCoupling,
        symmetry,
        partners,
        hamiltonian_energy

# -- The composed system and the model one-liners ---------------------------------

include("systems/spin_system.jl")
export  SpinSystem,
        is_hamiltonian,
        set_spins!,
        geometry,
        energy,
        magnetization,
        delta_energy

include("systems/ising.jl")
include("systems/blume_capel.jl")
include("systems/xy.jl")
include("systems/heisenberg.jl")
export  IsingSystem,
        VisionConeIsingSystem,
        HopfieldSystem,
        EdwardsAndersonSystem,
        BlumeCapelSystem,
        VisionConeBlumeCapelSystem,
        XYSystem,
        HeisenbergSystem

# -- Observables -------------------------------------------------------------------

include("observables/structure_factor.jl")
include("observables/correlation.jl")
export  structure_factor,
        correlation_length

# -- Updates -------------------------------------------------------------------------

include("updates/spin_flip.jl")
export  spin_flip!

include("updates/cluster.jl")
export  cluster_update!,
        Wolff,
        SwendsenWang,
        mean_cluster_size,
        mean_cluster_count

include("updates/spin_exchange.jl")
export  spin_exchange!

# -- Exact results ---------------------------------------------------------------------

include("exact_solutions/ising2d_exact.jl")
export  logdos_exact_ising2D

end # module MCXSpins
