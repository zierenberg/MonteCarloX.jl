"""
    SoftMatterSystems

Off-lattice soft matter systems in arbitrary spatial dimension D.

# Systems
    ParticleGas(; D=3, N, L, pair_potential, delta=0.1)
    BeadSpringPolymer(; D=3, num_poly, length_poly, L, pair_potential, bond_potential, ...)

# Updates
`translate!`

# Observables
`energy`, `energy_pair`, `energy_bond`, `energy_bending`,
`radius_of_gyration_sq`, `center_of_mass`, `end_to_end_distance_sq`,
`gyration_tensor`, `clusters`
"""
module SoftMatterSystems

using StaticArrays
using MonteCarloX: AbstractSystem,
                   AbstractImportanceSampling,
                   accept!

# ── Potentials ─────────────────────────────────────────────────────────────

abstract type AbstractSoftMatterSystem <: AbstractSystem end

include("potentials/abstractions.jl")
export  AbstractPairPotential,
        AbstractBondPotential,
        AbstractBendingPotential,
        NoPotential,
        NoBondPotential,
        NoBendingPotential,
        cutoff_sq

include("potentials/lennard_jones.jl")
export  LennardJonesPotential

include("potentials/fene.jl")
export  FENEPotential

include("potentials/bending.jl")
export  CosineBendingPotential

# ── Geometry ───────────────────────────────────────────────────────────────

include("geometry/periodic.jl")
export  wrap_coordinate,
        wrap_position,
        minimum_image_displacement,
        minimum_image_sq

# ── Systems ────────────────────────────────────────────────────────────────

include("systems/particle_gas.jl")
export  ParticleGas,
        num_particles,
        init!,
        energy,
        energy_pair

include("systems/bead_spring_polymer.jl")
export  BeadSpringPolymer,
        num_polymers,
        polymer_length,
        energy_bond,
        energy_bending

# ── Updates ────────────────────────────────────────────────────────────────

include("updates/translate.jl")
export  translate!

# ── Observables ────────────────────────────────────────────────────────────

include("observables/cluster.jl")
export  clusters,
        largest_cluster_size,
        second_largest_cluster_size,
        cluster_size_distribution

include("observables/polymer_observables.jl")
export  center_of_mass,
        radius_of_gyration_sq,
        end_to_end_distance_sq,
        gyration_tensor

end # module SoftMatterSystems
