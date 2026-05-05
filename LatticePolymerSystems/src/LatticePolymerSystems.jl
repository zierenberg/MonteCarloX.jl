"""
    LatticePolymerSystems

Lattice polymer systems in arbitrary spatial dimension D.

# Constructor
    LatticePolymer(; dims, polys, ...)                  # explicit polymer lengths
    LatticePolymer(; dims, num_poly, length_poly, ...)  # uniform homopolymers

# Updates
`slither_move!`, `translate_move!`, `pivot_move!`, `double_bridge_move!`

# Observables
`energy`, `radius_of_gyration_sq`, `center_of_mass`,
`end_to_end_distance_sq`, `gyration_tensor`, `clusters`
"""
module LatticePolymerSystems

using Graphs
using StaticArrays
using MonteCarloX: AbstractSystem,
                   AbstractImportanceSampling,
                   accept!

# ── Systems ─────────────────────────────────────────────────────────────────

include("systems/abstract_lattice_system.jl")
export  site_to_coords,
        coords_to_site,
        apply_pbc,
        lattice_difference,
        lattice_distance_sq

include("systems/lattice_polymer.jl")
export  LatticePolymer,
        num_polymers,
        polymer_length,
        init!,
        energy,
        site_contacts,
        site_energy

# ── Updates ─────────────────────────────────────────────────────────────────

include("updates/slither.jl")
export  slither_move!

include("updates/translate.jl")
export  translate_move!

include("updates/pivot.jl")
export  pivot_move!

include("updates/double_bridge.jl")
export  double_bridge_move!

# ── Observables ─────────────────────────────────────────────────────────────

include("observables/cluster.jl")
export  clusters,
        largest_cluster_size,
        second_largest_cluster_size,
        cluster_size_distribution

include("observables/polymer_observables.jl")
export  radius_of_gyration_sq,
        center_of_mass,
        end_to_end_distance_sq,
        gyration_tensor

end # module LatticePolymerSystems
