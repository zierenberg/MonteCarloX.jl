"""
    MCXLatticeMatter

Lattice polymer systems in arbitrary spatial dimension D.

# Constructor
    LatticePolymer(; dims, polys, ...)                  # explicit polymer lengths
    LatticePolymer(; dims, num_poly, length_poly, ...)  # uniform homopolymers

# Updates
`translate!`, `slither!`, `pivot!`, `double_bridge!`

# Observables
`energy`, `radius_of_gyration_sq`, `center_of_mass`,
`end_to_end_distance_sq`, `gyration_tensor`, `clusters`
"""
module MCXLatticeMatter

using StaticArrays
using MonteCarloX: AbstractSystem,
                   AbstractImportanceSampling,
                   accept!
import MonteCarloX: init!

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
        energy,
        site_contacts,
        site_energy

# ── Updates ─────────────────────────────────────────────────────────────────

include("updates/slither.jl")
export  slither!

include("updates/translate.jl")
export  translate!

include("updates/pivot.jl")
export  pivot!

include("updates/double_bridge.jl")
export  double_bridge!

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

end # module MCXLatticeMatter
