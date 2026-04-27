"""
    LatticePolymerSystems

Module for lattice-based particle systems (lattice gas, lattice polymers).

This module is designed to be used with MonteCarloX.jl and provides
concrete implementations of AbstractSystem for lattice particle models.
"""
module LatticePolymerSystems

# using for type definitions and utilities
using Random: randperm

using MonteCarloX: AbstractSystem,
                   AbstractImportanceSampling,
                   AbstractMetropolis,
                   accept!

# import for extensions of MonteCarloX functions
import MonteCarloX

export AbstractLatticeParticleSystem,
       # Geometry
       site_index,
       site_coords,
       build_cubic_neighbors,
       lattice_difference,
       # Systems
       LatticeGas,
       LatticePolymer,
       # Initialization
       init!,
       # Observables
       energy,
       num_contacts,
       delta_energy,
       # Updates
       kawasaki_move!,
       polymer_move!,
       # Cluster analysis
       flood_fill_clusters,
       largest_cluster_size,
       second_largest_cluster_size,
       cluster_size_distribution,
       # Polymer observables
       radius_of_gyration_sq,
       center_of_mass,
       end_to_end_distance_sq

include("abstractions.jl")
include("geometry/cubic_lattice.jl")
include("systems/lattice_gas.jl")
include("systems/lattice_polymer.jl")
include("updates/kawasaki.jl")
include("updates/polymer_moves.jl")
include("observables/cluster.jl")
include("observables/polymer_observables.jl")

end # module LatticePolymerSystems
