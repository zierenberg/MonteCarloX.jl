"""
    SoftMatterSystems

Module for off-lattice soft matter systems (particle gas, bead-spring polymers).

This module is designed to be used with MonteCarloX.jl and provides
concrete implementations of AbstractSystem with composable potentials.

Systems are parameterized by potential types, allowing different interaction
models to be combined at construction time:

```julia
# LJ gas
lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
gas = ParticleGas(100; L=10.0, pair_potential=lj)

# Flexible bead-spring polymer
poly = BeadSpringPolymer(4, 20; L=20.0,
    pair_potential = LennardJonesPotential(epsilon=1.0, sigma=1.0),
    bond_potential = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5))

# Semiflexible polymer (add bending stiffness)
poly = BeadSpringPolymer(4, 20; L=20.0,
    pair_potential = LennardJonesPotential(epsilon=1.0, sigma=1.0),
    bond_potential = FENEPotential(spring_constant=30.0, l0=0.0, l_max=1.5),
    bending_potential = CosineBendingPotential(5.0))
```
"""
module SoftMatterSystems

using StaticArrays: SVector
using Random

using MonteCarloX: AbstractSystem,
                   AbstractImportanceSampling,
                   AbstractMetropolis,
                   accept!

import MonteCarloX

export AbstractSoftMatterSystem,
       # Potentials
       AbstractPairPotential,
       AbstractBondPotential,
       AbstractBendingPotential,
       NoPotential,
       NoBondPotential,
       NoBendingPotential,
       LennardJonesPotential,
       FENEPotential,
       CosineBendingPotential,
       cutoff_sq,
       # Geometry
       wrap_coordinate,
       wrap_position,
       minimum_image_sq,
       minimum_image_displacement,
       # Systems
       ParticleGas,
       BeadSpringPolymer,
       # Initialization
       init!,
       # Observables
       energy,
       energy_pair,
       energy_bond,
       energy_bending,
       # Updates
       particle_move!,
       monomer_move!,
       # Cluster analysis
       geometric_clusters,
       largest_cluster_size,
       second_largest_cluster_size,
       cluster_size_distribution

include("abstractions.jl")
include("geometry/periodic.jl")
include("potentials/lennard_jones.jl")
include("potentials/fene.jl")
include("potentials/bending.jl")
include("systems/particle_gas.jl")
include("systems/bead_spring_polymer.jl")
include("updates/particle_shift.jl")
include("updates/monomer_displacement.jl")
include("observables/cluster.jl")

end # module SoftMatterSystems
