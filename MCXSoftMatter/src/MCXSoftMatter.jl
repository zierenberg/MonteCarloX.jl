module MCXSoftMatter

using StaticArrays
using MonteCarloX: AbstractSystem,
                   AbstractImportanceSampling,
                   AbstractMetropolis,
                   accept!
import MonteCarloX: init!

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

include("environments/abstract.jl")
include("environments/periodic_box.jl")
include("environments/cell_list.jl")
export  AbstractEnvironment,
        PeriodicBox,
        difference,
        constrain,
        is_valid,
        distance_sq,
        distance,
        CellList,
        NoCellList

include("molecules/abstract.jl")
export  AbstractMolecule,
        CacheMonatomic,
        CachePolymer,
        total_energy

include("molecules/monatomic.jl")
export  Monatomic

include("molecules/polymer.jl")
export  Polymer

include("systems/particle_system.jl")
include("systems/particle_gas.jl")
include("systems/bead_spring_polymer.jl")
export  ParticleSystem,
        ParticleGas,
        BeadSpringPolymer,
        num_particles,
        num_polymers,
        polymer_length,
        energy,
        energy_pair,
        energy_bond,
        energy_bending

include("updates/translate.jl")
export  translate!

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

end # module MCXSoftMatter
