module MonteCarloX

# Core dependencies
using Random
using StatsBase
using LinearAlgebra
using StaticArrays

# Core abstractions (new API)
include("abstractions.jl")

# Utilities
include("utils.jl")
include("event_handler.jl")
include("rng.jl")

# Measurement framework (new API)
include("measurements.jl")

# Equilibrium algorithms (new API)
include("equilibrium.jl")

# Legacy equilibrium (kept for compatibility, may be deprecated)
include("importance_sampling.jl")
include("reweighting.jl")

# Non-equilibrium algorithms
include("kinetic_monte_carlo.jl")
include("poisson_process.jl")
include("gillespie.jl")

# Cluster algorithms (to be moved to SpinSystems eventually)
include("cluster_wolff.jl")

# Include SpinSystems as a submodule
include("../SpinSystems/src/SpinSystems.jl")

# Export core abstractions
export AbstractSystem,
       AbstractLogWeight,
       AbstractAlgorithm,
       AbstractUpdate,
       AbstractMeasurement

# Export measurement framework
export Measurement,
       Measurements,
       MeasurementSchedule,
       IntervalSchedule,
       PreallocatedSchedule,
       measure!,
       is_complete

# Export equilibrium algorithms (new API)
export AbstractImportanceSampling,
       BoltzmannLogWeight,
       Metropolis,
       accept!,
       acceptance_rate,
       reset_statistics!

# Export legacy algorithms (for backward compatibility)
export accept,
       sweep

# Export non-equilibrium algorithms
export Gillespie,
       KineticMonteCarlo,
       PoissonProcess,
       InhomogeneousPoissonProcess,
       next_event,
       next_time,
       advance!,
       next,
       init

# Export reweighting utilities
export expectation_value_from_timeseries,
       distribution_from_timeseries

# Export helper functions
export log_sum,
       binary_search,
       random_element,
       kldivergence

# Export event handler types
export AbstractEventHandlerRate,
       ListEventRateSimple,
       ListEventRateActiveMask

# Export RNG utilities
export MutableRandomNumbers,
       reset

# Re-export SpinSystems module
export SpinSystems

end # module MonteCarloX
