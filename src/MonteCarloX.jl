module MonteCarloX

# Core dependencies
using Random
using StatsBase
using LinearAlgebra
using StaticArrays

# Core abstractions (shared by all algorithms)
include("abstractions.jl")

# Utilities
include("utils.jl")
include("event_handler.jl")
include("rng.jl")

# Measurement framework
include("measurements/measurements.jl")

# Algorithms (equilibrium and non-equilibrium)
include("algorithms/metropolis.jl")
include("algorithms/gillespie.jl")
include("algorithms/kinetic_monte_carlo.jl")
include("algorithms/poisson_process.jl")

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
