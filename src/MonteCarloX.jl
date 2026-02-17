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
include("rng.jl")

# Measurement framework
include("measurements/measurements.jl")

# Log weights (canonical ensemble)
include("weights/canonical.jl")

# Algorithms (equilibrium)
include("algorithms/importance_sampling.jl")  # Core importance sampling functions (accept!, etc.)
include("algorithms/metropolis.jl")  # Metropolis importance sampling

# Algorithms (non-equilibrium)
include("algorithms/event_handler.jl")  # Event handling for kinetic Monte Carlo

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
BoltzmannLogWeight,
       
# Export importance sampling algorithms
export AbstractImportanceSampling,
       BoltzmannLogWeight,
       Metropolis,
       accept!,
       acceptance_rate,
       reset_statistics!

# Export helper functions
export log_sum,
       binary_search,
       kldivergence

# Export event handler types
export AbstractEventHandlerRate,
       ListEventRateSimple,
       ListEventRateActiveMask

# Export RNG utilities
export MutableRandomNumbers,
       reset

end # module MonteCarloX
