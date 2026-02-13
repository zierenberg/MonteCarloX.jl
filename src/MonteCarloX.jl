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

# Algorithms
include("algorithms/event_handler.jl")  # Event handling for kinetic Monte Carlo
include("algorithms/importance_sampling.jl")  # Metropolis and importance sampling
include("algorithms/kinetic_monte_carlo.jl")  # Gillespie (simplest KMC) and general KMC
include("algorithms/poisson_process.jl")  # Poisson processes
# Future algorithms (placeholders):
# include("algorithms/multicanonical.jl")
# include("algorithms/parallel_tempering.jl")
# include("algorithms/population_annealing.jl")

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

# Export importance sampling algorithms
export AbstractImportanceSampling,
       BoltzmannLogWeight,
       Metropolis,
       accept!,
       acceptance_rate,
       reset_statistics!

# Export kinetic Monte Carlo algorithms
export Gillespie,
       KineticMonteCarlo,
       SimulationKineticMonteCarlo,
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
