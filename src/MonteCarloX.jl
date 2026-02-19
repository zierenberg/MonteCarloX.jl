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
include("tools/reweighting.jl")

# Measurement framework
include("measurements/measurements.jl")

# Log weights (canonical ensemble)
include("weights/canonical.jl")

# Event handlers (non-equilibrium)
include("event_handler/abstractions.jl")
include("event_handler/list_event_rate_simple.jl")
include("event_handler/list_event_rate_active_mask.jl")
include("event_handler/event_queue.jl")

# Algorithms (equilibrium)
include("algorithms/importance_sampling.jl")  # Core importance sampling functions (accept!, etc.)
include("algorithms/metropolis.jl")  # Metropolis importance sampling
include("algorithms/multicanonical.jl")

# Algorithms (non-equilibrium)
include("algorithms/kinetic_monte_carlo.jl")
include("algorithms/gillespie.jl")

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
    Multicanonical,
    WangLandau,
       accept!,
       acceptance_rate,
    reset_statistics!,
    update_weights!,
    update_weights

# Export kinetic Monte Carlo algorithms
export AbstractKineticMonteCarlo,
         Gillespie,
         next,
         step!,
         next_time,
         next_event,
         advance!

# Export helper functions
export log_sum,
       binary_search,
       kldivergence

# Export tools
export expectation_value_from_timeseries,
       distribution_from_timeseries,
       expectation_value_from_histogram,
       log_normalization

# Export event handler types
export AbstractEventHandlerRate,
    AbstractEventHandlerTime,
       ListEventRateSimple,
    ListEventRateActiveMask,
    EventQueue,
    get_time,
    set_time!,
    add!

# Export RNG utilities
export MutableRandomNumbers,
       reset

end # module MonteCarloX
