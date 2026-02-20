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
# include("tools/reweighting.jl")

# Measurement framework
include("measurements/measurements.jl")

# Log weights (canonical ensemble)
include("weights/canonical.jl")
include("weights/mutable.jl")

# Event handlers (non-equilibrium)
include("event_handler/abstractions.jl")
include("event_handler/list_event_rate_simple.jl")
include("event_handler/list_event_rate_active_mask.jl")
include("event_handler/event_queue.jl")

# Algorithms (equilibrium)
include("algorithms/importance_sampling.jl")  # Core importance sampling functions (accept!, etc.)
include("algorithms/metropolis.jl")  # Metropolis importance sampling
include("algorithms/heat_bath.jl")
include("algorithms/multicanonical.jl")
include("algorithms/parallel_sampling.jl")
include("algorithms/replica_exchange.jl")
include("algorithms/parallel_tempering.jl")
include("algorithms/parallel_multicanonical.jl")
include("algorithms/wang_landau.jl")

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
  reset!,
       is_complete

# Export importance sampling algorithms
export AbstractImportanceSampling,
         AbstractGeneralizedEnsemble,
         AbstractMetropolis,
         AbstractHeatBath,
       BoltzmannLogWeight,
         TabulatedLogWeight,
       Metropolis,
         Glauber,
         HeatBath,
         Multicanonical,
         ParallelSampling,
         ReplicaExchange,
         ParallelTempering,
         ParallelMulticanonical,
         WangLandau,
       accept!,
       acceptance_rate,
         reset_statistics!,
         reset_exchange_statistics!,
         log_acceptance_ratio,
         sweep_exchanges!,
         attempt_exchange!,
         exchange_rate,
         update_weight!,
         update_f!,
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
