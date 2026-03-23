module MonteCarloX

# Core dependencies
using Random
using StatsBase
using LinearAlgebra
using StaticArrays

# Core abstractions (shared by all algorithms)
include("abstract_system.jl")
export  AbstractSystem

include("algorithms/abstract_algorithm.jl")
export  AbstractAlgorithm,
        AbstractImportanceSampling,
        AbstractHeatBath,
        AbstractKineticMonteCarlo

include("ensembles/abstract_ensemble.jl")
export  AbstractEnsemble,
        update!

# Utilities
include("utils.jl")
export  log_sum,
        binary_search,
        kldivergence

include("rng.jl")
export  MutableRandomNumbers,
        reset!

# include("tools/reweighting.jl")

# Measurement framework
include("measurements/measurements.jl")
export  Measurement,
        Measurements,
        MeasurementSchedule,
        IntervalSchedule,
        PreallocatedSchedule,
        times,
        data,
        measure!,
        reset!,
        is_complete

# Binned utilities
include("structures/binned_object.jl")
export  BinnedObject,
        DiscreteBinning,
        ContinuousBinning,
        get_centers,
        get_edges,
        get_values,
        set!

# Ensembles
include("ensembles/function.jl")
export  FunctionEnsemble

include("ensembles/boltzmann.jl")
export  BoltzmannEnsemble

include("ensembles/multicanonical.jl")
export  MulticanonicalEnsemble

include("ensembles/wang_landau.jl")
export  WangLandauEnsemble

# Event handlers (non-equilibrium)
include("event_handler/abstractions.jl")
export  AbstractEventHandlerRate,
        AbstractEventHandlerTime

include("event_handler/list_event_rate_simple.jl")
export  ListEventRateSimple

include("event_handler/list_event_rate_active_mask.jl")
export  ListEventRateActiveMask

include("event_handler/event_queue.jl")
export  EventQueue,
        get_time,
        set_time!,
        add!

# Algorithms (equilibrium)
include("algorithms/importance_sampling.jl") # Core importance sampling functions (accept!, etc.)
export  AbstractImportanceSampling,
        AbstractHeatBath,
        ImportanceSampling,
        ensemble,
        logweight,
        accept!,
        acceptance_rate,
        reset!

include("algorithms/metropolis.jl") # Metropolis importance sampling
export  AbstractMetropolis,
        Metropolis,
        Glauber

include("algorithms/heat_bath.jl")
export  HeatBath

include("algorithms/multicanonical.jl")
export  Multicanonical

include("algorithms/parallel_multicanonical.jl")
export  ParallelMulticanonical,
        is_root,
        merge_histograms!,
        distribute_logweight!

include("algorithms/wang_landau.jl")
export  WangLandau

# Algorithms (non-equilibrium)
include("algorithms/kinetic_monte_carlo.jl")
export  AbstractKineticMonteCarlo,
        next,
        step!,
        next_time,
        next_event,
        event_source,
        modify!,
        advance!

include("algorithms/gillespie.jl")
export  Gillespie

end # module MonteCarloX
