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

# ── Infrastructure ──────────────────────────────────────────────────────────

include("infrastructure/binned_object.jl")
export  BinnedObject,
        DiscreteBinning,
        ContinuousBinning,
        get_centers,
        get_edges,
        get_values,
        set!

include("infrastructure/utils.jl")
export  log_sum,
        binary_search,
        kldivergence,
        distribution_from_logdos

include("infrastructure/rng.jl")
export  MutableRandomNumbers,
        reset!

include("infrastructure/parallel_backends.jl")
export  ThreadsBackend,
        MPIBackend,
        init,
        finalize!,
        rank,
        size,
        is_root

include("infrastructure/checkpointing.jl")
export  CheckpointSession,
        init_checkpoint,
        checkpoint!,
        restore,
        relink!

# ── Measurements ────────────────────────────────────────────────────────────

include("measurements/measurements.jl")
include("measurements/autocorrelations.jl")
export  Measurement,
        Measurements,
        MeasurementSchedule,
        IntervalSchedule,
        PreallocatedSchedule,
        integrated_autocorrelation_time,
        integrated_autocorrelation_times, #TODO: this needs to be solved in PT
        tau_int,
        times,
        data,
        measure!,
        reset!,
        is_complete

# ── Ensembles ───────────────────────────────────────────────────────────────

include("ensembles/function.jl")
export  FunctionEnsemble

include("ensembles/boltzmann.jl")
export  BoltzmannEnsemble

include("ensembles/multicanonical.jl")
export  MulticanonicalEnsemble

include("ensembles/wang_landau.jl")
export  WangLandauEnsemble

# ── Event handlers (non-equilibrium) ────────────────────────────────────────

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

# ── Algorithms (equilibrium) ────────────────────────────────────────────────

include("algorithms/importance_sampling.jl")
export  AbstractImportanceSampling,
        AbstractHeatBath,
        ImportanceSampling,
        ensemble,
        logweight,
        accept!,
        acceptance_rate,
        reset!

include("algorithms/metropolis.jl")
export  AbstractMetropolis,
        Metropolis,
        Glauber

include("algorithms/heat_bath.jl")
export  HeatBath

include("algorithms/multicanonical.jl")
export  Multicanonical

include("algorithms/parallel_chains.jl")
export  ParallelChains,
        algorithm,
        run!,
        merge!

include("algorithms/parallel_multicanonical.jl")
export  ParallelMulticanonical,
        merge_histograms!,
        distribute_logweight!

include("algorithms/replica_exchange.jl")
export  ReplicaExchange

include("algorithms/parallel_tempering.jl")
export  ParallelTempering,
        index,
        optimize_exchange_interval!,
        acceptance_rates,
        exchange_log_ratio,
        attempt_exchange_pair!,
        set_betas

include("algorithms/wang_landau.jl")
export  WangLandau

# ── Algorithms (non-equilibrium) ────────────────────────────────────────────

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
