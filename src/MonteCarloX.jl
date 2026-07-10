module MonteCarloX

# Core dependencies
using Random
using StatsBase
using LinearAlgebra
using StaticArrays

# Core abstractions (shared by all algorithms)
include("abstract_system.jl")
export  AbstractSystem,
        init!

include("algorithms/abstract_algorithm.jl")
export  AbstractAlgorithm,
        AbstractMarkovChainMonteCarlo,
        AbstractKineticMonteCarlo,
        steps

include("ensembles/abstract_ensemble.jl")
export  AbstractEnsemble,
        linear_logweight,
        set_logweight!,
        update_logweight!,
        update!

# ── Infrastructure ──────────────────────────────────────────────────────────

include("infrastructure/binned_object.jl")
export  BinnedObject,
        DiscreteBinning,
        ContinuousBinning,
        ArbitraryContinuousBinning,
        AbstractBoundary,
        ErrorBoundary,
        NegInfBoundary,
        ZeroBoundary,
        get_centers,
        get_edges,
        get_values,
        set!

include("infrastructure/utils.jl")
export  log_sum,
        binary_search,
        kldivergence

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

include("infrastructure/parallel_chains.jl")
export  ParallelChains,
        algorithm,
        on_root,
        with_parallel,
        merge!

include("infrastructure/checkpointing.jl")
export  CheckpointSession,
        init_checkpoint,
        checkpoint!,
        restore_checkpoint

include("infrastructure/monitoring.jl")
export  Roundtrips,
        flatness,
        extend!,
        smooth!

# ── Measurements ────────────────────────────────────────────────────────────

include("measurements/measurements.jl")
include("measurements/autocorrelations.jl")
include("measurements/diagnostics.jl")
export  Measurement,
        Measurements,
        MeasurementSchedule,
        IntervalSchedule,
        PreallocatedSchedule,
        integrated_autocorrelation_time,
        integrated_autocorrelation_times, #TODO: this needs to be solved in PT
        tau_int,
        rhat,
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

include("ensembles/constant.jl")
export  ConstantEnsemble

include("ensembles/multicanonical.jl")
export  MulticanonicalEnsemble,
        visited_range

include("ensembles/wang_landau.jl")
export  WangLandauEnsemble

# ── Reweighting ─────────────────────────────────────────────────────────────

include("infrastructure/reweighting.jl")
export  ImportanceWeights,
        reweight,
        log_normalization,
        ess

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

include("algorithms/balance.jl")
export  BalanceFunction,
        MetropolisBalance,
        GlauberBalance,
        acceptance_probability,
        transition_rate

include("algorithms/metropolis_hastings.jl")
export  AbstractMarkovChainMonteCarlo,
        MetropolisHastingsAlgorithm,
        MetropolisAlgorithm,
        GlauberAlgorithm,
        ensemble,
        balance,
        logweight,
        accept!,
        acceptance_rate,
        reset!

include("infrastructure/step_size.jl")
export  AdaptiveStep,
        step_size,
        adapt!

include("algorithms/rejection_sampling.jl")
export  RejectionSampling

include("algorithms/heat_bath.jl")
export  HeatBathAlgorithm

include("algorithms/flat_histogram.jl")
export  MulticanonicalAlgorithm,
        WangLandauAlgorithm,
        ParallelMulticanonical,
        merge_histograms!,
        distribute_logweight!

include("algorithms/replica_exchange.jl")
export  ReplicaExchange,
        ParallelTempering,
        index,
        optimize_exchange_interval!,
        acceptance_rates,
        exchange_log_ratio,
        attempt_exchange_pair!,
        set_betas

# ── Deprecated algorithm names (old API resolves with a warning) ─────────────
Base.@deprecate_binding AbstractImportanceSampling AbstractMarkovChainMonteCarlo
Base.@deprecate_binding ImportanceSampling MetropolisHastingsAlgorithm
Base.@deprecate_binding AbstractMetropolis AbstractMarkovChainMonteCarlo
Base.@deprecate_binding AbstractHeatBath AbstractMarkovChainMonteCarlo
Base.@deprecate_binding MetropolisHastings MetropolisHastingsAlgorithm
Base.@deprecate_binding MarkovChainMonteCarlo MetropolisHastingsAlgorithm
Base.@deprecate_binding MCMC MetropolisHastingsAlgorithm
Base.@deprecate_binding Metropolis MetropolisAlgorithm
Base.@deprecate_binding Glauber GlauberAlgorithm
Base.@deprecate_binding Multicanonical MulticanonicalAlgorithm
Base.@deprecate_binding WangLandau WangLandauAlgorithm
Base.@deprecate_binding HeatBath HeatBathAlgorithm
export  AbstractImportanceSampling, ImportanceSampling,
        AbstractMetropolis, AbstractHeatBath, HeatBath,
        MetropolisHastings, MarkovChainMonteCarlo, MCMC,
        Metropolis, Glauber, Multicanonical, WangLandau

# Deprecated ensemble-mutation names: logweight is read-only, mutations are explicit.
Base.@deprecate update!(ens::MulticanonicalEnsemble; kwargs...) update_logweight!(ens; kwargs...)
Base.@deprecate update!(ens::WangLandauEnsemble; kwargs...) update_logweight!(ens; kwargs...)
Base.@deprecate set!(ens::MulticanonicalEnsemble, args...; kwargs...) set_logweight!(ens, args...; kwargs...)

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

# Fenwick-tree rate handler: lives in event_handler/, included here so the KMC
# `next`/`next_event`/`next_time` it overloads already exist.
include("event_handler/event_rate_tree.jl")
export  EventRateTree,
        total_rate

# Rejection-free n-fold way over the local-states protocol (nsites/local_states/
# delta_energy/modify!/partners), driven by Gillespie/advance!.
include("algorithms/nfold_way.jl")
export  NFoldWay,
        nsites,
        local_states,
        partners,
        delta_energy,
        assert_linear_ensemble

end # module MonteCarloX
