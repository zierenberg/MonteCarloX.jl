# Importance Sampling - Core functionality
# Shared by all importance sampling algorithms (Metropolis, Heat Bath, etc.)
"""
    AbstractImportanceSampling <: AbstractAlgorithm

Base type for importance sampling algorithms (Metropolis, Heat Bath, etc.).

Importance sampling algorithms:
- Use accept/reject steps based on log weight ratios
- Track acceptance statistics
- Include an RNG and a log weight function
"""
abstract type AbstractImportanceSampling <: AbstractAlgorithm end

"""
    AbstractGeneralizedEnsemble <: AbstractImportanceSampling

Base type for generalized-ensemble samplers (e.g. multicanonical, Wang-Landau).
"""
abstract type AbstractGeneralizedEnsemble <: AbstractImportanceSampling end

"""
    AbstractMetropolis <: AbstractImportanceSampling

Base type for Metropolis-family samplers where acceptance is naturally
computed from a local state difference (e.g. ΔE).
"""
abstract type AbstractMetropolis <: AbstractImportanceSampling end

"""
    AbstractHeatBath <: AbstractAlgorithm

Base type for heat-bath style samplers.
"""
abstract type AbstractHeatBath <: AbstractAlgorithm end

"""
    ImportanceSampling <: AbstractImportanceSampling

Generic importance-sampling algorithm that operates on full-state
acceptance arguments `(x_new, x_old)` using a callable `logweight`.
"""
mutable struct ImportanceSampling{LW,RNG<:AbstractRNG} <: AbstractImportanceSampling
    rng::RNG
    logweight::LW
    steps::Int
    accepted::Int
end

ImportanceSampling(rng::AbstractRNG, logweight) = ImportanceSampling(rng, logweight, 0, 0)

@inline record_visit!(logweight, accepted::Bool, x_new, x_old) = nothing

"""
    accept!(alg::AbstractImportanceSampling, x_new, x_old)
    accept!(alg::AbstractMetropolis, delta_x)

Evaluate acceptance criterion for importance sampling with differences.

Updates step and acceptance counters in the algorithm.
Returns true if the move is accepted based on the Metropolis criterion:
- Accept if log_ratio > 0 (new state has higher weight)
- Accept with probability exp(log_ratio) otherwise

This is the core accept/reject step used by all importance sampling algorithms.
"""
function accept!(alg::AbstractImportanceSampling, x_new::T, x_old::T) where T
    log_ratio = alg.logweight(x_new) - alg.logweight(x_old)
    accepted = _accept!(alg, log_ratio)
    # TODO: check if this costs too much if not needed..
    # Problem: this is not passed on if logweight is a sum of logweights; hence we may have to make it vector/tuple of abstractLogWeiths?
    record_visit!(alg.logweight, accepted, x_new, x_old)
    return accepted
end
function accept!(alg::AbstractMetropolis, delta_x)
    log_ratio = alg.logweight(delta_x)
    return _accept!(alg, log_ratio)
end
# core function to evaluate acceptance and update counters
function _accept!(alg::AbstractImportanceSampling, log_ratio::Real)
    alg.steps += 1
    accepted = log_ratio > 0 || rand(alg.rng) < exp(log_ratio)
    alg.accepted += accepted 
    return accepted
end

"""
    acceptance_rate(alg::AbstractImportanceSampling)

Calculate the acceptance rate of the algorithm.

Returns the fraction of accepted moves: accepted/steps.
Returns 0.0 if no steps have been attempted yet.
"""
acceptance_rate(alg::AbstractImportanceSampling) = 
    alg.steps > 0 ? alg.accepted / alg.steps : 0.0

"""
    reset!(alg::AbstractImportanceSampling)

Reset step and acceptance counters to zero.

Useful when you want to measure acceptance rate for a specific
run phase without previous history.
"""
function reset!(alg::AbstractImportanceSampling)
    alg.steps = 0
    alg.accepted = 0
end

"""
    set!(alg::AbstractImportanceSampling, args...)

Forward `set!` to the algorithm's logweight object.
"""
set!(alg::AbstractImportanceSampling, args...) = set!(alg.logweight, args...)

"""
    set_logweight!(alg::AbstractImportanceSampling, args...)

Forward `set_logweight!` to the algorithm's logweight object.
"""
set_logweight!(alg::AbstractImportanceSampling, args...) = set!(alg, args...)

"""
    update!(alg::AbstractImportanceSampling, args...; kwargs...)

Forward `update!` to the algorithm's logweight object.
"""
update!(alg::AbstractImportanceSampling, args...; kwargs...) =
    update!(alg.logweight, args...; kwargs...)

"""
    update_weight!(alg::AbstractImportanceSampling, args...; kwargs...)

Forward `update_weight!` to the algorithm's logweight object.
"""
update_weight!(alg::AbstractImportanceSampling, args...; kwargs...) =
    update!(alg, args...; kwargs...)
