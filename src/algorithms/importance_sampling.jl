# Importance Sampling - Core functionality
# Shared by all importance sampling algorithms (Metropolis, Heat Bath, etc.)
"""
    AbstractImportanceSampling <: AbstractAlgorithm

Base type for importance sampling algorithms (Metropolis, Heat Bath, etc.).

Importance sampling algorithms:
- Use accept/reject steps based on log-weight ratios
- Track acceptance statistics
- Include an RNG and a callable ensemble score

The ensemble score is typically a log target density:
- Bayesian inference: `logposterior(theta) = loglikelihood(data, theta) + logprior(theta)`
- Statistical mechanics: `logweight(x) = -beta * E(x)`

Both levels are intentional in this API:
- `ensemble(alg)` names the architectural object carried by the algorithm
- `logweight(alg)` names its role in acceptance ratios

`ensemble(alg)` therefore defines the effective logweight used by `accept!`.
"""
abstract type AbstractImportanceSampling <: AbstractAlgorithm end

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
acceptance arguments `(x_new, x_old)` using a callable `ensemble`.

The callable may be a function or a log-weight object and should return a
scalar score such as a log density / log weight.
"""
mutable struct ImportanceSampling{LW,RNG<:AbstractRNG} <: AbstractImportanceSampling
    rng::RNG
    ensemble::LW
    steps::Int
    accepted::Int
end

@inline _as_ensemble(e::AbstractEnsemble) = e
@inline _as_ensemble(e) = FunctionEnsemble(e)

ImportanceSampling(rng::AbstractRNG, ensemble) = ImportanceSampling(rng, _as_ensemble(ensemble), 0, 0)

"""
    ensemble(alg::AbstractImportanceSampling)

Return the ensemble object carried by an importance-sampling algorithm.

This is the canonical accessor in the ensemble-first API.
Operationally, this object defines the logweight used in acceptance.
"""
@inline ensemble(alg::AbstractImportanceSampling) = getfield(alg, :ensemble)

@inline logweight(e::AbstractEnsemble, x) = e(x)

"""
    logweight(alg::AbstractImportanceSampling)

Return the algorithm ensemble via a logweight-oriented alias.
Equivalent to `ensemble(alg)`.

Use this accessor when reasoning about acceptance formulas.
"""
@inline logweight(alg::AbstractImportanceSampling) = ensemble(alg)

# Optional ensemble-level visit hooks used by generic accept!.
# Ensembles that need histogram/visit bookkeeping can specialize these.
@inline should_record_visit(ens) = false
@inline record_visit!(ens, x_vis) = nothing

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
    ens = ensemble(alg)
    log_ratio = logweight(ens, x_new) - logweight(ens, x_old)
    accepted = _accept!(alg, log_ratio)
    if should_record_visit(ens)
        x_vis = accepted ? x_new : x_old
        record_visit!(ens, x_vis)
    end
    return accepted
end
function accept!(alg::AbstractMetropolis, delta_x)
    log_ratio = logweight(ensemble(alg), delta_x)
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
@inline reset!(alg::AbstractImportanceSampling) = _reset!(alg)
function _reset!(alg::AbstractImportanceSampling)
    alg.steps = 0
    alg.accepted = 0
end

"""
    set!(alg::AbstractImportanceSampling, args...)

Forward `set!` to the algorithm's logweight object.
"""
set!(alg::AbstractImportanceSampling, args...) = set!(ensemble(alg), args...)

"""
    set_logweight!(alg::AbstractImportanceSampling, args...)

Forward `set_logweight!` to the algorithm's ensemble object.
Useful for multicanonical-style workflows.
"""
set_logweight!(alg::AbstractImportanceSampling, args...) = set!(alg, args...)

"""
    update!(alg::AbstractImportanceSampling, args...; kwargs...)

Forward `update!` to the algorithm's ensemble/logweight object.
"""
update!(alg::AbstractImportanceSampling, args...; kwargs...) =
    update!(ensemble(alg), args...; kwargs...)

