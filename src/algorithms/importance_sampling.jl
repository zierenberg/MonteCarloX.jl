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
    log_acceptance_ratio(alg::AbstractMetropolis, delta_state)

Default log-acceptance ratio for Metropolis-family updates from a local
state difference.
"""
@inline log_acceptance_ratio(alg::AbstractMetropolis, delta_state) = alg.logweight(delta_state)

"""
    log_acceptance_ratio(alg::AbstractImportanceSampling, state_new, state_old)

General log-acceptance ratio for importance sampling algorithms that require
absolute state values.
"""
@inline log_acceptance_ratio(alg::AbstractImportanceSampling, state_new, state_old) =
    alg.logweight(state_new) - alg.logweight(state_old)

"""
    accept!(alg::AbstractImportanceSampling, log_ratio::Real)

Evaluate acceptance criterion for importance sampling.

Updates step and acceptance counters in the algorithm.
Returns true if the move is accepted based on the Metropolis criterion:
- Accept if log_ratio > 0 (new state has higher weight)
- Accept with probability exp(log_ratio) otherwise

This is the core accept/reject step used by all importance sampling algorithms.
"""
function accept!(alg::AbstractImportanceSampling, log_ratio::Real)
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
    reset_statistics!(alg::AbstractImportanceSampling)

Reset step and acceptance counters to zero.

Useful when you want to measure acceptance rate for a specific
run phase without previous history.
"""
function reset_statistics!(alg::AbstractImportanceSampling)
    alg.steps = 0
    alg.accepted = 0
end
