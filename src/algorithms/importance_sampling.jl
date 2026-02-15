# Importance Sampling - Core functionality
# Shared by all importance sampling algorithms (Metropolis, Heat Bath, etc.)

"""
    accept!(alg::AbstractImportanceSampling, log_ratio::Real)

Evaluate acceptance criterion for importance sampling.

Updates step and acceptance counters in the algorithm.
Returns true if the move is accepted based on the Metropolis criterion:
- Accept if log_ratio > 0 (new state has higher weight)
- Accept with probability exp(log_ratio) otherwise

This is the core accept/reject step used by all importance sampling algorithms.
"""
@inline function accept!(alg::AbstractImportanceSampling, log_ratio::Real)
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
