# Equilibrium algorithms - Importance Sampling
# Based on examples/api.ipynb

using Random

"""
    AbstractImportanceSampling <: AbstractAlgorithm

Base type for importance sampling algorithms (Metropolis, Heat Bath, etc.).
"""
abstract type AbstractImportanceSampling <: AbstractAlgorithm end

"""
    accept!(alg::AbstractImportanceSampling, log_ratio::Real)

Evaluate acceptance criterion for importance sampling.

Updates step and acceptance counters in the algorithm.
Returns true if the move is accepted.
"""
@inline function accept!(alg::AbstractImportanceSampling, log_ratio::Real)
    alg.steps += 1
    accepted = log_ratio > 0 || rand(alg.rng) < exp(log_ratio)
    alg.accepted += accepted 
    return accepted
end

# Specific implementations

"""
    BoltzmannLogWeight <: AbstractLogWeight

Boltzmann weight function: w(E) = exp(-β*E).

# Fields
- `β::Real`: Inverse temperature
"""
struct BoltzmannLogWeight <: AbstractLogWeight
    β::Real
end

"""
    (lw::BoltzmannLogWeight)(E)

Evaluate log weight: log(w) = -β * sum(E).
E can be a scalar or vector of energy components.
"""
@inline (lw::BoltzmannLogWeight)(E) = -lw.β * sum(E)

"""
    Metropolis <: AbstractImportanceSampling

Metropolis algorithm for importance sampling.

# Fields
- `rng::AbstractRNG`: Random number generator
- `logweight::Base.Callable`: Log weight function
- `steps::Int`: Total number of steps attempted
- `accepted::Int`: Number of accepted steps
"""
mutable struct Metropolis <: AbstractImportanceSampling
    rng::AbstractRNG
    logweight::Base.Callable
    steps::Int
    accepted::Int
end

"""
    Metropolis(rng::AbstractRNG, logweight::Base.Callable)

Create a Metropolis sampler with a general log weight function.
"""
Metropolis(rng::AbstractRNG, logweight::Base.Callable) = 
    Metropolis(rng, logweight, 0, 0)

"""
    Metropolis(rng::AbstractRNG; β::Real)

Create a Metropolis sampler with Boltzmann weight at inverse temperature β.
"""
Metropolis(rng::AbstractRNG; β::Real) = 
    Metropolis(rng, BoltzmannLogWeight(β))

"""
    acceptance_rate(alg::AbstractImportanceSampling)

Calculate the acceptance rate of the algorithm.
"""
acceptance_rate(alg::AbstractImportanceSampling) = alg.steps > 0 ? alg.accepted / alg.steps : 0.0

"""
    reset_statistics!(alg::AbstractImportanceSampling)

Reset step and acceptance counters.
"""
function reset_statistics!(alg::AbstractImportanceSampling)
    alg.steps = 0
    alg.accepted = 0
end
