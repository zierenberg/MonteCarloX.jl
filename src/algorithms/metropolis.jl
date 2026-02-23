# Metropolis Algorithm
# Classic importance sampling with Metropolis-Hastings acceptance criterion
using Random

"""
    Metropolis <: AbstractImportanceSampling

Metropolis algorithm for importance sampling.

The Metropolis algorithm samples from a probability distribution 
proportional to exp(log_weight) using an accept/reject criterion.

# Fields
- `rng::AbstractRNG`: Random number generator
- `logweight::Union{AbstractLogWeight, Function}`: Log weight function
- `steps::Int`: Total number of steps attempted
- `accepted::Int`: Number of accepted steps

# Examples
```julia
# Create with Boltzmann weight
alg = Metropolis(Random.default_rng(), β=2.0)

# Create with custom log weight function
alg = Metropolis(Random.default_rng(), E -> -2.0 * sum(E))

# Create with a weight object
logweight = BoltzmannLogWeight(1.5)
alg = Metropolis(Random.default_rng(), logweight)
```
"""
mutable struct Metropolis{LW, RNG<:AbstractRNG} <: AbstractMetropolis
    rng::RNG
    logweight::LW
    steps::Int
    accepted::Int
end

"""
    Metropolis(rng::AbstractRNG, logweight::Union{AbstractLogWeight, Function})

Create a Metropolis sampler with a general log weight function.

# Arguments
- `rng::AbstractRNG`: Random number generator
- `logweight`: Either an `AbstractLogWeight` object or a callable function
"""
Metropolis(rng::AbstractRNG, logweight::Union{AbstractLogWeight, Function}) = 
    Metropolis(rng, logweight, 0, 0)

"""
    Metropolis(rng::AbstractRNG; β::Real)

Create a Metropolis sampler with Boltzmann weight at inverse temperature β.

This is a convenience constructor for the canonical ensemble.

# Arguments
- `rng::AbstractRNG`: Random number generator

# Keyword Arguments
- `β::Real`: Inverse temperature (β = 1/k_B T)
"""
Metropolis(rng::AbstractRNG; β::Real) = 
    Metropolis(rng, BoltzmannLogWeight(β))

"""
    Glauber <: AbstractMetropolis

Glauber sampler with logistic acceptance rule.

Uses the same proposal interface and log-ratio as Metropolis-family algorithms,
but acceptance is:

    p_accept = 1 / (1 + exp(-log_ratio))
"""
mutable struct Glauber{LW, RNG<:AbstractRNG} <: AbstractMetropolis
    rng::RNG
    logweight::LW
    steps::Int
    accepted::Int
end

Glauber(rng::AbstractRNG, logweight::Union{AbstractLogWeight, Function}) =
    Glauber(rng, logweight, 0, 0)

Glauber(rng::AbstractRNG; β::Real) =
    Glauber(rng, BoltzmannLogWeight(β))

function accept!(alg::Glauber, delta_state::Real)
    log_ratio = alg.logweight(delta_state)
    alg.steps += 1
    accepted = rand(alg.rng) < logistic(log_ratio)
    alg.accepted += accepted
    return accepted
end
