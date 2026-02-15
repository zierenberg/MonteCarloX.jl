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
weight = BoltzmannLogWeight(1.5)
alg = Metropolis(Random.default_rng(), weight)
```
"""
mutable struct Metropolis <: AbstractImportanceSampling
    rng::AbstractRNG
    logweight::Union{AbstractLogWeight, Function}
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
