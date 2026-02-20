# Gillespie algorithm
# Simplest user-facing kinetic Monte Carlo algorithm.

using Random

"""
    Gillespie <: AbstractKineticMonteCarlo

Gillespie algorithm for continuous-time event-driven sampling.

# Fields
- `rng::AbstractRNG`: Random number generator
- `steps::Int`: Number of events sampled
- `time::Float64`: Current simulation time
"""
mutable struct Gillespie{RNG<:AbstractRNG} <: AbstractKineticMonteCarlo
    rng::RNG
    steps::Int
    time::Float64
end

"""
    Gillespie(rng::AbstractRNG)

Create a Gillespie sampler with explicit RNG.
"""
Gillespie(rng::AbstractRNG) = Gillespie(rng, 0, 0.0)

"""
    Gillespie()

Create a Gillespie sampler using `Random.GLOBAL_RNG`.
"""
Gillespie() = Gillespie(Random.GLOBAL_RNG)
