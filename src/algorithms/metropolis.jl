"""
    AbstractMetropolis <: AbstractImportanceSampling

Base type for Metropolis-family samplers where acceptance is naturally
computed from a local state difference (e.g. ΔE).
"""
abstract type AbstractMetropolis <: AbstractImportanceSampling end

"""
    accept!(alg::AbstractMetropolis, delta_state)

Metropolis-family acceptance using a local state difference.
"""
function accept!(alg::AbstractMetropolis, delta_state)
    log_ratio = logweight(ensemble(alg), delta_state)
    return _accept!(alg, log_ratio)
end

"""
    Metropolis <: AbstractMetropolis

Metropolis algorithm for importance sampling.

The Metropolis algorithm samples from a probability distribution 
proportional to exp(log_weight) using an accept/reject criterion.

Unified view:
- Bayesian inference: `logweight(theta) = logposterior(theta)`
- Statistical mechanics: `logweight(x) = -beta * E(x)`

Both are passed as the same callable ensemble score.
In other words, the algorithm `ensemble` defines the operative logweight.

# Fields
- `rng::AbstractRNG`: Random number generator
- `ensemble`: Callable ensemble score (function or weight object)
- `steps::Int`: Total number of steps attempted
- `accepted::Int`: Number of accepted steps

# Examples
```julia
# Create with Boltzmann weight
alg = Metropolis(Random.default_rng(), β=2.0)

# Create with Bayesian log-posterior
logposterior(theta) = loglikelihood(theta) + logprior(theta)
alg = Metropolis(Random.default_rng(), logposterior)

# Create with custom callable score
alg = Metropolis(Random.default_rng(), x -> -0.5 * x^2)

# Create with a weight object
ens = BoltzmannEnsemble(β=1.5)
alg = Metropolis(Random.default_rng(), ens)
```

```julia
# Create with a tabulated or custom ensemble
ens = FunctionEnsemble(x -> -0.5 * x^2)
alg = Metropolis(Random.default_rng(), ens)
```
"""
mutable struct Metropolis{LW, RNG<:AbstractRNG} <: AbstractMetropolis
    rng::RNG
    ensemble::LW
    steps::Int
    accepted::Int
end

"""
    Metropolis(rng::AbstractRNG, ensemble)

Create a Metropolis sampler with a general callable ensemble score.

# Arguments
- `rng::AbstractRNG`: Random number generator
- `ensemble`: A callable object or function returning log weight / log density
"""
Metropolis(rng::AbstractRNG, ensemble) = 
    Metropolis(rng, _as_ensemble(ensemble), 0, 0)

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
    Metropolis(rng, BoltzmannEnsemble(β=β))

"""
    Glauber <: AbstractMetropolis

Glauber sampler with logistic acceptance rule.

Uses the same proposal interface and log-ratio as Metropolis-family algorithms,
but acceptance is:

    p_accept = 1 / (1 + exp(-log_ratio))
"""
mutable struct Glauber{LW, RNG<:AbstractRNG} <: AbstractMetropolis
    rng::RNG
    ensemble::LW
    steps::Int
    accepted::Int
end

Glauber(rng::AbstractRNG, ensemble) =
    Glauber(rng, _as_ensemble(ensemble), 0, 0)

Glauber(rng::AbstractRNG; β::Real) =
    Glauber(rng, BoltzmannEnsemble(β=β))

function accept!(alg::Glauber, delta_state::Real)
    log_ratio = logweight(ensemble(alg), delta_state)
    alg.steps += 1
    accepted = rand(alg.rng) < logistic(log_ratio)
    alg.accepted += accepted
    return accepted
end
