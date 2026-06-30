"""
    AbstractMetropolis <: AbstractImportanceSampling

Base type for Metropolis-family samplers where acceptance is naturally
computed from a local difference of the ensemble's logweight argument
(e.g. ΔE for `BoltzmannEnsemble`).

Requires a linear logweight: `logweight(ens, Δarg) == logweight(ens, arg+Δarg) - logweight(ens, arg)`.
Non-linear ensembles (e.g. `MulticanonicalEnsemble`, `WangLandauEnsemble`) must use
`ImportanceSampling` instead.
"""
abstract type AbstractMetropolis <: AbstractImportanceSampling end


"""
    accept!(alg::AbstractMetropolis, delta_arg) -> Bool

Metropolis-family acceptance using a local difference `delta_arg` of the
ensemble's logweight argument (e.g. `ΔE` for `BoltzmannEnsemble`). Only
valid for linear ensembles; see [`AbstractMetropolis`](@ref).
"""
@inline function accept!(alg::AbstractMetropolis, delta_arg)
    log_ratio = logweight(ensemble(alg), delta_arg)
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

# Create with a Boltzmann ensemble object
ens = BoltzmannEnsemble(β=1.5)
alg = Metropolis(Random.default_rng(), ens)

# Create with a linear callable (caller asserts linearity)
ens = FunctionEnsemble(x -> -0.5 * x, linear=true)
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
function Metropolis(rng::AbstractRNG, ensemble)
    ens = _as_ensemble(ensemble)
    linear_logweight(ens) || throw(ArgumentError(
        "$(typeof(ens)) does not have a linear logweight and cannot be used with Metropolis. " *
        "Use ImportanceSampling or a dedicated algorithm instead."))
    Metropolis(rng, ens, 0, 0)
end

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

function Glauber(rng::AbstractRNG, ensemble)
    ens = _as_ensemble(ensemble)
    linear_logweight(ens) || throw(ArgumentError(
        "$(typeof(ens)) does not have a linear logweight and cannot be used with Glauber. " *
        "Use ImportanceSampling or a dedicated algorithm instead."))
    Glauber(rng, ens, 0, 0)
end

Glauber(rng::AbstractRNG; β::Real) =
    Glauber(rng, BoltzmannEnsemble(β=β))

function accept!(alg::Glauber, delta_arg::Real)
    log_ratio = logweight(ensemble(alg), delta_arg)
    alg.steps += 1
    accepted = rand(alg.rng) < logistic(log_ratio)
    alg.accepted += accepted
    return accepted
end
