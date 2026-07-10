# The MetropolisHastingsAlgorithm engine — the accept/reject core shared by every algorithm built from
# propose + balance-accept (Metropolis, Glauber, Metropolis-Hastings, Multicanonical, Wang-Landau,
# …). The three orthogonal slots are the ensemble (what target), the balance function (which
# dynamics), and — supplied by the caller — the proposal (which move). The friendly named
# constructors live in metropolis.jl / glauber.jl / multicanonical.jl / wang_landau.jl.
"""
    MetropolisHastingsAlgorithm{E,B,RNG} <: AbstractMarkovChainMonteCarlo

Propose-and-accept Markov-chain Monte Carlo engine. Carries an `ensemble` (the target, via its
`logweight`), a [`BalanceFunction`](@ref) (the dynamics), plus `steps`/`accepted` counters. The
proposal is not stored here: it is the caller's move, which assembles the log acceptance-ratio
`logR` passed to [`accept!`](@ref).

The friendly named constructors [`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref),
[`MulticanonicalAlgorithm`](@ref) and [`WangLandauAlgorithm`](@ref) build this engine with the
appropriate ensemble × balance; construct it directly for a custom combination.

Unified view of the ensemble score:
- statistical mechanics: `logweight(x) = -β E(x)`
- Bayesian inference:    `logweight(θ) = logposterior(θ)`

# Fields
- `rng::RNG`: random number generator
- `ensemble::E`: callable ensemble score (log-weight object or function)
- `balance::B`: balance function (`MetropolisBalance`, `GlauberBalance`, …)
- `steps::Int`, `accepted::Int`: acceptance counters
"""
mutable struct MetropolisHastingsAlgorithm{E,B<:BalanceFunction,RNG<:AbstractRNG} <: AbstractMarkovChainMonteCarlo
    rng::RNG
    ensemble::E
    balance::B
    steps::Int
    accepted::Int
end

"""
    MetropolisHastingsAlgorithm(rng, ensemble, balance=MetropolisBalance())

Build the engine from an `rng`, a callable `ensemble` (wrapped via `FunctionEnsemble` if it is a
bare function), and a [`BalanceFunction`](@ref).
"""
MetropolisHastingsAlgorithm(rng::AbstractRNG, ensemble, balance::BalanceFunction=MetropolisBalance()) =
    MetropolisHastingsAlgorithm(rng, _as_ensemble(ensemble), balance, 0, 0)

"""
    ensemble(alg::AbstractMarkovChainMonteCarlo)

Return the ensemble object carried by an MCMC algorithm — the object whose `logweight` defines
the acceptance.
"""
@inline ensemble(alg::AbstractMarkovChainMonteCarlo) = getfield(alg, :ensemble)

"""
    balance(alg::MetropolisHastingsAlgorithm)

Return the [`BalanceFunction`](@ref) (Metropolis, Glauber, …) governing the acceptance dynamics.
"""
@inline balance(alg::MetropolisHastingsAlgorithm) = getfield(alg, :balance)

"""
    logweight(alg::AbstractMarkovChainMonteCarlo)

Return the algorithm's ensemble as a logweight callable. Equivalent to `logweight(ensemble(alg))`.
"""
@inline logweight(alg::AbstractMarkovChainMonteCarlo) = logweight(ensemble(alg))

"""
    accept!(alg::MetropolisHastingsAlgorithm, logR) -> Bool

Core acceptance step: apply the algorithm's [`balance`](@ref) function to the log acceptance-ratio

    logR = log( [π(x′) q(x′→x)] / [π(x) q(x→x′)] ),

update the counters, and return whether the move is accepted. The caller assembles `logR` —
typically `logweight(ensemble(alg), ΔE)` for a symmetric move on a linear ensemble, or via the
two-argument convenience below. This scalar contract is what keeps the *ensemble*, *balance* and
*proposal* concerns separate.
"""
@inline function accept!(alg::MetropolisHastingsAlgorithm, logR::Real)
    alg.steps += 1
    accepted = _sample_accept(alg.balance, alg.rng, logR)
    alg.accepted += accepted
    return accepted
end

"""
    accept!(alg::MetropolisHastingsAlgorithm, arg_new, arg_old) -> Bool

Convenience over [`accept!(alg, logR)`](@ref) that forms `logR = logweight(ens, arg_new) −
logweight(ens, arg_old)` from the ensemble's logweight arguments (energies for
`BoltzmannEnsemble`, parameter vectors for a Bayesian `FunctionEnsemble`, the binned reaction
coordinate for `MulticanonicalEnsemble`). This is the general path valid for any ensemble
(linear or not); it also drives multicanonical visit recording.
"""
@inline function accept!(alg::MetropolisHastingsAlgorithm, arg_new::T, arg_old::T) where T
    ens = ensemble(alg)
    logR = logweight(ens, arg_new) - logweight(ens, arg_old)
    accepted = accept!(alg, logR)
    if should_record_visit(ens)
        record_visit!(ens, accepted ? arg_new : arg_old)
    end
    return accepted
end

"""
    acceptance_rate(alg::MetropolisHastingsAlgorithm)

Fraction of accepted moves, `accepted / steps` (0.0 before any step).
"""
acceptance_rate(alg::MetropolisHastingsAlgorithm) =
    alg.steps > 0 ? alg.accepted / alg.steps : 0.0

"""
    reset!(alg::MetropolisHastingsAlgorithm)

Reset the `steps` and `accepted` counters to zero.
"""
@inline reset!(alg::MetropolisHastingsAlgorithm) = _reset!(alg)
function _reset!(alg::MetropolisHastingsAlgorithm)
    alg.steps = 0
    alg.accepted = 0
end

# ── Named dynamics: the classic points in ensemble × balance space ────────────

"""
    MetropolisAlgorithm(rng, ensemble)
    MetropolisAlgorithm(rng; β)

Metropolis-dynamics sampler: the engine with [`MetropolisBalance`](@ref). The keyword form is
the canonical-ensemble convenience (`BoltzmannEnsemble(β=β)`). Metropolis–Hastings needs no
separate type — a proposal with nonzero ratio folds into the `logR` passed to [`accept!`](@ref).
"""
MetropolisAlgorithm(rng::AbstractRNG, ensemble) =
    MetropolisHastingsAlgorithm(rng, ensemble, MetropolisBalance())
MetropolisAlgorithm(rng::AbstractRNG; β::Real) =
    MetropolisHastingsAlgorithm(rng, BoltzmannEnsemble(β=β), MetropolisBalance())

"""
    GlauberAlgorithm(rng, ensemble)
    GlauberAlgorithm(rng; β)

Glauber-dynamics sampler: the engine with [`GlauberBalance`](@ref) (logistic acceptance). For a
two-state local update this coincides with single-site heat bath.
"""
GlauberAlgorithm(rng::AbstractRNG, ensemble) =
    MetropolisHastingsAlgorithm(rng, ensemble, GlauberBalance())
GlauberAlgorithm(rng::AbstractRNG; β::Real) =
    MetropolisHastingsAlgorithm(rng, BoltzmannEnsemble(β=β), GlauberBalance())
