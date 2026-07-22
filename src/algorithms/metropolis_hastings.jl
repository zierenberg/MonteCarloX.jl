# The MetropolisHastingsAlgorithm engine — the accept/reject core shared by every algorithm built from
# propose + balance-accept (Metropolis, Glauber, Metropolis-Hastings, Multicanonical, Wang-Landau,
# …). The three orthogonal slots are the ensemble (what target), the balance function (which
# dynamics), and — supplied by the caller — the proposal (which move). The friendly named
# constructors live in metropolis.jl / glauber.jl / multicanonical.jl / wang_landau.jl.
"""
    MetropolisHastingsAlgorithm{E,B,RNG} <: AbstractMarkovChainMonteCarlo

Propose-and-accept Markov-chain Monte Carlo engine. Carries an `ensemble` (the target, via its
`logweight`), a [`BalanceFunction`](@ref) (the dynamics), plus `steps`/`accepted` counters. The
proposal is not stored here: it is the caller's move, which supplies coordinates to
[`accept!`](@ref) (or a whole log acceptance-ratio to [`accept_logratio!`](@ref)).

The named constructors [`MetropolisAlgorithm`](@ref), [`GlauberAlgorithm`](@ref),
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
    balance(alg::MetropolisHastingsAlgorithm)

Return the [`BalanceFunction`](@ref) (Metropolis, Glauber, …) governing the acceptance dynamics.
"""
@inline balance(alg::MetropolisHastingsAlgorithm) = getfield(alg, :balance)

"""
    accept_logratio!(alg::MetropolisHastingsAlgorithm, logR) -> Bool

Core acceptance primitive: apply the algorithm's [`balance`](@ref) function to the log
acceptance-ratio

    logR = log( [π(x′) q(x′→x)] / [π(x) q(x→x′)] ),

update the counters, and return whether the move is accepted. This is the raw contract — the
caller owns the *whole* `logR`, so it touches neither the ensemble nor the proposal. It is the
escape hatch for a bespoke target that is not expressed as an MCX ensemble; ordinary model code
should use [`accept!`](@ref), which lets the algorithm own the ensemble (the target π) while the
model supplies coordinates and, if the proposal is asymmetric, a `correction`.
"""
@inline function accept_logratio!(alg::MetropolisHastingsAlgorithm, logR::Real)
    alg.steps += 1
    accepted = _sample_accept(alg.balance, alg.rng, logR)
    alg.accepted += accepted
    return accepted
end

"""
    accept!(alg::MetropolisHastingsAlgorithm, Δarg; correction=0) -> Bool

Linear-ensemble acceptance: the algorithm owns the target π, so the model hands over only the
coordinate DIFFERENCE `Δarg` (e.g. the O(1) local `ΔE` of a spin flip) and the algorithm forms
`logR = logweight(ens, Δarg) + correction`. Requires a [`linear_logweight`](@ref) ensemble
(e.g. `BoltzmannEnsemble`); for a nonlinear ensemble the difference is not enough — pass the
absolute pair `accept!(alg, arg_new, arg_old)` instead.

`correction` is the log proposal-ratio `log[q(x′→x)/q(x→x′)]` (the Metropolis–Hastings term for
an asymmetric move); it also carries any additive log-target factor the ensemble argument does
not, such as a reference-process log-density. It defaults to `0` (symmetric proposal).
"""
@inline function accept!(alg::MetropolisHastingsAlgorithm, Δarg::Real; correction::Real=0)
    ens = ensemble(alg)
    assert_linear_ensemble(ens, "accept!(alg, Δarg)")
    logR = logweight(ens, Δarg)
    # For the default (symmetric) call `correction` is the literal 0, so `iszero` folds at
    # compile time and the `+ correction` add is elided — the spin fast path stays add-free.
    return accept_logratio!(alg, iszero(correction) ? logR : logR + correction)
end

"""
    accept!(alg::MetropolisHastingsAlgorithm, arg_new, arg_old; correction=0) -> Bool

General acceptance for any ensemble: the model hands over the ABSOLUTE logweight arguments
(energies for `BoltzmannEnsemble`, parameter vectors for a Bayesian `FunctionEnsemble`, the
binned reaction coordinate for `MulticanonicalEnsemble`) and the algorithm forms `logR =
logweight(ens, arg_new) − logweight(ens, arg_old) + correction`. Valid for linear and nonlinear
ensembles alike; it also drives multicanonical visit recording.

Argument-order convention: STATE pairs follow the acceptance-ratio order — numerator first,
`(arg_new, arg_old)`, as in `min(1, π(x′)/π(x))`. (Ensemble pairs, e.g. in `reweight`, follow the
flow order source → target instead.) See the one-argument form for the meaning of `correction`.
"""
@inline function accept!(alg::MetropolisHastingsAlgorithm, arg_new::T, arg_old::T; correction::Real=0) where T
    ens = ensemble(alg)
    logR = logweight(ens, arg_new) - logweight(ens, arg_old)
    # Default (symmetric) call: `correction` is the literal 0, `iszero` folds and the add is elided.
    accepted = accept_logratio!(alg, iszero(correction) ? logR : logR + correction)
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
separate type — an asymmetric proposal folds into the `correction` keyword of [`accept!`](@ref).
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
