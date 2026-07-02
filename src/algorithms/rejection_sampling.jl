"""
    RejectionSampling <: AbstractAlgorithm

Rejection sampler: the independent-sample analogue of the Metropolis accept/reject
step. Candidates are kept with probability `p(x) / g(x)`, where `g` is an `envelope`
that dominates the (unnormalized) `target` `p`, i.e. `p(x) ≤ g(x)` for all `x`. The
accepted candidates are then distributed exactly as `p`.

The candidate must be drawn from the *normalized* envelope. In the common case the
envelope is a proposal scaled by a constant, `g = M·q`, so one draws `x ~ q` and
supplies `envelope` as the log-weight `logq(x) + logM` — but any dominating function
works, which is why the bound is carried by the envelope ensemble rather than a
separate scalar.

The acceptance formula `logweight(target, x) - logweight(envelope, x)` is exactly the
log importance weight of [`reweight`](@ref): rejection and importance sampling share
one computation and differ only in what they do with it — rejection requires the
envelope to dominate and *accepts* with it, `reweight` keeps every draw and *weights*
with it. As with Metropolis, the caller owns the proposal loop and calls
[`accept!`](@ref) on each candidate; [`acceptance_rate`](@ref) then estimates
`Z_target / Z_envelope`.

`envelope` and `target` may be `AbstractEnsemble`s, bare log-weight callables, or —
with the `Distributions` extension loaded — `Distribution`s (wrapped via `logpdf`).

# Fields
- `rng::AbstractRNG`
- `envelope`: ensemble providing the dominating log-weight `logg`
- `target`: ensemble providing the unnormalized `logp`
- `steps::Int`, `accepted::Int`: proposal and acceptance counters
"""
mutable struct RejectionSampling{E,T,RNG<:AbstractRNG} <: AbstractAlgorithm
    rng::RNG
    envelope::E
    target::T
    steps::Int
    accepted::Int
end

"""
    RejectionSampling(rng, envelope, target)

Create a rejection sampler whose `envelope` dominates the unnormalized `target`,
`logweight(target, x) ≤ logweight(envelope, x)` for all `x`.
"""
function RejectionSampling(rng::AbstractRNG, envelope, target)
    return RejectionSampling(rng, _as_ensemble(envelope), _as_ensemble(target), 0, 0)
end

@inline steps(alg::RejectionSampling) = getfield(alg, :steps)

"""
    accept!(alg::RejectionSampling, x) -> Bool

Rejection acceptance for a candidate `x` drawn from the normalized envelope: accept
with probability `exp(logp(x) - logg(x)) = p(x) / g(x)`, updating the counters. Throws
if the envelope is violated (`logp(x) > logg(x)`), which would otherwise silently bias
the accepted samples toward the wrong distribution.
"""
@inline function accept!(alg::RejectionSampling, x)
    log_ratio = logweight(alg.target, x) - logweight(alg.envelope, x)
    log_ratio > 0 && throw(ArgumentError(
        "rejection envelope violated at x = $x: logp(x) = $(logweight(alg.target, x)) " *
        "exceeds logg(x) = $(logweight(alg.envelope, x)); the envelope must dominate the target"))
    alg.steps += 1
    accepted = rand(alg.rng) < exp(log_ratio)
    alg.accepted += accepted
    return accepted
end

"""
    acceptance_rate(alg::RejectionSampling)

Fraction of proposals accepted, `accepted / steps` (`0.0` before any proposal).
Estimates `Z_target / Z_envelope` — the rejection-sampling counterpart to the
`log Z` from importance-sampling [`reweight`](@ref).
"""
acceptance_rate(alg::RejectionSampling) = alg.steps > 0 ? alg.accepted / alg.steps : 0.0

"""
    reset!(alg::RejectionSampling)

Reset the proposal and acceptance counters to zero.
"""
function reset!(alg::RejectionSampling)
    alg.steps = 0
    alg.accepted = 0
    return alg
end
