#### Balance functions ####
#
# A balance function turns a log acceptance-ratio
#
#     logR = log( [π(x′) q(x′→x)] / [π(x) q(x→x′)] )
#
# into the dynamics of a Markov chain. The SAME function `f` has two readings, one per time
# convention:
#
#     acceptance_probability(balance, logR)  → discrete-time accept/reject probability  (accept!)
#     transition_rate(balance, logR)         → continuous-time transition rate          (n-fold / kMC)
#
# This is the single object behind "a Metropolis-vs-Glauber choice": one formula replaces the
# separate accept rules and the hard-coded n-fold rate rule. Detailed balance is the defining
# property
#
#     f(logR) / f(−logR) = exp(logR).
#
# Members:
#     MetropolisBalance  f(logR) = min(1, e^logR)              (Metropolis 1953 / Hastings 1970)
#     GlauberBalance     f(logR) = 1 / (1 + e^{−logR}) = σ(logR)  (Glauber ≡ Barker 1965)

"""
    BalanceFunction

Choice of Markov-chain dynamics: a map from a log acceptance-ratio `logR` to an acceptance
probability (discrete time) or an equal transition rate (continuous time). Members must satisfy
detailed balance, `f(logR) / f(-logR) = exp(logR)`. See [`MetropolisBalance`](@ref),
[`GlauberBalance`](@ref), [`acceptance_probability`](@ref), [`transition_rate`](@ref).
"""
abstract type BalanceFunction end

"""
    MetropolisBalance <: BalanceFunction

Metropolis rule `f(logR) = min(1, exp(logR))`: always accept an uphill move (`logR ≥ 0`),
otherwise accept with probability `exp(logR)`.
"""
struct MetropolisBalance <: BalanceFunction end

"""
    GlauberBalance <: BalanceFunction

Glauber / Barker rule `f(logR) = 1 / (1 + exp(-logR)) = σ(logR)` (the logistic sigmoid). For a
two-state local update this coincides with single-site heat bath.
"""
struct GlauberBalance <: BalanceFunction end

"""
    acceptance_probability(balance::BalanceFunction, logR) -> Float64

Probability of accepting a proposed move whose log acceptance-ratio is `logR`, under `balance`.
Reads the balance function in the discrete-time convention (used by [`accept!`](@ref)).
"""
@inline acceptance_probability(::MetropolisBalance, logR::Real) = logR ≥ 0 ? 1.0 : exp(logR)
@inline acceptance_probability(::GlauberBalance, logR::Real) = logistic(logR)

"""
    transition_rate(balance::BalanceFunction, logR) -> Float64

Transition rate implied by `balance` for a move with log acceptance-ratio `logR`. Reads the
*same* balance function in the continuous-time convention (used by kinetic Monte Carlo, e.g. the
n-fold way). For the shipped members this equals [`acceptance_probability`](@ref); the separate
name marks the reading, and leaves room for balance functions whose rate need not be ≤ 1.
"""
@inline transition_rate(balance::BalanceFunction, logR::Real) = acceptance_probability(balance, logR)

# Sample one accept/reject decision. The generic form draws a single `rand` and compares against
# the acceptance probability (matches Glauber's stream exactly). Metropolis overrides it with the
# short-circuit that consumes NO rand on an uphill move, preserving the historical Metropolis
# stream and the zero-overhead spin fast path.
@inline _sample_accept(balance::BalanceFunction, rng::AbstractRNG, logR::Real) =
    rand(rng) < acceptance_probability(balance, logR)
@inline _sample_accept(::MetropolisBalance, rng::AbstractRNG, logR::Real) =
    logR > 0 || rand(rng) < exp(logR)
