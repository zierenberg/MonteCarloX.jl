#### MCMC convergence diagnostics ####
#
# Standard quality measures for Markov chain output: an autocorrelation-based
# effective sample size for a single chain, and the Gelman–Rubin R̂ across chains.

"""
    ess(samples::AbstractVector{<:Real}) -> Float64

Autocorrelation-based effective sample size of a single Markov chain:
`length(samples) / (2 τ_int)`, where `τ_int` is the
[`integrated_autocorrelation_time`](@ref) (Sokal convention `1/2 + Σ_{t≥1} C(t)`).
Equals `length(samples)` for an uncorrelated chain and shrinks as autocorrelation
grows — the honest count of independent draws behind a Monte Carlo estimate.

(For the reliability of *importance weights*, see the separate
[`ess(::ImportanceWeights)`](@ref), which measures weight concentration, not autocorrelation.)
"""
ess(samples::AbstractVector{<:Real}) =
    length(samples) / (2 * integrated_autocorrelation_time(samples))

"""
    rhat(chains) -> Float64

Gelman–Rubin potential scale reduction factor R̂ from several independent chains
(`chains` a collection of scalar chains). It compares the between-chain variance to
the within-chain variance; R̂ → 1 as the chains mix to the same distribution, while
values above ≈1.01 signal that they have not yet converged to a common target.

```math
\\hat R = \\sqrt{\\frac{\\frac{N-1}{N} W + \\frac{1}{N} B}{W}},
```
with `W` the mean within-chain variance and `B = N · var(chain means)`.
"""
function rhat(chains::AbstractVector{<:AbstractVector{<:Real}})
    M = length(chains)
    M >= 2 || throw(ArgumentError("rhat needs at least 2 chains"))
    N = minimum(length, chains)
    N >= 2 || throw(ArgumentError("rhat needs chains of length >= 2"))
    means = [mean(view(c, 1:N)) for c in chains]
    vars  = [var(view(c, 1:N))  for c in chains]
    W = mean(vars)
    B = N * var(means)
    W > 0 || return 1.0
    var_plus = (N - 1) / N * W + B / N
    return sqrt(var_plus / W)
end
