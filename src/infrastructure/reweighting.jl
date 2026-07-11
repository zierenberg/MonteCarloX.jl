#### Reweighting ####
#
# Importance-sampling reweighting: turn samples drawn under a `source` ensemble into
# expectations under a `target` ensemble. Everything is kept in log space (offset-free,
# overflow-safe); the linear weights are materialized only on demand via `weights`.

"""
    ImportanceWeights(logw::AbstractVector{<:Real})

Per-sample log importance weights produced by [`reweight`](@ref):
`gᵢ = logweight(target, argᵢ) - logweight(source, argᵢ)`.

Stored in log space (offset-free, overflow-safe). The log of the summed weights
`log_sum_weights = log Σᵢ exp(gᵢ)` is computed once at construction via [`log_sum`](@ref) and
reused by [`weights`](@ref), [`log_normalization`](@ref) and [`ess`](@ref).

Use [`weights`](@ref) to obtain the actual (self-normalized) linear weights for use with
StatsBase, e.g. `mean(A, weights(iw))`, `var(A, weights(iw))`.
"""
struct ImportanceWeights{T<:Real,V<:AbstractVector{T}}
    logw::V
    log_sum_weights::T
end

function ImportanceWeights(logw::AbstractVector{<:Real})
    isempty(logw) && throw(ArgumentError("logw must be non-empty"))
    return ImportanceWeights(logw, log_sum(logw))
end

Base.length(iw::ImportanceWeights) = length(iw.logw)

"""
    reweight(args, source, target=ConstantEnsemble()) -> ImportanceWeights
    reweight(args, source => target)

Log importance weights that reweight samples drawn under `source` to expectations under
`target`, evaluated on the recorded `args` — the coordinate each ensemble's `logweight`
consumes (e.g. energies for a `BoltzmannEnsemble`, or the multicanonical coordinate).

`target` defaults to a flat [`ConstantEnsemble`](@ref), so `reweight(args, source)` simply
strips the `source` weighting (`wᵢ ∝ exp(-logweight(source, argᵢ))`) — recovering the
unbiased density, e.g. the density of states from a multicanonical run.

`source` and `target` may be `AbstractEnsemble`s or bare callables (wrapped as a
`FunctionEnsemble`). The unified log-weight view means the same call serves statistical
mechanics (`BoltzmannEnsemble`, `MulticanonicalEnsemble`) and Bayesian inference
(`FunctionEnsemble` of a log-posterior).

Argument-order convention: ENSEMBLE pairs follow the flow order, source → target — also
available as the self-documenting Pair form `reweight(args, source => target)`. (State
pairs, e.g. in `accept!`, follow the acceptance-ratio order — numerator first — instead.)

```julia
w      = reweight(energies, BoltzmannEnsemble(β=0.40) => BoltzmannEnsemble(β=0.44))
mean_E = mean(energies, weights(w))
logZ   = log_normalization(w)          # log(Z_target / Z_source)
neff   = ess(w)
```
"""
function reweight(args, source, target=ConstantEnsemble())
    s = _as_ensemble(source)
    t = _as_ensemble(target)
    logw = [logweight(t, a) - logweight(s, a) for a in args]
    return ImportanceWeights(logw)
end
reweight(args, (source, target)::Pair) = reweight(args, source, target)

"""
    reweight(logdensity::BinnedObject, logw::AbstractVector) -> ImportanceWeights

Per-bin importance weights from a binned log-density (e.g. a log-DOS) and an additional
log-weight aligned with `get_centers(logdensity)`: `gᵢ = logdensity.values[i] + logw[i]`.
Bins with non-finite log-density (`NaN` or `-Inf`, marking empty or forbidden bins)
receive zero weight.

The canonical distribution and mean energy at inverse temperature `β` from an exact
log-DOS read

```julia
logdos = logdos_exact_ising2D(L)
E      = get_centers(logdos)
P      = weights(reweight(logdos, -β .* E))   # P(E), sums to 1
mean_E = mean(E, P)
```
"""
function reweight(logdensity::BinnedObject, logw::AbstractVector)
    length(logw) == length(logdensity.values) ||
        throw(DimensionMismatch("logw must match the number of bins"))
    g = [isfinite(lg) ? lg + lw : -Inf for (lg, lw) in zip(logdensity.values, logw)]
    any(isfinite, g) || throw(ArgumentError("log-density has no finite entries"))
    return ImportanceWeights(g)
end

"""
    weights(iw::ImportanceWeights) -> StatsBase.AnalyticWeights

Self-normalized importance weights `exp(gᵢ - logZ)` (summing to 1) as StatsBase
`AnalyticWeights`, ready for `mean(A, weights(iw))`, `var`, `std`, `quantile`, …

`AnalyticWeights` gives the effective-sample-size (reliability) variance correction, which
is the correct one for importance weights and matches [`ess`](@ref).
"""
StatsBase.weights(iw::ImportanceWeights) = AnalyticWeights(exp.(iw.logw .- iw.log_sum_weights))

"""
    log_normalization(iw::ImportanceWeights) -> Real

`log(Z_target / Z_source) = logsumexp(g) - log N`, the self-normalized importance-sampling
estimate of the ratio of normalizing constants (free-energy difference / log-evidence ratio).
"""
log_normalization(iw::ImportanceWeights) = iw.log_sum_weights - log(length(iw))

"""
    ess(iw::ImportanceWeights) -> Real

Kish effective sample size `(Σ wᵢ)² / Σ wᵢ²`, computed in log space. Ranges from 1
(one sample dominates) to `length(iw)` (uniform weights). A value ≪ `length(iw)` signals
that the source and target distributions overlap too little for reliable reweighting.
"""
ess(iw::ImportanceWeights) = exp(2 * iw.log_sum_weights - log_sum(2 .* iw.logw))
