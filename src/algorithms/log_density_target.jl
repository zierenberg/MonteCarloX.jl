#### Log-density target (gradient-sampler / interop seam) ####
#
# A thin wrapper pairing a log-weight callable over a parameter vector with its
# dimension. This is the subset of "scores" that genuinely are densities over an
# unconstrained ℝⁿ — i.e. Bayesian targets — as opposed to `logweight` in general,
# whose argument may be a projected coordinate (an energy, a muca bin). Carrying the
# dimension lets such a target satisfy the LogDensityProblems.jl interface (via the
# `LogDensityProblemsExt` extension), so it can be driven by gradient-based samplers
# such as AdvancedHMC while remaining usable directly by MonteCarloX's own samplers.

"""
    LogDensityTarget(logdensity, dimension)

Wrap a log-density `logdensity` over a `dimension`-dimensional parameter vector.

`logdensity` may be a bare callable `θ -> Real` or any `AbstractEnsemble` (coerced
via the usual `logweight` protocol). The wrapper adds the one thing a callable lacks
for gradient-based sampling: the parameter-space `dimension`.

Usable directly as a target for [`MetropolisHastingsAlgorithm`](@ref), and — when
`LogDensityProblems` is loaded — it satisfies that interface (order-0), so an AD
package can equip it with gradients and hand it to AdvancedHMC/NUTS.

```julia
target = LogDensityTarget(logposterior, 2)      # 2 parameters
alg    = MetropolisHastingsAlgorithm(rng, target)        # MonteCarloX's own sampler
# or, with LogDensityProblems + ForwardDiff + AdvancedHMC loaded, drive it by NUTS.
```
"""
struct LogDensityTarget{E}
    ensemble::E
    dimension::Int
    function LogDensityTarget(logdensity, dimension::Integer)
        e = _as_ensemble(logdensity)
        return new{typeof(e)}(e, Int(dimension))
    end
end

@inline logweight(t::LogDensityTarget, x) = logweight(t.ensemble, x)
@inline logweight(t::LogDensityTarget) = logweight(t.ensemble)
@inline _as_ensemble(t::LogDensityTarget) = t.ensemble
linear_logweight(t::LogDensityTarget) = linear_logweight(t.ensemble)
