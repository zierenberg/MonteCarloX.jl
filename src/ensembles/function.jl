"""
    FunctionEnsemble(logweight; linear=false, dimension=nothing)

Wrap a log-weight callable in an ensemble. `logweight` maps a coordinate to an
**un-normalized** log weight — e.g. a Bayesian log-posterior, or `x -> -β*E(x)`.
It need not integrate to 1: that is exactly what separates a log-weight from a
`logpdf`, and what `log_normalization`/`ess` recover.

Set `linear=true` to allow use with Metropolis-family algorithms, asserting that
`logweight(Δarg) == logweight(arg + Δarg) - logweight(arg)`. The caller is
responsible for ensuring this property holds.

Set `dimension` when the coordinate is a parameter vector `θ ∈ ℝᵈ` and the
ensemble should be usable by gradient-based samplers: with `LogDensityProblems`
loaded, the ensemble then satisfies that interface (order-0), so an AD package
can equip it with gradients and hand it to AdvancedHMC/NUTS. MonteCarloX's own
samplers never need it.

```julia
alg    = MarkovChainMonteCarlo(rng, logposterior)          # dimension not needed
target = FunctionEnsemble(logposterior; dimension = 10)    # HMC-ready via LogDensityProblems
```
"""
struct FunctionEnsemble{F} <: AbstractEnsemble
    logweight::F
    linear::Bool
    dimension::Union{Int,Nothing}
end

function FunctionEnsemble(logweight; linear::Bool=false, dimension::Union{Integer,Nothing}=nothing)
    return FunctionEnsemble(logweight, linear, dimension === nothing ? nothing : Int(dimension))
end

linear_logweight(e::FunctionEnsemble) = e.linear

@inline logweight(e::FunctionEnsemble) = e.logweight
@inline logweight(e::FunctionEnsemble, arg) = e.logweight(arg)
