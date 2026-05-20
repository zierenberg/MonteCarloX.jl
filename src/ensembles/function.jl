"""
    FunctionEnsemble(f; linear=false)

Wrap an arbitrary callable (e.g. Bayesian logdensity/logposterior function)
in an ensemble object.

Set `linear=true` to allow use with Metropolis-family algorithms,
asserting that `f(Δx) == f(x + Δx) - f(x)`. The caller is responsible for
ensuring this property holds.
"""
struct FunctionEnsemble{F} <: AbstractEnsemble
    f::F
    linear::Bool
end

FunctionEnsemble(f; linear::Bool=false) = FunctionEnsemble(f, linear)

linear_logweight(e::FunctionEnsemble) = e.linear

@inline logweight(e::FunctionEnsemble) = e.f
@inline logweight(e::FunctionEnsemble, x) = e.f(x)
