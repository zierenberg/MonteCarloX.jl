"""
    FunctionEnsemble

Wrap an arbitrary callable (e.g. Bayesian logdensity/logposterior function)
in an ensemble object.
"""
struct FunctionEnsemble{F} <: AbstractEnsemble
    f::F
end

@inline logweight(e::FunctionEnsemble) = e.f
@inline logweight(e::FunctionEnsemble, x) = e.f(x)
