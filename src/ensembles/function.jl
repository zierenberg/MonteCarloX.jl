"""
    FunctionEnsemble(f; linear=false)

Wrap an arbitrary callable (e.g. Bayesian logdensity/logposterior function)
in an ensemble object.

Set `linear=true` to allow use with Metropolis-family algorithms,
asserting that `f(Δarg) == f(arg + Δarg) - f(arg)`. The caller is responsible
for ensuring this property holds.
"""
struct FunctionEnsemble{F} <: AbstractEnsemble
    f::F
    linear::Bool
end

FunctionEnsemble(f; linear::Bool=false) = FunctionEnsemble(f, linear)

linear_logweight(e::FunctionEnsemble) = e.linear

@inline logweight(e::FunctionEnsemble) = e.f
@inline logweight(e::FunctionEnsemble, arg) = e.f(arg)
