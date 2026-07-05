"""
    FunctionEnsemble(logweight; linear=false)

Wrap a log-weight callable in an ensemble. `logweight` maps a coordinate to an
**un-normalized** log weight — e.g. a Bayesian log-posterior, or `x -> -β*E(x)`.
It need not integrate to 1: that is exactly what separates a log-weight from a
`logpdf`, and what `log_normalization`/`ess` recover.

Set `linear=true` to allow use with Metropolis-family algorithms, asserting that
`logweight(Δarg) == logweight(arg + Δarg) - logweight(arg)`. The caller is
responsible for ensuring this property holds.
"""
struct FunctionEnsemble{F} <: AbstractEnsemble
    logweight::F
    linear::Bool
end

FunctionEnsemble(logweight; linear::Bool=false) = FunctionEnsemble(logweight, linear)

linear_logweight(e::FunctionEnsemble) = e.linear

@inline logweight(e::FunctionEnsemble) = e.logweight
@inline logweight(e::FunctionEnsemble, arg) = e.logweight(arg)
