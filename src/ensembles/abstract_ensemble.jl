"""
    AbstractEnsemble

Base type for all ensemble objects.
"""
abstract type AbstractEnsemble end

function Base.:(==)(a::T, b::T) where {T<:AbstractEnsemble}
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(T))
end

@inline _as_ensemble(e::AbstractEnsemble) = e
@inline _as_ensemble(e) = FunctionEnsemble(e)

"""
    linear_logweight(ens::AbstractEnsemble) -> Bool

Whether `logweight(ens, Δarg) == logweight(ens, arg + Δarg) - logweight(ens, arg)`
holds, i.e. the logweight is linear in its argument so that Metropolis-family
algorithms can work with argument differences alone.

Defaults to `false`. Ensembles that satisfy linearity must opt in.
"""
linear_logweight(::AbstractEnsemble) = false

"""
    logweight(ens::AbstractEnsemble)

Return a callable logweight object/function for an ensemble.
Concrete ensembles must implement this.
"""
function logweight(ens::AbstractEnsemble)
    throw(ArgumentError("logweight not implemented for ensemble type $(typeof(ens))"))
end

"""
    logweight(ens::AbstractEnsemble, arg)

Evaluate the ensemble's logweight on `arg` — the quantity the ensemble's
weight is parameterized by (e.g. energy for `BoltzmannEnsemble`, parameter
vector for a Bayesian `FunctionEnsemble`, reaction coordinate for a
`MulticanonicalEnsemble`). Concrete ensembles should provide this directly
or rely on a callable from `logweight(ens)`.
"""
function logweight(ens::AbstractEnsemble, arg)
    return logweight(ens)(arg)
end

# """
#     set!(ens::AbstractEnsemble, args...; kwargs...)

# Configure/modify an ensemble in-place.
# Concrete ensembles must specialize this when supported.
# """
# function set!(ens::AbstractEnsemble, args...; kwargs...)
#     throw(ArgumentError("set! not implemented for ensemble type $(typeof(ens))"))
# end

"""
    update!(ens::AbstractEnsemble, args...; kwargs...)

Perform in-place adaptation/update of an ensemble.
Concrete ensembles must specialize this when supported.
"""
function update!(ens::AbstractEnsemble, args...; kwargs...)
    throw(ArgumentError("update! not implemented for ensemble type $(typeof(ens))"))
end
