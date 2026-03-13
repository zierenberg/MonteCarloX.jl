"""
    AbstractEnsemble

Base type for all ensemble objects.
"""
abstract type AbstractEnsemble end

@inline _as_ensemble(e::AbstractEnsemble) = e
@inline _as_ensemble(e) = FunctionEnsemble(e)

"""
    logweight(ens::AbstractEnsemble)

Return a callable logweight object/function for an ensemble.
Concrete ensembles must implement this.
"""
function logweight(ens::AbstractEnsemble)
    throw(ArgumentError("logweight not implemented for ensemble type $(typeof(ens))"))
end

"""
    logweight(ens::AbstractEnsemble, x)

Evaluate logweight on state/value `x`.
Concrete ensembles should provide this directly or rely on a callable from
`logweight(ens)`.
"""
function logweight(ens::AbstractEnsemble, x)
    return logweight(ens)(x)
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
