"""
    AbstractEnsemble

Base type for all ensemble objects. An ensemble assigns a log-weight to the coordinate its
sampling is parameterized by — an energy (`BoltzmannEnsemble`), a parameter vector (a Bayesian
`FunctionEnsemble`), a binned reaction coordinate (`MulticanonicalEnsemble`).
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

Return the ensemble's logweight as a read-only callable (for tabulated ensembles: the
underlying binned object). To modify weights, use [`set_logweight!`](@ref) and
[`update_logweight!`](@ref).
"""
function logweight(ens::AbstractEnsemble)
    throw(ArgumentError("logweight not implemented for ensemble type $(typeof(ens))"))
end

"""
    logweight(ens::AbstractEnsemble, arg)

Evaluate the ensemble's logweight on `arg` — the quantity the ensemble's weight is
parameterized by (energy for `BoltzmannEnsemble`, parameter vector for a Bayesian
`FunctionEnsemble`, reaction coordinate for a `MulticanonicalEnsemble`).
"""
function logweight(ens::AbstractEnsemble, arg)
    return logweight(ens)(arg)
end

"""
    set_logweight!(ens::AbstractEnsemble, args...)

Overwrite (part of) an ensemble's tabulated logweight. Provided by adaptive ensembles
(multicanonical); see their methods for the accepted arguments.
"""
function set_logweight! end

"""
    update_logweight!(ens::AbstractEnsemble; kwargs...)

Adapt an ensemble's logweight in place from its recorded statistics (multicanonical: refine
the weights from the histogram; Wang-Landau: shrink the modification factor).
"""
function update_logweight! end

# Optional visit hooks used by the two-argument accept! of the Metropolis-Hastings engine.
# Ensembles that need histogram/visit bookkeeping (multicanonical) specialize these.
@inline should_record_visit(ens) = false
@inline record_visit!(_ens, _arg_vis) = nothing
