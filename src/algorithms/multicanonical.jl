"""
    MulticanonicalAlgorithm([rng,] bins; init=0.0, kwargs...)

Create a [`MetropolisHastingsAlgorithm`](@ref) engine (Metropolis balance) with a
`MulticanonicalEnsemble` built from `bins`. Multicanonical sampling varies only the ensemble
slot; the histogram bookkeeping lives in the ensemble, around the generic accept step.

Extra keyword arguments (`warn_overwrite`, `smooth_window`, …) are forwarded to the
`MulticanonicalEnsemble` constructor. If `rng` is omitted, the global RNG is used.
"""
function MulticanonicalAlgorithm(rng::AbstractRNG, bins; kwargs...)
    return MetropolisHastingsAlgorithm(rng, MulticanonicalEnsemble(bins; kwargs...))
end
MulticanonicalAlgorithm(bins; kwargs...) = MulticanonicalAlgorithm(Random.GLOBAL_RNG, bins; kwargs...)

"""
    MulticanonicalAlgorithm([rng,] ens::MulticanonicalEnsemble)

Wrap an existing multicanonical logweight in a [`MetropolisHastingsAlgorithm`](@ref) engine.
If `rng` is omitted, the global RNG is used.
"""
MulticanonicalAlgorithm(rng::AbstractRNG, ens::MulticanonicalEnsemble) = MetropolisHastingsAlgorithm(rng, ens)
MulticanonicalAlgorithm(ens::MulticanonicalEnsemble) = MulticanonicalAlgorithm(Random.GLOBAL_RNG, ens)

function reset!(alg::MetropolisHastingsAlgorithm{<:MulticanonicalEnsemble})
    ens = ensemble(alg)
    h = ens.histogram
    fill!(h.values, zero(eltype(h.values)))
    _reset!(alg) # reset acceptance stats
    return nothing
end
