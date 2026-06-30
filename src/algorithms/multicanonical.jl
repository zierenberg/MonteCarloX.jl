"""
    Multicanonical([rng,] bins; init=0.0, kwargs...)

Create a generic `MarkovChainMonteCarlo` algorithm with a
`MulticanonicalEnsemble` built from `bins`.

Extra keyword arguments (`warn_overwrite`, `smooth_window`, etc.) are
forwarded to the `MulticanonicalEnsemble` constructor.

If `rng` is not provided, the global RNG will be used.
"""
function Multicanonical(rng::AbstractRNG, bins; kwargs...)
    return MarkovChainMonteCarlo(rng, MulticanonicalEnsemble(bins; kwargs...))
end
Multicanonical(bins; kwargs...) = Multicanonical(Random.GLOBAL_RNG, bins; kwargs...)

"""
    Multicanonical([rng,] ens::MulticanonicalEnsemble)

Wrap an existing multicanonical logweight in a generic
`MarkovChainMonteCarlo` algorithm.

If `rng` is not provided, the global RNG will be used.
"""
Multicanonical(rng::AbstractRNG, ens::MulticanonicalEnsemble) = MarkovChainMonteCarlo(rng, ens)
Multicanonical(ens::MulticanonicalEnsemble) = Multicanonical(Random.GLOBAL_RNG, ens)

function reset!(alg::MarkovChainMonteCarlo{<:MulticanonicalEnsemble})
    ens = ensemble(alg)
    h = ens.histogram
    fill!(h.values, zero(eltype(h.values)))
    _reset!(alg) # reset acceptance stats
    return nothing
end