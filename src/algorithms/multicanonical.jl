"""
    Multicanonical([rng,] bins; init=0.0)

Create a generic `ImportanceSampling` algorithm with a
`MulticanonicalEnsemble` built from `bins`.

If `rng` is not provided, the global RNG will be used.
"""
function Multicanonical(rng::AbstractRNG, bins; init::Real=0.0)
    ens = bins isa BinnedObject ? MulticanonicalEnsemble(bins) : MulticanonicalEnsemble(bins; init=init)
    return ImportanceSampling(rng, ens)
end
Multicanonical(bins; init::Real=0.0) =
    Multicanonical(Random.GLOBAL_RNG, bins; init=init)

"""
    Multicanonical([rng,] ens::MulticanonicalEnsemble)

Wrap an existing multicanonical logweight in a generic
`ImportanceSampling` algorithm.

If `rng` is not provided, the global RNG will be used.
"""
Multicanonical(rng::AbstractRNG, ens::MulticanonicalEnsemble) = ImportanceSampling(rng, ens)
Multicanonical(ens::MulticanonicalEnsemble) = Multicanonical(Random.GLOBAL_RNG, ens)

function reset!(alg::ImportanceSampling{<:MulticanonicalEnsemble})
    ens = ensemble(alg)
    h = ens.histogram
    fill!(h.values, zero(eltype(h.values)))
    _reset!(alg) # reset acceptance stats
    return nothing
end

