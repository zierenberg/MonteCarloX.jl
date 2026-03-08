using Random

"""
    Multicanonical(rng, bins; init=0.0)

Create a generic `ImportanceSampling` algorithm with a
`MulticanonicalLogWeight` built from `bins`.
"""
function Multicanonical(rng::AbstractRNG, bins; init::Real=0.0)
    lw = MulticanonicalLogWeight(bins; init=init)
    return ImportanceSampling(rng, lw)
end

"""
    Multicanonical(bins; init=0.0)

Global-RNG convenience constructor.
"""
Multicanonical(bins; init::Real=0.0) =
    Multicanonical(Random.GLOBAL_RNG, bins; init=init)

"""
    Multicanonical(rng, lw::MulticanonicalLogWeight)

Wrap an existing multicanonical logweight in a generic
`ImportanceSampling` algorithm.
"""
Multicanonical(rng::AbstractRNG, lw::MulticanonicalLogWeight) = ImportanceSampling(rng, lw)
Multicanonical(lw::MulticanonicalLogWeight) = ImportanceSampling(Random.GLOBAL_RNG, lw)

function record_visit!(lw::MulticanonicalLogWeight, accepted::Bool, x_new, x_old)
    x_vis = accepted ? x_new : x_old

    if x_vis isa Tuple
        idx_new = _binindex(lw.histogram.bins, x_vis)
        if all(d -> (1 <= idx_new[d] <= size(lw.histogram.values, d)), 1:length(idx_new))
            lw.histogram[x_vis...] += 1
        end
    else
        idx_new = _binindex(lw.histogram.bins[1], x_vis)
        if 1 <= idx_new <= size(lw.histogram.values, 1)
            lw.histogram[x_vis] += 1
        end
    end

    return nothing
end

function reset!(alg::ImportanceSampling{<:MulticanonicalLogWeight})
    fill!(alg.logweight.histogram.values, zero(eltype(alg.logweight.histogram.values)))
    alg.steps = 0
    alg.accepted = 0
    return nothing
end

@inline set_logweight!(alg::ImportanceSampling{<:MulticanonicalLogWeight}, xrange, f::Function) = set!(alg.logweight, xrange, f)

@inline update_logweight!(alg::ImportanceSampling{<:MulticanonicalLogWeight}; mode=:simple) = update!(alg.logweight; mode=mode)