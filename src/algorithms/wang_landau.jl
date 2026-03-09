using Random

"""
    WangLandau([rng,] bins_or_logweight; init=0.0, logf=1.0)

Create a generic `ImportanceSampling` algorithm with a
`WangLandauEnsemble` built from `bins_or_logweight`.
If `rng` is not provided, the global RNG will be used.
"""
function WangLandau(rng::AbstractRNG, bins_or_logweight; init::Real=0.0, logf::Real=1.0)
    ens = bins_or_logweight isa BinnedObject ?
          WangLandauEnsemble(bins_or_logweight; logf=logf) :
          WangLandauEnsemble(bins_or_logweight; init=init, logf=logf)
    return ImportanceSampling(rng, ens)
end
WangLandau(bins_or_logweight; init::Real=0.0, logf::Real=1.0) =
    WangLandau(Random.GLOBAL_RNG, bins_or_logweight; init=init, logf=logf)

# access to the logweight object
@inline logweight(alg::ImportanceSampling{<:WangLandauEnsemble}) =
    logweight(ensemble(alg))

"""
    accept!(alg::ImportanceSampling{<:WangLandauEnsemble}, x_new, x_old)

Perform Metropolis acceptance and apply Wang-Landau local adaptation at the
visited state by updating the tabulated logweight.
"""
function accept!(alg::ImportanceSampling{<:WangLandauEnsemble}, x_new::Real, x_old::Real)
    ens = ensemble(alg)
    lw = logweight(alg)
    log_ratio = lw(x_new) - lw(x_old)
    accepted = _accept!(alg, log_ratio)
    x_vis = accepted ? x_new : x_old
    lw[x_vis] -= ens.logf
    return accepted
end

"""
    update!(alg::ImportanceSampling{<:WangLandauEnsemble})

Update Wang-Landau schedule by scaling the modification factor:
`logf <- power * logf` with default `power=0.5`.
"""
function update!(ens::WangLandauEnsemble; power::Real=0.5)
    ens.logf *= power
    return nothing
end

update!(alg::ImportanceSampling{<:WangLandauEnsemble}; kwargs...) =
    update!(ensemble(alg); kwargs...)
