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
    accept!(alg::ImportanceSampling{<:WangLandauEnsemble}, arg_new, arg_old) -> Bool

Perform Metropolis acceptance and apply Wang-Landau local adaptation at the
visited argument by decrementing the tabulated logweight by `ens.logf`.
See [`accept!`](@ref) on `AbstractImportanceSampling` for the meaning of
`arg_new`/`arg_old` (the ensemble's `logweight` argument, typically the
reaction coordinate, not the full state).
"""
function accept!(alg::ImportanceSampling{<:WangLandauEnsemble}, arg_new::Real, arg_old::Real)
    ens = ensemble(alg)
    lw = logweight(alg)
    log_ratio = lw(arg_new) - lw(arg_old)
    accepted = _accept!(alg, log_ratio)
    arg_vis = accepted ? arg_new : arg_old
    lw[arg_vis] -= ens.logf
    return accepted
end