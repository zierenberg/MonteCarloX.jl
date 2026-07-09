using Random

"""
    WangLandauAlgorithm([rng,] bins_or_logweight; init=0.0, logf=1.0)

Create a [`MetropolisHastingsAlgorithm`](@ref) engine (Metropolis balance) with a `WangLandauEnsemble`
built from `bins_or_logweight`. Like multicanonical, Wang-Landau varies only the ensemble slot;
the online-weight adaptation lives in the custom [`accept!`](@ref) below, around the generic
accept step. If `rng` is omitted, the global RNG is used.
"""
function WangLandauAlgorithm(rng::AbstractRNG, bins_or_logweight; init::Real=0.0, logf::Real=1.0)
    ens = bins_or_logweight isa BinnedObject ?
          WangLandauEnsemble(bins_or_logweight; logf=logf) :
          WangLandauEnsemble(bins_or_logweight; init=init, logf=logf)
    return MetropolisHastingsAlgorithm(rng, ens)
end
WangLandauAlgorithm(bins_or_logweight; init::Real=0.0, logf::Real=1.0) =
    WangLandauAlgorithm(Random.GLOBAL_RNG, bins_or_logweight; init=init, logf=logf)

"""
    accept!(alg::MetropolisHastingsAlgorithm{<:WangLandauEnsemble}, arg_new, arg_old) -> Bool

Metropolis acceptance followed by the Wang-Landau adaptation: decrement the tabulated logweight
at the visited argument by `ens.logf`. `arg_new`/`arg_old` are the ensemble's logweight arguments
(typically the reaction coordinate), not the full state.
"""
function accept!(alg::MetropolisHastingsAlgorithm{<:WangLandauEnsemble}, arg_new::Real, arg_old::Real)
    ens = ensemble(alg)
    lw = logweight(alg)
    log_ratio = lw(arg_new) - lw(arg_old)
    accepted = accept!(alg, log_ratio)
    arg_vis = accepted ? arg_new : arg_old
    lw[arg_vis] -= ens.logf
    return accepted
end
