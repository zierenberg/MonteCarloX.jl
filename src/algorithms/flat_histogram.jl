# ── Flat-histogram methods (multicanonical, Wang-Landau) ──────────────────────
#
# Flat-histogram sampling varies only the ENSEMBLE slot of the Metropolis-Hastings engine: the
# target is a tabulated logweight over a binned reaction coordinate, adapted until the recorded
# histogram is flat. The two classic members differ in WHEN they adapt:
#
#   Multicanonical — offline: sample with fixed weights, then refine them between iterations
#                    (update_logweight!; simple W -= log H or the error-weighted recursive rule)
#   Wang-Landau    — online: decrement the visited bin by logf after EVERY accept step, with a
#                    schedule that shrinks logf toward zero
#
# The weight tables, histograms, and their transformations (set_logweight!, update_logweight!,
# smooth!, flatness, extend!) live with the ensembles (src/ensembles/multicanonical.jl,
# wang_landau.jl); here are only the named constructors and the engine hooks.

"""
    MulticanonicalAlgorithm([rng,] bins; init=0.0, kwargs...)
    MulticanonicalAlgorithm([rng,] ens::MulticanonicalEnsemble)

[`MetropolisHastingsAlgorithm`](@ref) engine with a `MulticanonicalEnsemble` built from `bins`
(extra keywords are forwarded to the ensemble constructor) or wrapped from an existing ensemble.
Iterate: sample, `update_logweight!(ensemble(alg))`, `reset!(alg)`.
"""
function MulticanonicalAlgorithm(rng::AbstractRNG, bins; kwargs...)
    return MetropolisHastingsAlgorithm(rng, MulticanonicalEnsemble(bins; kwargs...))
end
MulticanonicalAlgorithm(bins; kwargs...) = MulticanonicalAlgorithm(Random.GLOBAL_RNG, bins; kwargs...)
MulticanonicalAlgorithm(rng::AbstractRNG, ens::MulticanonicalEnsemble) = MetropolisHastingsAlgorithm(rng, ens)
MulticanonicalAlgorithm(ens::MulticanonicalEnsemble) = MulticanonicalAlgorithm(Random.GLOBAL_RNG, ens)

function reset!(alg::MetropolisHastingsAlgorithm{<:MulticanonicalEnsemble})
    ens = ensemble(alg)
    h = ens.histogram
    fill!(h.values, zero(eltype(h.values)))
    _reset!(alg) # reset acceptance stats
    return nothing
end

"""
    WangLandauAlgorithm([rng,] bins_or_logweight; init=0.0, logf=1.0)

[`MetropolisHastingsAlgorithm`](@ref) engine with a `WangLandauEnsemble` built from
`bins_or_logweight`. The online weight adaptation lives in the custom [`accept!`](@ref) below;
shrink the modification factor with `update_logweight!(ensemble(alg); power)`.
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

# ── Parallel multicanonical: independent chains, merged weight refinement ────


"""
    ParallelMulticanonical(backend, alg)

Create `ParallelChains` for multicanonical algorithms.
Validates that the algorithm(s) carry a `MulticanonicalEnsemble`.
"""
function ParallelMulticanonical(backend::ThreadsBackend,
                                alg::AbstractVector{<:MetropolisHastingsAlgorithm{<:MulticanonicalEnsemble}})
    return ParallelChains(backend, alg)
end

function ParallelMulticanonical(backend::MPIBackend,
                                alg::MetropolisHastingsAlgorithm{<:MulticanonicalEnsemble})
    return ParallelChains(backend, alg)
end

"""
    merge_histograms!(pc)

Sum histograms across all chains into the root chain. After this call only the
root holds the merged histogram; other chains' buffers are unchanged.
Use `distribute_logweight!` to propagate the refined weights back.
"""
function merge_histograms!(pc::ParallelChains{ThreadsBackend, <:Vector{<:MetropolisHastingsAlgorithm{<:MulticanonicalEnsemble}}})
    n = size(pc)
    r = root_chain(pc)
    h_root = ensemble(algorithm(pc, r)).histogram.values
    for i in 1:n
        i == r && continue
        h_root .+= ensemble(algorithm(pc, i)).histogram.values
    end
    return nothing
end

function merge_histograms!(pc::ParallelChains{<:MPIBackend, <:MetropolisHastingsAlgorithm{<:MulticanonicalEnsemble}})
    MPI.Reduce!(ensemble(algorithm(pc)).histogram.values, +, pc.backend.comm; root=pc.backend.root)
    return nothing
end

"""
    distribute_logweight!(pc)

Broadcast logweights from the root chain to all other chains.
"""
function distribute_logweight!(pc::ParallelChains{ThreadsBackend, <:Vector{<:MetropolisHastingsAlgorithm{<:MulticanonicalEnsemble}}})
    r = root_chain(pc)
    root_lw = ensemble(algorithm(pc, r)).logweight.values
    n = size(pc)
    for i in 1:n
        i == r && continue
        ensemble(algorithm(pc, i)).logweight.values .= root_lw
    end
    return nothing
end

function distribute_logweight!(pc::ParallelChains{<:MPIBackend, <:MetropolisHastingsAlgorithm{<:MulticanonicalEnsemble}})
    MPI.Bcast!(ensemble(algorithm(pc)).logweight.values, pc.backend.root, pc.backend.comm)
    return nothing
end
