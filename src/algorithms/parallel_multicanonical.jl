"""
    ParallelMulticanonical

Parallel multicanonical sampling: independent chains with multicanonical
ensembles, plus `merge_histograms!` and `distribute_logweight!` for
iterative weight refinement.

`ParallelMulticanonical` is not a separate type — it is a `ParallelChains`
whose algorithms carry `MulticanonicalEnsemble`. The constructor and
convenience functions dispatch on this type signature.
"""

using MPI

"""
    ParallelMulticanonical(backend, alg)

Create `ParallelChains` for multicanonical algorithms.
Validates that the algorithm(s) carry a `MulticanonicalEnsemble`.
"""
function ParallelMulticanonical(backend::ThreadsBackend,
                                alg::AbstractVector{<:ImportanceSampling{<:MulticanonicalEnsemble}})
    return ParallelChains(backend, alg)
end

function ParallelMulticanonical(backend::MPIBackend,
                                alg::ImportanceSampling{<:MulticanonicalEnsemble})
    return ParallelChains(backend, alg)
end

"""
    merge_histograms!(pc)

Sum histograms across all chains. After this call every chain (or rank)
holds the merged histogram.
"""
function merge_histograms!(pc::ParallelChains{ThreadsBackend, <:Vector{<:ImportanceSampling{<:MulticanonicalEnsemble}}})
    n = size(pc)
    merged = sum(ensemble(algorithm(pc, i)).histogram.values for i in 1:n)
    for i in 1:n
        ensemble(algorithm(pc, i)).histogram.values .= merged
    end
    return nothing
end

function merge_histograms!(pc::ParallelChains{<:MPIBackend, <:ImportanceSampling{<:MulticanonicalEnsemble}})
    MPI.Reduce!(ensemble(algorithm(pc)).histogram.values, +, pc.backend.comm; root=pc.backend.root)
    return nothing
end

"""
    distribute_logweight!(pc)

Broadcast logweights from root to all chains.
"""
function distribute_logweight!(pc::ParallelChains{ThreadsBackend, <:Vector{<:ImportanceSampling{<:MulticanonicalEnsemble}}})
    root_lw = ensemble(algorithm(pc, 1)).logweight.values
    n = size(pc)
    for i in 2:n
        ensemble(algorithm(pc, i)).logweight.values .= root_lw
    end
    return nothing
end

function distribute_logweight!(pc::ParallelChains{<:MPIBackend, <:ImportanceSampling{<:MulticanonicalEnsemble}})
    MPI.Bcast!(ensemble(algorithm(pc)).logweight.values, pc.backend.root, pc.backend.comm)
    return nothing
end
