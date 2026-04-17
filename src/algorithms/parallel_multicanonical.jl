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

Sum histograms across all chains into the root chain. After this call only the
root holds the merged histogram; other chains' buffers are unchanged.
Use `distribute_logweight!` to propagate the refined weights back.
"""
function merge_histograms!(pc::ParallelChains{ThreadsBackend, <:Vector{<:ImportanceSampling{<:MulticanonicalEnsemble}}})
    n = size(pc)
    r = root_chain(pc)
    h_root = ensemble(algorithm(pc, r)).histogram.values
    for i in 1:n
        i == r && continue
        h_root .+= ensemble(algorithm(pc, i)).histogram.values
    end
    return nothing
end

function merge_histograms!(pc::ParallelChains{<:MPIBackend, <:ImportanceSampling{<:MulticanonicalEnsemble}})
    MPI.Reduce!(ensemble(algorithm(pc)).histogram.values, +, pc.backend.comm; root=pc.backend.root)
    return nothing
end

"""
    distribute_logweight!(pc)

Broadcast logweights from the root chain to all other chains.
"""
function distribute_logweight!(pc::ParallelChains{ThreadsBackend, <:Vector{<:ImportanceSampling{<:MulticanonicalEnsemble}}})
    r = root_chain(pc)
    root_lw = ensemble(algorithm(pc, r)).logweight.values
    n = size(pc)
    for i in 1:n
        i == r && continue
        ensemble(algorithm(pc, i)).logweight.values .= root_lw
    end
    return nothing
end

function distribute_logweight!(pc::ParallelChains{<:MPIBackend, <:ImportanceSampling{<:MulticanonicalEnsemble}})
    MPI.Bcast!(ensemble(algorithm(pc)).logweight.values, pc.backend.root, pc.backend.comm)
    return nothing
end
