"""
    ParallelChains{B, A}

Coordinator for independent parallel Markov chains that sample the same
target distribution. The backend type parameter `B` determines the
parallelism strategy:

- `ParallelChains{ThreadsBackend}`: local vector of algorithms, shared memory.
- `ParallelChains{MPIBackend}`: one algorithm per MPI rank, MPI communication.
"""

using MPI

struct ParallelChains{B, A}
    backend::B
    alg::A # This is either a vector of algorithms (Threads) or a single algorithm (MPI)
end

# access the algorithm(s)
@inline algorithm(pc::ParallelChains{ThreadsBackend}) = pc.alg
@inline algorithm(pc::ParallelChains{ThreadsBackend}, i::Integer) = pc.alg[i]
@inline algorithm(pc::ParallelChains{<:MPIBackend}) = pc.alg
@inline algorithm(pc::ParallelChains{<:MPIBackend}, ::Integer) = pc.alg

@inline rank(pc::ParallelChains) = rank(pc.backend)
@inline size(pc::ParallelChains) = size(pc.backend)
@inline is_root(pc::ParallelChains) = is_root(pc.backend)

# 1-based chain index of the root chain.
# Threads: conventionally chain 1. MPI: index corresponding to backend.root.
@inline root_chain(::ParallelChains{ThreadsBackend}) = 1
@inline root_chain(pc::ParallelChains{<:MPIBackend}) = pc.backend.root + 1

"""
    on_root(f, pc::ParallelChains)

Run a block on the root chain only; no-op on non-root ranks.

The callback may take zero or one argument.  When one argument is
accepted it receives the root chain index, useful for fetching
the algorithm:

    on_root(pmuca) do i
        alg = algorithm(pmuca, i)
        update!(ensemble(alg); mode=:simple)
    end

    on_root(pt) do
        println("done")
    end
"""
@inline function on_root(f, pc::ParallelChains{ThreadsBackend})
    i = root_chain(pc)
    return applicable(f, i) ? f(i) : f()
end

@inline function on_root(f, pc::ParallelChains{<:MPIBackend})
    if !is_root(pc)
        return nothing
    end
    i = root_chain(pc)
    return applicable(f, i) ? f(i) : f()
end

function ParallelChains(backend::ThreadsBackend, alg::AbstractVector{<:AbstractAlgorithm})
    length(alg) == size(backend) || throw(ArgumentError(
        "number of algorithms ($(length(alg))) must match backend size ($(size(backend)))"))
    collected = collect(alg)
    return ParallelChains{ThreadsBackend, typeof(collected)}(backend, collected)
end

function ParallelChains(backend::MPIBackend, alg::AbstractAlgorithm)
    return ParallelChains{typeof(backend), typeof(alg)}(backend, alg)
end


"""
    with_parallel(f, pc::ParallelChains)

Run `f` for each chain in parallel.

- Threads: calls `f(i, alg)` for each chain via `Threads.@threads`.
- MPI: calls `f(alg)` once on the local rank.

Note the different callback signatures: threads receives `(i, alg)` because
shared-memory code needs the chain index for storage; MPI receives `(alg)` only
because each rank owns exactly one chain.

    # Threads
    with_parallel(pc) do i, alg
        ...
    end

    # MPI
    with_parallel(pc) do alg
        ...
    end
"""
function with_parallel(f, pc::ParallelChains{ThreadsBackend})
    n = size(pc)
    Threads.@threads for i in 1:n
        f(i, algorithm(pc, i))
    end
    return nothing
end

function with_parallel(f, pc::ParallelChains{<:MPIBackend})
    f(algorithm(pc))
    return nothing
end


"""
    merge!(values, op, pc::ParallelChains)

Reduce `values` across all chains with binary operator `op`.

- Threads: `values` is a collection of per-chain results; returns `reduce(op, values)`.
- MPI: `values` is the local rank's array; performs in-place `MPI.Allreduce!`.
"""
function merge!(values, op, pc::ParallelChains{ThreadsBackend})
    return reduce(op, values)
end

function merge!(values::AbstractArray, op, pc::ParallelChains{<:MPIBackend})
    MPI.Allreduce!(values, op, pc.backend.comm)
    return values
end
