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
    run!(f!, pc::ParallelChains, results)

Run `f!(alg, results, i)` for each chain, writing results in-place.

- Threads: each thread calls `f!(alg, results, i)` for its chain index `i`.
- MPI: each rank calls `f!(alg, results, i)` with `i = rank(pc) + 1`.

The function `f!(alg, results, i)` receives the chain's algorithm, the result
container, and the 1-based chain index. It must store its output in-place.
"""
function run!(f!, pc::ParallelChains{ThreadsBackend}, results)
    n = size(pc)
    Threads.@threads for i in 1:n
        f!(algorithm(pc, i), results, i)
    end
    return results
end

function run!(f!, pc::ParallelChains{<:MPIBackend}, results)
    i = rank(pc) + 1
    f!(algorithm(pc), results, i)
    return results
end

# This is used as 
#all_results = [zeros(3) for _ in 1:size(pc)]

# run!(pc, all_results) do alg, results, i
#     x = randn(alg.rng)
#     Δ = 1.5

#     for _ in 1:burn_in
#         x_new = x + Δ * randn(alg.rng)
#         accept!(alg, x_new, x) && (x = x_new)
#     end
#     reset!(alg)

#     sum_x  = 0.0
#     sum_x2 = 0.0

#     for _ in 1:n_steps
#         x_new = x + Δ * randn(alg.rng)
#         accept!(alg, x_new, x) && (x = x_new)
#         sum_x  += x
#         sum_x2 += x^2
#     end

#     @inbounds begin
#         results[i][1] = sum_x
#         results[i][2] = sum_x2
#         results[i][3] = Float64(n_steps)
#     end
# end


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
