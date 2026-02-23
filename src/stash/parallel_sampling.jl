mutable struct ParallelSampling{A<:AbstractAlgorithm,V<:AbstractVector{A}} <: AbstractAlgorithm
    replicas::V
    rank::Int
    comm_size::Int
    local_indices::Vector{Int}
end

function ParallelSampling(
    replicas::AbstractVector{<:AbstractAlgorithm};
    rank::Integer = 0,
    comm_size::Integer = 1,
    local_indices::AbstractVector{<:Integer} = collect(1:length(replicas)),
)
    rank < 0 && throw(ArgumentError("`rank` must be non-negative"))
    comm_size < 1 && throw(ArgumentError("`comm_size` must be >= 1"))
    isempty(replicas) && throw(ArgumentError("`replicas` must not be empty"))

    indices = Int.(collect(local_indices))
    for idx in indices
        1 <= idx <= length(replicas) || throw(ArgumentError("`local_indices` contains out-of-range index $(idx)"))
    end

    return ParallelSampling(replicas, Int(rank), Int(comm_size), indices)
end

replica_count(ps::ParallelSampling) = length(ps.replicas)
local_replica_count(ps::ParallelSampling) = length(ps.local_indices)

function neighbor_exchange_pairs(nreplicas::Integer, phase::Integer = 0)
    nreplicas < 1 && throw(ArgumentError("`nreplicas` must be >= 1"))

    start_idx = iseven(phase) ? 1 : 2
    pairs = Tuple{Int,Int}[]
    for idx in start_idx:2:(nreplicas - 1)
        push!(pairs, (idx, idx + 1))
    end
    return pairs
end
