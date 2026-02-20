# WORK-IN-PROGRESS: right now it is not really helping yet.

mutable struct ParallelMulticanonical{PS<:ParallelSampling} <: AbstractAlgorithm
    parallel::PS
    master_rank::Int
end

function ParallelMulticanonical(
    replicas::AbstractVector{<:AbstractGeneralizedEnsemble};
    rank::Integer = 0,
    comm_size::Integer = 1,
    local_indices::AbstractVector{<:Integer} = collect(1:length(replicas)),
    master_rank::Integer = 0,
)
    parallel = ParallelSampling(
        replicas;
        rank=rank,
        comm_size=comm_size,
        local_indices=local_indices,
    )
    0 <= master_rank < comm_size || throw(ArgumentError("`master_rank` must satisfy 0 <= master_rank < comm_size"))

    return ParallelMulticanonical(
        parallel,
        Int(master_rank),
    )
end

replica_count(alg::ParallelMulticanonical) = replica_count(alg.parallel)
local_replica_count(alg::ParallelMulticanonical) = local_replica_count(alg.parallel)

function _merge_histograms(histograms::AbstractVector{<:Histogram})
    isempty(histograms) && throw(ArgumentError("`histograms` must not be empty"))

    merged = deepcopy(histograms[1])
    merged.weights .= zero(eltype(merged.weights))

    for hist in histograms
        hist.edges == merged.edges || throw(ArgumentError("all histograms must have identical bin edges"))
        merged.weights .+= hist.weights
    end

    return merged
end

function _distribute_logweight!(
    alg::ParallelMulticanonical,
    source_logweight::TabulatedLogWeight,
)
    source_hist = source_logweight.histogram
    for replica in alg.parallel.replicas
        replica.logweight isa TabulatedLogWeight ||
            throw(ArgumentError("all replicas must use `TabulatedLogWeight`"))
        target_hist = replica.logweight.histogram
        _assert_same_histogram_bins(target_hist, source_hist)
        _assert_same_shape(target_hist.weights, source_hist.weights)
        target_hist.weights .= source_hist.weights
    end

    return nothing
end

function update_weights!(
    alg::ParallelMulticanonical,
    histograms::AbstractVector{<:Histogram};
    mode::Symbol = :simple,
    master_replica_index::Integer = 1,
)
    n = replica_count(alg)
    length(histograms) == n || throw(ArgumentError("`histograms` length must match number of replicas"))
    1 <= master_replica_index <= n || throw(ArgumentError("`master_replica_index` is out of range"))

    merged = _merge_histograms(histograms)

    master_replica = alg.parallel.replicas[master_replica_index]
    update_weights!(master_replica, merged; mode=mode)

    _distribute_logweight!(alg, master_replica.logweight)

    return nothing
end
