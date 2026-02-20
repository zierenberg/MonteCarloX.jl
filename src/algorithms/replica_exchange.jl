using Random

mutable struct ReplicaExchange{PS<:ParallelSampling,RNG<:AbstractRNG} <: AbstractAlgorithm
    rng::RNG
    parallel::PS
    exchange_attempts::Int
    exchange_accepted::Int
end

function ReplicaExchange(
    replicas::AbstractVector{<:AbstractImportanceSampling};
    rng::AbstractRNG = Random.GLOBAL_RNG,
    rank::Integer = 0,
    comm_size::Integer = 1,
    local_indices::AbstractVector{<:Integer} = collect(1:length(replicas)),
)
    parallel = ParallelSampling(
        replicas;
        rank=rank,
        comm_size=comm_size,
        local_indices=local_indices,
    )

    return ReplicaExchange(
        rng,
        parallel,
        0,
        0,
    )
end

replica_count(alg::ReplicaExchange) = replica_count(alg.parallel)
local_replica_count(alg::ReplicaExchange) = local_replica_count(alg.parallel)

exchange_rate(alg::ReplicaExchange) =
    alg.exchange_attempts > 0 ? alg.exchange_accepted / alg.exchange_attempts : 0.0

function reset_exchange_statistics!(alg::ReplicaExchange)
    alg.exchange_attempts = 0
    alg.exchange_accepted = 0
    return nothing
end

@inline function replica_exchange_log_ratio(
    alg_i::AbstractImportanceSampling,
    alg_j::AbstractImportanceSampling,
    state_i,
    state_j,
)
    return alg_i.logweight(state_j) + alg_j.logweight(state_i) -
           alg_i.logweight(state_i) - alg_j.logweight(state_j)
end

function attempt_exchange!(
    alg::ReplicaExchange,
    states::AbstractVector,
    idx_i::Integer,
    idx_j::Integer,
)
    n = replica_count(alg)
    length(states) == n || throw(ArgumentError("`states` length must match number of replicas"))
    (1 <= idx_i <= n && 1 <= idx_j <= n) || throw(BoundsError(states, (idx_i, idx_j)))

    replica_i = alg.parallel.replicas[idx_i]
    replica_j = alg.parallel.replicas[idx_j]
    state_i = states[idx_i]
    state_j = states[idx_j]

    log_ratio = replica_exchange_log_ratio(replica_i, replica_j, state_i, state_j)

    alg.exchange_attempts += 1
    accepted = log_ratio > 0 || rand(alg.rng) < exp(log_ratio)
    if accepted
        states[idx_i], states[idx_j] = states[idx_j], states[idx_i]
        alg.exchange_accepted += 1
    end

    return accepted
end

function sweep_exchanges!(
    alg::ReplicaExchange,
    states::AbstractVector;
    phase::Integer = 0,
)
    accepted = 0
    for (idx_i, idx_j) in neighbor_exchange_pairs(replica_count(alg), phase)
        accepted += attempt_exchange!(alg, states, idx_i, idx_j)
    end
    return accepted
end
