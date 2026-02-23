using Random

mutable struct ParallelTempering{PS<:ParallelSampling,RNG<:AbstractRNG} <: AbstractAlgorithm
    rng::RNG
    parallel::PS
    exchange_attempts::Int
    exchange_accepted::Int
end

function ParallelTempering(
    replicas::AbstractVector{<:AbstractMetropolis};
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

    return ParallelTempering(
        rng,
        parallel,
        0,
        0,
    )
end

replica_count(alg::ParallelTempering) = replica_count(alg.parallel)
local_replica_count(alg::ParallelTempering) = local_replica_count(alg.parallel)

exchange_rate(alg::ParallelTempering) =
    alg.exchange_attempts > 0 ? alg.exchange_accepted / alg.exchange_attempts : 0.0

function reset_exchange_statistics!(alg::ParallelTempering)
    alg.exchange_attempts = 0
    alg.exchange_accepted = 0
    return nothing
end

@inline function exchange_log_ratio(
    alg_i::AbstractMetropolis,
    alg_j::AbstractMetropolis,
    energy_i::Real,
    energy_j::Real,
)
    β_i = inverse_temperature(alg_i)
    β_j = inverse_temperature(alg_j)
    return (energy_i - energy_j) * (β_i - β_j)
end

@inline function inverse_temperature(alg::AbstractMetropolis)
    lw = alg.logweight
    if lw isa BoltzmannLogWeight
        return lw.β
    end
    throw(ArgumentError("ParallelTempering exchange requires BoltzmannLogWeight to extract β"))
end

function attempt_exchange!(
    alg::ParallelTempering,
    energies::AbstractVector{<:Real},
    idx_i::Integer,
    idx_j::Integer,
)
    n = replica_count(alg)
    length(energies) == n || throw(ArgumentError("`energies` length must match number of replicas"))
    (1 <= idx_i <= n && 1 <= idx_j <= n) || throw(BoundsError(energies, (idx_i, idx_j)))

    replica_i = alg.parallel.replicas[idx_i]
    replica_j = alg.parallel.replicas[idx_j]
    energy_i = energies[idx_i]
    energy_j = energies[idx_j]

    log_ratio = exchange_log_ratio(replica_i, replica_j, energy_i, energy_j)

    alg.exchange_attempts += 1
    accepted = log_ratio > 0 || rand(alg.rng) < exp(log_ratio)
    if accepted
        energies[idx_i], energies[idx_j] = energies[idx_j], energies[idx_i]
        alg.exchange_accepted += 1
    end

    return accepted
end

function sweep_exchanges!(
    alg::ParallelTempering,
    energies::AbstractVector{<:Real};
    phase::Integer = 0,
)
    accepted = 0
    for (idx_i, idx_j) in neighbor_exchange_pairs(replica_count(alg), phase)
        accepted += attempt_exchange!(alg, energies, idx_i, idx_j)
    end
    return accepted
end
