"""
    ReplicaExchange

Generic replica-exchange coordinator from either a message backend
(e.g. `MPIBackend`, `DistributedBackend`) or a local vector of per-replica
algorithms.
"""

abstract type AbstractReplicaExchange <: AbstractAlgorithm end

mutable struct ReplicaExchangeMessage <: AbstractReplicaExchange
    alg::AbstractImportanceSampling
    backend::AbstractMessageBackend
    root::Int
    stage::Int
    indices::Vector{Int}
    steps::Vector{Int}
    accepted::Vector{Int}
end

mutable struct ReplicaExchangeVector{A<:AbstractImportanceSampling} <: AbstractReplicaExchange
    alg::Vector{A}
    stage::Int
    indices::Vector{Int}
    steps::Vector{Int}
    accepted::Vector{Int}
end

function ReplicaExchangeMessage(alg::AbstractImportanceSampling, backend::AbstractMessageBackend; root::Int=0)
    n = size(backend)
    0 <= root < n || throw(ArgumentError("`root` must satisfy 0 <= root < size"))
    indices = collect(1:n)
    nedges = max(0, n - 1)
    return ReplicaExchangeMessage(alg, backend, Int(root), 0, indices, zeros(Int, nedges), zeros(Int, nedges))
end

function ReplicaExchangeVector(algs::AbstractVector{<:AbstractImportanceSampling})
    n = length(algs)
    n >= 2 || throw(ArgumentError("need at least 2 local algorithms for replica exchange"))
    indices = collect(1:n)
    nedges = max(0, n - 1)
    return ReplicaExchangeVector(collect(algs), 0, indices, zeros(Int, nedges), zeros(Int, nedges))
end

# alg-first, backend second — canonical order matching ParallelMulticanonical
ReplicaExchange(alg::AbstractImportanceSampling, backend::AbstractMessageBackend; root::Int=0) =
    ReplicaExchangeMessage(alg, backend; root=root)
ReplicaExchange(alg::AbstractImportanceSampling, comm; root::Int=0) =
    ReplicaExchangeMessage(alg, MPIBackend(comm); root=root)
ReplicaExchange(algs::AbstractVector{<:AbstractImportanceSampling}; root::Int=0) =
    ReplicaExchangeVector(algs)

function reset!(rx::AbstractReplicaExchange)
    rx.stage = 0
    rx.indices .= eachindex(rx.indices)
    fill!(rx.steps, 0)
    fill!(rx.accepted, 0)
    return rx
end

function acceptance_rates(steps::AbstractVector{<:Real}, accepted::AbstractVector{<:Real})
    length(steps) == length(accepted) || throw(ArgumentError("steps and accepted must have the same length"))
    rates = zeros(Float64, length(steps))
    @inbounds for i in eachindex(rates)
        rates[i] = steps[i] > 0 ? accepted[i] / steps[i] : 0.0
    end
    return rates
end

acceptance_rates(rx::AbstractReplicaExchange) = acceptance_rates(rx.steps, rx.accepted)

function acceptance_rate(steps::AbstractVector{<:Real}, accepted::AbstractVector{<:Real})
    length(steps) == length(accepted) || throw(ArgumentError("steps and accepted must have the same length"))
    total_steps = sum(steps)
    return total_steps > 0 ? sum(accepted) / total_steps : 0.0
end

acceptance_rate(rx::AbstractReplicaExchange) = acceptance_rate(rx.steps, rx.accepted)

@inline rank(rx::ReplicaExchangeMessage) = rank(rx.backend)
@inline size(rx::ReplicaExchangeMessage) = size(rx.backend)
@inline rank(::ReplicaExchangeVector) = 0
@inline size(rx::ReplicaExchangeVector) = length(rx.alg)

@inline is_root(rx::ReplicaExchangeMessage) = rank(rx) == rx.root
@inline is_root(::ReplicaExchangeVector) = true

# Collective communication wrappers
barrier(rx::ReplicaExchangeMessage) = barrier(rx.backend)
barrier(::ReplicaExchangeVector) = nothing

allgather(value, rx::ReplicaExchangeMessage) = allgather(value, rx.backend)
allgather(value, rx::ReplicaExchangeVector) = [value]

allreduce!(values, op, rx::ReplicaExchangeMessage) = allreduce!(values, op, rx.backend)
allreduce!(values, op, ::ReplicaExchangeVector) = values

reduce(values, op, root::Int, rx::ReplicaExchangeMessage) = reduce(values, op, root, rx.backend)
reduce(values, op, root::Int, ::ReplicaExchangeVector) = copy(values)

bcast!(values, root::Int, rx::ReplicaExchangeMessage) = bcast!(values, root, rx.backend)
bcast!(values, root::Int, ::ReplicaExchangeVector) = values

gather(value, rx::ReplicaExchangeMessage; root::Int) = gather(value, rx.backend; root=root)
gather(value, ::ReplicaExchangeVector; root::Int) = [value]

gather_at_root(value, rx::ReplicaExchangeMessage) = gather(value, rx.backend; root=rx.root)
gather_at_root(value, ::ReplicaExchangeVector) = [value]

broadcast_from_root!(values, rx::ReplicaExchangeMessage) = bcast!(values, rx.root, rx.backend)
broadcast_from_root!(values, ::ReplicaExchangeVector) = values

function exchange_stats_at_root(rx::ReplicaExchangeMessage)
    steps_total = reduce(rx.steps, +, rx.root, rx)
    accepted_total = reduce(rx.accepted, +, rx.root, rx)
    barrier(rx)
    is_root(rx) || return nothing
    rates = acceptance_rates(steps_total, accepted_total)
    return (steps=steps_total, accepted=accepted_total, rates=rates)
end

@inline index(rx::AbstractReplicaExchange) = rx.indices[rank(rx) + 1]
@inline index(rx::ReplicaExchangeVector, replica::Integer) = rx.indices[Int(replica)]

function _resolve_pair(my_index::Int, stage::Int, nranks::Int)
    first = iseven(stage) ? 1 : 2
    offset = my_index - first

    if offset >= 0 && iseven(offset) && my_index < nranks
        return (active=true, pair_id=my_index, partner_index=my_index + 1)
    elseif offset > 0 && isodd(offset) && my_index - 1 >= first
        return (active=true, pair_id=my_index - 1, partner_index=my_index - 1)
    else
        return (active=false, pair_id=0, partner_index=0)
    end
end

function _partner_rank(rx::AbstractReplicaExchange, partner_index::Int)
    partner_pos = findfirst(==(partner_index), rx.indices)
    partner_pos === nothing && throw(ArgumentError("Replica-exchange partner index $partner_index not found in current ladder permutation"))
    return partner_pos - 1
end

"""
    exchange_log_ratio(ens_i, ens_j, x_i, x_j)

Replica-exchange swap log-ratio for two ensembles and their local observables.
"""
@inline function exchange_log_ratio(ens_i, ens_j, x_i::Real, x_j::Real)
    return (logweight(ens_i, x_j) - logweight(ens_i, x_i)) +
           (logweight(ens_j, x_i) - logweight(ens_j, x_j))
end

@inline _accept_exchange(log_ratio::Real, u::Real) = (log_ratio > 0) || (u < exp(log_ratio))

"""
    attempt_exchange_pair!(alg_i, alg_j, x_i, x_j, u)

Attempt one pair exchange using shared random number `u`.
If accepted, ensembles are swapped between `alg_i` and `alg_j`.
Returns `true` if accepted.
"""
function attempt_exchange_pair!(alg_i::AbstractImportanceSampling,
                                alg_j::AbstractImportanceSampling,
                                x_i::Real,
                                x_j::Real,
                                u::Real)
    isfinite(u) || throw(ArgumentError("shared random number `u` must be finite"))
    log_ratio = exchange_log_ratio(alg_i.ensemble, alg_j.ensemble, x_i, x_j)
    accepted = _accept_exchange(log_ratio, u)
    if accepted
        alg_i.ensemble, alg_j.ensemble = alg_j.ensemble, alg_i.ensemble
    end
    return accepted
end

function _update_pair!(rx::ReplicaExchangeMessage,
                       alg::AbstractImportanceSampling,
                       x::Real,
                       pair_id::Int,
                       partner_index::Int)
    my_index = index(rx)
    partner_rank = _partner_rank(rx, partner_index)
    is_owner = rank(rx) < partner_rank
    ens = alg.ensemble

    packet = (ensemble=ens, x=x, u=float(is_owner ? rand(alg.rng) : NaN))
    packet_p = exchange_packet(rx.backend, packet, partner_rank, rx.stage, is_owner)

    ens_p = packet_p.ensemble
    x_p = packet_p.x
    u = is_owner ? packet.u : packet_p.u
    isfinite(u) || throw(ArgumentError("Replica-exchange received non-finite shared random number `u`; check rank owner logic"))
    log_ratio = exchange_log_ratio(ens, ens_p, x, x_p)

    if is_owner
        rx.steps[pair_id] += 1
    end

    if _accept_exchange(log_ratio, u)
        if is_owner
            rx.accepted[pair_id] += 1
        end
        alg.ensemble = ens_p
        return partner_index
    end

    return my_index
end

function update!(rx::ReplicaExchangeMessage, alg::AbstractImportanceSampling, x::Real)
    barrier(rx)
    
    my_index = index(rx)
    pair = _resolve_pair(my_index, rx.stage, size(rx))

    new_index = my_index
    if pair.active
        new_index = _update_pair!(rx, alg, x, pair.pair_id, pair.partner_index)
    end

    rx.indices = allgather(new_index, rx)
    rx.stage = 1 - rx.stage
    
    barrier(rx)
    return nothing
end

function update!(rx::ReplicaExchangeMessage, x::Real)
    return update!(rx, rx.alg, x)
end

function update!(rx::ReplicaExchangeVector, xs::AbstractVector{<:Real})
    length(xs) == size(rx) || throw(ArgumentError("xs must have length size(rx)"))

    first = iseven(rx.stage) ? 1 : 2
    @inbounds for pair_id in first:2:(size(rx) - 1)
        ri = findfirst(==(pair_id), rx.indices)
        rj = findfirst(==(pair_id + 1), rx.indices)
        (ri === nothing || rj === nothing) && throw(ArgumentError("Replica-exchange local index permutation is inconsistent"))

        rx.steps[pair_id] += 1
        u = rand(rx.alg[ri].rng)
        did_accept = attempt_exchange_pair!(rx.alg[ri], rx.alg[rj], xs[ri], xs[rj], u)
        if did_accept
            rx.accepted[pair_id] += 1
            rx.indices[ri], rx.indices[rj] = rx.indices[rj], rx.indices[ri]
        end
    end

    rx.stage = 1 - rx.stage
    return nothing
end
