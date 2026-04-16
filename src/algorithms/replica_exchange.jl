"""
    ReplicaExchange{B, A}

Generic replica-exchange coordinator. The backend type parameter `B`
determines the parallelism strategy:

- `ReplicaExchange{ThreadsBackend}`: local vector of per-replica algorithms, shared memory.
- `ReplicaExchange{MPIBackend}`: one algorithm per MPI rank, MPI communication.
"""

using MPI

mutable struct ReplicaExchange{B, A}
    replica::ParallelChains{B, A}
    stage::Int
    indices::Vector{Int}
    steps::Vector{Int}
    accepted::Vector{Int}
end

@inline rank(rx::ReplicaExchange) = rank(rx.replica)
@inline size(rx::ReplicaExchange) = size(rx.replica)
@inline is_root(rx::ReplicaExchange) = is_root(rx.replica)
@inline algorithm(rx::ReplicaExchange, args...) = algorithm(rx.replica, args...)

# indexing
@inline index(rx::ReplicaExchange{<:ThreadsBackend}, i::Integer) = rx.indices[i]
@inline index(rx::ReplicaExchange{<:MPIBackend}) = rx.indices[rank(rx) + 1]

function ReplicaExchange(backend::ThreadsBackend, alg::AbstractVector{<:AbstractImportanceSampling})
    n = length(alg)
    n >= 2 || throw(ArgumentError("need at least 2 algorithms for replica exchange"))
    pc = ParallelChains(backend, alg)
    nedges = n - 1
    return ReplicaExchange(pc, 0, collect(1:n), zeros(Int, nedges), zeros(Int, nedges))
end

function ReplicaExchange(backend::MPIBackend, alg::AbstractImportanceSampling)
    pc = ParallelChains(backend, alg)
    n = size(backend)
    nedges = max(0, n - 1)
    return ReplicaExchange(pc, 0, collect(1:n), zeros(Int, nedges), zeros(Int, nedges))
end

function reset!(rx::ReplicaExchange)
    rx.stage = 0
    rx.indices .= eachindex(rx.indices)
    fill!(rx.steps, 0)
    fill!(rx.accepted, 0)
    return rx
end

function acceptance_rates(rx::ReplicaExchange{ThreadsBackend})
    return [s > 0 ? a / s : 0.0 for (s, a) in zip(rx.steps, rx.accepted)]
end

function acceptance_rates(rx::ReplicaExchange{<:MPIBackend})
    comm = rx.replica.backend.comm
    root = rx.replica.backend.root
    steps_total = MPI.Reduce(rx.steps, +, root, comm)
    accepted_total = MPI.Reduce(rx.accepted, +, root, comm)
    is_root(rx) || return Float64[]
    return [s > 0 ? a / s : 0.0 for (s, a) in zip(steps_total, accepted_total)]
end

function acceptance_rate(rx::ReplicaExchange{ThreadsBackend})
    total = sum(rx.steps)
    return total > 0 ? sum(rx.accepted) / total : 0.0
end

function acceptance_rate(rx::ReplicaExchange{<:MPIBackend})
    comm = rx.replica.backend.comm
    root = rx.replica.backend.root
    steps_total = MPI.Reduce(rx.steps, +, root, comm)
    accepted_total = MPI.Reduce(rx.accepted, +, root, comm)
    is_root(rx) || return 0.0
    total = sum(steps_total)
    return total > 0 ? sum(accepted_total) / total : 0.0
end

################ exchange logic ################

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

function _partner_rank(rx::ReplicaExchange, partner_index::Int)
    partner_pos = findfirst(==(partner_index), rx.indices)
    partner_pos === nothing && throw(ArgumentError("Replica-exchange partner index $partner_index not found in current ladder permutation"))
    return partner_pos - 1
end

################ exchange logic (Threads specific) ################
function update!(rx::ReplicaExchange{ThreadsBackend}, xs::AbstractVector{<:Real})
    length(xs) == size(rx) || throw(ArgumentError("xs must have length size(rx)"))

    first = iseven(rx.stage) ? 1 : 2
    @inbounds for pair_id in first:2:(size(rx) - 1)
        ri = findfirst(==(pair_id), rx.indices)
        rj = findfirst(==(pair_id + 1), rx.indices)
        (ri === nothing || rj === nothing) && throw(ArgumentError("Replica-exchange local index permutation is inconsistent"))

        rx.steps[pair_id] += 1
        u = rand(algorithm(rx, ri).rng)
        did_accept = attempt_exchange_pair!(algorithm(rx, ri), algorithm(rx, rj), xs[ri], xs[rj], u)
        if did_accept
            rx.accepted[pair_id] += 1
            rx.indices[ri], rx.indices[rj] = rx.indices[rj], rx.indices[ri]
        end
    end

    rx.stage = 1 - rx.stage
    return nothing
end



################ exchange logic (MPI specific) ################
function _exchange_packet_mpi(comm, packet, partner_rank::Int, tag::Integer, is_owner::Bool)
    if is_owner
        MPI.send(packet, comm; dest=partner_rank, tag=tag)
        return MPI.recv(comm; source=partner_rank, tag=tag)
    end
    recv_packet = MPI.recv(comm; source=partner_rank, tag=tag)
    MPI.send(packet, comm; dest=partner_rank, tag=tag)
    return recv_packet
end

function _update_pair!(rx::ReplicaExchange{<:MPIBackend},
                       alg::AbstractImportanceSampling,
                       x::Real,
                       pair_id::Int,
                       partner_index::Int)
    my_index = index(rx)
    partner_rank = _partner_rank(rx, partner_index)
    is_owner = rank(rx) < partner_rank
    ens = alg.ensemble

    packet = (ensemble=ens, x=x, u=float(is_owner ? rand(alg.rng) : NaN))
    packet_p = _exchange_packet_mpi(rx.replica.backend.comm, packet, partner_rank, rx.stage, is_owner)

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

function update!(rx::ReplicaExchange{<:MPIBackend}, x::Real)
    comm = rx.replica.backend.comm
    MPI.Barrier(comm)

    my_index = index(rx)
    pair = _resolve_pair(my_index, rx.stage, size(rx))

    new_index = my_index
    if pair.active
        new_index = _update_pair!(rx, algorithm(rx), x, pair.pair_id, pair.partner_index)
    end

    rx.indices = MPI.Allgather(new_index, comm)
    rx.stage = 1 - rx.stage

    MPI.Barrier(comm)
    return nothing
end

