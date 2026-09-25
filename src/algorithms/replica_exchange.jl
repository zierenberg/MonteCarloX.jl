"""
    ReplicaExchange{B, A}

Generic replica-exchange coordinator. The backend type parameter `B`
determines the parallelism strategy:

- `ReplicaExchange{ThreadsBackend}`: local vector of per-replica algorithms, shared memory.
- `ReplicaExchange{MPIBackend}`: one algorithm per MPI rank, MPI communication.
"""

mutable struct ReplicaExchange{B, A} <: AbstractAlgorithm
    replica::ParallelChains{B, A}
    stage::Int
    indices::Vector{Int}
    steps::Vector{Int}
    accepted::Vector{Int}
end

@inline steps(rx::ReplicaExchange) = sum(rx.steps)

@inline rank(rx::ReplicaExchange) = rank(rx.replica)
@inline size(rx::ReplicaExchange) = size(rx.replica)
@inline is_root(rx::ReplicaExchange) = is_root(rx.replica)
@inline algorithm(rx::ReplicaExchange, args...) = algorithm(rx.replica, args...)
@inline with_parallel(f, rx::ReplicaExchange) = with_parallel(f, rx.replica)

"""
    on_root(f, rx::ReplicaExchange)

Run a block on the root chain only; no-op on non-root ranks.
Consistent with `on_root(f, pc::ParallelChains)`.  The callback
may take zero or one argument (root chain index).
"""
@inline function on_root(f, rx::ReplicaExchange{ThreadsBackend})
    i = root_chain(rx.replica)
    return applicable(f, i) ? f(i) : f()
end

@inline function on_root(f, rx::ReplicaExchange{<:MPIBackend})
    if !is_root(rx)
        return nothing
    end
    i = root_chain(rx.replica)
    return applicable(f, i) ? f(i) : f()
end

# indexing
@inline ensemble_index(rx::ReplicaExchange{<:ThreadsBackend}, i::Integer) = rx.indices[i]
@inline ensemble_index(rx::ReplicaExchange{<:ThreadsBackend}) = rx.indices
@inline ensemble_index(rx::ReplicaExchange{<:MPIBackend}) = rx.indices[rank(rx) + 1]

function ReplicaExchange(backend::ThreadsBackend,
                         alg::AbstractVector{<:AbstractMarkovChainMonteCarlo})
    n = length(alg)
    n >= 2 || throw(ArgumentError("need at least 2 algorithms for replica exchange"))
    pc = ParallelChains(backend, alg)
    nedges = n - 1
    return ReplicaExchange(pc, 0, collect(1:n), zeros(Int, nedges), zeros(Int, nedges))
end

function ReplicaExchange(backend::MPIBackend, alg::AbstractMarkovChainMonteCarlo)
    pc = ParallelChains(backend, alg)
    n = size(backend)
    nedges = max(0, n - 1)
    return ReplicaExchange(pc, 0, collect(1:n), zeros(Int, nedges), zeros(Int, nedges))
end

"""
    ReplicaExchange(ensembles; seed=1000, rng=Xoshiro, algorithm=MetropolisAlgorithm, backend=nothing)

One replica per ensemble, each with its own RNG seeded `seed + i`.

```julia
ReplicaExchange([BoltzmannEnsemble(T = T) for T in Ts])      # parallel tempering
ReplicaExchange([window_ensemble(w) for w in windows])       # replica-exchange Wang-Landau
```

The replicas need not differ in temperature, or even be of one kind — any ensembles work, and
exchanges are attempted between neighbours in the given order.

`backend=nothing` or `::ThreadsBackend` gives threads mode; `::MPIBackend` gives MPI mode, the
rank-local replica selected by rank.
"""
function ReplicaExchange(ensembles::AbstractVector{<:AbstractEnsemble};
                         seed::Integer=1000,
                         rng=Xoshiro,
                         algorithm=MetropolisAlgorithm,
                         backend::Union{Nothing,ThreadsBackend,MPIBackend}=nothing)
    n = length(ensembles)
    n >= 2 || throw(ArgumentError("need at least 2 replicas"))
    backend === nothing && (backend = ThreadsBackend(n))
    size(backend) == n || throw(ArgumentError(
        "size(backend) (=$(size(backend))) must equal length(ensembles) (=$n)"))

    backend isa ThreadsBackend &&
        return ReplicaExchange(backend, [algorithm(rng(seed + i), ensembles[i]) for i in 1:n])
    i = rank(backend) + 1
    return ReplicaExchange(backend, algorithm(rng(seed + i), ensembles[i]))
end

function reset!(rx::ReplicaExchange)
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
    exchange_log_ratio(ens_i, ens_j, arg_i, arg_j)

Replica-exchange swap log-ratio for two ensembles and their local observables.
"""
@inline function exchange_log_ratio(ens_i, ens_j, arg_i, arg_j)
    return (logweight(ens_i, arg_j) - logweight(ens_i, arg_i)) +
           (logweight(ens_j, arg_i) - logweight(ens_j, arg_j))
end

@inline _accept_exchange(log_ratio::Real, u::Real) = (log_ratio > 0) || (u < exp(log_ratio))

"""
    attempt_exchange_pair!(alg_i, alg_j, arg_i, arg_j, u)

Attempt one pair exchange using shared random number `u`.
If accepted, ensembles are swapped between `alg_i` and `alg_j`.
Returns `true` if accepted.
"""
function attempt_exchange_pair!(alg_i::AbstractMarkovChainMonteCarlo,
                                alg_j::AbstractMarkovChainMonteCarlo,
                                arg_i,
                                arg_j,
                                u::Real)
    isfinite(u) || throw(ArgumentError("shared random number `u` must be finite"))
    log_ratio = exchange_log_ratio(alg_i.ensemble, alg_j.ensemble, arg_i, arg_j)
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

"""
    attempt_exchange!(rx::ReplicaExchange, coordinates)

Attempt one round of exchanges — every ladder edge of the current parity, alternating between
calls — using the replicas' reaction `coordinates` — whatever the ensembles score, one entry per
replica (threads) or the rank-local value (MPI).

Configurations never move: chain or rank `r` keeps its own system throughout, which is why
MonteCarloX never has to copy a model. What moves on acceptance is the **ensemble**, and
[`ensemble_index`](@ref)`(rx, r)` is the label that travels with it. Per-replica state that depends
on the rung must therefore be re-read afterwards.

Pairs with [`advance!`](@ref), which does the sampling half of the loop.
"""
function attempt_exchange!(rx::ReplicaExchange{ThreadsBackend}, xs::AbstractVector)
    length(xs) == size(rx) || throw(ArgumentError("xs must have length size(rx)"))

    # Inverse permutation: position[ladder_index] = chain position. Built once in O(n) so the
    # pair lookups below are O(1) instead of a findfirst scan (avoids O(n²) per exchange sweep).
    position = Vector{Int}(undef, size(rx))
    @inbounds for p in eachindex(rx.indices)
        position[rx.indices[p]] = p
    end

    first = iseven(rx.stage) ? 1 : 2
    @inbounds for pair_id in first:2:(size(rx) - 1)
        ri = position[pair_id]
        rj = position[pair_id + 1]

        rx.steps[pair_id] += 1
        u = rand(algorithm(rx, ri).rng)
        did_accept = attempt_exchange_pair!(algorithm(rx, ri), algorithm(rx, rj),
                                            xs[ri], xs[rj], u)
        if did_accept
            rx.accepted[pair_id] += 1
            rx.indices[ri], rx.indices[rj] = rx.indices[rj], rx.indices[ri]
            position[pair_id], position[pair_id + 1] = rj, ri
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

# Three-phase pair exchange. The log-ratio splits into two halves, each evaluable from one rank's
# own ensemble — logR = [W_i(x_j) − W_i(x_i)] + [W_j(x_i) − W_j(x_j)] — so the decision needs only
# the coordinates and one scalar per rank. The ensemble crosses the wire solely on acceptance,
# which matters for tabulated ensembles (muca, Wang-Landau) that carry O(bins) arrays. Both ranks
# sum the same two floats, so they agree bitwise and reach the same decision from the shared `u`.
function _update_pair!(rx::ReplicaExchange{<:MPIBackend},
                       alg::AbstractMarkovChainMonteCarlo,
                       x,
                       pair_id::Int,
                       partner_index::Int)
    my_index = ensemble_index(rx)
    partner_rank = _partner_rank(rx, partner_index)
    is_owner = rank(rx) < partner_rank
    comm = rx.replica.backend.comm
    ens = alg.ensemble
    tag = 3 * rx.stage

    # Phase 1 — coordinates and the shared random number. O(1); the ensemble stays put.
    local_packet = (x=x, u=float(is_owner ? rand(alg.rng) : NaN))
    partner_packet = _exchange_packet_mpi(comm, local_packet, partner_rank, tag, is_owner)
    u = is_owner ? local_packet.u : partner_packet.u
    isfinite(u) || throw(ArgumentError("Replica-exchange received non-finite shared random number `u`; check rank owner logic"))

    # Phase 2 — each rank owns the half of the log-ratio only its own ensemble can evaluate.
    half = logweight(ens, partner_packet.x) - logweight(ens, x)
    half_partner = _exchange_packet_mpi(comm, half, partner_rank, tag + 1, is_owner)
    log_ratio = half + half_partner

    if is_owner
        rx.steps[pair_id] += 1
    end

    if _accept_exchange(log_ratio, u)
        if is_owner
            rx.accepted[pair_id] += 1
        end
        # Phase 3 — only now is the ensemble itself sent.
        alg.ensemble = _exchange_packet_mpi(comm, ens, partner_rank, tag + 2, is_owner)
        return partner_index
    end

    return my_index
end

function attempt_exchange!(rx::ReplicaExchange{<:MPIBackend}, x)
    comm = rx.replica.backend.comm
    MPI.Barrier(comm)

    my_index = ensemble_index(rx)
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


# ── Parallel tempering: the β-ladder specialization of replica exchange ──────

# Constructors dispatching on backend
ParallelTempering(backend::ThreadsBackend, alg::AbstractVector{<:AbstractMarkovChainMonteCarlo}; kwargs...) =
    ReplicaExchange(backend, alg; kwargs...)
ParallelTempering(backend::MPIBackend, alg::AbstractMarkovChainMonteCarlo; kwargs...) =
    ReplicaExchange(backend, alg; kwargs...)

"""
    ParallelTempering(betas; seed=1000, rng=Xoshiro, backend=nothing, observable=identity)

[`ReplicaExchange`](@ref) over `BoltzmannEnsemble`s — `ReplicaExchange([BoltzmannEnsemble(β=β)
for β in betas])`. Equivalent to writing that out, and `BoltzmannEnsemble(T=T)` if temperatures
read better.

- `backend=nothing` (default): threads mode.
- `backend::ThreadsBackend`: threads mode.
- `backend::MPIBackend`: MPI mode; rank-local replica is selected by backend rank.
"""
ParallelTempering(betas::AbstractVector{<:Real}; kwargs...) =
    ReplicaExchange([BoltzmannEnsemble(β=float(β)) for β in betas]; kwargs...)

# ── Driving the ladder ──────────────────────────────────────────────────────

# Without this a ReplicaExchange, being an AbstractAlgorithm, would take the single-chain method.
@inline advance!(sweep!, rx::ReplicaExchange, states, n::Integer) =
    advance!(sweep!, rx.replica, states, n)

"""
    _group_samples(local_samples, n)

Group samples into per-ladder-index vectors. Dispatches on sample format:
- `AbstractVector{<:AbstractVector{<:Real}}`: already grouped by index, returned as-is.
- `AbstractVector{<:Tuple{<:Integer,<:Real}}`: flat `(index, value)` tuples sorted into bins.
"""
_group_samples(samples::AbstractVector{<:AbstractVector{<:Real}}, n::Int) = samples

function _group_samples(samples::AbstractVector{<:Tuple{<:Integer,<:Real}}, n::Int)
    grouped = [Float64[] for _ in 1:n]
    for (idx, e) in samples
        1 <= idx <= n || throw(ArgumentError("sample index $idx out of bounds for $n ladders"))
        push!(grouped[idx], float(e))
    end
    return grouped
end

"""
    optimize_exchange_interval!(pt, local_samples, sweeps_after_exchange; ...)

Adapt post-exchange local sweep counts from integrated autocorrelation times of
energy traces per ladder index.

`local_samples` can be either:
- `AbstractVector{<:AbstractVector{<:Real}}`: pre-grouped traces per ladder index.
- `AbstractVector{<:Tuple{<:Integer,<:Real}}`: flat `(ladder_index, value)` tuples.
"""
function optimize_exchange_interval!(pt::ReplicaExchange,
                                     local_samples,
                                     sweeps_after_exchange::AbstractVector{<:Integer};
                                     base_sweeps::Integer,
                                     min_sweeps::Integer=1,
                                     max_sweeps::Integer=typemax(Int),
                                     min_points::Integer=400,
                                     max_lag::Union{Nothing,Integer}=200)
    n = size(pt)
    length(sweeps_after_exchange) == n || throw(ArgumentError("sweeps_after_exchange must have length size(pt)"))

    if pt isa ReplicaExchange{<:MPIBackend}
        comm = pt.replica.backend.comm
        root = pt.replica.backend.root
        all_samples = MPI.gather(local_samples, comm; root=root)

        if is_root(pt)
            merged = [Float64[] for _ in 1:n]
            for rank_samples in all_samples
                grouped = _group_samples(rank_samples, n)
                for i in 1:n
                    append!(merged[i], grouped[i])
                end
            end
            taus = integrated_autocorrelation_times(merged; min_points=min_points, max_lag=max_lag)
            _retune_exchange_sweeps!(sweeps_after_exchange, taus, base_sweeps, min_sweeps, max_sweeps)
        end

        MPI.Bcast!(sweeps_after_exchange, root, comm)
        return sweeps_after_exchange[ensemble_index(pt)]
    end

    grouped = _group_samples(local_samples, n)
    taus = integrated_autocorrelation_times(grouped; min_points=min_points, max_lag=max_lag)
    _retune_exchange_sweeps!(sweeps_after_exchange, taus, base_sweeps, min_sweeps, max_sweeps)
    return sweeps_after_exchange[ensemble_index(pt)]
end

function _retune_exchange_sweeps!(sweeps_after_exchange::AbstractVector{<:Integer},
                                  taus::AbstractVector{<:Real},
                                  base_sweeps::Integer,
                                  min_sweeps::Integer,
                                  max_sweeps::Integer)
    finite_taus = filter(isfinite, taus)
    tau_ref = isempty(finite_taus) ? 1.0 : median(finite_taus)

    @inbounds for i in eachindex(sweeps_after_exchange)
        scale = isfinite(taus[i]) ? taus[i] / tau_ref : 1.0
        target = round(Int, base_sweeps * scale)
        sweeps_after_exchange[i] = clamp(target, min_sweeps, max_sweeps)
    end
    return sweeps_after_exchange
end

"""
    set_betas(n, βmin, βmax, mode; T=Float64)

Create a beta ladder of length `n` between `βmin` and `βmax`.

Modes:
- `:uniform`: linearly spaced from `βmax` to `βmin`.
- `:geometric`: geometrically spaced from `βmax` to `βmin`.
"""
function set_betas(nreplicas::Integer,
                   βmin::Real,
                   βmax::Real,
                   mode::Symbol;
                   T::Type{<:Real}=Float64)
    nreplicas >= 2 || throw(ArgumentError("nreplicas must be >= 2"))

    vals = if mode == :uniform
        range(float(βmax), float(βmin), length=Int(nreplicas))
    elseif mode == :geometric
        exp.(range(log(float(βmax)), log(float(βmin)), length=Int(nreplicas)))
    else
        throw(ArgumentError("unknown beta mode $(mode); use :uniform or :geometric"))
    end

    return Vector{T}(vals)
end
