"""
    ParallelTempering

Temperature-parameterized specialization layered on top of `ReplicaExchange`.
It provides beta-ladder construction helpers and convenience constructors.
"""

# Type aliases keep existing names while reusing replica-exchange internals.
const AbstractParallelTempering = AbstractReplicaExchange
const ParallelTemperingMessage = ReplicaExchangeMessage
const ParallelTemperingVector = ReplicaExchangeVector

# alg-first canonical order, matching ParallelMulticanonical
ParallelTempering(alg::AbstractImportanceSampling, backend::AbstractMessageBackend; root::Int=0) =
    ReplicaExchange(alg, backend; root=root)
ParallelTempering(algs::AbstractVector{<:AbstractImportanceSampling}; root::Int=0) =
    ReplicaExchange(algs; root=root)

"""
    ParallelTempering(alg, mode::Symbol; kwargs...)

Construct a message-based parallel-tempering coordinator by initializing a
backend via `init(mode; kwargs...)`.

Examples:
- `ParallelTempering(alg, :MPI)`
- `ParallelTempering(alg, :Distributed; addprocs=4)`
"""
function ParallelTempering(alg::AbstractImportanceSampling, mode::Symbol; root::Int=0, kwargs...)
    backend = init(mode; kwargs...)
    return ReplicaExchange(alg, backend; root=root)
end

"""
    ParallelTempering(betas; seed=0, rng=MersenneTwister, backend=nothing, root=0)

Convenience constructor that creates per-replica RNGs from `seed + i` and builds
`Metropolis` replicas over `betas`.

- `backend=nothing` (default): local vector mode.
- `backend::AbstractMessageBackend` or `Symbol`: message-based mode; rank-local
  replica is selected by backend rank.

This constructor intentionally fixes RNG initialization to `seed + i` for a
compact, reproducible setup.
"""
function ParallelTempering(betas::AbstractVector{<:Real};
                           seed::Integer=0,
                           rng=MersenneTwister,
                           backend::Union{Nothing,Symbol,AbstractMessageBackend}=nothing,
                           root::Int=0)
    vals = _assert_valid_betas!(collect(float.(betas)))
    n = length(vals)

    if backend === nothing
        algs = [Metropolis(rng(seed + i); β=vals[i]) for i in 1:n]
        return ReplicaExchange(algs; root=root)
    end

    be = backend isa Symbol ? init(backend) : backend
    size(be) == n || throw(ArgumentError("size(backend) (=$(size(be))) must equal length(betas) (=$n)"))
    i = rank(be) + 1
    alg = Metropolis(rng(seed + i); β=vals[i])
    return ReplicaExchange(alg, be; root=root)
end

"""
    optimize_exchange_interval!(pt, local_samples, sweeps_after_exchange;
                                base_sweeps, min_sweeps=1, max_sweeps=typemax(Int),
                                min_points=400, max_lag=200)

Adapt post-exchange local sweep counts from integrated autocorrelation times of
energy traces per ladder index.

- In message-backend mode, samples are gathered at root, the schedule is updated
  at root, then broadcast back to all ranks.
- In vector mode, adaptation is local and direct.

Returns the sweep count for the current local ladder index to be used until the
next adaptation step.
"""
function optimize_exchange_interval!(pt::AbstractReplicaExchange,
                                     local_samples::AbstractVector{<:AbstractVector{<:Real}},
                                     sweeps_after_exchange::AbstractVector{<:Integer};
                                     base_sweeps::Integer,
                                     min_sweeps::Integer=1,
                                     max_sweeps::Integer=typemax(Int),
                                     min_points::Integer=400,
                                     max_lag::Union{Nothing,Integer}=200)
    n = size(pt)
    length(local_samples) == n || throw(ArgumentError("local_samples must have length size(pt)"))
    length(sweeps_after_exchange) == n || throw(ArgumentError("sweeps_after_exchange must have length size(pt)"))
    base_sweeps >= 1 || throw(ArgumentError("base_sweeps must be >= 1"))
    min_sweeps >= 1 || throw(ArgumentError("min_sweeps must be >= 1"))
    max_sweeps >= min_sweeps || throw(ArgumentError("max_sweeps must be >= min_sweeps"))
    min_points >= 2 || throw(ArgumentError("min_points must be >= 2"))

    if pt isa ReplicaExchangeMessage
        traces_all = gather_at_root(local_samples, pt)

        if is_root(pt)
            merged = [Float64[] for _ in 1:n]
            for rank_samples in traces_all
                for i in 1:n
                    append!(merged[i], rank_samples[i])
                end
            end
            taus = integrated_autocorrelation_times(merged; min_points=min_points, max_lag=max_lag)
            _retune_exchange_sweeps!(sweeps_after_exchange, taus, base_sweeps, min_sweeps, max_sweeps)
        end

        broadcast_from_root!(sweeps_after_exchange, pt)
        return sweeps_after_exchange[index(pt)]
    end

    taus = integrated_autocorrelation_times(local_samples; min_points=min_points, max_lag=max_lag)
    _retune_exchange_sweeps!(sweeps_after_exchange, taus, base_sweeps, min_sweeps, max_sweeps)
    return sweeps_after_exchange[index(pt)]
end

function optimize_exchange_interval!(pt::AbstractReplicaExchange,
                                     local_samples::AbstractVector{<:Tuple{<:Integer,<:Real}},
                                     sweeps_after_exchange::AbstractVector{<:Integer};
                                     base_sweeps::Integer,
                                     min_sweeps::Integer=1,
                                     max_sweeps::Integer=typemax(Int),
                                     min_points::Integer=400,
                                     max_lag::Union{Nothing,Integer}=200)
    n = size(pt)
    length(sweeps_after_exchange) == n || throw(ArgumentError("sweeps_after_exchange must have length size(pt)"))
    base_sweeps >= 1 || throw(ArgumentError("base_sweeps must be >= 1"))
    min_sweeps >= 1 || throw(ArgumentError("min_sweeps must be >= 1"))
    max_sweeps >= min_sweeps || throw(ArgumentError("max_sweeps must be >= min_sweeps"))
    min_points >= 2 || throw(ArgumentError("min_points must be >= 2"))

    if pt isa ReplicaExchangeMessage
        traces_all = gather_at_root(local_samples, pt)

        if is_root(pt)
            merged = [Float64[] for _ in 1:n]
            for rank_samples in traces_all
                append_indexed_samples!(merged, rank_samples)
            end
            taus = integrated_autocorrelation_times(merged; min_points=min_points, max_lag=max_lag)
            _retune_exchange_sweeps!(sweeps_after_exchange, taus, base_sweeps, min_sweeps, max_sweeps)
        end

        broadcast_from_root!(sweeps_after_exchange, pt)
        return sweeps_after_exchange[index(pt)]
    end

    grouped = [Float64[] for _ in 1:n]
    append_indexed_samples!(grouped, local_samples)
    taus = integrated_autocorrelation_times(grouped; min_points=min_points, max_lag=max_lag)
    _retune_exchange_sweeps!(sweeps_after_exchange, taus, base_sweeps, min_sweeps, max_sweeps)
    return sweeps_after_exchange[index(pt)]
end

function append_indexed_samples!(grouped::AbstractVector{<:AbstractVector{Float64}},
                                 samples::AbstractVector{<:Tuple{<:Integer,<:Real}})
    n = length(grouped)
    for (idx, e) in samples
        1 <= idx <= n || throw(ArgumentError("sample index $idx out of bounds for $n ladders"))
        push!(grouped[idx], float(e))
    end
    return grouped
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

@inline function _assert_valid_betas!(betas::AbstractVector{<:Real})
    n = length(betas)
    n >= 2 || throw(ArgumentError("need at least 2 replicas"))
    @inbounds for i in eachindex(betas)
        β = betas[i]
        isfinite(β) || throw(ArgumentError("beta values must be finite"))
        β > 0 || throw(ArgumentError("beta values must be positive"))
    end
    return betas
end

function set_betas!(betas::AbstractVector{<:Real}, values::AbstractVector{<:Real})
    length(betas) == length(values) || throw(ArgumentError("values must match length(betas)"))
    betas .= values
    return _assert_valid_betas!(betas)
end

function set_betas(nreplicas::Integer,
                   βmin::Real,
                   βmax::Real,
                   mode::Symbol;
                   T::Type{<:Real}=Float64)
    nreplicas >= 2 || throw(ArgumentError("nreplicas must be >= 2"))
    βmin > 0 && βmax > 0 || throw(ArgumentError("betas must be positive"))
    βmax > βmin || throw(ArgumentError("βmax must be larger than βmin"))

    vals = if mode == :uniform
        range(float(βmax), float(βmin), length=Int(nreplicas))
    elseif mode == :geometric
        exp.(range(log(float(βmax)), log(float(βmin)), length=Int(nreplicas)))
    else
        throw(ArgumentError("unknown beta mode $(mode); use :uniform or :geometric"))
    end

    betas = Vector{T}(vals)
    return _assert_valid_betas!(betas)
end

function set_betas(nreplicas::Integer, values::AbstractVector{<:Real})
    length(values) == Int(nreplicas) || throw(ArgumentError("values must have length nreplicas"))
    betas = collect(float.(values))
    return _assert_valid_betas!(betas)
end

function retune_betas!(betas::AbstractVector{<:Real}, rates::AbstractVector{<:Real}; target::Real=0.3, damping::Real=0.5)
    n = length(betas)
    n >= 3 || throw(ArgumentError("need at least 3 betas to retune interior points"))
    length(rates) == n - 1 || throw(ArgumentError("rates must have length length(betas)-1"))
    target > 0 || throw(ArgumentError("target must be positive"))

    descending = betas[1] > betas[end]
    abs_gaps = abs.(diff(float.(betas)))
    total_gap = sum(abs_gaps)

    @inbounds for i in eachindex(abs_gaps)
        r = clamp(float(rates[i]), 1e-6, 1.0)
        abs_gaps[i] *= exp(damping * log(r / float(target)))
    end

    scaled = abs_gaps .* (total_gap / sum(abs_gaps))
    beta0 = float(betas[1])

    @inbounds for i in 2:n
        betas[i] = descending ? beta0 - sum(view(scaled, 1:(i - 1))) : beta0 + sum(view(scaled, 1:(i - 1)))
    end

    return betas
end
