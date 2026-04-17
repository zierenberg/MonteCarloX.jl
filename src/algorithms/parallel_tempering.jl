"""
    ParallelTempering

Temperature-parameterized specialization layered on top of `ReplicaExchange`.
It provides beta-ladder construction helpers and convenience constructors.
"""

# Constructors dispatching on backend
ParallelTempering(backend::ThreadsBackend, alg::AbstractVector{<:AbstractImportanceSampling}) =
    ReplicaExchange(backend, alg)
ParallelTempering(backend::MPIBackend, alg::AbstractImportanceSampling) =
    ReplicaExchange(backend, alg)

"""
    ParallelTempering(betas; seed=1000, rng=Xoshiro, backend=nothing)

Convenience constructor that creates per-replica RNGs from `seed + i` and builds
`Metropolis` replicas over `betas`.

- `backend=nothing` (default): threads mode.
- `backend::ThreadsBackend`: threads mode.
- `backend::MPIBackend`: MPI mode; rank-local replica is selected by backend rank.
"""
function ParallelTempering(betas::AbstractVector{<:Real};
                           seed::Integer=1000,
                           rng=Xoshiro,
                           backend::Union{Nothing,ThreadsBackend,MPIBackend}=nothing)
    n = length(betas)
    n >= 2 || throw(ArgumentError("need at least 2 replicas"))
    vals = collect(float.(betas))

    # if nothing: create a threads backend with one thread per beta
    if backend === nothing
        backend = ThreadsBackend(n)
        alg = [Metropolis(rng(seed + i); β=vals[i]) for i in 1:n]
        return ReplicaExchange(backend, alg)
    end

    if backend isa ThreadsBackend
        size(backend) == n || throw(ArgumentError("size(backend) (=$(size(backend))) must equal length(betas) (=$n)"))
        alg = [Metropolis(rng(seed + i); β=vals[i]) for i in 1:n]
        return ReplicaExchange(backend, alg)
    end

    if backend isa MPIBackend
        size(backend) == n || throw(ArgumentError("size(backend) (=$(size(backend))) must equal length(betas) (=$n)"))
        i = rank(backend) + 1
        alg = Metropolis(rng(seed + i); β=vals[i])
        return ReplicaExchange(backend, alg)
    end

    throw(ArgumentError("unsupported backend type $(typeof(backend))"))
end

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
        return sweeps_after_exchange[index(pt)]
    end

    grouped = _group_samples(local_samples, n)
    taus = integrated_autocorrelation_times(grouped; min_points=min_points, max_lag=max_lag)
    _retune_exchange_sweeps!(sweeps_after_exchange, taus, base_sweeps, min_sweeps, max_sweeps)
    return sweeps_after_exchange[index(pt)]
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
