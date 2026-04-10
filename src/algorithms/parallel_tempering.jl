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
    ParallelTempering(rngs, betas; backend=nothing, root=0)

Temperature-explicit convenience constructor. Creates one `Metropolis` replica
per (rng, beta) pair.

- `backend=nothing` (default): local vector mode, all replicas in-process.
- `backend::AbstractMessageBackend` or `Symbol`: message-based mode. Each rank
  selects its own (rng, beta) by rank index. Use `init(:MPI)` or `init(:Distributed)`
  to obtain a backend.

Examples:
```julia
# local vector (default, safe for serial/threaded use)
pt = ParallelTempering(rngs, betas)

# MPI: backend returned by init carries rank info
backend = init(:MPI)
pt = ParallelTempering(rngs, betas; backend=backend)
# pt.alg is the Metropolis algorithm for this rank
```
"""
function ParallelTempering(rngs::AbstractVector,
                           betas::AbstractVector{<:Real};
                           backend::Union{Nothing,Symbol,AbstractMessageBackend}=nothing,
                           root::Int=0)
    vals = _assert_valid_betas!(collect(float.(betas)))
    n = length(vals)
    length(rngs) == n || throw(ArgumentError("length(rngs) must equal length(betas)"))

    if backend === nothing
        algs = [Metropolis(rngs[i]; β=vals[i]) for i in 1:n]
        return ReplicaExchange(algs; root=root)
    end

    be = backend isa Symbol ? init(backend) : backend
    size(be) == n || throw(ArgumentError("size(backend) (=$(size(be))) must equal length(betas) (=$n)"))
    i = rank(be) + 1
    alg = Metropolis(rngs[i]; β=vals[i])
    return ReplicaExchange(alg, be; root=root)
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
    return ParallelTempering((β, i) -> Metropolis(rng(seed + i); β=β), betas; backend=backend, root=root)
end

"""
    ParallelTempering(build_replica, betas; backend=nothing, root=0)

Create replicas from a beta ladder via a factory function.
`build_replica(beta, i)` is called with the beta value and 1-based index.

- `backend=nothing` (default): vector mode, all replicas built in-process.
- Otherwise: message mode, only the local replica for this rank is built.
"""
function ParallelTempering(build_replica::Function,
                           betas::AbstractVector{<:Real};
                           backend::Union{Nothing,Symbol,AbstractMessageBackend}=nothing,
                           root::Int=0)
    vals = _assert_valid_betas!(collect(float.(betas)))

    if backend === nothing
        algs = [build_replica(β, i) for (i, β) in enumerate(vals)]
        return ReplicaExchange(algs; root=root)
    end

    be = backend isa Symbol ? init(backend) : backend
    n = length(vals)
    size(be) == n || throw(ArgumentError("size(backend) (=$(size(be))) must equal length(betas) (=$n)"))
    i = rank(be) + 1
    alg = build_replica(vals[i], i)
    return ReplicaExchange(alg, be; root=root)
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
