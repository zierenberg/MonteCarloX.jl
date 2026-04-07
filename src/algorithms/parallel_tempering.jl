using MPI

"""
    ParallelTempering <: AbstractAlgorithm

Parallel tempering helper with MPI metadata and exchange bookkeeping. 
Each rank hosts one replica. Neighbor relations are tracked through the ladder-index permutation `indices` and swaps are done for Boltzmann ensembles only.

Open Challenges:
- think about passing the full ensemble between pairs for clarity and future generalization. This should be sufficient lightweight, as it is similarly sized as a bigger float...
"""
mutable struct ParallelTempering <: AbstractImportanceSampling
    comm::Any
    rank::Int
    size::Int
    root::Int
    stage::Int
    indices::Vector{Int}
    steps::Vector{Int}
    accepted::Vector{Int}
end

function ParallelTempering(comm; root::Int=0)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    0 <= root < size || throw(ArgumentError("`root` must satisfy 0 <= root < size"))
    indices = collect(1:Int(size))
    nedges = max(0, Int(size) - 1)
    return ParallelTempering(comm, Int(rank), Int(size), Int(root), 0, indices, zeros(Int, nedges), zeros(Int, nedges))
end

function reset!(pt::ParallelTempering)
    pt.stage = 0
    pt.indices .= eachindex(pt.indices)
    fill!(pt.steps, 0)
    fill!(pt.accepted, 0)
    return pt
end

function acceptance_rates(pt::ParallelTempering)
    rates = zeros(Float64, length(pt.steps))
    @inbounds for i in eachindex(rates)
        rates[i] = pt.steps[i] > 0 ? pt.accepted[i] / pt.steps[i] : 0.0
    end
    return rates
end

function acceptance_rate(pt::ParallelTempering)
    total_steps = sum(pt.steps)
    return total_steps > 0 ? sum(pt.accepted) / total_steps : 0.0
end

# PT-specific accept helper: updates per-pair counters and returns decision.
function _accept!(pt::ParallelTempering, pair_id::Int, log_ratio::Real, u::Real)
    pt.steps[pair_id] += 1
    accepted = (log_ratio > 0) || (u < exp(log_ratio))
    pt.accepted[pair_id] += accepted
    return accepted
end

@inline function is_root(pt::ParallelTempering)
    return pt.rank == pt.root
end

"""
    index(pt::ParallelTempering)

Ladder index currently held by this rank.
Index 1 is the coldest end; index `pt.size` is the warmest end.
"""
@inline index(pt::ParallelTempering) = pt.indices[pt.rank + 1]

# Even stage activates pair IDs 1,3,5,... and odd stage activates 2,4,6,...
# Pair ID k corresponds to neighboring labels (k, k+1).
# Returns (active, pair_id, partner_index).
function _resolve_pair(my_index::Int, stage::Int, nranks::Int)
    first  = iseven(stage) ? 1 : 2
    offset = my_index - first

    if offset >= 0 && iseven(offset) && my_index < nranks
        return (active=true, pair_id=my_index, partner_index=my_index + 1)
    elseif offset > 0 && isodd(offset) && my_index - 1 >= first
        return (active=true, pair_id=my_index - 1, partner_index=my_index - 1)
    else
        return (active=false, pair_id=0, partner_index=0)
    end
end

function _update_pair!(pt::ParallelTempering,
                       alg::AbstractImportanceSampling,
                       x::Real,
                       pair_id::Int,
                       partner_index::Int)
    my_index = index(pt)
    partner_rank = findfirst(==(partner_index), pt.indices) - 1
    ens = alg.ensemble
    ens isa BoltzmannEnsemble || throw(ArgumentError("ParallelTempering currently supports BoltzmannEnsemble only"))
    beta = ens.beta

    # Lower rank appends a shared random float drawn from the local algorithm
    # RNG so PT swaps are reproducible from algorithm seeding alone.
    sendbuf = pt.rank < partner_rank ?
        [float(beta), float(x), rand(alg.rng)] :
        [float(beta), float(x), 0.0]
    recvbuf = similar(sendbuf)
    MPI.Sendrecv!(sendbuf, partner_rank, pt.stage,
                    recvbuf, partner_rank, pt.stage, pt.comm)

    beta_p    = recvbuf[1]
    x_p       = recvbuf[2]
    u         = pt.rank < partner_rank ? sendbuf[3] : recvbuf[3]
    ens_p     = BoltzmannEnsemble(convert(typeof(beta), beta_p))
    log_ratio = (logweight(ens, x_p) - logweight(ens, x)) +
            (logweight(ens_p, x) - logweight(ens_p, x_p))

    if pt.rank < partner_rank
        # Owner rank decides and records this pair attempt exactly once.
        accepted = _accept!(pt, pair_id, log_ratio, u)
    else
        # Non-owner receives the decision via shared `u`; no counter update.
        accepted = (log_ratio > 0) || (u < exp(log_ratio))
    end

    if accepted
        alg.ensemble = BoltzmannEnsemble(convert(typeof(beta), beta_p))
        return partner_index
    end

    return my_index
end

"""
    update!(pt::ParallelTempering, alg::AbstractImportanceSampling, x::Real)

Attempt one parallel-tempering swap step for scalar swap observable `x`.
"""
function update!(pt::ParallelTempering, alg::AbstractImportanceSampling, x::Real)
    my_slot = index(pt)
    pair = _resolve_pair(my_slot, pt.stage, pt.size)

    new_slot = my_slot
    if pair.active
        new_slot = _update_pair!(pt, alg, x, pair.pair_id, pair.partner_index)
    end

    # Propagate label updates to all ranks with one collective — O(nranks) scalars
    new_indices = MPI.Allgather(Int32(new_slot), pt.comm)
    @inbounds for r in eachindex(pt.indices)
        pt.indices[r] = Int(new_indices[r])
    end

    pt.stage = 1 - pt.stage
    return nothing
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

"""
    set_betas!(betas, values)

Set `betas` explicitly from a user-provided vector.
"""
function set_betas!(betas::AbstractVector{<:Real}, values::AbstractVector{<:Real})
    length(betas) == length(values) || throw(ArgumentError("values must match length(betas)"))
    betas .= values
    return _assert_valid_betas!(betas)
end

"""
    set_betas(nreplicas, βmin, βmax, mode::Symbol; T=Float64)

Create a beta ladder with chosen spacing mode.
`βmax` is the cold end and becomes the first entry.
"""
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

"""
    set_betas(nreplicas, values)

Create a beta ladder by copying explicit values.
"""
function set_betas(nreplicas::Integer, values::AbstractVector{<:Real})
    length(values) == Int(nreplicas) || throw(ArgumentError("values must have length nreplicas"))
    betas = collect(float.(values))
    return _assert_valid_betas!(betas)
end

"""
    retune_betas!(betas, rates; target=0.3, damping=0.5)

Retune interior beta spacing from measured pair acceptance rates while
keeping end points fixed. This is a lightweight hook for iterative ladder
optimization.
"""
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