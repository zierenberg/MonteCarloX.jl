using MPI

mutable struct ExchangeStats
    attempts::Vector{Int}
    accepts::Vector{Int}
end

ExchangeStats(nedges::Integer) = ExchangeStats(zeros(Int, max(0, nedges)), zeros(Int, max(0, nedges)))

function reset!(stats::ExchangeStats)
    fill!(stats.attempts, 0)
    fill!(stats.accepts, 0)
    return stats
end

function acceptance_rates(stats::ExchangeStats)
    rates = zeros(Float64, length(stats.attempts))
    @inbounds for i in eachindex(rates)
        rates[i] = stats.attempts[i] > 0 ? stats.accepts[i] / stats.attempts[i] : 0.0
    end
    return rates
end

@inline function _record_exchange!(stats::ExchangeStats, edge::Int, accepted::Bool)
    stats.attempts[edge] += 1
    stats.accepts[edge] += accepted
    return nothing
end

"""
    ParallelTempering <: AbstractAlgorithm

Parallel tempering helper with MPI metadata and exchange bookkeeping.
Each rank typically hosts one replica and exchanges inverse temperatures
(`beta`) with neighboring ranks.
"""
mutable struct ParallelTempering <: AbstractAlgorithm
    comm::Any
    rank::Int
    size::Int
    root::Int
    stage::Int
    stats::ExchangeStats
end

function ParallelTempering(comm; root::Int=0)
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    0 <= root < size || throw(ArgumentError("`root` must satisfy 0 <= root < size"))
    stats = ExchangeStats(max(0, Int(size) - 1))
    return ParallelTempering(comm, Int(rank), Int(size), Int(root), 0, stats)
end

@inline function is_root(pt::ParallelTempering)
    return pt.rank == pt.root
end

@inline function exchange_partner(pt::ParallelTempering, stage::Integer)
    partner = iseven(pt.rank + Int(stage)) ? (pt.rank + 1) : (pt.rank - 1)
    return 0 <= partner < pt.size ? partner : -1
end

@inline pt_log_acceptance(beta_i::Real, beta_j::Real, E_i::Real, E_j::Real) = (beta_i - beta_j) * (E_i - E_j)

"""
    inverse_temperature(ens::BoltzmannEnsemble)

Return inverse temperature `beta` for a Boltzmann ensemble.
"""
@inline inverse_temperature(ens::BoltzmannEnsemble) = ens.beta

"""
    inverse_temperature(alg::AbstractImportanceSampling)

Return inverse temperature `beta` for importance-sampling algorithms
with Boltzmann ensembles.
"""
@inline inverse_temperature(alg::AbstractImportanceSampling) = inverse_temperature(ensemble(alg))

function set_inverse_temperature!(alg::AbstractImportanceSampling, beta::Real)
    ens = ensemble(alg)
    ens isa BoltzmannEnsemble || throw(ArgumentError("set_inverse_temperature! currently supports BoltzmannEnsemble only"))
    alg.ensemble = BoltzmannEnsemble(convert(typeof(ens.beta), beta))
    return alg
end

"""
    attempt_exchange!(pt, beta, energy; rng=Random.default_rng(), stage=nothing)

Attempt one MPI neighbor exchange for a rank-local replica. The method
exchanges temperatures (`beta`), not states.

If `stage` is omitted, alternating even/odd staging is used automatically.
"""
function attempt_exchange!(pt::ParallelTempering,
                           beta::Real,
                           energy::Real;
                           rng::AbstractRNG=Random.default_rng(),
                           stage=nothing)
    st = stage === nothing ? pt.stage : Int(stage)
    partner = exchange_partner(pt, st)

    if partner < 0
        if stage === nothing
            pt.stage = 1 - pt.stage
        end
        return (accepted=false, beta=beta, partner=partner, log_ratio=0.0)
    end

    sendbuf = [float(beta), float(energy)]
    recvbuf = similar(sendbuf)
    tag_state = 100 + st
    MPI.Sendrecv!(sendbuf, partner, tag_state, recvbuf, partner, tag_state, pt.comm)

    beta_partner = recvbuf[1]
    energy_partner = recvbuf[2]
    log_ratio = pt_log_acceptance(beta, beta_partner, energy, energy_partner)

    tag_acc = 200 + st
    if pt.rank < partner
        accepted = (log_ratio >= 0) || (rand(rng) < exp(log_ratio))
        flag = Int32(accepted)
        MPI.Send([flag], partner, tag_acc, pt.comm)
    else
        recv_flag = Vector{Int32}(undef, 1)
        MPI.Recv!(recv_flag, partner, tag_acc, pt.comm)
        accepted = recv_flag[1] == 1
    end

    edge = min(pt.rank, partner) + 1
    _record_exchange!(pt.stats, edge, accepted)

    if stage === nothing
        pt.stage = 1 - pt.stage
    end

    beta_new = accepted ? beta_partner : beta
    return (accepted=accepted, beta=beta_new, partner=partner, log_ratio=log_ratio)
end

"""
    attempt_exchange!(pt, alg, energy; rng=alg.rng, stage=nothing)

MPI neighbor exchange for an importance-sampling algorithm with Boltzmann
ensemble. On acceptance the algorithm temperature is updated in-place.
"""
function attempt_exchange!(pt::ParallelTempering, alg::AbstractImportanceSampling, energy::Real; rng::AbstractRNG=alg.rng, stage=nothing)
    beta = inverse_temperature(alg)
    out = attempt_exchange!(pt, beta, energy; rng=rng, stage=stage)
    if out.accepted
        set_inverse_temperature!(alg, out.beta)
    end
    return out
end

"""
    attempt_exchange_pairs!(rng, betas, energies, stage; stats=nothing)

Perform nearest-neighbor exchanges in a local replica array using an
even/odd stage. Useful for threaded runs where multiple replicas live
inside one process.
"""
function attempt_exchange_pairs!(rng::AbstractRNG,
                                 betas::AbstractVector{<:Real},
                                 energies::AbstractVector{<:Real},
                                 stage::Integer;
                                 stats::Union{Nothing,ExchangeStats}=nothing)
    length(betas) == length(energies) || throw(ArgumentError("betas and energies must have the same length"))
    first_i = iseven(Int(stage)) ? 1 : 2
    last_i = length(betas) - 1
    @inbounds for i in first_i:2:last_i
        j = i + 1
        log_ratio = pt_log_acceptance(betas[i], betas[j], energies[i], energies[j])
        accepted = (log_ratio >= 0) || (rand(rng) < exp(log_ratio))
        if accepted
            betas[i], betas[j] = betas[j], betas[i]
        end
        if stats !== nothing
            _record_exchange!(stats, i, accepted)
        end
    end
    return betas
end

"""
    attempt_exchange_pairs!(rng, algs, energies, stage; stats, labels)

Perform nearest-neighbor exchanges, swapping `ensemble` fields directly
between adjacent algorithm objects. No new ensemble objects are allocated.

When `labels` is provided (integer vector with `labels[r]` = original
β-ladder slot for replica `r`), it is swapped alongside the ensembles.
Stats are recorded per original adjacent temperature pair: only when
the two slots currently hold adjacent ladder indices (`|labels[i]-labels[j]|==1`).
"""
function attempt_exchange_pairs!(rng::AbstractRNG,
                                    algs::AbstractVector{<:AbstractImportanceSampling},
                                    energies::AbstractVector{<:Real},
                                    stage::Integer;
                                    stats::Union{Nothing,ExchangeStats}=nothing,
                                    labels::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    n = length(algs)
    n == length(energies) || throw(ArgumentError("algs and energies must have the same length"))
    first_i = iseven(Int(stage)) ? 1 : 2
    @inbounds for i in first_i:2:(n - 1)
        j = i + 1
        βi = inverse_temperature(algs[i])
        βj = inverse_temperature(algs[j])
        log_ratio = pt_log_acceptance(βi, βj, energies[i], energies[j])
        accepted = (log_ratio >= 0) || (rand(rng) < exp(log_ratio))
        if accepted
            ensi = algs[i].ensemble
            algs[i].ensemble = algs[j].ensemble
            algs[j].ensemble = ensi
            if labels !== nothing
                labeli = labels[i]
                labels[i] = labels[j]
                labels[j] = labeli
            end
        end
        if stats !== nothing
            if labels !== nothing
                lo, hi = minmax(labels[i], labels[j])
                hi == lo + 1 && _record_exchange!(stats, lo, accepted)
            else
                _record_exchange!(stats, i, accepted)
            end
        end
    end
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
        step = scaled[i - 1]
        betas[i] = descending ? beta0 - sum(view(scaled, 1:(i - 1))) : beta0 + sum(view(scaled, 1:(i - 1)))
    end

    return betas
end