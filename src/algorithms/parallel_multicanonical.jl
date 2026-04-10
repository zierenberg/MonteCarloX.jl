"""
    ParallelMulticanonical

Construct a parallel multicanonical coordinator from either a message backend
(e.g. `MPIBackend`) or a local vector of per-replica algorithms.
"""

mutable struct ParallelMulticanonicalMessage <: AbstractAlgorithm
    alg::AbstractImportanceSampling
    backend::AbstractMessageBackend
    root::Int
end

mutable struct ParallelMulticanonicalVector{A<:AbstractImportanceSampling} <: AbstractAlgorithm
    alg::Vector{A}
end

function _validate_parallel_multicanonical_alg(alg::AbstractImportanceSampling)
    ens = ensemble(alg)
    hasproperty(ens, :histogram) || throw(ArgumentError("parallel multicanonical requires an ensemble with `histogram`"))
    hasproperty(ens, :logweight) || throw(ArgumentError("parallel multicanonical requires an ensemble with `logweight`"))

    hist = getproperty(ens, :histogram)
    lw = getproperty(ens, :logweight)
    hasproperty(hist, :values) || throw(ArgumentError("`histogram` must provide a `values` array"))
    hasproperty(lw, :values) || throw(ArgumentError("`logweight` must provide a `values` array"))
    length(hist.values) == length(lw.values) || throw(ArgumentError("`histogram.values` and `logweight.values` must have matching lengths"))
    return alg
end

function ParallelMulticanonicalMessage(backend::AbstractMessageBackend, alg::AbstractImportanceSampling; root::Int=0)
    n = size(backend)
    0 <= root < n || throw(ArgumentError("`root` must satisfy 0 <= root < size"))
    return ParallelMulticanonicalMessage(_validate_parallel_multicanonical_alg(alg), backend, Int(root))
end

function ParallelMulticanonicalVector(algs::AbstractVector{<:AbstractImportanceSampling})
    length(algs) >= 1 || throw(ArgumentError("need at least 1 algorithm"))
    validated = map(_validate_parallel_multicanonical_alg, collect(algs))
    return ParallelMulticanonicalVector(validated)
end

ParallelMulticanonical(backend::AbstractMessageBackend, alg::AbstractImportanceSampling; root::Int=0) =
    ParallelMulticanonicalMessage(backend, alg; root=root)
ParallelMulticanonical(alg::AbstractImportanceSampling, backend::AbstractMessageBackend; root::Int=0) =
    ParallelMulticanonicalMessage(backend, alg; root=root)
ParallelMulticanonical(alg::AbstractImportanceSampling, mode::Symbol; root::Int=0, kwargs...) =
    ParallelMulticanonical(init(mode; kwargs...), alg; root=root)
ParallelMulticanonical(algs::AbstractVector{<:AbstractImportanceSampling}) =
    ParallelMulticanonicalVector(algs)

@inline is_root(pmuca::ParallelMulticanonicalMessage) = rank(pmuca.backend) == pmuca.root
@inline is_root(::ParallelMulticanonicalVector) = true

@inline rank(pmuca::ParallelMulticanonicalMessage) = rank(pmuca.backend)
@inline size(pmuca::ParallelMulticanonicalMessage) = size(pmuca.backend)
@inline rank(::ParallelMulticanonicalVector) = 0
@inline size(pmuca::ParallelMulticanonicalVector) = length(pmuca.alg)

# Collective communication wrappers
barrier(pmuca::ParallelMulticanonicalMessage) = barrier(pmuca.backend)
barrier(::ParallelMulticanonicalVector) = nothing

allgather(value, pmuca::ParallelMulticanonicalMessage) = allgather(value, pmuca.backend)
allgather(value, pmuca::ParallelMulticanonicalVector) = [value]

allreduce!(values, op, pmuca::ParallelMulticanonicalMessage) = allreduce!(values, op, pmuca.backend)
allreduce!(values, op, ::ParallelMulticanonicalVector) = values

reduce(values, op, root::Int, pmuca::ParallelMulticanonicalMessage) = reduce(values, op, root, pmuca.backend)
reduce(values, op, root::Int, ::ParallelMulticanonicalVector) = copy(values)

bcast!(values, root::Int, pmuca::ParallelMulticanonicalMessage) = bcast!(values, root, pmuca.backend)
bcast!(values, root::Int, ::ParallelMulticanonicalVector) = values

gather(value, pmuca::ParallelMulticanonicalMessage; root::Int) = gather(value, pmuca.backend; root=root)
gather(value, pmuca::ParallelMulticanonicalVector; root::Int) = [value]

# --- merge_histograms! ---

function merge_histograms!(pmuca::ParallelMulticanonicalMessage)
    barrier(pmuca)
    allreduce!(ensemble(pmuca.alg).histogram.values, +, pmuca)
    barrier(pmuca)
    return nothing
end

function merge_histograms!(pmuca::ParallelMulticanonicalVector)
    merged = sum(ensemble(alg).histogram.values for alg in pmuca.alg)
    for alg in pmuca.alg
        ensemble(alg).histogram.values .= merged
    end
    return nothing
end

# --- distribute_logweight! ---

function distribute_logweight!(pmuca::ParallelMulticanonicalMessage)
    barrier(pmuca)
    bcast!(ensemble(pmuca.alg).logweight.values, pmuca.root, pmuca)
    barrier(pmuca)
    return nothing
end

function distribute_logweight!(pmuca::ParallelMulticanonicalVector)
    root_lw = ensemble(pmuca.alg[1]).logweight.values
    for alg in pmuca.alg[2:end]
        ensemble(alg).logweight.values .= root_lw
    end
    return nothing
end
