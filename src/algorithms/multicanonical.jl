# Multicanonical / generalized-ensemble scaffolding
using Random
"""
    Multicanonical <: AbstractImportanceSampling

Generalized-ensemble importance sampling algorithm.

The algorithm tracks standard acceptance statistics and stores a tabulated
log-weight estimate.

Notation used here:
- `Ω(x)`: density of states
- `S(x) = log Ω(x)`: microcanonical entropy estimate
- `logWeight(x) = -S(x)` stored in `alg.logweight`
"""
mutable struct Multicanonical{RNG<:AbstractRNG} <: AbstractGeneralizedEnsemble
    rng::RNG
    logweight::TabulatedLogWeight
    histogram::Histogram
    steps::Int
    accepted::Int
end

function Multicanonical(rng::AbstractRNG, logweight)
    logweight isa TabulatedLogWeight ||
        throw(ArgumentError("`logweight` must be a `TabulatedLogWeight`"))
    histogram = zero(logweight.histogram)
    return Multicanonical{typeof(rng)}(rng, logweight, histogram, 0, 0)
end
Multicanonical(logweight) = Multicanonical(Random.GLOBAL_RNG, logweight)

# dispatch the accept function so that histogram is updated on every call to `accept`
function accept!(alg::Multicanonical, x_new::Real, x_old::Real)
    log_ratio = alg.logweight(x_new) - alg.logweight(x_old)
    accepted = _accept!(alg, log_ratio)
    if accepted
        alg.histogram[x_new] += 1
    else
        alg.histogram[x_old] += 1
    end
    return accepted
end

function reset!(alg::Multicanonical)
    fill!(alg.histogram.weights, 0.0)
    alg.steps = 0
    alg.accepted = 0
    return nothing
end

"""
    update_weights!(
    alg::Multicanonical,
    mode::Symbol = :simple,
)

Update multicanonical log-weights from a sampling histogram in-place.

Currently implemented mode:
- `:simple`: for visited bins, perform
    `logWeight(E) <- logWeight(E) - log(H(E))`.

This mutates `alg.logweight` and returns `nothing`.
"""
function update_weight!(
    alg::Multicanonical;
    mode::Symbol = :simple,
)
    if mode != :simple
        throw(ArgumentError("unsupported mode=$(mode), currently only :simple"))
    end

    @inbounds for idx in eachindex(alg.histogram.weights)
        h = alg.histogram.weights[idx]
        logh = h > 0 ? log(h) : 0.0
        alg.logweight.histogram.weights[idx] -= logh
    end

    return nothing
end