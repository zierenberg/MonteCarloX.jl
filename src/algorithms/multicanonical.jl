# Multicanonical / generalized-ensemble scaffolding

using Random

"""
    Multicanonical <: AbstractImportanceSampling

Generalized-ensemble importance sampling algorithm.

The algorithm tracks standard acceptance statistics and stores a tabulated
log-weight estimate.

Notation used here:
- `Ω(E)`: density of states
- `S(E) = log Ω(E)`: microcanonical entropy estimate
- `logWeight(E) = S(E)` stored in `alg.logweight`
"""
mutable struct Multicanonical{LW,RNG<:AbstractRNG} <: AbstractGeneralizedEnsemble
    rng::RNG
    logweight::LW
    steps::Int
    accepted::Int
end

function Multicanonical(rng::AbstractRNG, logweight)
    logweight isa TabulatedLogWeight ||
        throw(ArgumentError("`logweight` must be a `TabulatedLogWeight`"))
    return Multicanonical{typeof(logweight),typeof(rng)}(rng, logweight, 0, 0)
end
Multicanonical(logweight) = Multicanonical(Random.GLOBAL_RNG, logweight)

@inline function _assert_same_bins(log_weight::Histogram, histogram::Histogram)
    if log_weight.edges != histogram.edges
        throw(ArgumentError("`log_weight` and `histogram` must have identical bin edges"))
    end
    return nothing
end

"""
    update_weights!(alg::Multicanonical, histogram::Histogram; mode=:simple)

Update multicanonical log-weights from a sampling histogram in-place.

Currently implemented mode:
- `:simple`: for visited bins, perform
    `logWeight(E) <- logWeight(E) - log(H(E))`.

This mutates `alg.logweight` and returns `nothing`.
"""
function update_weights!(
    alg::Multicanonical,
    histogram::Histogram;
    mode::Symbol = :simple,
)
    if !(alg.logweight isa TabulatedLogWeight)
        throw(ArgumentError("`alg.logweight` must be a TabulatedLogWeight"))
    end

    log_weight = alg.logweight.histogram

    if mode != :simple
        throw(ArgumentError("unsupported mode=$(mode), currently only :simple"))
    end

    _assert_same_bins(log_weight, histogram)

    log_hist = similar(log_weight.weights, Float64)
    @inbounds for idx in eachindex(histogram.weights)
        h = histogram.weights[idx]
        log_hist[idx] = h > 0 ? log(h) : 0.0
    end

    log_weight.weights .-= log_hist

    return nothing
end

update_weights(args...; kwargs...) = update_weights!(args...; kwargs...)
