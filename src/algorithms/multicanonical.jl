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

TODO: 
- implement check that new state is within the domain of the log-weight table and define behavior if not (e.g. reject move)
"""
mutable struct Multicanonical{RNG<:AbstractRNG} <: AbstractGeneralizedEnsemble
    rng::RNG
    logweight::BinnedLogWeight
    histogram::BinnedLogWeight # highjack logweight type for histogram, but should be compatible with logweight type
    steps::Int
    accepted::Int
end

function Multicanonical(rng::AbstractRNG, logweight)
    logweight isa BinnedLogWeight ||
        throw(ArgumentError("`logweight` must be a `BinnedLogWeight`"))
    histogram = zero(logweight)
    return Multicanonical{typeof(rng)}(rng, logweight, histogram, 0, 0)
end
Multicanonical(logweight) = Multicanonical(Random.GLOBAL_RNG, logweight)

# dispatch the accept function so that histogram is updated on every call to `accept`
function accept!(alg::Multicanonical, x_new::Real, x_old::Real)
    # catch outside-of-domain moves and reject them (not efficient but simple to implement for now)
    # this should could be handled within the logweight type?
    idx_new = _binindex(alg.histogram.bins[1], x_new)
    if idx_new < 1 || idx_new > size(alg.histogram.weights, 1)
        alg.histogram[x_old] += 1
        alg.steps += 1
        return false
    end
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
    set_logweight!(alg::Multicanonical, range, f)

Set multicanonical log-weights in a selected value range by evaluating `f`
on each bin center.

`range` can be either `(left, right)` or any `AbstractRange` whose endpoints
define the interval.

For each center `x` in the selected interval, this applies
`alg.logweight[x] = f(x)`.
"""
function set_logweight!(
    alg::Multicanonical,
    xrange::Union{Tuple{<:Real,<:Real},AbstractRange{<:Real}},
    f::Function,
)
    length(size(alg.logweight.weights)) == 1 ||
        throw(ArgumentError("`set_logweight!` currently supports only 1D binned log-weights"))

    centers = alg.logweight.bins[1].centers
    n = length(centers)

    xleft, xright = if xrange isa Tuple
        Float64(min(xrange[1], xrange[2])), Float64(max(xrange[1], xrange[2]))
    else
        Float64(min(first(xrange), last(xrange))), Float64(max(first(xrange), last(xrange)))
    end

    idx_left = clamp(searchsortedfirst(centers, xleft), 1, n)
    idx_right = clamp(searchsortedlast(centers, xright), 1, n)
    idx_left <= idx_right ||
        throw(ArgumentError("selected range does not overlap any bin centers"))

    if centers[idx_left] > xright || centers[idx_right] < xleft
        throw(ArgumentError("selected range does not overlap any bin centers"))
    end

    @inbounds for i in idx_left:idx_right
        x = centers[i]
        alg.logweight.weights[i] = Float64(f(x))
    end

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
        alg.logweight.weights[idx] -= logh
    end

    return nothing
end