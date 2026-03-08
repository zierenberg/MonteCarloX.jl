struct MulticanonicalLogWeight{BO<:BinnedObject} <: AbstractLogWeight
    logweight::BO
    histogram::BO

    function MulticanonicalLogWeight(logweight::BO, histogram::BO) where {BO<:BinnedObject}
        _assert_same_domain(logweight, histogram)
        new{BO}(logweight, histogram)
    end
end

function MulticanonicalLogWeight(bins; init::Real=0.0)
    logweight = BinnedObject(bins, float(init))
    histogram = zero(logweight)
    return MulticanonicalLogWeight(logweight, histogram)
end

@inline function (lw::MulticanonicalLogWeight)(x)
    if x isa Tuple
        return lw.logweight(x...)
    else
        return lw.logweight(x)
    end
end

"""
    set!(lw::MulticanonicalLogWeight, range, f)

Set multicanonical log-weights in a selected value range by evaluating `f`
on each bin center.

`range` can be either `(left, right)` or any `AbstractRange` whose endpoints
define the interval.
"""
function set!(
    lw::MulticanonicalLogWeight,
    xrange::Union{Tuple{<:Real,<:Real},AbstractRange{<:Real}},
    f::Function,
)
    length(size(lw.logweight.values)) == 1 ||
        throw(ArgumentError("`set!` currently supports only 1D binned log-weights"))

    centers = collect(lw.logweight.bins[1])
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
        lw.logweight.values[i] = Float64(f(x))
    end

    return nothing
end

"""
    update!(lw::MulticanonicalLogWeight; mode=:simple)

Update multicanonical log-weights from the sampling histogram in-place.
"""
function update!(
    lw::MulticanonicalLogWeight;
    mode::Symbol = :simple,
)
    if mode != :simple
        throw(ArgumentError("unsupported mode=$(mode), currently only :simple"))
    end

    @inbounds for idx in eachindex(lw.histogram.values)
        h = lw.histogram.values[idx]
        logh = h > 0 ? log(h) : 0.0
        lw.logweight.values[idx] -= logh
    end

    return nothing
end

update_weight!(lw::MulticanonicalLogWeight; mode::Symbol=:simple) = update!(lw; mode=mode)